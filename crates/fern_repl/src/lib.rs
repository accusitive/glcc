//! fern REPL core.
//!
//! Definitions accumulate in one session file. Expressions are wrapped in
//! a generated no-argument function, checked, lowered, and added to a
//! persistent JIT image ([`fern_exec::ExecSession`]) that keeps the
//! runtime and every earlier definition compiled. The core never touches
//! stdin/stdout so it can be driven by tests and, later, by the shell
//! front-end.
//!
//! Results print for `Int`, `Float`, `Bool`, `String`, and `Unit`; other
//! values report their type. Definitions persist in one session, and
//! failed input rolls back to the last good program.

use fern::ast::ItemKind;
use fern::hir::BodyKey;
use fern::span::{FileId, Span};
use fern::ty::Ty;
use fern_exec::ExecSession;

pub struct Repl {
    session: fern::Session,
    file: FileId,
    modules: Vec<FileId>,
    source: String,
    next_expr: u32,
    exec: ExecSession,
}

impl Default for Repl {
    fn default() -> Self {
        Repl::new()
    }
}

impl Repl {
    pub fn new() -> Repl {
        let mut session = fern::Session::new();
        let file = session.add_file("<repl>", "");
        Repl {
            session,
            file,
            modules: Vec::new(),
            source: String::new(),
            next_expr: 0,
            exec: ExecSession::new(),
        }
    }

    /// Set the directory used to resolve `import` paths.
    pub fn set_module_root(&mut self, root: impl Into<std::path::PathBuf>) {
        self.session.set_module_root(root);
    }

    /// Validate and persist a definition. Returns the signature lines to
    /// show. On any error the previous definitions are restored.
    pub fn define(&mut self, chunk: &str) -> Result<Vec<String>, Vec<fern::Diag>> {
        let previous = self.source.clone();
        let modules_before = self.modules.len();
        let (chunk_items, chunk_diags) = parse_chunk(chunk);
        if !chunk_diags.is_empty() {
            return Err(chunk_diags);
        }

        for import in chunk_imports(chunk) {
            match self.session.load_import(import) {
                Ok(file) => {
                    if !self.modules.contains(&file) {
                        self.modules.push(file);
                    }
                }
                Err(diags) => {
                    self.modules.truncate(modules_before);
                    return Err(diags);
                }
            }
        }

        let current = parse_chunk(&self.source).0;
        let mut redefined = false;
        for defined in &chunk_items {
            if let Some(item) = current.iter().find(|item| item.name == defined.name) {
                blank_range(&mut self.source, item.span);
                redefined = true;
            }
        }
        if !self.source.is_empty() && !self.source.ends_with('\n') {
            self.source.push('\n');
        }
        self.source.push_str(chunk);
        if !self.source.ends_with('\n') {
            self.source.push('\n');
        }
        self.sync();

        let files = self.files();
        match self.session.analyze(&files) {
            Ok(analysis) => {
                if redefined {
                    self.exec.reset();
                    self.session.reset_lowering();
                }
                Ok(chunk_items
                    .iter()
                    .filter_map(|defined| signature_line(&analysis, &defined.name))
                    .collect())
            }
            Err(diags) => {
                self.source = previous;
                self.modules.truncate(modules_before);
                self.sync();
                Err(diags)
            }
        }
    }

    /// Check and run one expression, returning its rendered value.
    pub fn eval_expr(&mut self, expr: &str) -> Result<String, Vec<fern::Diag>> {
        let previous = self.source.clone();
        let name = format!("__repl_{}", self.next_expr);
        self.next_expr += 1;

        if !self.source.is_empty() && !self.source.ends_with('\n') {
            self.source.push('\n');
        }
        self.source.push_str(&format!("let {name} = ({expr})\n"));
        self.sync();

        let files = self.files();
        let analysis = match self.session.analyze(&files) {
            Ok(analysis) => analysis,
            Err(diags) => {
                self.source = previous;
                self.sync();
                return Err(diags);
            }
        };

        let Some(ret) = return_type(&analysis, &name) else {
            self.source = previous;
            self.sync();
            return Err(vec![fern::Diag::new(
                format!("no checked body for `{name}`"),
                Span::new(0, 0),
            )]);
        };

        let module = match self.session.compile_shared(&files) {
            Ok(module) => module,
            Err(diags) => {
                self.source = previous;
                self.sync();
                return Err(diags);
            }
        };

        self.exec.load(&module);
        let value = self.exec.call(&name, &ret).render();
        self.source = previous;
        self.sync();
        Ok(value)
    }

    pub fn clear(&mut self) {
        self.source.clear();
        self.modules.clear();
        self.sync();
        self.exec.reset();
        self.session.reset_lowering();
    }

    /// What the last `eval_expr`'s lowering did. Tests use it to prove
    /// that a heavy library is not re-lowered on every input.
    pub fn lower_stats(&self) -> fern::LowerStats {
        self.session.lower_stats()
    }

    pub fn render(&self, diag: &fern::Diag) -> String {
        self.session.render(diag)
    }

    fn sync(&mut self) {
        self.session.set_text(self.file, self.source.clone());
    }

    fn files(&self) -> Vec<FileId> {
        let mut files = vec![self.session.prelude(), self.file];
        files.extend(self.modules.iter().copied());
        files
    }
}

#[derive(Clone)]
struct NamedSpan {
    name: String,
    span: Span,
}

fn parse_chunk(src: &str) -> (Vec<NamedSpan>, Vec<fern::Diag>) {
    let tokens = match fern::lexer::lex(src) {
        Ok(tokens) => tokens,
        Err(diag) => return (Vec::new(), vec![diag]),
    };
    let parsed = fern::parser::parse_recovering(tokens);
    let items = parsed
        .program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fun(fun) => Some((fun.name.clone(), item.span)),
            ItemKind::Let(value) => Some((value.name.clone(), item.span)),
            ItemKind::Data(data) => Some((data.name.clone(), item.span)),
            ItemKind::Class(class) => Some((class.name.clone(), item.span)),
            ItemKind::Extern(extern_item) => Some((extern_item.name.clone(), item.span)),
            ItemKind::Instance(_) | ItemKind::Import(_) | ItemKind::Use(_) => None,
        })
        .map(|(name, span)| NamedSpan { name, span })
        .collect();
    (items, parsed.diags)
}

fn chunk_imports(src: &str) -> Vec<Vec<String>> {
    let tokens = match fern::lexer::lex(src) {
        Ok(tokens) => tokens,
        Err(_) => return Vec::new(),
    };
    fern::parser::parse_recovering(tokens)
        .program
        .items
        .into_iter()
        .filter_map(|item| match item.kind {
            ItemKind::Import(import) => Some(import.path),
            _ => None,
        })
        .collect()
}

fn blank_range(source: &mut String, span: Span) {
    let lo = span.lo as usize;
    let hi = (span.hi as usize).min(source.len());
    if lo >= hi || lo >= source.len() {
        return;
    }
    let blanks: String = source[lo..hi]
        .chars()
        .map(|c| if c == '\n' { '\n' } else { ' ' })
        .collect();
    source.replace_range(lo..hi, &blanks);
}

fn return_type(analysis: &fern::Analysis, name: &str) -> Option<Ty> {
    if let Some(id) = analysis.defs.fn_ids.get(name) {
        return analysis
            .hir
            .bodies
            .iter()
            .find(|body| body.key == BodyKey::Fn(*id))
            .map(|body| body.ret.clone());
    }
    let id = *analysis.defs.value_ids.get(name)?;
    analysis
        .hir
        .bodies
        .iter()
        .find(|body| body.key == BodyKey::Value(id))
        .map(|body| body.ret.clone())
}

fn signature_line(analysis: &fern::Analysis, name: &str) -> Option<String> {
    if let Some(id) = analysis.defs.fn_ids.get(name) {
        analysis.defs.fns[id.index()]
            .scheme
            .as_ref()
            .map(|scheme| format!("{name} : {}", display_scheme(scheme)))
    } else if let Some(id) = analysis.defs.value_ids.get(name) {
        analysis.defs.values[id.index()]
            .scheme
            .as_ref()
            .map(|scheme| format!("{name} : {}", display_scheme(scheme)))
    } else if analysis.defs.data_ids.contains_key(name) {
        Some(format!("data {name}"))
    } else if analysis.defs.class_ids.contains_key(name) {
        Some(format!("class {name}"))
    } else {
        None
    }
}

fn display_scheme(scheme: &fern::defs::Scheme) -> String {
    let mut names = Vec::new();
    for (index, (var, _)) in scheme.vars.iter().enumerate() {
        names.push((var.as_str(), var_letter(index)));
    }
    let text = rename_vars(&scheme.ty.to_string(), &names);
    if names.is_empty() {
        text
    } else {
        let vars: Vec<String> = names.iter().map(|(_, letter)| letter.clone()).collect();
        format!("forall {}. {text}", vars.join(" "))
    }
}

fn var_letter(index: usize) -> String {
    if index < 26 {
        ((b'a' + index as u8) as char).to_string()
    } else {
        format!("t{index}")
    }
}

/// Replace `t$<digits>` tokens in a rendered type with scheme letters.
fn rename_vars(text: &str, names: &[(&str, String)]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find("t$") {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        let end = tail
            .char_indices()
            .find(|(_, c)| !c.is_ascii_digit() && *c != '$' && *c != 't')
            .map(|(i, _)| i)
            .unwrap_or(tail.len());
        let token = &tail[..end];
        match names.iter().find(|(var, _)| *var == token) {
            Some((_, letter)) => out.push_str(letter),
            None => out.push_str(token),
        }
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

pub fn is_definition(chunk: &str) -> bool {
    let trimmed = chunk.trim_start();
    ["data ", "class ", "instance ", "extern ", "fun ", "let ", "import ", "use "]
        .iter()
        .any(|keyword| trimmed.starts_with(keyword))
}

pub fn input_incomplete(chunk: &str) -> bool {
    let tokens = match fern::lexer::lex(chunk) {
        Ok(tokens) => tokens,
        Err(diag) => return diag.incomplete,
    };
    if tokens.len() == 1 {
        return false;
    }
    if is_definition(chunk) {
        return fern::parser::parse_recovering(tokens).incomplete;
    }
    match fern::parser::parse_expr_fragment(tokens) {
        Ok(_) => false,
        Err(diag) => diag.incomplete,
    }
}
