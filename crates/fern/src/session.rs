//! The shared compilation session.
//!
//! A session owns source files and memoizes per-file parse results by
//! revision. All front-ends (batch, REPL, LSP) go through this type so
//! there is exactly one notion of "the current program".
//!
//! Queries are pure functions of interned inputs. Incrementality is
//! deliberately coarse for now: parse results are cached per file, and
//! the checked/lowered program is cached against the revisions of the
//! files that make up the unit.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::Program;
use crate::defs::Defs;
use crate::hir::HirProgram;
use crate::intern::Interner;
use crate::span::{self, Diag, FileId, KResult};
use crate::{check, defs, lexer, lower, parser, prelude, MirModule};

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub program: Program,
    pub diags: Vec<Diag>,
    pub incomplete: bool,
}

struct SourceFile {
    path: String,
    text: String,
    revision: u64,
}

struct ParseCache {
    revision: u64,
    result: Rc<ParseResult>,
}

struct UnitCache {
    key: Vec<(FileId, u64)>,
    module: Rc<MirModule>,
}

/// The checked program: collected definitions plus HIR. Front-ends that
/// only need diagnostics (LSP) stop here; batch and REPL lower it.
pub struct Analysis {
    pub defs: Rc<Defs>,
    pub hir: Rc<HirProgram>,
}

struct AnalysisCache {
    key: Vec<(FileId, u64)>,
    analysis: Rc<Analysis>,
}

pub struct Session {
    files: Vec<SourceFile>,
    parse_cache: RefCell<HashMap<FileId, ParseCache>>,
    analysis_cache: RefCell<Option<AnalysisCache>>,
    unit_cache: RefCell<Option<UnitCache>>,
    intern: RefCell<Interner>,
    unit_compiles: Cell<u64>,
}

impl Default for Session {
    fn default() -> Self {
        Session::new()
    }
}

impl Session {
    /// A session with the built-in prelude as file zero.
    pub fn new() -> Session {
        Session {
            files: vec![SourceFile {
                path: "<prelude>".to_string(),
                text: prelude::SOURCE.to_string(),
                revision: 0,
            }],
            parse_cache: RefCell::new(HashMap::new()),
            analysis_cache: RefCell::new(None),
            unit_cache: RefCell::new(None),
            intern: RefCell::new(Interner::new()),
            unit_compiles: Cell::new(0),
        }
    }

    pub fn add_file(&mut self, path: impl Into<String>, text: impl Into<String>) -> FileId {
        let file = FileId(self.files.len() as u32);
        self.files.push(SourceFile {
            path: path.into(),
            text: text.into(),
            revision: 0,
        });
        file
    }

    pub fn set_text(&mut self, file: FileId, text: impl Into<String>) {
        let entry = &mut self.files[file.index()];
        entry.text = text.into();
        entry.revision += 1;
        self.parse_cache.borrow_mut().remove(&file);
        *self.analysis_cache.borrow_mut() = None;
        *self.unit_cache.borrow_mut() = None;
    }

    pub fn text(&self, file: FileId) -> &str {
        &self.files[file.index()].text
    }

    pub fn path(&self, file: FileId) -> &str {
        &self.files[file.index()].path
    }

    pub fn revision(&self, file: FileId) -> u64 {
        self.files[file.index()].revision
    }

    pub fn prelude(&self) -> FileId {
        FileId::PRELUDE
    }

    /// Parse one file, reusing the cached result while its revision is
    /// unchanged.
    pub fn parse(&self, file: FileId) -> Rc<ParseResult> {
        if let Some(cached) = self.parse_cache.borrow().get(&file) {
            if cached.revision == self.revision(file) {
                return cached.result.clone();
            }
        }
        let text = self.text(file);
        let result = Rc::new(match lexer::lex_in(file, text) {
            Ok(tokens) => {
                let parsed = parser::parse_recovering(tokens);
                ParseResult {
                    program: parsed.program,
                    diags: parsed.diags,
                    incomplete: parsed.incomplete,
                }
            }
            Err(diag) => ParseResult {
                program: Program { items: Vec::new() },
                incomplete: diag.incomplete,
                diags: vec![diag],
            },
        });
        self.parse_cache.borrow_mut().insert(
            file,
            ParseCache {
                revision: self.revision(file),
                result: result.clone(),
            },
        );
        result
    }

    /// Merge the parsed items of `files`, in order, into one program.
    pub fn program(&self, files: &[FileId]) -> (Program, Vec<Diag>, bool) {
        let mut items = Vec::new();
        let mut diags = Vec::new();
        let mut incomplete = false;
        for &file in files {
            let parsed = self.parse(file);
            diags.extend(parsed.diags.iter().cloned());
            incomplete |= parsed.incomplete;
            items.extend(parsed.program.items.iter().cloned());
        }
        (Program { items }, diags, incomplete)
    }

    /// Parse and check `files`, without lowering. This is the query LSP
    /// consumes: it needs diagnostics and HIR, not machine code.
    pub fn analyze(&self, files: &[FileId]) -> Result<Rc<Analysis>, Vec<Diag>> {
        let key: Vec<(FileId, u64)> = files.iter().map(|&f| (f, self.revision(f))).collect();
        if let Some(cached) = self.analysis_cache.borrow().as_ref() {
            if cached.key == key {
                return Ok(cached.analysis.clone());
            }
        }

        let (program, diags, _) = self.program(files);
        if !diags.is_empty() {
            return Err(diags);
        }
        self.unit_compiles.set(self.unit_compiles.get() + 1);
        let mut collected = defs::collect(&program).map_err(|d| vec![d])?;
        let hir = check::check(&mut collected, &program).map_err(|d| vec![d])?;
        let analysis = Rc::new(Analysis {
            defs: Rc::new(collected),
            hir: Rc::new(hir),
        });
        *self.analysis_cache.borrow_mut() = Some(AnalysisCache {
            key,
            analysis: analysis.clone(),
        });
        Ok(analysis)
    }

    /// Compile `files` as one unit. Returns every diagnostic on failure;
    /// batch callers report the first one.
    pub fn compile(&self, files: &[FileId]) -> Result<MirModule, Vec<Diag>> {
        let key: Vec<(FileId, u64)> = files.iter().map(|&f| (f, self.revision(f))).collect();
        if let Some(cached) = self.unit_cache.borrow().as_ref() {
            if cached.key == key {
                return Ok((*cached.module).clone());
            }
        }

        let analysis = self.analyze(files)?;
        let module = lower::lower(&analysis.defs, &analysis.hir).map_err(|d| vec![d])?;
        *self.unit_cache.borrow_mut() = Some(UnitCache {
            key,
            module: Rc::new(module.clone()),
        });
        Ok(module)
    }

    /// Render a diagnostic using the session's file table.
    pub fn render(&self, diag: &Diag) -> String {
        let sources: Vec<&str> = self.files.iter().map(|f| f.text.as_str()).collect();
        span::render_in(&sources, diag)
    }

    /// Stable hash of a file's text, for callers that key caches by
    /// content rather than revision.
    pub fn intern_path(&self, path: &str) -> crate::intern::Sym {
        self.intern.borrow_mut().intern(path)
    }
}

/// One-shot convenience: compile `src` with the prelude.
pub fn compile_str(src: &str) -> KResult<MirModule> {
    let mut session = Session::new();
    let file = session.add_file("<input>", src);
    session
        .compile(&[session.prelude(), file])
        .map_err(|mut diags| diags.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(session: &Session, file: FileId) -> Vec<FileId> {
        vec![session.prelude(), file]
    }

    #[test]
    fn parse_is_cached_until_the_revision_changes() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun main() = 1");
        let first = session.parse(file);
        let second = session.parse(file);
        assert!(Rc::ptr_eq(&first, &second));

        session.set_text(file, "fun main() = 2");
        let third = session.parse(file);
        assert!(!Rc::ptr_eq(&second, &third));
    }

    #[test]
    fn unchanged_unit_does_not_recompile() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun main() = 1");
        let files = unit(&session, file);

        session.compile(&files).expect("compiles");
        session.compile(&files).expect("compiles again");
        assert_eq!(session.unit_compiles.get(), 1);

        session.set_text(file, "fun main() = 2");
        let files = unit(&session, file);
        session.compile(&files).expect("compiles after edit");
        assert_eq!(session.unit_compiles.get(), 2);
    }

    #[test]
    fn analysis_is_reused_for_an_unchanged_unit() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun main() = 1");
        let files = unit(&session, file);
        let first = session.analyze(&files).expect("checks");
        let second = session.analyze(&files).expect("checks again");
        assert!(Rc::ptr_eq(&first, &second));
        assert_eq!(session.unit_compiles.get(), 1);

        session.set_text(file, "fun main() = 2");
        let third = session.analyze(&unit(&session, file)).expect("checks");
        assert!(!Rc::ptr_eq(&first, &third));
    }

    #[test]
    fn session_surfaces_recovered_items_and_diags() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun broken( = 1\nfun ok() = 2\n");
        let parsed = session.parse(file);
        assert!(!parsed.diags.is_empty());
        let names: Vec<&str> = parsed
            .program
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                crate::ast::ItemKind::Fun(fun) => Some(fun.name.as_str()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"ok"), "{names:?}");
    }

    #[test]
    fn multiple_files_merge_in_order() {
        let mut session = Session::new();
        let a = session.add_file("a.fern", "fun twice(x) = x * 2");
        let b = session.add_file("b.fern", "fun main() = twice(21)");
        let files = [session.prelude(), a, b];
        let (program, diags, _) = session.program(&files);
        assert!(diags.is_empty(), "{diags:?}");
        let user_items = program
            .items
            .iter()
            .filter(|item| item.span.file == a || item.span.file == b)
            .count();
        assert_eq!(user_items, 2);
        let module = session.compile(&files).unwrap_or_else(|d| {
            panic!("compile failed: {}", d[0].msg);
        });
        assert!(module.funcs.iter().any(|f| f.name == "main"));
    }
}
