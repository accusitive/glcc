//! The shared compilation session.
//!
//! A session owns source files and memoizes per-file parse results by
//! revision. All front-ends (batch, REPL, LSP) go through this type so
//! there is exactly one notion of "the current program".
//!
//! Queries are pure functions of interned inputs. Parse results are
//! cached per file; checking and lowering are incremental across item
//! appends. When a new program keeps a stable item prefix (identified by
//! span, source hash, and kind) and only appends functions, values, or
//! externs, the session rechecks just the new items and the lowerer emits
//! just their bodies. Anything else falls back to a whole-program pass.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::ast::{self, Program};
use crate::defs::{Defs, FnId, ValueId};
use crate::hir::{BodyKey, HirProgram};
use crate::intern::Interner;
use crate::span::{self, Diag, FileId, KResult, Span};
use crate::{MirModule, check, defs, lexer, lower, parser, prelude, resolve};

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub program: Program,
    pub diags: Vec<Diag>,
    pub incomplete: bool,
}

struct SourceFile {
    path: String,
    module_path: Vec<String>,
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

/// Which top-level item a [`ItemKey`] names, for the cheap kinds that an
/// incremental check can append without invalidating earlier items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKindKey {
    Data,
    Class,
    Instance,
    Extern,
    Fun,
    Let,
    Import,
    Use,
}

/// Identity of one top-level item: where it sits and a hash of its
/// source text. Both parts must match for two items to be the same, so
/// an equal-length token edit still invalidates the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemKey {
    pub span: Span,
    pub hash: u64,
    pub kind: ItemKindKey,
}

/// The checked program: collected definitions plus HIR. Front-ends that
/// only need diagnostics (LSP) stop here; batch and REPL lower it.
pub struct Analysis {
    pub defs: Rc<Defs>,
    pub hir: Rc<HirProgram>,
    /// One key per top-level item, in program order.
    pub items: Vec<ItemKey>,
    /// How many trailing items the last check processed. Zero when the
    /// result came straight from the cache.
    pub checked: Cell<usize>,
    /// Whether the last check reused a prefix. A full check may have
    /// changed class dispatch, so it must not resume an old lowering.
    pub(crate) incremental: bool,
    /// Root bodies added by the last check, for an incremental lowerer.
    pub(crate) roots: RefCell<Vec<BodyKey>>,
    /// Names of the root items behind those bodies.
    pub(crate) root_names: RefCell<Vec<String>>,
    /// Import graph and SCCs for this analysis.
    pub module_graph: ModuleGraph,
}

struct AnalysisCache {
    key: Vec<(FileId, u64)>,
    analysis: Rc<Analysis>,
}

/// The lowering state kept between inputs: the MIR module so far, the
/// root names it holds, and the item list it was built from.
struct LowerState {
    items: Vec<ItemKey>,
    roots: HashSet<String>,
    cache: lower::LowerCache,
}

/// One file in the import graph. The prelude is not a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleNode {
    pub file: FileId,
    pub path: Vec<String>,
    pub imports: Vec<FileId>,
    /// Hash of the exported interface: signatures, data/class declarations,
    /// uses/imports, and instances. Function bodies are excluded.
    pub interface: u64,
}

/// A strongly connected component of the import graph. A component with
/// more than one file, or one file with a self-edge, is an import cycle
/// and therefore one codegen unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleGroup {
    pub files: Vec<FileId>,
    /// Combined interface hash of every file in the component.
    pub interface: u64,
}

/// The reachable import graph over a file list. Groups are ordered so that
/// dependencies come before their importers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleGraph {
    pub nodes: Vec<ModuleNode>,
    pub groups: Vec<ModuleGroup>,
}

/// What the last [`Session::compile`] did, for performance tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LowerStats {
    /// True when the whole program was lowered from scratch.
    pub full: bool,
    /// How many root bodies the check selected: the appended ones on an
    /// incremental pass, or every root on a full one.
    pub new_roots: usize,
    /// How many bodies the lowerer actually emitted.
    pub bodies: usize,
}

pub struct Session {
    files: Vec<SourceFile>,
    module_root: PathBuf,
    module_paths: Vec<PathBuf>,
    std_paths: Vec<PathBuf>,
    modules: HashMap<Vec<String>, FileId>,
    module_files: Vec<FileId>,
    loaded_files: HashMap<PathBuf, FileId>,
    parse_cache: RefCell<HashMap<FileId, ParseCache>>,
    analysis_cache: RefCell<Option<AnalysisCache>>,
    unit_cache: RefCell<Option<UnitCache>>,
    lower_state: RefCell<Option<LowerState>>,
    lower_stats: Cell<LowerStats>,
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
                module_path: Vec::new(),
                text: prelude::SOURCE.to_string(),
                revision: 0,
            }],
            module_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            module_paths: env_search_paths(),
            std_paths: default_std_paths(),
            modules: HashMap::new(),
            module_files: Vec::new(),
            loaded_files: HashMap::new(),
            parse_cache: RefCell::new(HashMap::new()),
            analysis_cache: RefCell::new(None),
            unit_cache: RefCell::new(None),
            lower_state: RefCell::new(None),
            lower_stats: Cell::new(LowerStats::default()),
            intern: RefCell::new(Interner::new()),
            unit_compiles: Cell::new(0),
        }
    }

    pub fn add_file(&mut self, path: impl Into<String>, text: impl Into<String>) -> FileId {
        let file = FileId(self.files.len() as u32);
        self.files.push(SourceFile {
            path: path.into(),
            module_path: Vec::new(),
            text: text.into(),
            revision: 0,
        });
        file
    }

    pub fn add_module_file(&mut self, path: impl Into<String>, module_path: Vec<String>, text: impl Into<String>) -> FileId {
        let file = FileId(self.files.len() as u32);
        self.files.push(SourceFile {
            path: path.into(),
            module_path,
            text: text.into(),
            revision: 0,
        });
        file
    }

    pub fn module_path(&self, file: FileId) -> &[String] {
        &self.files[file.index()].module_path
    }

    pub fn module_files(&self) -> &[FileId] {
        &self.module_files
    }

    /// Build the import graph over `files` and group cycles (Tarjan SCC).
    /// The prelude is omitted; the entry/root file and every module are
    /// nodes. Dependencies are ordered before their importers.
    pub fn module_graph(&self, files: &[FileId]) -> ModuleGraph {
        let files: Vec<FileId> = files.iter().copied().filter(|file| *file != FileId::PRELUDE).collect();
        let index_of: HashMap<FileId, usize> = files.iter().enumerate().map(|(i, f)| (*f, i)).collect();
        let by_path: HashMap<Vec<String>, FileId> = files
            .iter()
            .copied()
            .filter_map(|file| {
                let path = self.module_path(file);
                if path.is_empty() { None } else { Some((path.to_vec(), file)) }
            })
            .collect();

        let mut nodes = Vec::with_capacity(files.len());
        let mut adjacency = vec![Vec::new(); files.len()];
        for (i, &file) in files.iter().enumerate() {
            let parsed = self.parse(file);
            let mut imports = Vec::new();
            for item in &parsed.program.items {
                let ast::ItemKind::Import(import) = &item.kind else {
                    continue;
                };
                if let Some(target) = by_path.get(&import.path).copied()
                    && let Some(&target_index) = index_of.get(&target)
                        && !imports.contains(&target) {
                            imports.push(target);
                            adjacency[i].push(target_index);
                        }
            }
            nodes.push(ModuleNode {
                file,
                path: self.module_path(file).to_vec(),
                imports,
                interface: interface_hash(self.text(file), &parsed.program.items, self.module_path(file)),
            });
        }

        let components = tarjan(&adjacency);
        let groups = components
            .into_iter()
            .map(|mut component| {
                component.sort_unstable();
                let interface = component
                    .iter()
                    .fold(0xcbf29ce484222325u64, |hash, i| hash.wrapping_mul(0x100000001b3) ^ nodes[*i].interface);
                ModuleGroup {
                    files: component.into_iter().map(|i| files[i]).collect(),
                    interface,
                }
            })
            .collect();
        ModuleGraph { nodes, groups }
    }

    pub fn set_module_root(&mut self, root: impl Into<PathBuf>) {
        self.module_root = root.into();
    }

    /// Add one directory to the ordinary module search path. It is tried
    /// after the entry directory and before any standard-library root.
    pub fn add_module_search_path(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        if !self.module_paths.contains(&path) {
            self.module_paths.push(path);
        }
    }

    /// Add a directory that contains the contents of `std` directly, so
    /// `import std::foldable` looks for `<path>/foldable.fern`. This is
    /// tried after the entry directory and `FERN_PATH`.
    pub fn add_std_search_path(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        if !self.std_paths.contains(&path) {
            self.std_paths.push(path);
        }
    }

    /// Load an entry file and every module it imports, returning the files
    /// in prelude-independent order: the entry first, then imports.
    pub fn load_entry(&mut self, entry: &Path) -> Result<Vec<FileId>, Vec<Diag>> {
        let canonical = fs::canonicalize(entry).map_err(|error| vec![Diag::new(format!("cannot read `{}`: {error}", entry.display()), Span::new(0, 0))])?;
        self.module_root = canonical.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
        let text = fs::read_to_string(&canonical).map_err(|error| vec![Diag::new(format!("cannot read `{}`: {error}", canonical.display()), Span::new(0, 0))])?;
        let file = self.add_module_file(canonical.display().to_string(), Vec::new(), text);
        if let Err(diag) = self.load_imports(file) {
            return Err(vec![diag]);
        }
        let mut files = vec![FileId::PRELUDE, file];
        files.extend(self.module_files.iter().copied());
        Ok(files)
    }

    /// Load one module into the session. REPL imports use this with the
    /// session's module root; the compiler driver uses [`Session::load_entry`].
    pub fn load_import(&mut self, path: Vec<String>) -> Result<FileId, Vec<Diag>> {
        self.load_module(path, Span::new(0, 0)).map_err(|diag| vec![diag])
    }

    fn load_imports(&mut self, file: FileId) -> KResult<()> {
        let parsed = self.parse(file);
        let imports: Vec<(Vec<String>, Span)> = parsed
            .program
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ast::ItemKind::Import(import) => Some((import.path.clone(), import.span)),
                _ => None,
            })
            .collect();
        for (path, span) in imports {
            self.load_module(path, span)?;
        }
        Ok(())
    }

    fn load_module(&mut self, path: Vec<String>, span: Span) -> KResult<FileId> {
        if let Some(file) = self.modules.get(&path) {
            return Ok(*file);
        }
        if path.is_empty() {
            return Err(Diag::new("an import needs a module path", span));
        }
        let source = self.find_module(&path, span)?;
        let canonical = fs::canonicalize(&source).map_err(|error| Diag::new(format!("cannot read `{}`: {error}", source.display()), span))?;
        if let Some(file) = self.loaded_files.get(&canonical) {
            let loaded = self.module_path(*file).join("::");
            return Err(Diag::new(format!("`{}` is already loaded as module `{loaded}`", canonical.display()), span));
        }
        let text = fs::read_to_string(&canonical).map_err(|error| Diag::new(format!("cannot read `{}`: {error}", canonical.display()), span))?;
        let file = self.add_module_file(canonical.display().to_string(), path.clone(), text);
        self.modules.insert(path, file);
        self.module_files.push(file);
        self.loaded_files.insert(canonical, file);
        self.load_imports(file)?;
        Ok(file)
    }

    fn find_module(&self, path: &[String], span: Span) -> KResult<PathBuf> {
        let mut tried = Vec::new();
        if let Some(found) = find_in_root(&self.module_root, path, &mut tried) {
            return Ok(found);
        }
        for root in &self.module_paths {
            if let Some(found) = find_in_root(root, path, &mut tried) {
                return Ok(found);
            }
        }

        if path.first().is_some_and(|segment| segment == "std") && path.len() > 1 {
            for root in &self.std_paths {
                if let Some(found) = find_in_root(root, &path[1..], &mut tried) {
                    return Ok(found);
                }
            }
        }

        let tried = tried.iter().map(|candidate| format!("`{}`", candidate.display())).collect::<Vec<_>>().join(", ");
        Err(Diag::new(format!("cannot find module `{}` (tried {tried})", path.join("::")), span))
    }

    pub fn set_text(&mut self, file: FileId, text: impl Into<String>) {
        let entry = &mut self.files[file.index()];
        entry.text = text.into();
        entry.revision += 1;
        self.parse_cache.borrow_mut().remove(&file);
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
        if let Some(cached) = self.parse_cache.borrow().get(&file)
            && cached.revision == self.revision(file) {
                return cached.result.clone();
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
                program: Program::single(Vec::new()),
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
    ///
    /// Module zero is always the prelude and module one is the root. Both
    /// share an empty canonical prefix, so single-file programs keep using
    /// bare names; their IDs still differ for ownership checks.
    pub fn program(&self, files: &[FileId]) -> (Program, Vec<Diag>, bool) {
        let mut modules = vec![ast::ModuleInfo { path: Vec::new() }, ast::ModuleInfo { path: Vec::new() }];
        let mut module_ids: HashMap<Vec<String>, ast::ModuleId> = HashMap::new();
        let mut items = Vec::new();
        let mut diags = Vec::new();
        let mut incomplete = false;
        for &file in files {
            let parsed = self.parse(file);
            diags.extend(parsed.diags.iter().cloned());
            incomplete |= parsed.incomplete;
            let module = if file == FileId::PRELUDE {
                ast::ModuleId::ROOT
            } else {
                let path = self.module_path(file);
                if path.is_empty() {
                    ast::ModuleId(1)
                } else {
                    *module_ids.entry(path.to_vec()).or_insert_with(|| {
                        let id = ast::ModuleId(modules.len() as u32);
                        modules.push(ast::ModuleInfo { path: path.to_vec() });
                        id
                    })
                }
            };
            items.extend(parsed.program.items.iter().cloned().map(|mut item| {
                item.module = module;
                item
            }));
        }
        (Program { modules, items }, diags, incomplete)
    }

    /// Parse and check `files`, without lowering. This is the query LSP
    /// consumes: it needs diagnostics and HIR, not machine code.
    ///
    /// The check reuses the previous result whenever the new item list
    /// extends a stable prefix with `fun`/`let`/`extern` items. The REPL
    /// relies on this: evaluating an expression appends one generated
    /// `fun`, so only that function is checked.
    pub fn analyze(&self, files: &[FileId]) -> Result<Rc<Analysis>, Vec<Diag>> {
        let key: Vec<(FileId, u64)> = files.iter().map(|&f| (f, self.revision(f))).collect();
        if let Some(cached) = self.analysis_cache.borrow().as_ref()
            && cached.key == key {
                // The roots stay in place: a compile right after analyze
                // must still see which bodies were newly checked.
                cached.analysis.checked.set(0);
                return Ok(cached.analysis.clone());
            }

        let (mut program, diags, _) = self.program(files);
        if !diags.is_empty() {
            return Err(diags);
        }
        let modular =
            program.modules.iter().any(|module| !module.path.is_empty()) || program.items.iter().any(|item| matches!(item.kind, ast::ItemKind::Import(_) | ast::ItemKind::Use(_)));
        let graph = self.module_graph(files);
        if modular {
            resolve::qualify(&mut program).map_err(|diag| vec![diag])?;
        }
        let items = self.item_keys(&program);
        self.unit_compiles.set(self.unit_compiles.get() + 1);

        let previous = self.analysis_cache.borrow().as_ref().map(|c| c.analysis.clone());
        let analysis = match (modular, previous) {
            (true, Some(previous)) => self.analyze_modules(program, items, graph, previous)?,
            (true, None) => self.analyze_full(program, items, graph)?,
            (false, Some(previous)) => self.analyze_from(program, items, graph, previous)?,
            (false, None) => self.analyze_full(program, items, graph)?,
        };
        *self.analysis_cache.borrow_mut() = Some(AnalysisCache { key, analysis: analysis.clone() });
        Ok(analysis)
    }

    fn analyze_full(&self, program: Program, items: Vec<ItemKey>, module_graph: ModuleGraph) -> Result<Rc<Analysis>, Vec<Diag>> {
        let mut collected = defs::collect(&program).map_err(|d| vec![d])?;
        let hir = check::check(&mut collected, &program).map_err(|d| vec![d])?;
        let (roots, names) = roots_of(&program, 0);
        let checked = item_count(&program);
        Ok(Rc::new(Analysis {
            defs: Rc::new(collected),
            hir: Rc::new(hir),
            items,
            checked: Cell::new(checked),
            incremental: false,
            roots: RefCell::new(roots),
            root_names: RefCell::new(names),
            module_graph,
        }))
    }

    fn analyze_from(&self, program: Program, items: Vec<ItemKey>, module_graph: ModuleGraph, previous: Rc<Analysis>) -> Result<Rc<Analysis>, Vec<Diag>> {
        let common = common_prefix(&previous.items, &items);
        if common == items.len() {
            if items.len() == previous.items.len() {
                previous.checked.set(0);
                previous.roots.borrow_mut().clear();
                previous.root_names.borrow_mut().clear();
                return Ok(previous);
            }
            return self.analyze_full(program, items, module_graph);
        }
        if common == 0 || !plain_items(&items[common..]) {
            return self.analyze_full(program, items, module_graph);
        }

        let (fn_count, value_count, _) = item_counts(&program.items[..common]);
        let mut collected = defs::collect(&program).map_err(|d| vec![d])?;
        if !inherit_schemes(&mut collected, &previous.defs, fn_count, value_count) {
            return self.analyze_full(program, items, module_graph);
        }
        let start_hir = trim_hir(&previous, &program, common);
        let hir = match check::check_from(&mut collected, &program, common, start_hir) {
            Ok(hir) => hir,
            Err(diag) => return Err(vec![diag]),
        };
        let (roots, names) = roots_of(&program, common);
        let checked = item_count(&program) - common;
        Ok(Rc::new(Analysis {
            defs: Rc::new(collected),
            hir: Rc::new(hir),
            items,
            checked: Cell::new(checked),
            incremental: true,
            roots: RefCell::new(roots),
            root_names: RefCell::new(names),
            module_graph,
        }))
    }

    fn analyze_modules(&self, program: Program, items: Vec<ItemKey>, module_graph: ModuleGraph, previous: Rc<Analysis>) -> Result<Rc<Analysis>, Vec<Diag>> {
        if !same_item_shape(&items, &previous.items) || !same_module_files(&module_graph, &previous.module_graph) {
            return self.analyze_full(program, items, module_graph);
        }

        let affected = affected_files(&previous.module_graph, &module_graph);
        let mut selected: HashSet<usize> = items
            .iter()
            .zip(&previous.items)
            .enumerate()
            .filter_map(|(index, (current, old))| (current.hash != old.hash).then_some(index))
            .collect();
        for (index, item) in program.items.iter().enumerate() {
            if affected.contains(&item.span.file) {
                selected.insert(index);
            }
        }

        if selected.is_empty() {
            return Ok(Rc::new(Analysis {
                defs: previous.defs.clone(),
                hir: previous.hir.clone(),
                items,
                checked: Cell::new(0),
                incremental: true,
                roots: RefCell::new(Vec::new()),
                root_names: RefCell::new(Vec::new()),
                module_graph,
            }));
        }

        let mut collected = defs::collect(&program).map_err(|d| vec![d])?;
        let start_hir = (*previous.hir).clone();
        let hir = check::check_selected(&mut collected, &program, &selected, start_hir).map_err(|diag| vec![diag])?;
        let (roots, names) = roots_of(&program, 0);
        let checked = selected.len();
        Ok(Rc::new(Analysis {
            defs: Rc::new(collected),
            hir: Rc::new(hir),
            items,
            checked: Cell::new(checked),
            incremental: false,
            roots: RefCell::new(roots),
            root_names: RefCell::new(names),
            module_graph,
        }))
    }

    /// Compile `files` as one unit. Returns every diagnostic on failure;
    /// batch callers report the first one.
    pub fn compile(&self, files: &[FileId]) -> Result<MirModule, Vec<Diag>> {
        Ok((*self.compile_shared(files)?).clone())
    }

    /// [`Session::compile`], returning a shared handle to the module
    /// snapshot instead of an owned copy. Long-lived front-ends (the
    /// REPL) keep the handle for the duration of one input.
    pub fn compile_shared(&self, files: &[FileId]) -> Result<Rc<MirModule>, Vec<Diag>> {
        let key: Vec<(FileId, u64)> = files.iter().map(|&f| (f, self.revision(f))).collect();
        if let Some(cached) = self.unit_cache.borrow().as_ref()
            && cached.key == key {
                return Ok(cached.module.clone());
            }

        let analysis = self.analyze(files)?;
        let previous = self.lower_state.borrow_mut().take();
        let (cache, stats, roots) = self.lower_analysis(&analysis, previous)?;
        let module = Rc::new(cache.module().clone());
        *self.lower_state.borrow_mut() = Some(LowerState {
            items: analysis.items.clone(),
            roots,
            cache,
        });
        *self.unit_cache.borrow_mut() = Some(UnitCache { key, module: module.clone() });
        self.lower_stats.set(stats);
        Ok(module)
    }

    fn lower_analysis(&self, analysis: &Rc<Analysis>, previous: Option<LowerState>) -> Result<(lower::LowerCache, LowerStats, HashSet<String>), Vec<Diag>> {
        let roots = analysis.roots.borrow().clone();
        let new_names: HashSet<String> = analysis.root_names.borrow().iter().cloned().collect();
        if let Some(previous) = previous {
            let common = common_prefix(&previous.items, &analysis.items);
            let append_only =
                analysis.incremental && common > 0 && analysis.items.len() > common && analysis.items.len() >= previous.items.len() && new_names.is_disjoint(&previous.roots);
            if append_only {
                let before = previous.cache.lowered();
                let new_count = new_names.len();
                let cache = lower::lower_more(&analysis.defs, &analysis.hir, previous.cache, &roots).map_err(|d| vec![d])?;
                let mut roots = previous.roots;
                roots.extend(new_names);
                let stats = LowerStats {
                    full: false,
                    new_roots: new_count,
                    bodies: cache.lowered() - before,
                };
                return Ok((cache, stats, roots));
            }
        }
        let cache = lower::lower_with_cache(&analysis.defs, &analysis.hir).map_err(|d| vec![d])?;
        let stats = LowerStats {
            full: true,
            new_roots: roots.len(),
            bodies: cache.lowered(),
        };
        let mut all_roots: HashSet<String> = analysis.defs.fns.iter().map(|f| f.name.clone()).collect();
        all_roots.extend(analysis.defs.values.iter().map(|v| v.name.clone()));
        Ok((cache, stats, all_roots))
    }

    /// Drop the incremental lowering state. The next compile rebuilds the
    /// MIR module from the current HIR. Front-ends call this when a name is
    /// redefined, so a stale body cannot be reused under the same symbol.
    pub fn reset_lowering(&self) {
        *self.lower_state.borrow_mut() = None;
        *self.unit_cache.borrow_mut() = None;
        self.lower_stats.set(LowerStats::default());
    }

    /// What the last [`Session::compile`] did. Tests use it to prove that
    /// an added definition does not re-lower the program.
    pub fn lower_stats(&self) -> LowerStats {
        self.lower_stats.get()
    }

    /// Identity of every top-level item in `program`'s order.
    fn item_keys(&self, program: &Program) -> Vec<ItemKey> {
        program
            .items
            .iter()
            .map(|item| {
                let text = if item.span.file == FileId::UNSPECIFIED {
                    ""
                } else {
                    self.text(item.span.file).get(item.span.lo as usize..item.span.hi as usize).unwrap_or("")
                };
                ItemKey {
                    span: item.span,
                    hash: fnv1a(text.as_bytes()),
                    kind: item_kind(&item.kind),
                }
            })
            .collect()
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
    session.compile(&[session.prelude(), file]).map_err(|mut diags| diags.remove(0))
}

fn same_item_shape(current: &[ItemKey], previous: &[ItemKey]) -> bool {
    current.len() == previous.len() && current.iter().zip(previous).all(|(a, b)| a.span.file == b.span.file && a.kind == b.kind)
}

fn same_module_files(current: &ModuleGraph, previous: &ModuleGraph) -> bool {
    current.nodes.len() == previous.nodes.len() && current.nodes.iter().zip(&previous.nodes).all(|(a, b)| a.file == b.file && a.path == b.path)
}

fn affected_files(previous: &ModuleGraph, current: &ModuleGraph) -> HashSet<FileId> {
    let previous_interfaces: HashMap<FileId, u64> = previous.nodes.iter().map(|node| (node.file, node.interface)).collect();
    let mut affected: HashSet<FileId> = current
        .nodes
        .iter()
        .filter(|node| previous_interfaces.get(&node.file) != Some(&node.interface))
        .map(|node| node.file)
        .collect();
    let mut reverse: HashMap<FileId, Vec<FileId>> = HashMap::new();
    for node in &current.nodes {
        for &import in &node.imports {
            reverse.entry(import).or_default().push(node.file);
        }
    }
    let mut queue: Vec<FileId> = affected.iter().copied().collect();
    while let Some(file) = queue.pop() {
        if let Some(importers) = reverse.get(&file) {
            for &importer in importers {
                if affected.insert(importer) {
                    queue.push(importer);
                }
            }
        }
    }
    affected
}

fn hash_str(hash: &mut u64, text: &str) {
    for byte in text.bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
    *hash ^= 0xff;
    *hash = hash.wrapping_mul(0x100000001b3);
}

fn interface_hash(source: &str, items: &[ast::Item], module_path: &[String]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash_str(&mut hash, &module_path.join("::"));
    for item in items {
        match &item.kind {
            ast::ItemKind::Fun(fun) => {
                hash_str(&mut hash, if fun.is_async { "async fun" } else { "fun" });
                hash_str(&mut hash, &fun.name);
                for param in &fun.params {
                    hash_str(&mut hash, &param.ty.display());
                }
                hash_str(&mut hash, &fun.ret.display());
            }
            ast::ItemKind::Let(value) => {
                hash_str(&mut hash, "let");
                hash_str(&mut hash, &value.name);
                match &value.ty {
                    Some(ty) => hash_str(&mut hash, &ty.display()),
                    None => {
                        let lo = item.span.lo as usize;
                        let hi = (item.span.hi as usize).min(source.len());
                        hash_str(&mut hash, source.get(lo..hi).unwrap_or(""));
                    }
                }
            }
            ast::ItemKind::Data(data) => {
                hash_str(&mut hash, "data");
                hash_str(&mut hash, &data.name);
                for param in &data.params {
                    hash_str(&mut hash, param);
                }
                for ctor in &data.ctors {
                    hash_str(&mut hash, &ctor.name);
                    for field in &ctor.fields {
                        hash_str(&mut hash, &field.display());
                    }
                }
            }
            ast::ItemKind::Class(class) => {
                hash_str(&mut hash, "class");
                hash_str(&mut hash, &class.name);
                for param in &class.params {
                    hash_str(&mut hash, param);
                }
                for method in &class.methods {
                    hash_str(&mut hash, &method.name);
                    hash_str(&mut hash, &method.ty.display());
                }
            }
            ast::ItemKind::Instance(instance) => {
                hash_str(&mut hash, "instance");
                hash_str(&mut hash, &instance.class);
                for bound in &instance.ctx {
                    hash_str(&mut hash, &bound.display());
                }
                hash_str(&mut hash, &instance.self_ty.display());
                for method in &instance.methods {
                    hash_str(&mut hash, &method.name);
                }
            }
            ast::ItemKind::Extern(extern_decl) => {
                hash_str(&mut hash, "extern");
                hash_str(&mut hash, &extern_decl.name);
                hash_str(&mut hash, extern_decl.symbol.as_deref().unwrap_or(""));
                hash_str(&mut hash, &extern_decl.ty.display());
            }
            ast::ItemKind::Import(import) => {
                hash_str(&mut hash, "import");
                hash_str(&mut hash, &import.path.join("::"));
                hash_str(&mut hash, import.alias.as_deref().unwrap_or(""));
            }
            ast::ItemKind::Use(use_decl) => {
                hash_str(&mut hash, "use");
                hash_str(&mut hash, &use_decl.path.join("::"));
                hash_str(&mut hash, use_decl.alias.as_deref().unwrap_or(""));
            }
        }
    }
    hash
}

fn env_search_paths() -> Vec<PathBuf> {
    std::env::var_os("FERN_PATH")
        .map(|value| std::env::split_paths(&value).filter(|path| !path.as_os_str().is_empty()).collect())
        .unwrap_or_default()
}

fn default_std_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = std::env::var_os("FERN_STD") {
        paths.push(PathBuf::from(path));
    }

    let development = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../stdlib/std");
    paths.push(development);

    #[cfg(unix)]
    paths.push(PathBuf::from("/usr/share/fern/std"));

    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
    if let Some(home) = home {
        paths.push(PathBuf::from(home).join(".local/share/fern/std"));
    }
    paths
}

fn find_in_root(root: &Path, path: &[String], tried: &mut Vec<PathBuf>) -> Option<PathBuf> {
    if path.is_empty() {
        return None;
    }
    let joined = path.join("/");
    let leaf = root.join(format!("{joined}.fern"));
    if leaf.is_file() {
        return Some(leaf);
    }
    tried.push(leaf);

    let package = root.join(&joined).join(format!("{}.fern", path.last().expect("non-empty module path")));
    if package.is_file() {
        return Some(package);
    }
    tried.push(package);
    None
}

fn tarjan(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct State<'a> {
        adjacency: &'a [Vec<usize>],
        next: usize,
        indices: Vec<Option<usize>>,
        low: Vec<usize>,
        stack: Vec<usize>,
        on_stack: Vec<bool>,
        out: Vec<Vec<usize>>,
    }

    impl State<'_> {
        fn visit(&mut self, node: usize) {
            let index = self.next;
            self.next += 1;
            self.indices[node] = Some(index);
            self.low[node] = index;
            self.stack.push(node);
            self.on_stack[node] = true;

            for &next in &self.adjacency[node] {
                if self.indices[next].is_none() {
                    self.visit(next);
                    self.low[node] = self.low[node].min(self.low[next]);
                } else if self.on_stack[next] {
                    self.low[node] = self.low[node].min(self.indices[next].expect("visited"));
                }
            }

            if self.low[node] == index {
                let mut component = Vec::new();
                loop {
                    let member = self.stack.pop().expect("stack holds the component");
                    self.on_stack[member] = false;
                    component.push(member);
                    if member == node {
                        break;
                    }
                }
                self.out.push(component);
            }
        }
    }

    let mut state = State {
        adjacency,
        next: 0,
        indices: vec![None; adjacency.len()],
        low: vec![0; adjacency.len()],
        stack: Vec::new(),
        on_stack: vec![false; adjacency.len()],
        out: Vec::new(),
    };
    for node in 0..adjacency.len() {
        if state.indices[node].is_none() {
            state.visit(node);
        }
    }
    state.out
}

fn item_kind(kind: &ast::ItemKind) -> ItemKindKey {
    match kind {
        ast::ItemKind::Data(_) => ItemKindKey::Data,
        ast::ItemKind::Class(_) => ItemKindKey::Class,
        ast::ItemKind::Instance(_) => ItemKindKey::Instance,
        ast::ItemKind::Extern(_) => ItemKindKey::Extern,
        ast::ItemKind::Fun(_) => ItemKindKey::Fun,
        ast::ItemKind::Let(_) => ItemKindKey::Let,
        ast::ItemKind::Import(_) => ItemKindKey::Import,
        ast::ItemKind::Use(_) => ItemKindKey::Use,
    }
}

/// An item that can be appended without re-checking earlier ones: it
/// introduces no instance that the class solver could pick up for code
/// already checked.
fn plain_items(items: &[ItemKey]) -> bool {
    items.iter().all(|item| matches!(item.kind, ItemKindKey::Fun | ItemKindKey::Let | ItemKindKey::Extern))
}

fn common_prefix(left: &[ItemKey], right: &[ItemKey]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

fn item_counts(items: &[ast::Item]) -> (usize, usize, usize) {
    let (mut fns, mut values, mut instances) = (0, 0, 0);
    for item in items {
        match &item.kind {
            ast::ItemKind::Fun(_) => fns += 1,
            ast::ItemKind::Let(_) => values += 1,
            ast::ItemKind::Instance(_) => instances += 1,
            ast::ItemKind::Data(_) | ast::ItemKind::Class(_) | ast::ItemKind::Extern(_) | ast::ItemKind::Import(_) | ast::ItemKind::Use(_) => {}
        }
    }
    (fns, values, instances)
}

fn item_count(program: &Program) -> usize {
    program.items.len()
}

/// Root bodies (and their names) for the items from `from` on. Instance
/// methods are lowered as part of a full lower, so a resumed lower only
/// needs the `fun`/`let` items.
fn roots_of(program: &Program, from: usize) -> (Vec<BodyKey>, Vec<String>) {
    let (mut fn_index, mut value_index, _) = item_counts(&program.items[..from]);
    let mut roots = Vec::new();
    let mut names = Vec::new();
    for item in &program.items[from..] {
        match &item.kind {
            ast::ItemKind::Fun(f) => {
                roots.push(BodyKey::Fn(FnId(fn_index as u32)));
                names.push(f.name.clone());
                fn_index += 1;
            }
            ast::ItemKind::Let(v) => {
                roots.push(BodyKey::Value(ValueId(value_index as u32)));
                names.push(v.name.clone());
                value_index += 1;
            }
            ast::ItemKind::Data(_) | ast::ItemKind::Class(_) | ast::ItemKind::Instance(_) | ast::ItemKind::Extern(_) | ast::ItemKind::Import(_) | ast::ItemKind::Use(_) => {}
        }
    }
    (roots, names)
}

/// Copy the checked schemes of the prefix items into a freshly collected
/// [`Defs`]. Returns false when the prefix names no longer line up and a
/// full check is required.
fn inherit_schemes(collected: &mut Defs, previous: &Defs, fn_count: usize, value_count: usize) -> bool {
    if fn_count > collected.fns.len() || fn_count > previous.fns.len() {
        return false;
    }
    if value_count > collected.values.len() || value_count > previous.values.len() {
        return false;
    }
    for index in 0..fn_count {
        if collected.fns[index].name != previous.fns[index].name {
            return false;
        }
        collected.fns[index].scheme = previous.fns[index].scheme.clone();
    }
    for index in 0..value_count {
        if collected.values[index].name != previous.values[index].name {
            return false;
        }
        collected.values[index].scheme = previous.values[index].scheme.clone();
    }
    true
}

/// The previous HIR with every body that belongs to an item after the
/// shared prefix dropped. Stale closures are kept: ids stay unique and
/// only reachable bodies are lowered.
fn trim_hir(previous: &Analysis, program: &Program, common: usize) -> HirProgram {
    let (fn_count, value_count, instance_count) = item_counts(&program.items[..common]);
    let mut hir = (*previous.hir).clone();
    hir.bodies.retain(|body| match body.key {
        BodyKey::Fn(id) => id.index() < fn_count,
        BodyKey::Value(id) => id.index() < value_count,
        BodyKey::Method { instance, .. } => instance < instance_count,
    });
    hir
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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
        let file = session.add_file("t.fern", "fun main() : Int = 1");
        let first = session.parse(file);
        let second = session.parse(file);
        assert!(Rc::ptr_eq(&first, &second));

        session.set_text(file, "fun main() : Int = 2");
        let third = session.parse(file);
        assert!(!Rc::ptr_eq(&second, &third));
    }

    #[test]
    fn unchanged_unit_does_not_recompile() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun main() : Int = 1");
        let files = unit(&session, file);

        session.compile(&files).expect("compiles");
        session.compile(&files).expect("compiles again");
        assert_eq!(session.unit_compiles.get(), 1);

        session.set_text(file, "fun main() : Int = 2");
        let files = unit(&session, file);
        session.compile(&files).expect("compiles after edit");
        assert_eq!(session.unit_compiles.get(), 2);
    }

    #[test]
    fn analysis_is_reused_for_an_unchanged_unit() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun main() : Int = 1");
        let files = unit(&session, file);
        let first = session.analyze(&files).expect("checks");
        let second = session.analyze(&files).expect("checks again");
        assert!(Rc::ptr_eq(&first, &second));
        assert_eq!(session.unit_compiles.get(), 1);

        session.set_text(file, "fun main() : Int = 2");
        let third = session.analyze(&unit(&session, file)).expect("checks");
        assert!(!Rc::ptr_eq(&first, &third));
    }

    #[test]
    fn session_surfaces_recovered_items_and_diags() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun broken( = 1\nfun ok() : Int = 2\n");
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
        let a = session.add_file("a.fern", "fun twice(x : Int) : Int = x * 2");
        let b = session.add_file("b.fern", "fun main() : Int = twice(21)");
        let files = [session.prelude(), a, b];
        let (program, diags, _) = session.program(&files);
        assert!(diags.is_empty(), "{diags:?}");
        let user_items = program.items.iter().filter(|item| item.span.file == a || item.span.file == b).count();
        assert_eq!(user_items, 2);
        let module = session.compile(&files).unwrap_or_else(|d| {
            panic!("compile failed: {}", d[0].msg);
        });
        assert!(module.funcs.iter().any(|f| f.name == "main"));
    }

    #[test]
    fn appending_a_definition_rechecks_only_it() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun a() : Int = 1\nfun b() : Int = 2\n");
        let files = unit(&session, file);
        session.analyze(&files).expect("checks");

        session.set_text(file, "fun a() : Int = 1\nfun b() : Int = 2\nfun c() : Int = 3\n");
        let second = session.analyze(&files).expect("checks again");
        assert_eq!(session.unit_compiles.get(), 2);
        assert_eq!(second.checked.get(), 1, "only the appended item is checked");
    }

    #[test]
    fn appended_functions_can_forward_reference_each_other() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun a() : Int = 1\n");
        let files = unit(&session, file);
        session.analyze(&files).expect("checks");

        session.set_text(file, "fun a() : Int = 1\nfun f() : Int = g()\nfun g() : Int = 2\n");
        let second = session.analyze(&files).expect("checks forward references");
        assert_eq!(second.checked.get(), 2, "both appended bodies are checked");
    }

    #[test]
    fn replacing_a_temporary_expression_rechecks_only_it() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun a() : Int = 1\n");
        let files = unit(&session, file);
        session.analyze(&files).expect("checks");

        session.set_text(file, "fun a() : Int = 1\nfun __repl_0() : Int = (a())\n");
        let first = session.analyze(&files).expect("checks the expression");
        assert_eq!(first.checked.get(), 1);

        session.set_text(file, "fun a() : Int = 1\nfun __repl_1() : Int = (a() + 1)\n");
        let second = session.analyze(&files).expect("checks the next expression");
        assert_eq!(second.checked.get(), 1, "only the new expression is checked");
        assert!(!second.hir.bodies.iter().any(|body| body.name == "__repl_0"), "the old expression's body is dropped");
        assert!(second.hir.bodies.iter().any(|body| body.name == "__repl_1"));
    }

    #[test]
    fn editing_the_last_item_rechecks_from_there() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun a() : Int = 1\nfun b() : Int = 2\n");
        let files = unit(&session, file);
        session.analyze(&files).expect("checks");

        session.set_text(file, "fun a() : Int = 1\nfun b() : Int = 20\n");
        let second = session.analyze(&files).expect("checks the edit");
        assert_eq!(second.checked.get(), 1, "only the edited item is checked");
    }

    #[test]
    fn appending_lowers_only_the_new_body() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun a() : Int = 1\nfun b() : Int = 2\n");
        let files = unit(&session, file);
        session.compile_shared(&files).expect("compiles");
        assert!(session.lower_stats().full);

        session.set_text(file, "fun a() : Int = 1\nfun b() : Int = 2\nfun c() : Int = 3\n");
        let module = session.compile_shared(&files).expect("compiles again");
        let stats = session.lower_stats();
        assert!(!stats.full, "the new item resumes the previous lowering");
        assert_eq!(stats.new_roots, 1);
        assert_eq!(stats.bodies, 1, "only c is emitted");
        assert!(module.funcs.iter().any(|f| f.name == "c"));
    }

    #[test]
    fn replacing_an_expression_keeps_earlier_bodies() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun a() : Int = 1\n");
        let files = unit(&session, file);
        session.compile_shared(&files).expect("compiles the definitions");

        session.set_text(file, "fun a() : Int = 1\nfun __repl_0() : Int = (a())\n");
        let module = session.compile_shared(&files).expect("compiles");
        assert!(module.funcs.iter().any(|f| f.name == "__repl_0"));
        assert!(!session.lower_stats().full);

        session.set_text(file, "fun a() : Int = 1\nfun __repl_1() : Int = (a() + 1)\n");
        let module = session.compile_shared(&files).expect("compiles again");
        assert!(!session.lower_stats().full, "the prefix is reused");
        assert_eq!(session.lower_stats().bodies, 1, "only the new expression is emitted");
        assert!(module.funcs.iter().any(|f| f.name == "__repl_1"));
        assert!(module.funcs.iter().any(|f| f.name == "a"), "a is still there");
    }

    #[test]
    fn editing_a_definition_drops_the_stale_lowering() {
        let mut session = Session::new();
        let file = session.add_file("t.fern", "fun a() : Int = 1\nfun b() : Int = 2\n");
        let files = unit(&session, file);
        session.compile_shared(&files).expect("compiles");

        session.set_text(file, "fun a() : Int = 1\nfun b() : Int = 20\n");
        let module = session.compile_shared(&files).expect("recompiles");
        assert!(session.lower_stats().full, "a changed body forces a fresh module");
        assert_eq!(module.funcs.iter().filter(|f| f.name == "b").count(), 1);
    }
}
