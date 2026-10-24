//! Module name resolution.
//!
//! Source item names are module-local. This pass rewrites every top-level
//! declaration and every non-local reference to its canonical
//! `module::name`, then drops `import` and `use` items. Later passes can
//! keep using one flat namespace.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::span::{Diag, KResult, Span};
use crate::ty::Ty;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sort {
    Type,
    Class,
    Value,
    Ctor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    Type,
    Value,
    Ctor,
    Class,
}

#[derive(Debug, Clone, Default)]
struct ModuleScope {
    path: Vec<String>,
    imports: HashMap<String, ModuleId>,
    type_uses: HashMap<String, String>,
    value_uses: HashMap<String, String>,
    ctor_uses: HashMap<String, String>,
    locals: HashMap<String, Vec<(Sort, String)>>,
}

#[derive(Default)]
struct Scopes {
    levels: Vec<Vec<String>>,
}

impl Scopes {
    fn push(&mut self) {
        self.levels.push(Vec::new());
    }

    fn pop(&mut self) {
        self.levels.pop();
    }

    fn bind(&mut self, names: Vec<String>) {
        if let Some(level) = self.levels.last_mut() {
            level.extend(names);
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.levels.iter().any(|level| level.iter().any(|n| n == name))
    }
}

struct Qualifier {
    modules: Vec<ModuleScope>,
    by_path: HashMap<String, ModuleId>,
    sorts: HashMap<String, Sort>,
    owners: HashMap<String, ModuleId>,
    methods: HashSet<String>,
}

pub fn qualify(program: &mut Program) -> KResult<()> {
    let qualifier = Qualifier::new(program)?;
    let mut out = Vec::with_capacity(program.items.len());
    for mut item in std::mem::take(&mut program.items) {
        let module = item.module;
        let drop = match &mut item.kind {
            ItemKind::Import(_) | ItemKind::Use(_) => true,
            ItemKind::Data(data) => {
                data.name = qualifier.local(module, &data.name);
                for ctor in &mut data.ctors {
                    ctor.name = qualifier.local(module, &ctor.name);
                    for field in &mut ctor.fields {
                        qualifier.ty(module, field)?;
                    }
                }
                false
            }
            ItemKind::Class(class) => {
                class.name = qualifier.local(module, &class.name);
                for method in &mut class.methods {
                    qualifier.ty(module, &mut method.ty)?;
                }
                false
            }
            ItemKind::Instance(instance) => {
                instance.class = qualifier.class_name(module, &instance.class, instance.span)?;
                for bound in &mut instance.ctx {
                    qualifier.ty(module, bound)?;
                }
                qualifier.ty(module, &mut instance.self_ty)?;
                qualifier.check_orphan(module, &instance.class, &instance.self_ty, instance.span)?;
                for method in &mut instance.methods {
                    let mut scopes = Scopes::default();
                    scopes.push();
                    for param in &mut method.params {
                        qualifier.pat(module, param)?;
                        scopes.bind(param.binds());
                    }
                    qualifier.expr(module, &mut method.body, &mut scopes)?;
                }
                false
            }
            ItemKind::Extern(extern_decl) => {
                extern_decl.name = qualifier.local(module, &extern_decl.name);
                qualifier.ty(module, &mut extern_decl.ty)?;
                false
            }
            ItemKind::Fun(fun) => {
                fun.name = qualifier.local(module, &fun.name);
                let mut scopes = Scopes::default();
                scopes.push();
                for param in &mut fun.params {
                    qualifier.ty(module, &mut param.ty)?;
                    qualifier.pat(module, &mut param.pat)?;
                    scopes.bind(param.pat.binds());
                }
                qualifier.ty(module, &mut fun.ret)?;
                qualifier.expr(module, &mut fun.body, &mut scopes)?;
                false
            }
            ItemKind::Let(value) => {
                value.name = qualifier.local(module, &value.name);
                if let Some(ty) = &mut value.ty {
                    qualifier.ty(module, ty)?;
                }
                let mut scopes = Scopes::default();
                qualifier.expr(module, &mut value.body, &mut scopes)?;
                false
            }
        };
        if !drop {
            out.push(item);
        }
    }
    program.items = out;
    Ok(())
}

impl Qualifier {
    fn new(program: &Program) -> KResult<Qualifier> {
        let mut modules = Vec::with_capacity(program.modules.len());
        let mut by_path = HashMap::new();
        for (index, info) in program.modules.iter().enumerate() {
            if !info.path.is_empty() {
                by_path.insert(info.path.join("::"), ModuleId(index as u32));
            }
            modules.push(ModuleScope {
                path: info.path.clone(),
                ..ModuleScope::default()
            });
        }

        let mut qualifier = Qualifier {
            modules,
            by_path,
            sorts: HashMap::new(),
            owners: HashMap::new(),
            methods: HashSet::new(),
        };

        for item in &program.items {
            let module = item.module;
            if module.index() >= qualifier.modules.len() {
                return Err(Diag::new("item belongs to an unknown module", item.span));
            }
            match &item.kind {
                ItemKind::Data(data) => {
                    qualifier.register(module, &data.name, Sort::Type);
                    for ctor in &data.ctors {
                        qualifier.register(module, &ctor.name, Sort::Ctor);
                    }
                }
                ItemKind::Class(class) => {
                    qualifier.register(module, &class.name, Sort::Class);
                    for method in &class.methods {
                        qualifier.methods.insert(method.name.clone());
                    }
                }
                ItemKind::Extern(extern_decl) => qualifier.register(module, &extern_decl.name, Sort::Value),
                ItemKind::Fun(fun) => qualifier.register(module, &fun.name, Sort::Value),
                ItemKind::Let(value) => qualifier.register(module, &value.name, Sort::Value),
                ItemKind::Instance(_) | ItemKind::Import(_) | ItemKind::Use(_) => {}
            }
        }

        for item in &program.items {
            match &item.kind {
                ItemKind::Import(import) => qualifier.add_import(item.module, import)?,
                ItemKind::Use(use_decl) => qualifier.add_use(item.module, use_decl)?,
                _ => {}
            }
        }
        Ok(qualifier)
    }

    fn register(&mut self, module: ModuleId, name: &str, sort: Sort) {
        let canonical = self.canonical(module, name);
        self.modules[module.index()].locals.entry(name.to_string()).or_default().push((sort, canonical.clone()));
        self.sorts.insert(canonical.clone(), sort);
        if matches!(sort, Sort::Type | Sort::Class) {
            self.owners.insert(canonical, module);
        }
    }

    fn canonical(&self, module: ModuleId, name: &str) -> String {
        let path = &self.modules[module.index()].path;
        if path.is_empty() { name.to_string() } else { format!("{}::{name}", path.join("::")) }
    }

    fn local(&self, module: ModuleId, name: &str) -> String {
        self.canonical(module, name)
    }

    fn add_import(&mut self, module: ModuleId, import: &ImportDecl) -> KResult<()> {
        let path = import.path.join("::");
        let target = self.by_path.get(&path).copied().ok_or_else(|| Diag::new(format!("unknown module `{path}`"), import.span))?;
        let alias = import
            .alias
            .clone()
            .or_else(|| import.path.last().cloned())
            .ok_or_else(|| Diag::new("empty import path", import.span))?;
        let scope = &mut self.modules[module.index()];
        if let Some(existing) = scope.imports.insert(alias.clone(), target)
            && existing != target {
                return Err(Diag::new(format!("import alias `{alias}` is already bound to another module"), import.span));
            }
        scope.imports.entry(import.path.join(".")).or_insert(target);
        Ok(())
    }

    fn add_use(&mut self, module: ModuleId, use_decl: &UseDecl) -> KResult<()> {
        if use_decl.path.len() < 2 {
            return Err(Diag::new("`use` needs a module path and an item name", use_decl.span));
        }
        let item = use_decl.path.last().expect("checked length").clone();
        let module_path = use_decl.path[..use_decl.path.len() - 1].join("::");
        let target = self
            .by_path
            .get(&module_path)
            .copied()
            .ok_or_else(|| Diag::new(format!("unknown module `{module_path}`"), use_decl.span))?;
        let alias = use_decl.alias.clone().unwrap_or_else(|| item.clone());
        let locals = self.modules[target.index()]
            .locals
            .get(&item)
            .cloned()
            .ok_or_else(|| Diag::new(format!("module `{module_path}` has no item `{item}`"), use_decl.span))?;
        let scope = &mut self.modules[module.index()];
        for (sort, canonical) in locals {
            let map = match sort {
                Sort::Type | Sort::Class => &mut scope.type_uses,
                Sort::Value => &mut scope.value_uses,
                Sort::Ctor => &mut scope.ctor_uses,
            };
            if let Some(existing) = map.insert(alias.clone(), canonical.clone())
                && existing != canonical {
                    return Err(Diag::new(format!("`{alias}` is already imported"), use_decl.span));
                }
        }
        Ok(())
    }

    fn lookup_local(&self, module: ModuleId, name: &str, expect: Expect) -> Option<String> {
        let accept = |sort: Sort| match expect {
            Expect::Type => matches!(sort, Sort::Type | Sort::Class),
            Expect::Value => sort == Sort::Value,
            Expect::Ctor => sort == Sort::Ctor,
            Expect::Class => sort == Sort::Class,
        };
        let find = |id: ModuleId| {
            self.modules
                .get(id.index())?
                .locals
                .get(name)?
                .iter()
                .find(|(sort, _)| accept(*sort))
                .map(|(_, canonical)| canonical.clone())
        };
        find(module).or_else(|| if module != ModuleId::ROOT { find(ModuleId::ROOT) } else { None })
    }

    fn lookup_direct(&self, module: ModuleId, name: &str, expect: Expect) -> Option<String> {
        let accept = |sort: Sort| match expect {
            Expect::Type => matches!(sort, Sort::Type | Sort::Class),
            Expect::Value => sort == Sort::Value,
            Expect::Ctor => sort == Sort::Ctor,
            Expect::Class => sort == Sort::Class,
        };
        self.modules
            .get(module.index())?
            .locals
            .get(name)?
            .iter()
            .find(|(sort, _)| accept(*sort))
            .map(|(_, canonical)| canonical.clone())
    }

    fn lookup_use(&self, module: ModuleId, name: &str, expect: Expect) -> Option<String> {
        let scope = &self.modules[module.index()];
        let map = match expect {
            Expect::Type | Expect::Class => &scope.type_uses,
            Expect::Value => &scope.value_uses,
            Expect::Ctor => &scope.ctor_uses,
        };
        map.get(name).cloned()
    }

    fn resolve_name(&self, module: ModuleId, name: &str, expect: Expect, span: Span) -> KResult<String> {
        if name.contains("::") {
            return Ok(name.to_string());
        }
        if name.contains('.') {
            return self.resolve_member(module, name, expect, span);
        }
        if let Some(canonical) = self.lookup_local(module, name, expect) {
            return Ok(canonical);
        }
        if let Some(canonical) = self.lookup_use(module, name, expect) {
            if expect == Expect::Class && self.sorts.get(&canonical) != Some(&Sort::Class) {
                return Err(Diag::new(format!("`{name}` is not a class"), span));
            }
            return Ok(canonical);
        }
        if expect == Expect::Value && self.methods.contains(name) {
            return Ok(name.to_string());
        }
        if expect == Expect::Type && Ty::intros().contains(&name) {
            return Ok(name.to_string());
        }
        if expect == Expect::Type && name.chars().next().is_some_and(char::is_lowercase) {
            return Ok(name.to_string());
        }
        let what = match expect {
            Expect::Type | Expect::Class => "type",
            Expect::Value => "value",
            Expect::Ctor => "constructor",
        };
        Err(Diag::new(format!("unknown {what} `{name}`"), span))
    }

    fn resolve_member(&self, module: ModuleId, name: &str, expect: Expect, span: Span) -> KResult<String> {
        let parts: Vec<&str> = name.split('.').collect();
        let scope = &self.modules[module.index()];
        for prefix_len in (1..parts.len()).rev() {
            let prefix = parts[..prefix_len].join(".");
            if let Some(target) = scope.imports.get(&prefix).copied() {
                if parts.len() - prefix_len != 1 {
                    return Err(Diag::new(format!("`{name}` does not name a single module item"), span));
                }
                let item = parts[prefix_len];
                return self
                    .lookup_member(target, item, expect, span)
                    .ok_or_else(|| Diag::new(format!("module `{prefix}` has no matching `{item}`"), span));
            }
        }
        Err(Diag::new(format!("unknown module in `{name}`; add an `import` first"), span))
    }

    fn lookup_member(&self, module: ModuleId, name: &str, expect: Expect, span: Span) -> Option<String> {
        let canonical = self.lookup_direct(module, name, expect)?;
        if expect == Expect::Class && self.sorts.get(&canonical) != Some(&Sort::Class) {
            return None;
        }
        let _ = span;
        Some(canonical)
    }

    fn ty(&self, module: ModuleId, ty: &mut TyExpr) -> KResult<()> {
        match &mut ty.kind {
            TyExprKind::Name(name) => {
                let span = ty.span;
                *name = self.resolve_name(module, name, Expect::Type, span)?;
            }
            TyExprKind::Paren(inner) => self.ty(module, inner)?,
            TyExprKind::Tuple(items) => {
                for item in items {
                    self.ty(module, item)?;
                }
            }
            TyExprKind::App(head, arg) => {
                self.ty(module, head)?;
                self.ty(module, arg)?;
            }
            TyExprKind::Arrow(left, right) => {
                self.ty(module, left)?;
                self.ty(module, right)?;
            }
        }
        Ok(())
    }

    fn class_name(&self, module: ModuleId, name: &str, span: Span) -> KResult<String> {
        let canonical = self.resolve_name(module, name, Expect::Class, span)?;
        if self.sorts.get(&canonical) == Some(&Sort::Class) {
            Ok(canonical)
        } else {
            Err(Diag::new(format!("`{name}` is not a class"), span))
        }
    }

    fn value_name(&self, module: ModuleId, name: &str, span: Span) -> KResult<String> {
        self.resolve_name(module, name, Expect::Value, span)
    }

    fn ctor_name(&self, module: ModuleId, name: &str, span: Span) -> KResult<String> {
        self.resolve_name(module, name, Expect::Ctor, span)
    }

    fn check_orphan(&self, module: ModuleId, class: &str, self_ty: &TyExpr, span: Span) -> KResult<()> {
        let class_owner = self.owners.get(class).copied();
        let mut names = Vec::new();
        collect_type_names(self_ty, &mut names);
        let local = names.iter().any(|name| self.owners.get(*name) == Some(&module));
        if class_owner == Some(module) || local {
            Ok(())
        } else {
            Err(Diag::new(
                format!("orphan instance: neither `{class}` nor `{}` belongs to this module", self_ty.display()),
                span,
            ))
        }
    }

    fn expr(&self, module: ModuleId, expr: &mut Expr, scopes: &mut Scopes) -> KResult<()> {
        match &mut expr.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::BoolConst(_) | ExprKind::Str(_) | ExprKind::Unit => {}
            ExprKind::CtorCall { name, args } => {
                let span = expr.span;
                *name = self.ctor_name(module, name, span)?;
                for arg in args {
                    self.expr(module, arg, scopes)?;
                }
            }
            ExprKind::Ctor { name } => {
                let span = expr.span;
                *name = self.ctor_name(module, name, span)?;
            }
            ExprKind::Var(name) => {
                if !scopes.contains(name) {
                    let span = expr.span;
                    *name = self.value_name(module, name, span)?;
                }
            }
            ExprKind::Call(callee, args) => {
                self.expr(module, callee, scopes)?;
                for arg in args {
                    self.expr(module, arg, scopes)?;
                }
            }
            ExprKind::Record { ctor, fields } => {
                let span = expr.span;
                *ctor = self.ctor_name(module, ctor, span)?;
                for (_, value) in fields {
                    self.expr(module, value, scopes)?;
                }
            }
            ExprKind::Field { .. } => self.resolve_field(module, expr, scopes)?,
            ExprKind::Try(inner) => self.expr(module, inner, scopes)?,
            ExprKind::Await(inner) => self.expr(module, inner, scopes)?,
            ExprKind::ClassSel { class, args, .. } => {
                let span = expr.span;
                *class = self.class_name(module, class, span)?;
                for arg in args {
                    self.expr(module, arg, scopes)?;
                }
            }
            ExprKind::Lambda { params, body } => {
                scopes.push();
                for param in params {
                    self.pat(module, param)?;
                    scopes.bind(param.binds());
                }
                self.expr(module, body, scopes)?;
                scopes.pop();
            }
            ExprKind::Let { binds, body } => {
                scopes.push();
                for (pat, value) in binds {
                    self.expr(module, value, scopes)?;
                    self.pat(module, pat)?;
                    scopes.bind(pat.binds());
                }
                self.expr(module, body, scopes)?;
                scopes.pop();
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(module, scrutinee, scopes)?;
                for arm in arms {
                    scopes.push();
                    self.pat(module, &mut arm.pat)?;
                    scopes.bind(arm.pat.binds());
                    if let Some(guard) = &mut arm.guard {
                        self.expr(module, guard, scopes)?;
                    }
                    self.expr(module, &mut arm.body, scopes)?;
                    scopes.pop();
                }
            }
            ExprKind::Tuple(items) | ExprKind::List(items) => {
                for item in items {
                    self.expr(module, item, scopes)?;
                }
            }
            ExprKind::If { cond, then_body, else_body } => {
                self.expr(module, cond, scopes)?;
                self.expr(module, then_body, scopes)?;
                self.expr(module, else_body, scopes)?;
            }
            ExprKind::Block(inner) => self.expr(module, inner, scopes)?,
            ExprKind::BinOp(_, left, right) => {
                self.expr(module, left, scopes)?;
                self.expr(module, right, scopes)?;
            }
            ExprKind::UnOp(_, operand) => self.expr(module, operand, scopes)?,
            ExprKind::Shell { command, .. } => self.shell(module, command, scopes)?,
        }
        Ok(())
    }

    fn resolve_field(&self, module: ModuleId, expr: &mut Expr, scopes: &mut Scopes) -> KResult<()> {
        let span = expr.span;
        if let Some(path) = field_path(expr) {
            let first = &path[0];
            let imports = &self.modules[module.index()].imports;
            let module_path = !scopes.contains(first) && (1..path.len()).any(|n| imports.contains_key(&path[..n].join(".")));
            if module_path {
                let member = path.last().expect("field paths are non-empty");
                let expect = if member.as_bytes()[0].is_ascii_uppercase() { Expect::Ctor } else { Expect::Value };
                let name = self.resolve_member(module, &path.join("."), expect, span)?;
                *expr = match expect {
                    Expect::Ctor => Expr {
                        kind: ExprKind::Ctor { name },
                        span,
                    },
                    _ => Expr { kind: ExprKind::Var(name), span },
                };
                return Ok(());
            }
        }
        if let ExprKind::Field { base, .. } = &mut expr.kind {
            self.expr(module, base, scopes)?;
        }
        Ok(())
    }

    fn pat(&self, module: ModuleId, pat: &mut Pat) -> KResult<()> {
        match pat {
            Pat::Wild | Pat::Bind(_) | Pat::Int(_) | Pat::Bool(_) => {}
            Pat::Paren(inner) => self.pat(module, inner)?,
            Pat::Ctor { name, fields } => {
                let span = Span::new(0, 0);
                *name = self.ctor_name(module, name, span)?;
                for field in fields {
                    self.pat(module, field)?;
                }
            }
            Pat::Tuple(items) | Pat::List(items) => {
                for item in items {
                    self.pat(module, item)?;
                }
            }
            Pat::Comp(left, right) => {
                self.pat(module, left)?;
                self.pat(module, right)?;
            }
        }
        Ok(())
    }

    fn shell(&self, module: ModuleId, command: &mut ShellCommand, scopes: &mut Scopes) -> KResult<()> {
        for stage in &mut command.stages {
            for word in &mut stage.words {
                for part in &mut word.parts {
                    match part {
                        ShellWordPart::Lit(_) => {}
                        ShellWordPart::Interp(expr) => self.expr(module, expr, scopes)?,
                        ShellWordPart::Capture(inner) => self.shell(module, inner, scopes)?,
                    }
                }
            }
        }
        Ok(())
    }
}

fn field_path(expr: &Expr) -> Option<Vec<String>> {
    match &expr.kind {
        ExprKind::Var(name) => Some(vec![name.clone()]),
        ExprKind::Field { base, field } => {
            let mut path = field_path(base)?;
            path.push(field.clone());
            Some(path)
        }
        _ => None,
    }
}

fn collect_type_names<'a>(ty: &'a TyExpr, out: &mut Vec<&'a str>) {
    match &ty.kind {
        TyExprKind::Name(name) => out.push(name),
        TyExprKind::Paren(inner) => collect_type_names(inner, out),
        TyExprKind::Tuple(items) => {
            for item in items {
                collect_type_names(item, out);
            }
        }
        TyExprKind::App(head, arg) => {
            collect_type_names(head, out);
            collect_type_names(arg, out);
        }
        TyExprKind::Arrow(left, right) => {
            collect_type_names(left, out);
            collect_type_names(right, out);
        }
    }
}
