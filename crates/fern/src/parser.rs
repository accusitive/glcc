//! Recursive-descent parser for fern.
//!
//! fern is a small pure ML-family surface. Case carries no grammatical
//! meaning: every name is one `Ident` token and resolution decides
//! whether it is a constructor, a type, or a value.
//!
//! ```text
//! item     := type | class | instance | extern | fn | let | import | use
//! type     := 'type' Name ('<' params '>')? '='
//!             '|' ctor ('|' ctor)*          -- sum
//!           | '{' (name ':' ty ',')* '}'    -- record (single constructor
//!                                             named like the type)
//! extern   := 'extern' string? name ':' [for<...>] ty
//! ctor     := Name ('(' ty (',' ty)* ')')?
//! class    := 'class' Name ('<' params '>')? '{' (name ':' [for<...>] ty ';')* '}'
//! instance := 'instance' ('<' params '>')? Name '<' ty '>' ('when' ty (',' ty)*)?
//!             '{' method* '}'
//! method   := name '(' pat-list? ')' '=' expr ';'?   (params may be empty)
//! fn       := 'async'? 'fn' name '(' pat-list? ')' ':' [for<...>] ty '->' ty '=' expr
//!             -- the parameter list is one pattern matched against the
//!             -- annotation's domain; a nullary fn is 'fn f() : -> ty'
//! let      := 'let' name (':' [for<...>] ty)? '=' expr
//! expr     := if | match | block | closure | prefix
//! closure  := '|' pat-list? '|' expr | '||' expr
//! if       := 'if' expr 'then' expr 'else' expr
//! match    := 'match' expr '{' ('case' pat ('when' expr)? '=>' expr ';')* '}'
//! block    := '{' ('let' pat '=' expr ';')* (expr ';')* expr? '}'
//! ty       := [for<...>] ty_app ('->' ty)?
//! ty_app   := Name ('<' ty (',' ty)* '>')?
//! ```
//!
//! Binary operators, loosest first: `||`, `&&`, comparisons, `|>`
//! (left-assoc value pipe), `::` (right-assoc cons), `+ -`, `* /`;
//! unary `-`; postfix `.await`, `?`, `.field`, calls `f(x, y)`.

use crate::ast::*;
use crate::lexer::{Tok, Token};
use crate::span::{Diag, KResult, Span};

#[derive(Debug, Clone)]
pub struct ParseOutput {
    pub program: Program,
    pub diags: Vec<Diag>,
    pub incomplete: bool,
}

/// Parse every item, reporting failures and resyncing at the next
/// top-level item instead of stopping at the first error.
pub fn parse_recovering(tokens: Vec<Token>) -> ParseOutput {
    let mut parser = Parser { tokens, pos: 0 };
    parser.program_recovering()
}

/// Parse a single expression, as found inside `${...}`.
pub fn parse_expr_fragment(tokens: Vec<Token>) -> KResult<Expr> {
    let mut parser = Parser { tokens, pos: 0 };
    let expr = parser.expr()?;
    parser.expect(&Tok::Eof, "end of the interpolation expression")?;
    Ok(expr)
}

/// Fail-fast wrapper over [`parse_recovering`]: returns the first
/// diagnostic and discards the partially parsed program.
pub fn parse(tokens: Vec<Token>) -> KResult<Program> {
    let mut out = parse_recovering(tokens);
    if out.diags.is_empty() { Ok(out.program) } else { Err(out.diags.remove(0)) }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        self.tokens.get(self.pos).map(|t| &t.tok).unwrap_or(&Tok::Eof)
    }

    fn bump(&mut self) -> Token {
        let token = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        token.unwrap_or(Token {
            tok: Tok::Eof,
            span: Span::new(0, 0),
        })
    }

    /// Span of the upcoming token.
    fn span(&self) -> Span {
        self.tokens.get(self.pos).map(|t| t.span).unwrap_or(Span::new(0, 0))
    }

    /// Span of the token just consumed.
    fn prev_span(&self) -> Span {
        self.tokens.get(self.pos.saturating_sub(1)).map(|t| t.span).unwrap_or(Span::new(0, 0))
    }

    fn is(&self, want: &Tok) -> bool {
        self.peek() == want
    }

    /// Mark an error as incomplete when the input ended while parsing.
    fn complete_if_eof(&self, diag: Diag) -> Diag {
        if self.is(&Tok::Eof) && !diag.incomplete {
            Diag::incomplete(diag.msg, diag.span)
        } else {
            diag
        }
    }

    fn expect(&mut self, want: &Tok, what: &str) -> KResult<Token> {
        if self.is(want) {
            return Ok(self.bump());
        }
        Err(self.complete_if_eof(Diag::new(format!("expected {what}, found {}", self.peek().describe()), self.span())))
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if self.is(want) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn ident(&mut self, what: &str) -> KResult<String> {
        match self.bump() {
            Token { tok: Tok::Ident(name), .. } => Ok(name),
            other => Err(Diag::new(format!("expected {what}, found {}", other.tok.describe()), other.span)),
        }
    }

    // ------------------------------------------------------------ items --

    fn program_recovering(&mut self) -> ParseOutput {
        let mut items = Vec::new();
        let mut diags = Vec::new();
        let mut incomplete = false;
        loop {
            self.eat_seps();
            if self.is(&Tok::Eof) {
                break;
            }
            let start = self.pos;
            match self.item() {
                Ok(item) => items.push(item),
                Err(diag) => {
                    let diag = self.complete_if_eof(diag);
                    incomplete |= diag.incomplete;
                    diags.push(diag);
                    self.synchronize(start);
                }
            }
        }
        ParseOutput {
            program: Program::single(items),
            diags,
            incomplete,
        }
    }

    /// Skip to the next top-level item keyword, ignoring keywords nested
    /// inside braces, parentheses, or brackets. Always advances when the
    /// failing item consumed nothing, so the recovery loop terminates.
    /// An item keyword already at the cursor belongs to the next item and
    /// is left for the caller.
    fn synchronize(&mut self, item_start: usize) {
        let lo = item_start.min(self.tokens.len());
        let hi = self.pos.min(self.tokens.len());
        let mut open = Vec::new();
        // Only unmatched braces carry over from the failed item: a block
        // or body can contain nested keywords, while an unclosed parameter
        // list is abandoned at the next item keyword.
        for token in &self.tokens[lo..hi] {
            match token.tok {
                Tok::LBrace => open.push(Delim::Brace),
                Tok::RBrace => close_delim(&mut open, Delim::Brace),
                _ => {}
            }
        }
        let start = self.pos;
        while !self.is(&Tok::Eof) {
            if open.is_empty() && is_item_start(self.peek()) {
                break;
            }
            match self.peek() {
                Tok::LBrace => open.push(Delim::Brace),
                Tok::LParen => open.push(Delim::Paren),
                Tok::LBracket => open.push(Delim::Bracket),
                Tok::RBrace => close_delim(&mut open, Delim::Brace),
                Tok::RParen => close_delim(&mut open, Delim::Paren),
                Tok::RBracket => close_delim(&mut open, Delim::Bracket),
                _ => {}
            }
            self.bump();
        }
        if self.pos == start && !self.is(&Tok::Eof) && (self.pos == item_start || !is_item_start(self.peek())) {
            self.bump();
        }
    }

    fn eat_seps(&mut self) {
        while self.is(&Tok::Semi) {
            self.bump();
        }
    }

    fn item(&mut self) -> KResult<Item> {
        let lo = self.span();
        let kind = match self.peek() {
            Tok::Type => ItemKind::Type(self.type_item()?),
            Tok::Class => ItemKind::Class(self.class_item()?),
            Tok::Instance => ItemKind::Instance(self.instance_item()?),
            Tok::Extern => ItemKind::Extern(self.extern_item()?),
            Tok::Async | Tok::Fn => ItemKind::Fn(self.fn_item()?),
            Tok::Let => ItemKind::Let(self.let_item()?),
            Tok::Import => ItemKind::Import(self.import_item()?),
            Tok::Use => ItemKind::Use(self.use_item()?),
            Tok::Priv => {
                return Err(Diag::new("`priv` is reserved for module privacy and is not implemented yet", self.span()));
            }
            other => {
                return Err(self.complete_if_eof(Diag::new(
                    format!(
                        "expected an item (`type`, `class`, `instance`, `extern`, `fn`, `let`, `import`, or `use`), found {}",
                        other.describe()
                    ),
                    self.span(),
                )));
            }
        };
        Ok(Item {
            module: ModuleId::ROOT,
            span: lo.to(self.prev_span()),
            kind,
        })
    }

    /// `type Name<P> = | Ctor(Ty) | ...` (leading bar required) or
    /// `type Name<P> = { field : Ty, ... }` (a record, whose single
    /// constructor is named like the type).
    fn type_item(&mut self) -> KResult<TypeDecl> {
        let start = self.span();
        self.bump();
        let name = self.ident("a type name")?;
        let params = self.type_params()?;
        self.expect(&Tok::Eq, "`=` after the type head")?;
        let mut ctors = Vec::new();
        if self.is(&Tok::LBrace) {
            self.bump();
            let mut fields = Vec::new();
            let mut names = Vec::new();
            if !self.is(&Tok::RBrace) {
                loop {
                    let field_span = self.span();
                    let field = self.ident("a record field name")?;
                    if names.contains(&field) {
                        return Err(Diag::new(format!("duplicate record field `{field}`"), field_span));
                    }
                    names.push(field);
                    self.expect(&Tok::Colon, "`:` after the record field name")?;
                    fields.push(self.ty_expr()?);
                    if self.eat(&Tok::Comma) {
                        continue;
                    }
                    break;
                }
            }
            self.expect(&Tok::RBrace, "`}` after the record fields")?;
            ctors.push(CtorDecl {
                name: name.clone(),
                fields,
                field_names: Some(names),
            });
        } else if self.eat(&Tok::Pipe) {
            let mut ctor_names = Vec::new();
            loop {
                let ctor_span = self.span();
                let ctor = self.ident("a constructor name")?;
                if ctor_names.contains(&ctor) {
                    return Err(Diag::new(format!("duplicate constructor `{ctor}`"), ctor_span));
                }
                ctor_names.push(ctor.clone());
                let mut fields = Vec::new();
                if self.is(&Tok::LParen) {
                    self.bump();
                    if !self.is(&Tok::RParen) {
                        loop {
                            fields.push(self.ty_expr()?);
                            if self.eat(&Tok::Comma) {
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)` after the constructor fields")?;
                }
                ctors.push(CtorDecl {
                    name: ctor,
                    fields,
                    field_names: None,
                });
                if self.eat(&Tok::Pipe) {
                    continue;
                }
                break;
            }
        } else {
            return Err(Diag::new(
                "expected `|` and constructors or `{` and record fields after `=` (type aliases are not supported)",
                self.span(),
            ));
        }
        Ok(TypeDecl {
            span: start.to(self.prev_span()),
            name,
            params,
            ctors,
        })
    }

    /// `<A, B>` after a `type`/`class`/`instance` head.
    fn type_params(&mut self) -> KResult<Vec<String>> {
        if !self.eat(&Tok::Lt) {
            return Ok(Vec::new());
        }
        let mut params = vec![self.ident("a type parameter")?];
        while self.eat(&Tok::Comma) {
            params.push(self.ident("a type parameter")?);
        }
        if let Some(i) = (0..params.len()).find(|&i| params[i + 1..].contains(&params[i])) {
            return Err(Diag::new(format!("duplicate type parameter `{}`", params[i]), self.span()));
        }
        self.expect(&Tok::Gt, "`>` closing the type parameters")?;
        Ok(params)
    }

    fn class_item(&mut self) -> KResult<ClassDecl> {
        let start = self.span();
        self.bump();
        let name = self.ident("a class name")?;
        let params = self.type_params()?;
        self.expect(&Tok::LBrace, "`{` opening the class body")?;
        let mut method_names = Vec::new();
        let mut methods = Vec::new();
        loop {
            self.eat_seps();
            if self.is(&Tok::RBrace) || self.is(&Tok::Eof) {
                break;
            }
            let name = self.ident("a method name")?;
            if method_names.contains(&name) {
                return Err(Diag::new(format!("duplicate class method `{name}`"), self.span()));
            }
            method_names.push(name.clone());
            self.expect(&Tok::Colon, "`:` before the method type")?;
            let ty = self.ty_expr()?;
            methods.push(MethodSig { name, ty });
        }
        self.expect(&Tok::RBrace, "`}` closing the class body")?;
        Ok(ClassDecl {
            span: start.to(self.prev_span()),
            name,
            params,
            methods,
        })
    }

    fn instance_item(&mut self) -> KResult<InstanceDecl> {
        let start = self.span();
        self.bump();
        let params = self.type_params()?;
        let class = self.ident("a class name")?;
        self.expect(&Tok::Lt, "`<` and the instance type after the class name")?;
        let mut args = Vec::new();
        loop {
            args.push(self.ty_expr()?);
            if self.eat(&Tok::Comma) {
                continue;
            }
            break;
        }
        self.expect(&Tok::Gt, "`>` closing the instance type")?;
        let mut ctx = Vec::new();
        if self.eat(&Tok::When) {
            loop {
                ctx.push(self.ty_expr()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
        }
        self.expect(&Tok::LBrace, "`{` opening the instance body")?;
        let mut method_names = Vec::new();
        let mut methods = Vec::new();
        loop {
            self.eat_seps();
            if self.is(&Tok::RBrace) || self.is(&Tok::Eof) {
                break;
            }
            let name = self.ident("a method name")?;
            if method_names.contains(&name) {
                return Err(Diag::new(format!("duplicate method `{name}`"), self.span()));
            }
            method_names.push(name.clone());
            let param = self.param_list_pattern()?;
            self.expect(&Tok::Eq, "`=` before the method body")?;
            let body = self.expr()?;
            methods.push(MethodBody {
                name,
                param,
                body,
                span: start.to(self.prev_span()),
            });
        }
        self.expect(&Tok::RBrace, "`}` closing the instance body")?;
        Ok(InstanceDecl {
            span: start.to(self.prev_span()),
            params,
            class,
            args,
            ctx,
            methods,
        })
    }

    fn extern_item(&mut self) -> KResult<ExternDecl> {
        let start = self.span();
        self.bump();
        let symbol = if matches!(self.peek(), Tok::Str(_)) {
            let Token { tok: Tok::Str(symbol), .. } = self.bump() else {
                unreachable!("checked above");
            };
            Some(symbol)
        } else {
            None
        };
        let name = self.ident("an extern name")?;
        self.expect(&Tok::Colon, "`:` before the extern signature")?;
        let ty = self.ty_expr()?;
        Ok(ExternDecl {
            span: start.to(self.prev_span()),
            name,
            symbol,
            ty,
        })
    }

    fn fn_item(&mut self) -> KResult<FnDecl> {
        let start = self.span();
        let is_async = self.eat(&Tok::Async);
        self.expect(&Tok::Fn, "`fn` after `async`")?;
        let name = self.ident("a function name")?;
        let param = self.param_list_pattern()?;
        self.expect(&Tok::Colon, "`:` and a type annotation after the parameters")?;
        let ann = self.ty_expr()?;
        let (_, bare) = ann.strip_for();
        if !matches!(bare.kind, TyExprKind::Arrow(..)) {
            return Err(Diag::new(
                "a function annotation must be a function type: write `->` before the return type (a nullary function is `fn f() : -> Int`)",
                self.span(),
            ));
        }
        self.expect(&Tok::Eq, "`=` before the function body")?;
        let body = self.expr()?;
        Ok(FnDecl {
            span: start.to(self.prev_span()),
            name,
            param,
            ann,
            body,
            is_async,
        })
    }

    /// The `(pat, pat, ...)` parameter list as ONE pattern: empty is the
    /// unit pattern, a single pattern stays bare, several become a tuple
    /// pattern.
    fn param_list_pattern(&mut self) -> KResult<Pat> {
        self.expect(&Tok::LParen, "`(`")?;
        if self.eat(&Tok::RParen) {
            return Ok(Pat::Tuple(Vec::new()));
        }
        let first = self.pat()?;
        if self.eat(&Tok::Comma) {
            let mut items = vec![first];
            loop {
                items.push(self.pat()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
            self.expect(&Tok::RParen, "`)` closing the parameters")?;
            Ok(Pat::Tuple(items))
        } else {
            self.expect(&Tok::RParen, "`)` closing the parameters")?;
            Ok(first)
        }
    }

    fn let_item(&mut self) -> KResult<ValueDecl> {
        let start = self.span();
        self.bump();
        let name = self.ident("a value name")?;
        let ty = if self.eat(&Tok::Colon) { Some(self.ty_expr()?) } else { None };
        self.expect(&Tok::Eq, "`=` after the value name")?;
        let body = self.expr()?;
        Ok(ValueDecl {
            span: start.to(self.prev_span()),
            name,
            ty,
            body,
        })
    }

    fn import_item(&mut self) -> KResult<ImportDecl> {
        let start = self.span();
        self.bump();
        let path = self.module_path("a module path")?;
        let alias = if self.eat(&Tok::As) { Some(self.ident("an import alias")?) } else { None };
        Ok(ImportDecl {
            span: start.to(self.prev_span()),
            path,
            alias,
        })
    }

    fn use_item(&mut self) -> KResult<UseDecl> {
        let start = self.span();
        self.bump();
        let mut path = vec![self.ident("a module path")?];
        while self.eat(&Tok::ColonColon) {
            path.push(self.ident("a module path or imported item")?);
        }
        let alias = if self.eat(&Tok::As) { Some(self.ident("an import alias")?) } else { None };
        Ok(UseDecl {
            span: start.to(self.prev_span()),
            path,
            alias,
        })
    }

    fn module_path(&mut self, what: &str) -> KResult<Vec<String>> {
        let mut path = vec![self.ident(what)?];
        while self.eat(&Tok::ColonColon) {
            path.push(self.ident("a module path segment")?);
        }
        Ok(path)
    }

    // ------------------------------------------------------------ types --

    /// `ty := 'for' '<' params '>' ty | '->' ty | ty_app ('->' ty)?`.
    /// Arrows are right-assoc; an empty domain is the unit tuple, the
    /// type of a nullary function.
    fn ty_expr(&mut self) -> KResult<TyExpr> {
        if self.is(&Tok::For) {
            let start = self.span();
            self.bump();
            self.expect(&Tok::Lt, "`<` after `for`")?;
            let mut params = vec![self.ident("a type parameter")?];
            while self.eat(&Tok::Comma) {
                params.push(self.ident("a type parameter")?);
            }
            if let Some(i) = (0..params.len()).find(|&i| params[i + 1..].contains(&params[i])) {
                return Err(Diag::new(format!("duplicate type parameter `{}`", params[i]), start));
            }
            self.expect(&Tok::Gt, "`>` closing the `for` binders")?;
            let ty = self.ty_expr()?;
            return Ok(TyExpr {
                kind: TyExprKind::For {
                    params,
                    ty: Box::new(ty),
                },
                span: start.to(self.prev_span()),
            });
        }
        if self.eat(&Tok::Arrow) {
            let rest = self.ty_expr()?;
            let span = rest.span;
            return Ok(TyExpr {
                kind: TyExprKind::Arrow(Box::new(TyExpr {
                    kind: TyExprKind::Tuple(Vec::new()),
                    span,
                }), Box::new(rest)),
                span,
            });
        }
        let first = self.ty_app()?;
        if self.eat(&Tok::Arrow) {
            let rest = self.ty_expr()?;
            let span = first.span.to(self.prev_span());
            Ok(TyExpr {
                kind: TyExprKind::Arrow(Box::new(first), Box::new(rest)),
                span,
            })
        } else {
            Ok(first)
        }
    }

    /// `Name<T1, T2>`: bracket application. Juxtaposition application is
    /// gone; complex arguments need their own brackets or parens.
    fn ty_app(&mut self) -> KResult<TyExpr> {
        let first = self.ty_atom()?;
        if self.is(&Tok::Lt) {
            let head_span = first.span;
            self.bump();
            let mut args = Vec::new();
            if !self.is(&Tok::Gt) {
                loop {
                    args.push(self.ty_expr()?);
                    if self.eat(&Tok::Comma) {
                        continue;
                    }
                    break;
                }
            }
            self.expect(&Tok::Gt, "`>` closing the type arguments")?;
            return Ok(TyExpr {
                kind: TyExprKind::App(Box::new(first), args),
                span: head_span.to(self.prev_span()),
            });
        }
        Ok(first)
    }

    fn ty_atom(&mut self) -> KResult<TyExpr> {
        let span = self.span();
        match self.peek() {
            Tok::Ident(_) => {
                let mut name = self.ident("a type")?;
                while self.eat(&Tok::Dot) {
                    let segment = self.ident("a type name after `.`")?;
                    name.push('.');
                    name.push_str(&segment);
                }
                Ok(TyExpr {
                    kind: TyExprKind::Name(name),
                    span: span.to(self.prev_span()),
                })
            }
            Tok::LParen => {
                self.bump();
                if self.is(&Tok::RParen) {
                    self.bump();
                    return Ok(TyExpr {
                        kind: TyExprKind::Tuple(Vec::new()),
                        span: span.to(self.prev_span()),
                    });
                }
                let first = self.ty_expr()?;
                if self.is(&Tok::Comma) {
                    let mut items = vec![first];
                    while self.eat(&Tok::Comma) {
                        items.push(self.ty_expr()?);
                    }
                    self.expect(&Tok::RParen, "`)` closing the tuple type")?;
                    Ok(TyExpr {
                        kind: TyExprKind::Tuple(items),
                        span: span.to(self.prev_span()),
                    })
                } else {
                    self.expect(&Tok::RParen, "`)` closing the parenthesized type")?;
                    Ok(TyExpr {
                        kind: TyExprKind::Paren(Box::new(first)),
                        span: span.to(self.prev_span()),
                    })
                }
            }
            other => Err(Diag::new(format!("expected a type, found {}", other.describe()), span)),
        }
    }

    // --------------------------------------------------------- patterns --

    /// `pat := pat_atom ('::' pat)?`: `::` is right-assoc.
    fn pat(&mut self) -> KResult<Pat> {
        let first = self.pat_atom_shell()?;
        if self.is(&Tok::ColonColon) {
            self.bump();
            let rest = self.pat()?;
            return Ok(Pat::Comp(Box::new(first), Box::new(rest)));
        }
        Ok(first)
    }

    fn pat_atom_shell(&mut self) -> KResult<Pat> {
        let pat = self.pat_atom()?;
        Ok(match pat {
            Pat::Paren(inner) => *inner,
            other => other,
        })
    }

    fn pat_atom(&mut self) -> KResult<Pat> {
        match self.peek() {
            Tok::Underscore => {
                self.bump();
                Ok(Pat::Wild)
            }
            Tok::Int(v) => {
                let v = *v;
                self.bump();
                Ok(Pat::Int(v))
            }
            Tok::Char(c) => {
                let c = *c;
                self.bump();
                Ok(Pat::Char(c))
            }
            Tok::True => {
                self.bump();
                Ok(Pat::Bool(true))
            }
            Tok::False => {
                self.bump();
                Ok(Pat::Bool(false))
            }
            Tok::Float(_) => Err(Diag::new("float literals cannot be matched; compare with `==` instead".to_string(), self.span())),
            Tok::Ident(_) => {
                // A bare name is a binding or a nullary constructor;
                // resolution decides. With fields it is unambiguously a
                // constructor pattern.
                let mut name = self.ident("a pattern")?;
                while self.eat(&Tok::Dot) {
                    name.push('.');
                    name.push_str(&self.ident("a constructor name after `.`")?);
                }
                if self.is(&Tok::LParen) {
                    self.bump();
                    let mut fields = Vec::new();
                    if !self.is(&Tok::RParen) {
                        loop {
                            fields.push(self.pat()?);
                            if self.eat(&Tok::Comma) {
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)` after the constructor pattern")?;
                    Ok(Pat::Ctor { name, fields })
                } else if name.contains('.') {
                    Ok(Pat::Ctor { name, fields: Vec::new() })
                } else {
                    Ok(Pat::Bind(name))
                }
            }
            Tok::LParen => {
                self.bump();
                if self.is(&Tok::RParen) {
                    self.bump();
                    return Ok(Pat::Tuple(Vec::new()));
                }
                let first = self.pat()?;
                if self.is(&Tok::Comma) {
                    let mut items = vec![first];
                    while self.eat(&Tok::Comma) {
                        items.push(self.pat()?);
                    }
                    self.expect(&Tok::RParen, "`)` closing the tuple pattern")?;
                    Ok(Pat::Tuple(items))
                } else {
                    self.expect(&Tok::RParen, "`)` closing the pattern")?;
                    Ok(Pat::Paren(Box::new(first)))
                }
            }
            Tok::LBracket => {
                self.bump();
                let mut items = Vec::new();
                if !self.is(&Tok::RBracket) {
                    loop {
                        items.push(self.pat()?);
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&Tok::RBracket, "`]` closing the list pattern")?;
                Ok(Pat::List(items))
            }
            other => Err(Diag::new(format!("expected a pattern, found {}", other.describe()), self.span())),
        }
    }

    // ------------------------------------------------------- expressions --

    fn expr(&mut self) -> KResult<Expr> {
        let lo = self.span();
        let kind = self.expr_raw()?;
        Ok(Expr {
            kind,
            span: lo.to(self.prev_span()),
        })
    }

    fn expr_raw(&mut self) -> KResult<ExprKind> {
        match self.peek() {
            Tok::If => self.if_expr(),
            Tok::Match => self.match_expr(),
            Tok::LBrace => {
                let body = self.block()?;
                Ok(ExprKind::Block(Box::new(body)))
            }
            _ => Ok(self.binary(0)?.kind),
        }
    }

    fn if_expr(&mut self) -> KResult<ExprKind> {
        self.bump();
        let cond = self.expr()?;
        self.expect(&Tok::Then, "`then`")?;
        let then_body = self.expr()?;
        self.expect(&Tok::Else, "`else`")?;
        let else_body = self.expr()?;
        Ok(ExprKind::If {
            cond: Box::new(cond),
            then_body: Box::new(then_body),
            else_body: Box::new(else_body),
        })
    }

    fn match_expr(&mut self) -> KResult<ExprKind> {
        self.bump();
        let scrutinee = self.expr()?;
        self.expect(&Tok::LBrace, "`{` opening the match arms")?;
        let mut arms = Vec::new();
        loop {
            self.eat_seps();
            if self.is(&Tok::RBrace) || self.is(&Tok::Eof) {
                break;
            }
            arms.push(self.match_arm()?);
        }
        self.expect(&Tok::RBrace, "`}` closing the match arms")?;
        Ok(ExprKind::Match {
            scrutinee: Box::new(scrutinee),
            arms,
        })
    }

    fn match_arm(&mut self) -> KResult<MatchArm> {
        self.expect(&Tok::Case, "`case` starting an arm")?;
        let pat = self.pat()?;
        let guard = if self.eat(&Tok::When) { Some(self.expr()?) } else { None };
        self.expect(&Tok::FatArrow, "`=>` before the arm body")?;
        let body = self.expr()?;
        Ok(MatchArm { pat, guard, body })
    }

    /// `{ let pat = expr; ...; tail? }`: statements in written order; the
    /// final expression (with no `;`) is the tail.
    fn block(&mut self) -> KResult<Expr> {
        let lo = self.span();
        self.expect(&Tok::LBrace, "`{`")?;
        let mut binds: Vec<(Pat, Expr)> = Vec::new();
        loop {
            self.eat_seps();
            match self.peek() {
                Tok::RBrace | Tok::Eof => break,
                Tok::Let => {
                    self.bump();
                    let pat = self.pat()?;
                    self.expect(&Tok::Eq, "`=` after the pattern")?;
                    let value = self.expr()?;
                    binds.push((pat, value));
                    self.expect(&Tok::Semi, "`;` after a `let` statement")?;
                }
                _ => {
                    let expr = self.expr()?;
                    if self.eat(&Tok::Semi) {
                        binds.push((Pat::Wild, expr));
                    } else {
                        self.expect(&Tok::RBrace, "`;` or `}` after the expression")?;
                        let out = Expr {
                            kind: ExprKind::Let { binds, body: Box::new(expr) },
                            span: lo.to(self.prev_span()),
                        };
                        return Ok(out);
                    }
                }
            }
        }
        self.expect(&Tok::RBrace, "`}` closing the block")?;
        let body = Expr { kind: ExprKind::Unit, span: lo };
        Ok(Expr {
            kind: ExprKind::Let { binds, body: Box::new(body) },
            span: lo.to(self.prev_span()),
        })
    }

    fn binary(&mut self, level: usize) -> KResult<Expr> {
        if level >= BIN_LEVELS.len() {
            return self.prefix();
        }
        if level == PIPE_LEVEL {
            let mut lhs = self.binary(level + 1)?;
            while self.is(&Tok::PipeGt) {
                self.bump();
                let rhs = self.binary(level + 1)?;
                let span = lhs.span.to(rhs.span);
                lhs = Expr {
                    kind: ExprKind::Call(Box::new(rhs), vec![lhs]),
                    span,
                };
            }
            return Ok(lhs);
        }
        let ops = BIN_LEVELS[level];
        if ops.is_empty() {
            // Cons level: right-assoc.
            let lhs = self.binary(level + 1)?;
            if self.is(&Tok::ColonColon) {
                self.bump();
                let rhs = self.binary(level)?;
                let span = lhs.span.to(rhs.span);
                return Ok(Expr {
                    kind: ExprKind::BinOp(BinOp::Cons, Box::new(lhs), Box::new(rhs)),
                    span,
                });
            }
            return Ok(lhs);
        }
        let mut lhs = self.binary(level + 1)?;
        loop {
            let op = match self.peek() {
                Tok::PipePipe => BinOp::Or,
                Tok::AmpAmp => BinOp::And,
                Tok::EqEq => BinOp::Eq,
                Tok::Ne => BinOp::Ne,
                Tok::Lt => BinOp::Lt,
                Tok::Le => BinOp::Le,
                Tok::Gt => BinOp::Gt,
                Tok::Ge => BinOp::Ge,
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                _ => break,
            };
            if !ops.contains(&op) {
                break;
            }
            self.bump();
            let rhs = self.binary(level + 1)?;
            let span = lhs.span.to(rhs.span);
            lhs = Expr {
                kind: ExprKind::BinOp(op, Box::new(lhs), Box::new(rhs)),
                span,
            };
        }
        Ok(lhs)
    }

    fn prefix(&mut self) -> KResult<Expr> {
        if self.is(&Tok::Minus) {
            let lo = self.span();
            self.bump();
            let operand = self.prefix()?;
            return Ok(Expr {
                kind: ExprKind::UnOp(UnOp::Neg, Box::new(operand)),
                span: lo.to(self.prev_span()),
            });
        }
        if self.is(&Tok::Pipe) || self.is(&Tok::PipePipe) {
            return self.closure();
        }
        self.postfix()
    }

    /// `|pat, pat| body` or `|| body`: the body is a full expression, so
    /// it extends through binary operators but stops at `,`/`)`/`;`.
    fn closure(&mut self) -> KResult<Expr> {
        let lo = self.span();
        let zero_params = self.eat(&Tok::PipePipe);
        let mut params = Vec::new();
        if !zero_params {
            self.expect(&Tok::Pipe, "`|` opening the closure parameters")?;
            if !self.is(&Tok::Pipe) {
                loop {
                    let mut pat = self.pat()?;
                    if self.eat(&Tok::Colon) {
                        let ty = self.ty_expr()?;
                        pat = Pat::Typed(Box::new(pat), ty);
                    }
                    params.push(pat);
                    if self.eat(&Tok::Comma) {
                        continue;
                    }
                    break;
                }
            }
            self.expect(&Tok::Pipe, "`|` closing the closure parameters")?;
        }
        let body = self.expr()?;
        Ok(Expr {
            kind: ExprKind::Lambda {
                params,
                body: Box::new(body),
            },
            span: lo.to(self.prev_span()),
        })
    }

    fn postfix(&mut self) -> KResult<Expr> {
        let mut atom = self.atom()?;
        loop {
            if self.eat(&Tok::Question) {
                let span = atom.span.to(self.prev_span());
                atom = Expr {
                    kind: ExprKind::Try(Box::new(atom)),
                    span,
                };
                continue;
            }
            if self.is(&Tok::LBrace) {
                // `name { field = value, ... }`: a record literal. The
                // lookahead keeps `{`-blocks after expressions working.
                let Some(name) = expr_path(&atom) else {
                    break;
                };
                let opens_record = matches!(
                    (self.tokens.get(self.pos + 1), self.tokens.get(self.pos + 2)),
                    (Some(Token { tok: Tok::RBrace, .. }), _) | (Some(Token { tok: Tok::Ident(_), .. }), Some(Token { tok: Tok::Eq, .. }))
                );
                if !opens_record {
                    break;
                }
                let open_span = atom.span;
                self.bump();
                let mut fields = Vec::new();
                if !self.is(&Tok::RBrace) {
                    loop {
                        let field = self.ident("a record field name")?;
                        self.expect(&Tok::Eq, "`=` after the record field name")?;
                        let value = self.expr()?;
                        fields.push((field, value));
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&Tok::RBrace, "`}` after the record fields")?;
                atom = Expr {
                    kind: ExprKind::Record { ctor: name, fields },
                    span: open_span.to(self.prev_span()),
                };
                continue;
            }
            if self.eat(&Tok::Dot) {
                if self.eat(&Tok::Await) {
                    let span = atom.span.to(self.prev_span());
                    atom = Expr {
                        kind: ExprKind::Await(Box::new(atom)),
                        span,
                    };
                    continue;
                }
                let member = self.ident("a member name after `.`")?;
                let span = atom.span.to(self.prev_span());
                atom = Expr {
                    kind: ExprKind::Field {
                        base: Box::new(atom),
                        field: member,
                    },
                    span,
                };
                continue;
            }
            if self.is(&Tok::LParen) {
                let callee_span = atom.span;
                self.bump();
                let mut args = Vec::new();
                if !self.is(&Tok::RParen) {
                    loop {
                        args.push(self.expr()?);
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&Tok::RParen, "`)` after the call arguments")?;
                atom = Expr {
                    kind: ExprKind::Call(Box::new(atom), args),
                    span: callee_span.to(self.prev_span()),
                };
                continue;
            }
            break;
        }
        Ok(atom)
    }

    fn atom(&mut self) -> KResult<Expr> {
        let lo = self.span();
        if self.is(&Tok::Eof) {
            return Err(Diag::incomplete("expected an expression here", lo));
        }
        match self.peek() {
            Tok::Int(_) => {
                let Token { tok: Tok::Int(v), span } = self.bump() else { unreachable!() };
                Ok(Expr { kind: ExprKind::Int(v), span })
            }
            Tok::Float(_) => {
                let Token { tok: Tok::Float(v), span } = self.bump() else { unreachable!() };
                Ok(Expr { kind: ExprKind::Float(v), span })
            }
            Tok::Char(_) => {
                let Token { tok: Tok::Char(c), span } = self.bump() else { unreachable!() };
                Ok(Expr { kind: ExprKind::Char(c), span })
            }
            Tok::Str(_) => {
                let Token { tok: Tok::Str(s), span } = self.bump() else { unreachable!() };
                Ok(Expr { kind: ExprKind::Str(s), span })
            }
            Tok::True | Tok::False => {
                let Token { tok, span } = self.bump();
                Ok(Expr {
                    kind: ExprKind::BoolConst(tok == Tok::True),
                    span,
                })
            }
            Tok::Underscore => {
                self.bump();
                Err(Diag::new("`_` is not a value; use it only in patterns", lo))
            }
            Tok::Ident(_) => {
                let name = self.ident("a value")?;
                if self.eat(&Tok::ColonColon) {
                    let method = self.ident("a method name")?;
                    self.expect(&Tok::LParen, "`(` after the method name: a class method is always applied")?;
                    let mut args = Vec::new();
                    if !self.is(&Tok::RParen) {
                        loop {
                            args.push(self.expr()?);
                            if self.eat(&Tok::Comma) {
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect(&Tok::RParen, "`)` after the call arguments")?;
                    return Ok(Expr {
                        kind: ExprKind::ClassSel { class: name, method, args },
                        span: lo.to(self.prev_span()),
                    });
                }
                Ok(Expr {
                    kind: ExprKind::Var(name),
                    span: lo.to(self.prev_span()),
                })
            }
            Tok::Builtin(_) => {
                let Token { tok: Tok::Builtin(name), span } = self.bump() else {
                    unreachable!("checked above");
                };
                Ok(Expr {
                    kind: ExprKind::Var(crate::prelude::builtin_name(&name)),
                    span,
                })
            }
            Tok::Shell { .. } => {
                let Token {
                    tok: Tok::Shell { src, capture },
                    span,
                } = self.bump()
                else {
                    unreachable!("checked above");
                };
                let base = span.lo + if capture { 2 } else { 1 };
                let raw = crate::shell::parse_raw(span.file, &src, base)?;
                let command = self.convert_shell(raw)?;
                Ok(Expr {
                    kind: ExprKind::Shell {
                        command: Box::new(command),
                        capture,
                    },
                    span,
                })
            }
            Tok::LParen => {
                self.bump();
                self.paren_expr(lo)
            }
            Tok::LBracket => self.list_expr(lo),
            Tok::If | Tok::Match | Tok::LBrace => {
                let kind = self.expr_raw()?;
                Ok(Expr {
                    kind,
                    span: lo.to(self.prev_span()),
                })
            }
            other => Err(Diag::new(format!("expected an expression, found {}", other.describe()), lo)),
        }
    }

    /// After an opening `(`: unit, grouping, or a tuple.
    fn paren_expr(&mut self, open: Span) -> KResult<Expr> {
        if self.is(&Tok::RParen) {
            self.bump();
            return Ok(Expr {
                kind: ExprKind::Unit,
                span: open.to(self.prev_span()),
            });
        }
        let first = self.expr()?;
        if self.is(&Tok::Comma) {
            let mut items = vec![first];
            while self.eat(&Tok::Comma) {
                items.push(self.expr()?);
            }
            self.expect(&Tok::RParen, "`)` closing the tuple")?;
            Ok(Expr {
                kind: ExprKind::Tuple(items),
                span: open.to(self.prev_span()),
            })
        } else {
            self.expect(&Tok::RParen, "`)` closing the parenthesized expression")?;
            Ok(Expr {
                kind: first.kind,
                span: first.span,
            })
        }
    }

    /// `pat (',' pat)*` followed by `) ->`, or `() ->`. Returns `None`
    /// when the shape is not a lambda. Errors are abortive (the caller
    /// rewinds either way).
    fn convert_shell(&mut self, raw: crate::shell::RawCommand) -> KResult<ShellCommand> {
        let stages = raw.stages.into_iter().map(|stage| self.convert_stage(stage)).collect::<KResult<Vec<_>>>()?;
        Ok(ShellCommand { stages, span: raw.span })
    }

    fn convert_stage(&mut self, raw: crate::shell::RawStage) -> KResult<ShellStage> {
        let words = raw.words.into_iter().map(|word| self.convert_word(word)).collect::<KResult<Vec<_>>>()?;
        Ok(ShellStage { words, span: raw.span })
    }

    fn convert_word(&mut self, raw: crate::shell::RawWord) -> KResult<ShellWord> {
        let parts = raw
            .parts
            .into_iter()
            .map(|part| match part {
                crate::shell::RawPart::Lit(text) => Ok(ShellWordPart::Lit(text)),
                crate::shell::RawPart::Interp { src, span } => {
                    let tokens = crate::lexer::lex_fragment(span.file, &src, span.lo)?;
                    let expr = parse_expr_fragment(tokens)?;
                    Ok(ShellWordPart::Interp(expr))
                }
                crate::shell::RawPart::Capture(command) => Ok(ShellWordPart::Capture(Box::new(self.convert_shell(*command)?))),
            })
            .collect::<KResult<Vec<_>>>()?;
        Ok(ShellWord { parts, span: raw.span })
    }

    fn list_expr(&mut self, open: Span) -> KResult<Expr> {
        self.bump();
        let mut items = Vec::new();
        if !self.is(&Tok::RBracket) {
            loop {
                items.push(self.expr()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
        }
        self.expect(&Tok::RBracket, "`]` closing the list")?;
        Ok(Expr {
            kind: ExprKind::List(items),
            span: open.to(self.prev_span()),
        })
    }
}

fn expr_path(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Var(name) => Some(name.clone()),
        ExprKind::Field { base, field } => Some(format!("{}.{}", expr_path(base)?, field)),
        _ => None,
    }
}

fn is_item_start(tok: &Tok) -> bool {
    matches!(
        tok,
        Tok::Type | Tok::Class | Tok::Instance | Tok::Extern | Tok::Async | Tok::Fn | Tok::Let | Tok::Import | Tok::Use | Tok::Priv
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Delim {
    Brace,
    Paren,
    Bracket,
}

/// Remove the nearest matching opening delimiter, discarding any
/// unterminated delimiters nested inside it.
fn close_delim(open: &mut Vec<Delim>, close: Delim) {
    if let Some(index) = open.iter().rposition(|&delim| delim == close) {
        open.truncate(index);
    }
}

/// `|>` binds tighter than comparisons but looser than `::`.
const PIPE_LEVEL: usize = 3;

const BIN_LEVELS: &[&[BinOp]] = &[
    &[BinOp::Or],
    &[BinOp::And],
    &[BinOp::Eq, BinOp::Ne, BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge],
    // Value pipe: handled before the operator table.
    &[],
    // Cons: right-assoc, sits between the pipe and addition.
    &[],
    &[BinOp::Add, BinOp::Sub],
    &[BinOp::Mul, BinOp::Div],
];
