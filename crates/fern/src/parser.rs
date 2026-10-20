//! Recursive-descent parser for fern.
//!
//! fern is a small pure ML-family surface:
//!
//! ```text
//! item     := data | class | instance | fun | let
//! data     := 'data' Upper lower* '=' ctor ('|' ctor)*
//! ctor     := Upper ( '(' ty (',' ty)* ')' )?
//! class    := 'class' Upper lower* '{' (lower ':' ty ';')* '}'
//! instance := 'instance' Upper ty ('when' ty (',' ty)*)? '{' method* '}'
//! method   := lower '(' pat-list? ')' '=' expr ';'?       (params may be empty)
//! fun      := 'fun' lower '(' pat-list? ')' '=' expr
//! let      := 'let' lower '=' expr
//! expr     := if | match | block | lambda | prefix
//! if       := 'if' expr 'then' expr 'else' expr
//! match    := 'match' expr '{' ('case' pat ('when' expr)? '=>' expr ';')* '}'
//! block    := '{' ('let' pat '=' expr ';')* (expr ';')* expr? '}'
//! ```
//!
//! Binary operators, loosest first: `||`, `&&`, comparisons, `::`
//! (right-assoc cons), `+ -`, `* /`; unary `-`; postfix calls `f(x, y)`.

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

/// Fail-fast wrapper over [`parse_recovering`]: returns the first
/// diagnostic and discards the partially parsed program.
pub fn parse(tokens: Vec<Token>) -> KResult<Program> {
    let mut out = parse_recovering(tokens);
    if out.diags.is_empty() {
        Ok(out.program)
    } else {
        Err(out.diags.remove(0))
    }
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
        Err(self.complete_if_eof(Diag::new(
            format!("expected {what}, found {}", self.peek().describe()),
            self.span(),
        )))
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if self.is(want) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn upper(&mut self, what: &str) -> KResult<String> {
        match self.bump() {
            Token { tok: Tok::Upper(name), .. } => Ok(name),
            other => Err(Diag::new(format!("expected {what}, found {}", other.tok.describe()), other.span)),
        }
    }

    fn lower(&mut self, what: &str) -> KResult<String> {
        match self.bump() {
            Token { tok: Tok::Lower(name), .. } => Ok(name),
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
            program: Program { items },
            diags,
            incomplete,
        }
    }

    /// Skip to the next top-level item keyword, ignoring keywords nested
    /// inside braces, parentheses, or brackets. Always advances when the
    /// failing item consumed nothing, so the recovery loop terminates.
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
        if self.pos == start && !self.is(&Tok::Eof) {
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
            Tok::Data => ItemKind::Data(self.data_item()?),
            Tok::Class => ItemKind::Class(self.class_item()?),
            Tok::Instance => ItemKind::Instance(self.instance_item()?),
            Tok::Fun => ItemKind::Fun(self.fun_item()?),
            Tok::Let => ItemKind::Let(self.let_item()?),
            other => {
                return Err(self.complete_if_eof(Diag::new(
                    format!("expected an item (`data`, `class`, `instance`, `fun`, or `let`), found {}", other.describe()),
                    self.span(),
                )));
            }
        };
        Ok(Item {
            span: lo.to(self.prev_span()),
            kind,
        })
    }

    fn data_item(&mut self) -> KResult<DataDecl> {
        let start = self.span();
        self.bump();
        let name = self.upper("a data type name")?;
        let mut params = Vec::new();
        while matches!(self.peek(), Tok::Lower(_)) {
            params.push(self.lower("a type parameter")?);
        }
        self.expect(&Tok::Eq, "`=` after the data head")?;
        let mut ctor_names = Vec::new();
        let mut ctors = Vec::new();
        loop {
            let ctor_span = self.span();
            let name = self.upper("a constructor name")?;
            if ctor_names.contains(&name) {
                return Err(Diag::new(format!("duplicate constructor `{name}`"), ctor_span));
            }
            ctor_names.push(name.clone());
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
            ctors.push(CtorDecl { name, fields });
            if self.eat(&Tok::Pipe) {
                continue;
            }
            break;
        }
        Ok(DataDecl {
            span: start.to(self.prev_span()),
            name,
            params,
            ctors,
        })
    }

    fn class_item(&mut self) -> KResult<ClassDecl> {
        let start = self.span();
        self.bump();
        let name = self.upper("a class name")?;
        let mut params = Vec::new();
        while matches!(self.peek(), Tok::Lower(_)) {
            params.push(self.lower("a class parameter")?);
        }
        if let Some(i) = (0..params.len()).find(|&i| params[i + 1..].contains(&params[i])) {
            return Err(Diag::new(format!("duplicate class parameter `{}`", params[i]), start));
        }
        self.expect(&Tok::LBrace, "`{` opening the class body")?;
        let mut method_names = Vec::new();
        let mut methods = Vec::new();
        loop {
            self.eat_seps();
            if self.is(&Tok::RBrace) || self.is(&Tok::Eof) {
                break;
            }
            let name = self.lower("a method name")?;
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
        let class = self.upper("a class name")?;
        let self_ty = self.ty_expr()?;
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
            let name = self.lower("a method name")?;
            if method_names.contains(&name) {
                return Err(Diag::new(format!("duplicate method `{name}`"), self.span()));
            }
            method_names.push(name.clone());
            self.expect(&Tok::LParen, "`(` after the method name")?;
            let params = if self.is(&Tok::RParen) {
                Vec::new()
            } else {
                let mut params = Vec::new();
                loop {
                    params.push(self.pat()?);
                    if self.eat(&Tok::Comma) {
                        continue;
                    }
                    break;
                }
                params
            };
            self.expect(&Tok::RParen, "`)` after the method parameters")?;
            self.expect(&Tok::Eq, "`=` before the method body")?;
            let body = self.expr()?;
            methods.push(MethodBody {
                name,
                params,
                body,
                span: start.to(self.prev_span()),
            });
        }
        self.expect(&Tok::RBrace, "`}` closing the instance body")?;
        Ok(InstanceDecl {
            span: start.to(self.prev_span()),
            class,
            ctx,
            self_ty,
            methods,
        })
    }

    fn fun_item(&mut self) -> KResult<FnDecl> {
        let start = self.span();
        self.bump();
        let name = self.lower("a function name")?;
        self.expect(&Tok::LParen, "`(` after the function name")?;
        let params = if self.is(&Tok::RParen) {
            Vec::new()
        } else {
            let mut params = Vec::new();
            loop {
                params.push(self.pat()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
            params
        };
        self.expect(&Tok::RParen, "`)` after the function parameters")?;
        self.expect(&Tok::Eq, "`=` before the function body")?;
        let body = self.expr()?;
        Ok(FnDecl {
            span: start.to(self.prev_span()),
            name,
            params,
            body,
        })
    }

    fn let_item(&mut self) -> KResult<ValueDecl> {
        let start = self.span();
        self.bump();
        let name = self.lower("a value name")?;
        self.expect(&Tok::Eq, "`=` after the value name")?;
        let body = self.expr()?;
        Ok(ValueDecl {
            span: start.to(self.prev_span()),
            name,
            body,
        })
    }

    // ------------------------------------------------------------ types --

    /// `ty := ty_app ('->' ty)*`: arrows are right-assoc.
    fn ty_expr(&mut self) -> KResult<TyExpr> {
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

    fn ty_app(&mut self) -> KResult<TyExpr> {
        let first = self.ty_atom()?;
        let mut out = first;
        while matches!(self.peek(), Tok::Upper(_) | Tok::Lower(_) | Tok::LParen) {
            let arg = self.ty_atom()?;
            let span = out.span.to(arg.span);
            out = TyExpr {
                kind: TyExprKind::App(Box::new(out), Box::new(arg)),
                span,
            };
        }
        Ok(out)
    }

    fn ty_atom(&mut self) -> KResult<TyExpr> {
        let span = self.span();
        match self.peek() {
            Tok::Upper(_) | Tok::Lower(_) => {
                let name = match self.bump() {
                    Token {
                        tok: Tok::Upper(name) | Tok::Lower(name),
                        ..
                    } => name,
                    _ => unreachable!("checked above"),
                };
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
            Tok::True => {
                self.bump();
                Ok(Pat::Bool(true))
            }
            Tok::False => {
                self.bump();
                Ok(Pat::Bool(false))
            }
            Tok::Float(_) => Err(Diag::new("float literals cannot be matched; compare with `==` instead".to_string(), self.span())),
            Tok::Lower(_) => {
                let name = self.lower("a pattern")?;
                Ok(Pat::Bind(name))
            }
            Tok::Upper(_) => {
                let name = self.upper("a pattern")?;
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
                } else {
                    Ok(Pat::Ctor { name, fields: Vec::new() })
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
        self.postfix()
    }

    fn postfix(&mut self) -> KResult<Expr> {
        let mut atom = self.atom()?;
        loop {
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
                atom = match atom.kind {
                    // `Ctor(args)`: a saturated constructor call.
                    ExprKind::Ctor { name } => Expr {
                        kind: ExprKind::CtorCall { name, args },
                        span: callee_span.to(self.prev_span()),
                    },
                    _ => Expr {
                        kind: ExprKind::Call(Box::new(atom), args),
                        span: callee_span.to(self.prev_span()),
                    },
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
            return Err(Diag::new("expected an expression here", lo));
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
            Tok::Upper(_) => {
                let name = self.upper("a value")?;
                if self.eat(&Tok::ColonColon) {
                    let method = self.lower("a method name")?;
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
                    kind: ExprKind::Ctor { name },
                    span: lo.to(self.prev_span()),
                })
            }
            Tok::Lower(_) => {
                let name = self.lower("a value")?;
                // Bare `x -> body` is a one-parameter lambda.
                if self.is(&Tok::Arrow) {
                    self.bump();
                    let body = self.expr()?;
                    return Ok(Expr {
                        kind: ExprKind::Lambda {
                            params: vec![Pat::Bind(name)],
                            body: Box::new(body),
                        },
                        span: lo.to(self.prev_span()),
                    });
                }
                Ok(Expr {
                    kind: ExprKind::Var(name),
                    span: lo.to(self.prev_span()),
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

    /// After an opening `(`: unit, grouping, tuple, or a paren-list lambda.
    fn paren_expr(&mut self, open: Span) -> KResult<Expr> {
        // Lambda attempt: `() ->`, `(pat, pat, ...) ->`. Rewind on any
        // failure and reparse as a plain expression; patterns are a small
        // subset of the expression grammar, so nothing is lost.
        let mark = self.pos;
        match self.try_lambda_pats() {
            Ok(Some(params)) => {
                self.expect(&Tok::Arrow, "`->` for a lambda")?;
                let body = self.expr()?;
                return Ok(Expr {
                    kind: ExprKind::Lambda { params, body: Box::new(body) },
                    span: open.to(self.prev_span()),
                });
            }
            _ => self.pos = mark,
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
    fn try_lambda_pats(&mut self) -> KResult<Option<Vec<Pat>>> {
        if self.is(&Tok::RParen) {
            self.bump();
            return Ok(self.is(&Tok::Arrow)).map(|is_arrow| if is_arrow { Some(Vec::new()) } else { None });
        }
        if !matches!(self.peek(), Tok::Underscore | Tok::Lower(_) | Tok::Upper(_) | Tok::LParen | Tok::LBracket) {
            return Ok(None);
        }
        let mut params = Vec::new();
        loop {
            params.push(self.pat()?);
            if self.eat(&Tok::Comma) {
                continue;
            }
            break;
        }
        self.expect(&Tok::RParen, "`)` closing the parameters")?;
        if self.is(&Tok::Arrow) { Ok(Some(params)) } else { Ok(None) }
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

fn is_item_start(tok: &Tok) -> bool {
    matches!(
        tok,
        Tok::Data | Tok::Class | Tok::Instance | Tok::Fun | Tok::Let
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

const BIN_LEVELS: &[&[BinOp]] = &[
    &[BinOp::Or],
    &[BinOp::And],
    &[BinOp::Eq, BinOp::Ne, BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge],
    // Cons: right-assoc, sits between comparisons and addition.
    &[],
    &[BinOp::Add, BinOp::Sub],
    &[BinOp::Mul, BinOp::Div],
];
