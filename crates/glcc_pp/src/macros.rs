//! Function-like macro parameter lists: parse the
//! parenthesized list into names plus variadic-ness, and split call
//! arguments on top-level commas with full delimiter-depth tracking
//! (`()`/`[]`/`{}` all nest; strings are atomic tokens).

use glcc_frontend::lexer::{Interner, PpToken, Punctuator};

use crate::expand::Tok;
use crate::{err, PpError};

#[derive(Debug, Clone)]
pub struct MacroParameterList {
    pub parameters: Vec<std::string::String>,
    pub variadic: bool,
    /// GNU `name...` spelling: the last entry of `parameters` names the
    /// variadic pack (clang `MacroInfo::isGNUVarargs`).
    pub named_variadic: bool,
}

/// Parse `(a, b, ...)`-style inner tokens (between the parens) into a
/// parameter list. `loc` names the definition site for diagnostics.
pub fn parse_parameters(inner: &[Tok], loc: &str, interner: &Interner) -> Result<MacroParameterList, PpError> {
    // Drop whitespace; commas separate (no nesting possible in a
    // well-formed list, but tolerate stray groups by depth-splitting).
    let args = split_top_level(inner);
    let mut parameters = Vec::new();
    let mut variadic = false;
    let mut named_variadic = false;
    // `()` and `(void)`-shaped empties: a lone empty piece means no params.
    let args: Vec<&[Tok]> = if args.len() == 1 && args[0].iter().all(|t| matches!(t.tok, PpToken::Ws(_))) {
        Vec::new()
    } else {
        args
    };
    for (i, piece) in args.iter().enumerate() {
        let words: Vec<&Tok> = piece.iter().filter(|t| !matches!(t.tok, PpToken::Ws(_))).collect();
        let last = i + 1 == args.len();
        match words.as_slice() {
            [] => return Err(err(format!("{loc}: empty macro parameter"))),
            [t] if matches!(t.tok, PpToken::Punct(Punctuator::Ellipsis)) => {
                if !last {
                    return Err(err(format!("{loc}: variadic '...' must be the final parameter")));
                }
                variadic = true;
            }
            [t, e] if matches!(e.tok, PpToken::Punct(Punctuator::Ellipsis)) => {
                // GNU `name...` spelling, handled token-level.
                let base = match &t.tok {
                    PpToken::Ident(sym) => interner.resolve(*sym).unwrap_or("").to_string(),
                    _ => String::new(),
                };
                if !last || !is_ident_text(&base) {
                    return Err(err(format!("{loc}: malformed variadic parameter")));
                }
                if parameters.iter().any(|p| p == &base) {
                    return Err(err(format!("{loc}: duplicate macro parameter '{base}'")));
                }
                parameters.push(base);
                variadic = true;
                named_variadic = true;
            }
            [t] => {
                let name = resolve_ident(t, interner);
                if !is_ident_text(&name) {
                    return Err(err(format!("{loc}: macro parameters must be identifiers, not '{name}'")));
                }
                if parameters.iter().any(|p| p == &name) {
                    return Err(err(format!("{loc}: duplicate macro parameter '{name}'")));
                }
                parameters.push(name);
            }
            _ => {
                return Err(err(format!("{loc}: macro parameters must be single identifiers or '...'")));
            }
        }
    }
    Ok(MacroParameterList { parameters, variadic, named_variadic })
}

fn resolve_ident(t: &Tok, interner: &Interner) -> std::string::String {
    match t.tok {
        PpToken::Ident(sym) => interner.resolve(sym).unwrap_or("").to_string(),
        _ => std::string::String::new(),
    }
}

fn is_ident_text(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c == '_' || c.is_ascii_alphabetic() => chars.all(|c| c == '_' || c.is_ascii_alphanumeric()),
        _ => false,
    }
}

/// Split tokens on top-level commas, tracking `()`/`[]`/`{}` depth.
/// Strings/chars are atomic tokens, so they never disturb the depth.
/// Split macro call arguments on top-level commas: only parentheses nest
/// (clang `ReadMacroCallArgumentList` counts parens alone, so a comma
/// inside `{}`/`[]` still separates arguments).
pub fn split_call_args(tokens: &[Tok]) -> Vec<&[Tok]> {
    let mut out = Vec::new();
    let mut depth = 0u32;
    let mut start = 0;
    for (i, t) in tokens.iter().enumerate() {
        match t.tok {
            PpToken::Punct(Punctuator::LParen) => depth += 1,
            PpToken::Punct(Punctuator::RParen) => depth = depth.saturating_sub(1),
            PpToken::Punct(Punctuator::Comma) if depth == 0 => {
                out.push(&tokens[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&tokens[start..]);
    out
}

pub fn split_top_level(tokens: &[Tok]) -> Vec<&[Tok]> {
    let mut out = Vec::new();
    let mut depth = 0u32;
    let mut start = 0;
    for (i, t) in tokens.iter().enumerate() {
        match t.tok {
            PpToken::Punct(Punctuator::LParen) | PpToken::Punct(Punctuator::LBracket) | PpToken::Punct(Punctuator::LBrace) => {
                depth += 1;
            }
            PpToken::Punct(Punctuator::RParen) | PpToken::Punct(Punctuator::RBracket) | PpToken::Punct(Punctuator::RBrace) => {
                depth = depth.saturating_sub(1);
            }
            PpToken::Punct(Punctuator::Comma) if depth == 0 => {
                out.push(&tokens[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&tokens[start..]);
    out
}
