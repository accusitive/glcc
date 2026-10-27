//! Diagnostic codes and spelling suggestions.
//!
//! [`DiagCode`] gives every diagnostic a stable, rust-style code
//! (`E0301`). Codes are attached where the error is built: static
//! messages shared by several sites get theirs from
//! [`DiagCode::for_message`], so the same semantic error carries the
//! same code no matter which parser/checker path reported it.
//!
//! [`did_you_mean`] implements the bounded edit-distance search behind
//! "help: a variable with a similar name exists" suggestions.

use alloc::{vec, vec::Vec};

/// Stable diagnostic code shared by semantically identical errors.
///
/// Codes are grouped by area: `E01xx` declarations, `E02xx` name
/// resolution, `E03xx` expressions, `E04xx` initializers, `E05xx`
/// builtins/layout, `E06xx` constant expressions, `E07xx` builtins,
/// `E08xx` attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagCode {
    // ---- declarations ----
    /// A name declared twice in one scope, or with an incompatible type.
    ConflictingRedeclaration,
    /// A second definition of an object or function.
    Redefinition,
    /// A struct/union/enum tag defined twice in one scope.
    RedefinitionOfTag,
    /// A declaration with no declarator (`int;`, `struct S;`).
    DeclarationDoesNotDeclareAnything,
    MultipleStorageClasses,
    StorageClassInParameter,
    ThreadLocalInParameter,
    ThreadLocalConflicts,
    VariablyModifiedAtFileScope,
    VoidObject,
    AsyncVariadic,
    InvalidRestrict,
    /// `[*]` in a type outside function prototype scope.
    StarBoundOutsidePrototype,

    // ---- name resolution ----
    UndeclaredIdentifier,
    NoSuchMember,
    AmbiguousMember,
    UndefinedLabel,
    TypedefUsedAsValue,

    // ---- expressions ----
    IncompatibleAssignment,
    InvalidBinaryOperands,
    InvalidCompoundAssignmentOperands,
    InvalidRelationalOperands,
    InvalidEqualityOperands,
    NotAnLvalue,
    InvalidCastOperands,
    NotAFunction,
    PointerTypeMismatchInConditional,
    CompoundLiteralFunctionType,
    AutoInitializerNotExpression,
    OverflowResultNotPointer,

    // ---- initializers and designators ----
    ArrayDesignatorNotInteger,
    FieldDesignatorInArrayInitializer,
    FieldDesignatorOnNonStruct,
    UnterminatedInitializer,

    // ---- builtins and layout ----
    InvalidVaArg,
    OffsetofLayout,
    InvalidOffsetofDesignator,
    UnsupportedAtomicBuiltin,

    // ---- constant expressions ----
    ConstexprInitializerNotConstant,

    // ---- attributes ----
    InvalidAlignment,
}

impl DiagCode {
    /// The spelling shown after the severity (`error[E0301]`).
    pub fn as_str(self) -> &'static str {
        match self {
            DiagCode::ConflictingRedeclaration => "E0101",
            DiagCode::Redefinition => "E0102",
            DiagCode::RedefinitionOfTag => "E0103",
            DiagCode::DeclarationDoesNotDeclareAnything => "E0104",
            DiagCode::MultipleStorageClasses => "E0105",
            DiagCode::StorageClassInParameter => "E0106",
            DiagCode::ThreadLocalInParameter => "E0107",
            DiagCode::ThreadLocalConflicts => "E0108",
            DiagCode::VariablyModifiedAtFileScope => "E0109",
            DiagCode::VoidObject => "E0110",
            DiagCode::AsyncVariadic => "E0111",
            DiagCode::InvalidRestrict => "E0112",
            DiagCode::StarBoundOutsidePrototype => "E0113",

            DiagCode::UndeclaredIdentifier => "E0201",
            DiagCode::NoSuchMember => "E0202",
            DiagCode::AmbiguousMember => "E0203",
            DiagCode::UndefinedLabel => "E0204",
            DiagCode::TypedefUsedAsValue => "E0205",

            DiagCode::IncompatibleAssignment => "E0301",
            DiagCode::InvalidBinaryOperands => "E0302",
            DiagCode::InvalidCompoundAssignmentOperands => "E0303",
            DiagCode::InvalidRelationalOperands => "E0304",
            DiagCode::InvalidEqualityOperands => "E0305",
            DiagCode::NotAnLvalue => "E0306",
            DiagCode::InvalidCastOperands => "E0307",
            DiagCode::NotAFunction => "E0308",
            DiagCode::PointerTypeMismatchInConditional => "E0309",
            DiagCode::CompoundLiteralFunctionType => "E0310",
            DiagCode::AutoInitializerNotExpression => "E0311",
            DiagCode::OverflowResultNotPointer => "E0312",

            DiagCode::ArrayDesignatorNotInteger => "E0401",
            DiagCode::FieldDesignatorInArrayInitializer => "E0402",
            DiagCode::FieldDesignatorOnNonStruct => "E0403",
            DiagCode::UnterminatedInitializer => "E0404",

            DiagCode::InvalidVaArg => "E0501",
            DiagCode::OffsetofLayout => "E0502",
            DiagCode::InvalidOffsetofDesignator => "E0503",

            DiagCode::ConstexprInitializerNotConstant => "E0601",

            DiagCode::UnsupportedAtomicBuiltin => "E0701",

            DiagCode::InvalidAlignment => "E0801",
        }
    }

    /// The code for a static diagnostic message. Formatted messages
    /// (which embed a spelling or type) attach their code explicitly at
    /// the construction site.
    pub fn for_message(message: &str) -> Option<DiagCode> {
        Some(match message {
            "conflicting redeclaration" => DiagCode::ConflictingRedeclaration,
            "redefinition" | "redefinition of function" => DiagCode::Redefinition,
            "redefinition of struct/union tag" | "redefinition of enum tag" => DiagCode::RedefinitionOfTag,
            "declaration does not declare anything" => DiagCode::DeclarationDoesNotDeclareAnything,
            "multiple storage class specifiers" => DiagCode::MultipleStorageClasses,
            "storage class specifier in parameter" => DiagCode::StorageClassInParameter,
            "thread-local storage in parameter" => DiagCode::ThreadLocalInParameter,
            "conflicting thread-local specifier" => DiagCode::ThreadLocalConflicts,
            "variably modified type at file scope" => DiagCode::VariablyModifiedAtFileScope,
            "declaration of `void` object" => DiagCode::VoidObject,
            "async function cannot be variadic" => DiagCode::AsyncVariadic,
            "invalid use of 'restrict'" => DiagCode::InvalidRestrict,
            "`[*]` outside function prototype scope" => DiagCode::StarBoundOutsidePrototype,

            "incompatible types in assignment" => DiagCode::IncompatibleAssignment,
            "invalid operands to binary operator" => DiagCode::InvalidBinaryOperands,
            "invalid operands to compound assignment" => DiagCode::InvalidCompoundAssignmentOperands,
            "invalid operands to relational operator" => DiagCode::InvalidRelationalOperands,
            "invalid operands to `==`/`!=`" => DiagCode::InvalidEqualityOperands,
            "operand of `++`/`--` must be an lvalue" => DiagCode::NotAnLvalue,
            "invalid operands to cast" => DiagCode::InvalidCastOperands,
            "called object is not a function" => DiagCode::NotAFunction,
            "pointer type mismatch in conditional expression" => DiagCode::PointerTypeMismatchInConditional,
            "compound literal has function type" => DiagCode::CompoundLiteralFunctionType,
            "`auto` initializer must be an expression" => DiagCode::AutoInitializerNotExpression,
            "overflow builtin result must be a pointer to integer" => DiagCode::OverflowResultNotPointer,

            "array designator must have integer type" => DiagCode::ArrayDesignatorNotInteger,
            "field designator in array initializer" => DiagCode::FieldDesignatorInArrayInitializer,
            "field designator for non-struct type" => DiagCode::FieldDesignatorOnNonStruct,
            "unterminated initializer" => DiagCode::UnterminatedInitializer,

            "invalid type in `va_arg`" => DiagCode::InvalidVaArg,
            "cannot compute layout for `offsetof`" => DiagCode::OffsetofLayout,
            "invalid `.field` designator in `offsetof`" => DiagCode::InvalidOffsetofDesignator,
            "`constexpr` initializer is not a constant expression" => DiagCode::ConstexprInitializerNotConstant,
            _ => return None,
        })
    }
}

/// Default `help:` suggestion for a static diagnostic message. Messages
/// whose suggestion needs the offending spelling attach theirs at the
/// construction site instead.
pub fn help_for_message(message: &str) -> Option<&'static str> {
    Some(match message {
        "declaration does not declare anything" => "remove the declaration or add a declarator",
        "missing type specifier; C89 implicit `int` assumed" => "write the type explicitly, such as `int`, to silence this warning",
        "missing type specifier; ISO C99 and later do not support implicit `int`" => "add an explicit type specifier, such as `int`",
        "field designator in array initializer" => "use an `[index]` designator for array elements",
        _ => return None,
    })
}

/// Maximum edit distance for which `needle` is considered a misspelling
/// of a candidate. Very short names get no suggestions: at one or two
/// characters almost every other identifier is "close".
fn suggestion_threshold(needle_len: usize) -> usize {
    match needle_len {
        0..=2 => 0,
        n => (n / 3).clamp(1, 3),
    }
}

/// The closest candidate to `needle`, if any is within the suggestion
/// threshold. Ties prefer the lexicographically smaller spelling so the
/// result does not depend on hash-map iteration order.
pub fn did_you_mean<'a>(needle: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let needle_len = needle.chars().count();
    let threshold = suggestion_threshold(needle_len);
    if threshold == 0 {
        return None;
    }
    let mut best: Option<(usize, &'a str)> = None;
    for candidate in candidates {
        if candidate == needle {
            continue;
        }
        let candidate_len = candidate.chars().count();
        if needle_len.abs_diff(candidate_len) > threshold {
            continue;
        }
        let distance = edit_distance(needle, candidate, threshold);
        if distance > threshold {
            continue;
        }
        match best {
            Some((best_distance, best_candidate))
                if distance > best_distance || (distance == best_distance && candidate >= best_candidate) => {}
            _ => best = Some((distance, candidate)),
        }
    }
    best.map(|(_, candidate)| candidate)
}

/// Levenshtein distance, giving up (returning `limit + 1`) once no
/// cell in the current row can stay within `limit`.
fn edit_distance(a: &str, b: &str, limit: usize) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > limit {
        return limit + 1;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        let mut row_min = cur[0];
        for (j, cb) in b.iter().enumerate() {
            let substitution = prev[j] + if ca == cb { 0 } else { 1 };
            let insertion = cur[j] + 1;
            let deletion = prev[j + 1] + 1;
            cur[j + 1] = substitution.min(insertion).min(deletion);
            row_min = row_min.min(cur[j + 1]);
        }
        if row_min > limit {
            return limit + 1;
        }
        core::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_typos_within_the_threshold() {
        let names = ["count", "counter", "print", "value"];
        assert_eq!(did_you_mean("cout", names), Some("count"));
        assert_eq!(did_you_mean("prnt", names), Some("print"));
        assert_eq!(did_you_mean("valuee", names), Some("value"));
    }

    #[test]
    fn rejects_distant_and_short_names() {
        let names = ["alpha", "beta", "gamma"];
        assert_eq!(did_you_mean("delta", names), None);
        assert_eq!(did_you_mean("a", names), None);
        assert_eq!(did_you_mean("x", ["y"]), None);
    }

    #[test]
    fn codes_are_stable_for_shared_messages() {
        assert_eq!(DiagCode::for_message("conflicting redeclaration"), Some(DiagCode::ConflictingRedeclaration));
        assert_eq!(DiagCode::for_message("invalid operands to binary operator"), Some(DiagCode::InvalidBinaryOperands));
        assert_eq!(DiagCode::for_message("some formatted message"), None);
        assert_eq!(DiagCode::InvalidBinaryOperands.as_str(), "E0302");
    }
}
