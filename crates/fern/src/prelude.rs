//! The built-in prelude: a fixed source fragment compiled before user
//! items, with spans tagged by [`crate::PRELUDE_TAG`].
//!
//! Keeping the prelude in fern source means it is checked and lowered by
//! the same pipeline as user code. It gives list syntax (`[...]`, `::`)
//! its constructors, `Maybe`, boolean `not`, and equality over the
//! built-in types.

/// The prelude, in fern syntax.
pub const SOURCE: &str = "\
data List a = Nil | Cons(a, List a)
data Maybe a = Nothing | Just(a)
class Eq a {
    eq : a -> a -> Bool
}
instance Eq Int {
    eq(x, y) = x == y
}
instance Eq Bool {
    eq(x, y) = x == y
}
instance Eq Float {
    eq(x, y) = x == y
}
instance Eq (List a) when Eq a {
    eq(xs, ys) = match xs {
        case Nil => match ys { case Nil => true case Cons(_, _) => false }
        case Cons(x, xs) => match ys {
            case Nil => false
            case Cons(y, ys) => eq(x, y) && eq(xs, ys)
        }
    }
}
instance Eq (Maybe a) when Eq a {
    eq(mx, my) = match mx {
        case Nothing => match my { case Nothing => true case Just(_) => false }
        case Just(x) => match my { case Nothing => false case Just(y) => eq(x, y) }
    }
}

fun not(x) = if x then false else true
";
