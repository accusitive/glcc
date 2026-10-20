//! The built-in prelude: a fixed source fragment compiled before user
//! items, with spans in [`crate::FileId::PRELUDE`].
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

fun not(x : Bool) : Bool = if x then false else true

extern \"fern_print\" print_raw : String -> Int
extern \"fern_str_concat\" concat : String -> String -> String
extern \"fern_str_identity\" shell_arg : String -> String
extern \"fern_str_of_int\" str_of_int : Int -> String
extern \"fern_str_eq\" str_eq : String -> String -> Bool
extern \"fern_str_len\" str_len : String -> Int
extern \"fern_str_find\" find_raw : String -> String -> Int
extern \"fern_str_slice\" slice_raw : String -> Int -> Int -> String
extern \"fern_getenv\" getenv : String -> String
extern \"fern_chdir\" chdir_raw : String -> Int
extern \"fern_setenv\" setenv_raw : String -> String -> Int
extern \"fern_exit\" exit : Int -> Int
extern \"fern_argv_nil\" argv_nil : Int -> Argv
extern \"fern_argv_cons\" argv_cons : String -> Argv -> Argv
extern \"fern_spawn_status\" spawn_status_raw : String -> Argv -> Int
extern \"fern_spawn_capture\" spawn_capture_raw : String -> Argv -> String
extern \"fern_pipe_nil\" pipe_nil : Int -> Pipe
extern \"fern_pipe_cons\" pipe_cons : Argv -> Pipe -> Pipe
extern \"fern_pipeline_status\" pipeline_status_raw : Pipe -> Int
extern \"fern_pipeline_capture\" pipeline_capture_raw : Pipe -> String

fun print(s : String) : Unit = { let _ = print_raw(s); () }
fun println(s : String) : Unit = { let _ = print_raw(concat(s, \"\\n\")); () }
fun str_of_bool(b : Bool) : String = if b then \"true\" else \"false\"

fun str_slice(s : String, start : Int, len : Int) : String = slice_raw(s, start, len)
fun str_find(needle : String, s : String) : Int = find_raw(s, needle)
fun str_contains(needle : String, s : String) : Bool = find_raw(s, needle) >= 0
fun str_split(sep : String, s : String) : List String =
    if str_len(sep) == 0 then [s]
    else {
        let at = find_raw(s, sep);
        if at < 0 then [s]
        else {
            let n = str_len(sep);
            Cons(slice_raw(s, 0, at), str_split(sep, slice_raw(s, at + n, str_len(s) - at - n)))
        }
    }
fun str_lines(s : String) : List String = str_split(\"\\n\", s)
fun str_join(sep : String, xs : List String) : String = match xs {
    case Nil => \"\"
    case Cons(x, rest) => match rest {
        case Nil => x
        case _ => concat(x, concat(sep, str_join(sep, rest)))
    }
}
fun ok(status : Int) : Bool = status == 0
fun cd(path : String) : Bool = chdir_raw(path) == 0
fun set_env(name : String, value : String) : Bool = setenv_raw(name, value) == 0

fun argv(xs : List String) : Argv = match xs {
    case Nil => argv_nil(0)
    case Cons(x, rest) => argv_cons(x, argv(rest))
}
fun spawn_status(name : String, args : List String) : Int = spawn_status_raw(name, argv(args))
fun spawn_capture(name : String, args : List String) : String = spawn_capture_raw(name, argv(args))
fun spawn(name : String, args : List String) : Bool = spawn_status(name, args) == 0

fun shell_pipe_of(xs : List Argv) : Pipe = match xs {
    case Nil => pipe_nil(0)
    case Cons(stage, rest) => pipe_cons(stage, shell_pipe_of(rest))
}
fun shell_pipeline(pipe : Pipe) : Bool = pipeline_status_raw(pipe) == 0
fun shell_pipeline_capture(pipe : Pipe) : String = pipeline_capture_raw(pipe)

instance Eq String {
    eq(x, y) = str_eq(x, y)
}
";

/// `#name` names a builtin. `#env` and `#set` are the shell-facing
/// spellings of `getenv` and `set_env`; every other name passes through.
pub fn builtin_name(name: &str) -> String {
    match name {
        "env" => "getenv".to_string(),
        "set" => "set_env".to_string(),
        _ => name.to_string(),
    }
}
