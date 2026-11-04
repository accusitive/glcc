//! The built-in prelude: a fixed source fragment compiled before user
//! items, with spans in [`crate::FileId::PRELUDE`].
//!
//! Keeping the prelude in fern source means it is checked and lowered by
//! the same pipeline as user code. It gives list syntax (`[...]`, `::`)
//! its constructors, `Maybe`, boolean `not`, and equality over the
//! built-in types.

/// The prelude, in fern syntax.
pub const SOURCE: &str = "\
type List<A> = | Nil | Cons(A, List<A>)
type Maybe<A> = | Nothing | Just(A)
class Eq<A> {
    eq: (A, A) -> Bool
}
instance Eq<Int> {
    eq(x, y) = x == y
}
instance Eq<Bool> {
    eq(x, y) = x == y
}
instance Eq<Float> {
    eq(x, y) = x == y
}
instance Eq<Char> {
    eq(x, y) = x == y
}
instance<A> Eq<List<A>> when Eq<A> {
    eq(xs, ys) = match xs {
        case Nil => match ys { case Nil => true case Cons(_, _) => false }
        case Cons(x, xs) => match ys {
            case Nil => false
            case Cons(y, ys) => eq(x, y) && eq(xs, ys)
        }
    }
}
instance<A> Eq<Maybe<A>> when Eq<A> {
    eq(mx, my) = match mx {
        case Nothing => match my { case Nothing => true case Just(_) => false }
        case Just(x) => match my { case Nothing => false case Just(y) => eq(x, y) }
    }
}

fn not(x) : Bool -> Bool = if x then false else true

extern \"fern_print\" print_raw : String -> Int
extern \"fern_str_concat\" concat : (String, String) -> String
extern \"fern_str_identity\" shell_arg : String -> String
extern \"fern_str_of_int\" str_of_int : Int -> String
extern \"fern_str_of_byte\" str_of_byte : Int -> String
extern \"fern_str_eq\" str_eq : (String, String) -> Bool
extern \"fern_str_len\" str_len : String -> Int
extern \"fern_str_find\" find_raw : (String, String) -> Int
extern \"fern_str_slice\" slice_raw : (String, Int, Int) -> String
extern \"fern_getenv\" getenv : String -> String
extern \"fern_read_file\" read_file_raw : String -> String
extern \"fern_path_exists\" path_exists_raw : String -> Bool
extern \"fern_path_newer\" path_newer_raw : (String, String) -> Bool
extern \"fern_chdir\" chdir_raw : String -> Int
extern \"fern_setenv\" setenv_raw : (String, String) -> Int
extern \"fern_exit\" exit : Int -> Int
extern \"fern_argv_nil\" argv_nil : Int -> Argv
extern \"fern_argv_cons\" argv_cons : (String, Argv) -> Argv
extern \"fern_spawn_status\" spawn_status_raw : (String, Argv) -> Int
extern \"fern_spawn_capture\" spawn_capture_raw : (String, Argv) -> String
extern \"fern_pipe_nil\" pipe_nil : Int -> Pipe
extern \"fern_pipe_cons\" pipe_cons : (Argv, Pipe) -> Pipe
extern \"fern_pipeline_status\" pipeline_status_raw : Pipe -> Int
extern \"fern_pipeline_capture\" pipeline_capture_raw : Pipe -> String
extern \"fern_block_on_int\" block_on_int : Future<Int> -> Int
extern \"fern_block_on_float\" block_on_float : Future<Float> -> Float
extern \"fern_block_on_bool\" block_on_bool : Future<Bool> -> Bool
extern \"fern_block_on_string\" block_on_string : Future<String> -> String
extern \"fern_block_on_unit\" block_on_unit : Future<Unit> -> Unit

fn print(s) : String -> Unit = { let _ = print_raw(s); () }
fn println(s) : String -> Unit = { let _ = print_raw(concat(s, \"\\n\")); () }
fn str_of_bool(b) : Bool -> String = if b then \"true\" else \"false\"
fn read_file(path) : String -> String = read_file_raw(path)
fn path_exists(path) : String -> Bool = path_exists_raw(path)
fn path_newer(target, prereq) : (String, String) -> Bool = path_newer_raw(target, prereq)

fn str_slice(s, start, len) : (String, Int, Int) -> String = slice_raw(s, start, len)
fn str_find(needle, s) : (String, String) -> Int = find_raw(s, needle)
fn str_contains(needle, s) : (String, String) -> Bool = find_raw(s, needle) >= 0
fn str_split(sep, s) : (String, String) -> List<String> =
    if str_len(sep) == 0 then [s]
    else {
        let at = find_raw(s, sep);
        if at < 0 then [s]
        else {
            let n = str_len(sep);
            Cons(slice_raw(s, 0, at), str_split(sep, slice_raw(s, at + n, str_len(s) - at - n)))
        }
    }
fn str_lines(s) : String -> List<String> = str_split(\"\\n\", s)
fn str_join(sep, xs) : (String, List<String>) -> String = match xs {
    case Nil => \"\"
    case Cons(x, rest) => match rest {
        case Nil => x
        case _ => concat(x, concat(sep, str_join(sep, rest)))
    }
}
fn ok(status) : Int -> Bool = status == 0
fn cd(path) : String -> Bool = chdir_raw(path) == 0
fn set_env(name, value) : (String, String) -> Bool = setenv_raw(name, value) == 0

fn argv(xs) : List<String> -> Argv = match xs {
    case Nil => argv_nil(0)
    case Cons(x, rest) => argv_cons(x, argv(rest))
}
fn spawn_status(name, args) : (String, List<String>) -> Int = spawn_status_raw(name, argv(args))
fn spawn_capture(name, args) : (String, List<String>) -> String = spawn_capture_raw(name, argv(args))
fn spawn(name, args) : (String, List<String>) -> Bool = spawn_status(name, args) == 0

fn shell_pipe_of(xs) : List<Argv> -> Pipe = match xs {
    case Nil => pipe_nil(0)
    case Cons(stage, rest) => pipe_cons(stage, shell_pipe_of(rest))
}
fn shell_pipeline(pipe) : Pipe -> Bool = pipeline_status_raw(pipe) == 0
fn shell_pipeline_capture(pipe) : Pipe -> String = pipeline_capture_raw(pipe)

instance Eq<String> {
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
