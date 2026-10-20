//! A persistent JIT image for fern programs.
//!
//! One [`ExecSession`] owns one Cranelift [`JITModule`]. The C runtime is
//! lowered once; each [`ExecSession::load`] adds the functions and data
//! the MIR module introduces and skips the symbols that are already
//! compiled and finalized. Calls from new code into old code resolve
//! through the module's existing declarations.
//!
//! This is what makes the REPL usable with a heavy library: evaluating
//! an expression compiles that expression, not the library it imports.

use std::collections::{HashMap, HashSet};

use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::FuncId;
use fern::ty::Ty;
use glcc_mir::ir as mir;

unsafe extern "C" {
    fn memcpy(dst: *mut core::ffi::c_void, src: *const core::ffi::c_void, n: usize) -> *mut core::ffi::c_void;
    fn memset(dst: *mut core::ffi::c_void, c: i32, n: usize) -> *mut core::ffi::c_void;
    fn write(fd: i32, buf: *const core::ffi::c_void, n: usize) -> isize;
    fn abort();
}

/// A value returned by a JITed fern function, rendered by the REPL.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    /// A value the REPL cannot print; the type is shown instead.
    Other(Ty),
}

impl Value {
    pub fn render(&self) -> String {
        match self {
            Value::Unit => "()".to_string(),
            Value::Int(value) => value.to_string(),
            Value::Float(value) => value.to_string(),
            Value::Bool(value) => value.to_string(),
            Value::String(value) => value.clone(),
            Value::Other(ty) => format!("= <{ty}>"),
        }
    }
}

/// A persistent JIT session: one module, grown one program at a time.
pub struct ExecSession {
    jit: JITModule,
    funcs: HashMap<String, FuncId>,
    defined_funcs: HashSet<String>,
    defined_data: HashSet<String>,
}

impl Default for ExecSession {
    fn default() -> Self {
        ExecSession::new()
    }
}

impl ExecSession {
    /// A session with the C runtime already compiled and finalized.
    pub fn new() -> ExecSession {
        let mut session = ExecSession {
            jit: build_jit(),
            funcs: HashMap::new(),
            defined_funcs: HashSet::new(),
            defined_data: HashSet::new(),
        };
        session.load_runtime();
        session
    }

    /// Lower `module` into the image, defining only the symbols that are
    /// new. Every symbol is declared, so calls into earlier code bind to
    /// the already-finalized functions.
    pub fn load(&mut self, module: &mir::Module) {
        let target = glcc_lower_cl::Target::native();
        let defined_funcs = &self.defined_funcs;
        let defined_data = &self.defined_data;
        let ids = glcc_lower_cl::lower_module_selected(
            module,
            &mut self.jit,
            target,
            &|name| !defined_funcs.contains(name),
            &|name| !defined_data.contains(name),
        );
        for (id, fid) in ids {
            self.funcs.insert(module.funcs[id].name.clone(), fid);
        }
        for func in &module.funcs {
            self.defined_funcs.insert(func.name.clone());
        }
        for global in &module.globals {
            self.defined_data.insert(global.name.clone());
        }
        self.jit.finalize_definitions().expect("jit finalize");
    }

    /// Drop every user definition and start from a fresh image holding
    /// only the runtime. The REPL uses this when a definition replaces
    /// an earlier one, so stale symbol bodies cannot win.
    pub fn reset(&mut self) {
        self.jit = build_jit();
        self.funcs.clear();
        self.defined_funcs.clear();
        self.defined_data.clear();
        self.load_runtime();
    }

    /// Call a zero-argument function by name.
    pub fn call(&mut self, name: &str, ret: &Ty) -> Value {
        let Some(&id) = self.funcs.get(name) else {
            return Value::Other(ret.clone());
        };
        let addr = self.jit.get_finalized_function(id);
        unsafe {
            if ret.is_unit() || is_named(ret, "Unit") {
                let f: extern "C" fn() = std::mem::transmute(addr);
                f();
                Value::Unit
            } else if is_named(ret, "Int") {
                let f: extern "C" fn() -> i64 = std::mem::transmute(addr);
                Value::Int(f())
            } else if is_named(ret, "Float") {
                let f: extern "C" fn() -> f64 = std::mem::transmute(addr);
                Value::Float(f())
            } else if is_named(ret, "Bool") {
                let f: extern "C" fn() -> bool = std::mem::transmute(addr);
                Value::Bool(f())
            } else if is_named(ret, "String") {
                let f: extern "C" fn() -> *const core::ffi::c_char = std::mem::transmute(addr);
                let ptr = f();
                if ptr.is_null() {
                    Value::String("null".to_string())
                } else {
                    Value::String(std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned())
                }
            } else {
                let f: extern "C" fn() = std::mem::transmute(addr);
                f();
                Value::Other(ret.clone())
            }
        }
    }

    /// Whether `name` is defined in this image.
    pub fn contains(&self, name: &str) -> bool {
        self.funcs.contains_key(name)
    }

    fn load_runtime(&mut self) {
        let runtime = fern::runtime::module();
        let ids = glcc_lower_cl::lower_module(&runtime, &mut self.jit, glcc_lower_cl::Target::native());
        for (id, fid) in ids {
            self.funcs.insert(runtime.funcs[id].name.clone(), fid);
        }
        for func in &runtime.funcs {
            self.defined_funcs.insert(func.name.clone());
        }
        for global in &runtime.globals {
            self.defined_data.insert(global.name.clone());
        }
        self.jit.finalize_definitions().expect("jit finalize runtime");
    }
}

fn build_jit() -> JITModule {
    let mut builder = JITBuilder::with_flags(
        &[("preserve_frame_pointers", "true")],
        cranelift_module::default_libcall_names(),
    )
    .expect("jit builder");
    builder.symbol("memcpy", memcpy as *const u8);
    builder.symbol("memset", memset as *const u8);
    builder.symbol("write", write as *const u8);
    builder.symbol("abort", abort as *const u8);
    JITModule::new(builder)
}

fn is_named(ty: &Ty, want: &str) -> bool {
    matches!(ty, Ty::Named { name, .. } if name == want)
}
