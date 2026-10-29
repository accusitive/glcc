//! wasm32 entry points: the whole glcc pipeline as a browser cdylib.
//!
//! The page copies source bytes into a [`glcc_alloc`] buffer and calls
//! [`glcc_compile`]; the JSON lands behind [`glcc_result_ptr`] and the
//! emitted object behind [`glcc_object_ptr`] for the server's
//! disassembly endpoint. There is no filesystem and no `clang`: the
//! native `glcc_pp` preprocessor always runs.
//!
//! Rust panics abort inside wasm, so the exporter installs a panic
//! hook that stashes the message in linear memory; the page reads it
//! via [`glcc_panic_ptr`] after catching the trap and starts a fresh
//! instance for the next compile.

use std::sync::{Mutex, Once};

use crate::{Options, opt_from_code, output_json, std_from_code, target_from_code};

static RESULT: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static OBJECT: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static PANIC: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static HOOK: Once = Once::new();

fn install_panic_hook() {
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            // `{info}` carries both location and payload.
            let msg = format!("{info}");
            if let Ok(mut p) = PANIC.lock() {
                *p = msg.into_bytes();
            }
        }));
    });
}

/// Allocate `len` bytes for the caller to fill with the source text.
#[unsafe(no_mangle)]
pub extern "C" fn glcc_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

/// Release a buffer previously handed out by [`glcc_alloc`].
///
/// # Safety
/// `ptr` must come from `glcc_alloc(len)` and must not be used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn glcc_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
    }
}

/// Compile `src[..len]` and stash the JSON result plus object bytes.
/// Returns the JSON length; the pointer is [`glcc_result_ptr`].
///
/// `opt`: 0 = -O0, 1 = speed, 2 = size. `target`: 0 = x86_64,
/// 1 = aarch64, 2 = riscv64. `flags`: bit 0 = `-fglcc-async`, bits
/// 1..=3 = standard (`std_from_code`). Preprocessing always goes
/// through `glcc_pp`; clang is not available inside wasm.
///
/// # Safety
/// `src` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn glcc_compile(src: *const u8, len: usize, opt: u32, target: u32, flags: u32) -> usize {
    install_panic_hook();
    if let Ok(mut p) = PANIC.lock() {
        p.clear();
    }
    let bytes = unsafe { std::slice::from_raw_parts(src, len) };
    let source = String::from_utf8_lossy(bytes);
    let opts = Options {
        opt: opt_from_code(opt),
        target: Some(target_from_code(target)),
        preprocess: true,
        pp_native: Some(true),
        cpp_args: Vec::new(),
        async_enabled: flags & 1 != 0,
        std: std_from_code((flags >> 1) & 0b111),
    };
    let out = crate::compile_object(&source, &opts);
    let json = output_json(&out);
    let object = out.object;
    let json_len = json.len();
    if let Ok(mut r) = RESULT.lock() {
        *r = json.into_bytes();
    }
    if let Ok(mut o) = OBJECT.lock() {
        *o = object;
    }
    json_len
}

/// Pointer to the JSON from the last [`glcc_compile`].
#[unsafe(no_mangle)]
pub extern "C" fn glcc_result_ptr() -> *const u8 {
    RESULT.lock().map(|r| r.as_ptr()).unwrap_or(std::ptr::null())
}

/// Pointer to the object bytes from the last [`glcc_compile`].
#[unsafe(no_mangle)]
pub extern "C" fn glcc_object_ptr() -> *const u8 {
    OBJECT.lock().map(|o| o.as_ptr()).unwrap_or(std::ptr::null())
}

#[unsafe(no_mangle)]
pub extern "C" fn glcc_object_len() -> usize {
    OBJECT.lock().map(|o| o.len()).unwrap_or(0)
}

/// Length of the panic message left by a trapped call; the message
/// itself is behind [`glcc_panic_ptr`].
#[unsafe(no_mangle)]
pub extern "C" fn glcc_panic_len() -> usize {
    PANIC.lock().map(|p| p.len()).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn glcc_panic_ptr() -> *const u8 {
    PANIC.lock().map(|p| p.as_ptr()).unwrap_or(std::ptr::null())
}
