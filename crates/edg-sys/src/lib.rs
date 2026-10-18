//! Raw FFI into EDG: the callable C-generating front end, driven
//! in-process by glcpp's driver.

use std::ffi::{c_char, CString};
use std::path::Path;

extern "C" {
    fn glcpp_edg_frontend(argc: i32, argv: *mut *mut c_char) -> i32;
    fn glcpp_edg_frontend_capture(argc: i32, argv: *mut *mut c_char, buf: *mut *mut c_char, len: *mut u64) -> i32;
    fn glcpp_edg_free(ptr: *mut c_char);
}

fn build_argv<P: AsRef<Path>>(args: &[P]) -> (Vec<CString>, Vec<*mut c_char>) {
    let c_args: Vec<CString> = args
        .iter()
        .map(|a| {
            let s = a.as_ref().as_os_str().to_string_lossy();
            CString::new(s.as_bytes()).expect("argument with interior NUL")
        })
        .collect();
    let mut ptrs: Vec<*mut c_char> = c_args.iter().map(|c| c.as_ptr() as *mut c_char).collect();
    ptrs.push(std::ptr::null_mut());
    (c_args, ptrs)
}

/// Run the EDG cpfe front end on the given arguments (first element is the
/// program name). Returns EDG's exit status.
pub fn frontend<P: AsRef<Path>>(args: &[P]) -> i32 {
    let (_store, mut ptrs) = build_argv(args);
    unsafe { glcpp_edg_frontend((ptrs.len() - 1) as i32, ptrs.as_mut_ptr()) }
}

/// Like `frontend`, but captures the process's stdout for the duration of
/// the call (used for `-E`). Returns (exit status, captured bytes).
pub fn frontend_capture<P: AsRef<Path>>(args: &[P]) -> (i32, Vec<u8>) {
    let (_store, mut ptrs) = build_argv(args);
    let mut buf: *mut c_char = std::ptr::null_mut();
    let mut len: u64 = 0;
    let rc = unsafe { glcpp_edg_frontend_capture((ptrs.len() - 1) as i32, ptrs.as_mut_ptr(), &mut buf, &mut len) };
    let bytes = if buf.is_null() {
        Vec::new()
    } else {
        let slice = unsafe { std::slice::from_raw_parts(buf as *const u8, len as usize) };
        let v = slice.to_vec();
        unsafe { glcpp_edg_free(buf) };
        v
    };
    (rc, bytes)
}
