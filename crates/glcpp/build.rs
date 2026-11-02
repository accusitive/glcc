// Build script: bake the staged EDG C++ runtime location into glcpp.
// edg-sys publishes `cargo:libc_dir` metadata (Cargo exposes it as
// `DEP_EDG_LIBC_DIR` since edg-sys sets `links = "edg"`); `rustc-env`
// makes it visible to `env!` in this crate, which forwards it to the
// driver via `EdgOps`.
fn main() {
    // The script still runs when only the workspace is selected and the
    // EDG-backed binary is skipped; only `cxx` builds need the runtime dir.
    if std::env::var_os("CARGO_FEATURE_CXX").is_none() {
        println!("cargo:rerun-if-changed=build.rs");
        return;
    }
    // edg-sys sets `links = "edg"`, so its `cargo:libc_dir` metadata
    // arrives as `DEP_EDG_LIBC_DIR`. It is absent on hosts where the
    // EDG runtime cannot be built (non-x86; see edg-sys/build.rs) --
    // the driver already warns and links against libstdc++ only.
    if let Ok(dir) = std::env::var("DEP_EDG_LIBC_DIR")
        && !dir.is_empty()
    {
        println!("cargo:rustc-env=GLCPP_LIBC_DIR={dir}");
    }
}
