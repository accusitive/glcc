//! `glcpp` command driver: currently the same stub as `glcc`, tagged as
//! the C++ flavor. EDG wiring is parked until the backend returns.

use glcc::driver;

fn main() {
    driver::install_edg_ops(driver::EdgOps {
        frontend: |args| edg_sys::frontend(args),
        frontend_capture: |args| edg_sys::frontend_capture(args),
        libc_dir: option_env!("GLCPP_LIBC_DIR").map(std::path::PathBuf::from),
    });
    let code = driver::run(std::env::args_os().skip(1).collect(), driver::Flavor::Cpp);
    std::process::exit(code);
}
