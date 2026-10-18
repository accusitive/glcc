use glcc::driver;

fn main() {
    let code = driver::run(std::env::args_os().skip(1).collect(), driver::Flavor::C);
    std::process::exit(code);
}
