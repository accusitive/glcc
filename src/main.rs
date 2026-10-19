use glcc::driver;

fn main() {
    // Parsing recurses on nested C constructs and debug frames are large.
    // Windows gives the main thread 1 MiB by default, which real inputs
    // (sqlite's shell.c) exceed; run the driver on a roomier stack.
    const STACK_SIZE: usize = 32 * 1024 * 1024;
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || driver::run(args, driver::Flavor::C))
        .expect("spawn compiler thread");
    std::process::exit(handle.join().unwrap_or(101));
}
