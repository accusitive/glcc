# GLCC - George Lewis's C Compiler

C compiler targeting x86_64 linux/windows, risc-v linux, and aarch64 linux
It was once mostly handwritten but has now been ~90%+ diluted to AI slop. Its tested extensively against existing test suites so I presume its ok. don't use in production
The parser should be fairly portable/extensible if you want to programmatically parse C code and reason about it from Rust without dependencies (works on web for example)

## Building

    cargo build
    cargo test --workspace

The C++ front end (`glcpp`) is opt-in and needs the EDG submodule:

    git submodule update --init edg
    cargo install --path . --locked
    cargo install --path crates/glcpp --features cxx --locked
    ./install.sh             # stage the runtimes in ~/.local/lib/glcc

`cargo install` builds and installs both compilers. `./install.sh` only
builds/copies the runtime files they link against: `glccrt.o` for C, and
EDG's `libC.a` for C++.

## License

Licensed under either of the following, at your option:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)


### what is fred.txt?
an agent made it for a completely unknown reason and im not removing it. 
### what is fern?
agentslop language used to make sure hte MIR is fairly language agnostic. its designed as a functional language. not used for anything