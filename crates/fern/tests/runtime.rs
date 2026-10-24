//! The embedded C runtime: compilation, verification, and the async ABI.

#[test]
fn runtime_compiles_and_exports_async_hooks() {
    let module = fern::runtime::module();
    let errors = glcc_mir::verify::verify_module(&module);
    assert!(errors.is_empty(), "runtime MIR verification failed:\n{}", errors.join("\n"));

    for name in [
        "fern_enter",
        "fern_pin",
        "fern_leave",
        "fern_alloc",
        "__glcc_frame_alloc",
        "__glcc_frame_free",
        "fern_block_on_int",
        "fern_block_on_float",
        "fern_block_on_bool",
        "fern_block_on_string",
        "fern_block_on_unit",
        "fern_io_sleep",
        "fern_io_read_file",
        "fern_io_write_file",
        "fern_io_read_stdin",
        "fern_io_read_line",
    ] {
        assert!(module.funcs.iter().any(|f| f.name == name), "runtime is missing `{name}`");
    }
}
