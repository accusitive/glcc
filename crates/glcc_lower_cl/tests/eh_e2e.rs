//! P2 acceptance for `docs/eh-plan.md`: a hand-written MIR module whose
//! thrower calls `__cxa_throw` (libstdc++) and whose catcher has a
//! `.lpad`, compiled through the real object pipeline —
//! [`glcc_lower_cl::lower_module_collecting`] +
//! [`glcc_lower_cl::emit_unwind`] — then linked with a C++ main and run.
//! The unwinder must walk into the Cranelift frame, match `int` against
//! the synthesized LSDA, and hand the landing pad the selector (`throw 42`
//! from `__cxa_throw` itself is the libstdc++ side of the round trip).

use cranelift_codegen::settings::{self, Configurable};
use cranelift_object::{ObjectBuilder, ObjectModule};
use cranelift_module::Module;

const MIR: &str = r#"
.extern @__cxa_allocate_exception sig=(i64) -> ptr(void)
.extern @__cxa_throw sig=(ptr(void), ptr(void), ptr(void)) -> void
.extern @__gxx_personality_v0 sig=() -> i32
.dataextern @_ZTIi ty=ptr(void)

.func @throw_int sig=(i32) -> void
.params %0:i32
b0:
  %1:ptr(void) = call.ext @__cxa_allocate_exception i64 4
  store %1, %0
  %2:ptr(void) = globaladdr @_ZTIi
  call.ext @__cxa_throw ptr(void) %1, ptr(void) %2, ptr(void) null
  unreachable

.func @catch_int sig=(i32) -> i32 personality=@__gxx_personality_v0
.params %0:i32
b0:
  invoke @throw_int sig=(i32) -> void, i32 %0 to b1 unwind b2
b1():
  ret -1
b2(%3:ptr(void), %4:i64) .lpad [catch @_ZTIi]:
  %5:i32 = eh.typeid @_ZTIi
  ret %5
"#;

const MAIN_CPP: &str = r#"
#include <cstdio>

extern "C" int catch_int(int x);

int main() {
    int sel = catch_int(42);
    std::printf("catch_int selector: %d\n", sel);
    return sel == 1 ? 0 : 1;
}
"#;

#[test]
fn eh_object_pipeline_unwinds_end_to_end() {
    let module = glcc_mir::parse_module(MIR).expect("EH module parses");
    assert!(glcc_mir::parse_module(MIR).is_ok()); // parse already ran verification

    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").expect("opt level");
    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .expect("host features")
        .finish(flags)
        .expect("host isa");
    // EH modules own `.eh_frame` (personality CIE + LSDAs).
    let mut builder = ObjectBuilder::new(
        isa.clone(),
        "glcc_eh_e2e".to_string(),
        cranelift_module::default_libcall_names(),
    )
    .expect("object builder");
    builder.unwind_info(false);
    let mut obj = ObjectModule::new(builder);
    let mut eh = Vec::new();
    glcc_lower_cl::lower_module_collecting(
        &module,
        &mut obj,
        glcc_lower_cl::Target::native(),
        &mut String::new(),
        &mut Vec::new(),
        &mut eh,
    );
    let mut product = obj.finish();
    assert!(eh.iter().any(|f| f.eh), "EH functions collected");
    let names: Vec<String> = module.globals.iter().map(|g| g.name.clone()).collect();
    glcc_lower_cl::emit_unwind(&mut product, &eh, &names, &*isa).expect("emit unwind metadata");
    let bytes = product.emit().expect("object emit");

    let out_dir = std::env::temp_dir().join("glcc-eh-e2e");
    std::fs::create_dir_all(&out_dir).unwrap();
    std::fs::write(out_dir.join("eh.o"), &bytes).unwrap();
    std::fs::write(out_dir.join("main.cpp"), MAIN_CPP).unwrap();
    let run = |cmd: &mut std::process::Command| {
        let out = cmd.output().expect("spawn");
        assert!(
            out.status.success(),
            "{cmd:?} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        out
    };
    run(std::process::Command::new("g++")
        .arg("-no-pie")
        .arg("main.cpp")
        .arg("eh.o")
        .arg("-o")
        .arg("eh")
        .current_dir(&out_dir));
    let out = run(std::process::Command::new("./eh").current_dir(&out_dir));
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{stdout}");
    assert!(stdout.contains("catch_int selector: 1"), "{stdout}");
}
