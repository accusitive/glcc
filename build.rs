// Build script: stage a glcc runtime like install.sh does, so a fresh
// `cargo build` already gives the driver a runtime to link with.
//
// Two layouts, by host:
//   - ELF hosts (Linux and the BSDs): `~/.local/lib/glcc/glccrt.o`, the
//     SysV runtime with the vendored soft-float quad helpers (entries
//     renamed `__glcc_*`, see rt/softfloat/glcc-fe.c) relocatably linked
//     in; needs `cc` and `ld`.
//   - x86-64 Windows hosts: `~/.local/lib/glcc/glccrt-win64.lib`, the
//     Win64 flavor. Clang assembles rt/glccrt.c's setjmp/longjmp block
//     (LLVM's integrated assembler emits COFF; there is no GAS), glcc
//     itself compiles the quad soft-float subset for the Windows target
//     (`$GLCC`, else the binary a previous `cargo build` left behind),
//     and MSVC's lib.exe combines the objects.
//
// Missing tools warn rather than fail: the compiler binary is useful
// without a runtime, and on hosts with neither layout it still builds.
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const CC_FLAGS: [&str; 6] = ["-c", "-Os", "-fPIC", "-ftls-model=initial-exec", "-fno-asynchronous-unwind-tables", "-fno-stack-protector"];

/// `__glcc_*` renames keeping the vendored soft-float off the toolchain
/// runtime's names, as install.sh does.
const RENAMES: [&str; 11] = [
    "-D__addtf3=__glcc_addtf3",
    "-D__subtf3=__glcc_subtf3",
    "-D__multf3=__glcc_multf3",
    "-D__divtf3=__glcc_divtf3",
    "-D__extendxftf2=__glcc_extendxftf2",
    "-D__trunctfxf2=__glcc_trunctfxf2",
    "-D__floattixf=__glcc_floattixf",
    "-D__floatuntixf=__glcc_floatuntixf",
    "-D__fixxfti=__glcc_fixxfti",
    "-D__fixunsxfti=__glcc_fixunsxfti",
    "-D__clzti2=__glcc_clzti2",
];

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    println!("cargo:rerun-if-changed=rt/glccrt.c");
    if cfg!(target_arch = "x86_64") {
        println!("cargo:rerun-if-changed=rt/softfloat");
    }
    if host_uses_elf() {
        stage_elf(&root);
    } else if cfg!(all(windows, target_arch = "x86_64")) {
        stage_windows(&root);
    } else {
        println!("cargo:warning=runtime staging is implemented for ELF and x86-64 Windows hosts; skipping on this platform");
    }
}

fn stage_elf(root: &Path) {
    let Some(cc) = find_tool("CC", "cc") else {
        println!("cargo:warning=no C compiler (cc); skipping glccrt.o staging");
        return;
    };
    let ld = if cfg!(target_arch = "x86_64") {
        match find_tool("", "ld") {
            Some(ld) => Some(ld),
            None => {
                println!("cargo:warning=no linker (ld) for soft-float staging; skipping glccrt.o");
                return;
            }
        }
    } else {
        None
    };
    let dst_dir = home_dir().join(".local/lib/glcc");
    if !dst_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&dst_dir) {
            panic!("cannot create {}: {e}", dst_dir.display());
        }
    }
    let out = dst_dir.join("glccrt.o");
    if is_fresh(&out, root) {
        return;
    }
    compile(&cc, &root.join("rt/glccrt.c"), &out);
    if let Some(ld) = &ld {
        combine_softfloat(&cc, ld, root, &out);
    }
}

/// The Win64 runtime: rt/glccrt.c's `_WIN32` asm block plus the parts of
/// the vendored soft-float the Windows target can reach (`__glcc_*` quad
/// arithmetic; the x87/`long double` TUs are SysV-only and glcc-fe-win.c
/// replaces the x87 glcc-fe.c), combined into one library.
fn stage_windows(root: &Path) {
    let out = home_dir().join(".local/lib/glcc/glccrt-win64.lib");
    if is_fresh(&out, root) {
        return;
    }
    let Some(clang) = glcc_toolchain::find_clang() else {
        println!("cargo:warning=no clang to assemble the Win64 runtime; skipping glccrt-win64.lib staging");
        return;
    };
    let Some(glcc) = find_glcc(root) else {
        println!("cargo:warning=glcc not found ($GLCC unset, no binary under target/); run `cargo build` again to stage glccrt-win64.lib");
        return;
    };
    let lib_exe = glcc_toolchain::msvc_dirs()
        .and_then(|d| d.bin.as_ref())
        .map(|bin| bin.join("lib.exe"))
        .filter(|lib| lib.is_file());
    let Some(lib_exe) = lib_exe else {
        println!("cargo:warning=MSVC lib.exe not found; skipping glccrt-win64.lib staging");
        return;
    };
    let dst_dir = out.parent().expect("staged lib has a parent");
    if !dst_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(dst_dir) {
            panic!("cannot create {}: {e}", dst_dir.display());
        }
    }
    let objdir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("win64-runtime");
    std::fs::create_dir_all(&objdir).expect("cannot create Win64 runtime objdir");

    let rt_obj = objdir.join("glccrt-win64.obj");
    let args: Vec<OsString> = vec![
        "-c".into(),
        "--target=x86_64-pc-windows-msvc".into(),
        root.join("rt/glccrt.c").into(),
        "-o".into(),
        rt_obj.clone().into(),
    ];
    run(&clang, &args, "clang assembly of rt/glccrt.c");

    // glcc needs no clang predefines here: rt/softfloat keys off
    // `__SIZEOF_FLOAT128__` (int_types.h) and `__clang_major__` (the
    // complex-quad typedef is dead weight for these TUs); grumbling
    // about `__glccrt_f80_*` cannot arise because those TUs stay out.
    const TUS: [&str; 6] = ["addtf3", "subtf3", "multf3", "divtf3", "clzti2", "glcc-fe-win"];
    const WIN_RENAMES: [&str; 5] = [
        "-D__addtf3=__glcc_addtf3",
        "-D__subtf3=__glcc_subtf3",
        "-D__multf3=__glcc_multf3",
        "-D__divtf3=__glcc_divtf3",
        "-D__clzti2=__glcc_clzti2",
    ];
    let softfloat = root.join("rt/softfloat");
    let mut objects = vec![rt_obj];
    for tu in TUS {
        let obj = objdir.join(format!("{tu}.obj"));
        let mut args: Vec<OsString> = vec!["--target=x86_64-windows".into(), "-c".into(), "-Os".into(), format!("-I{}", softfloat.display()).into()];
        args.push("-U__clang_major__".into());
        args.push("-D__SIZEOF_FLOAT128__=16".into());
        args.extend(WIN_RENAMES.map(OsString::from));
        args.push(softfloat.join(format!("{tu}.c")).into());
        args.push("-o".into());
        args.push(obj.clone().into());
        run(&glcc, &args, &format!("glcc compile of {tu}.c"));
        objects.push(obj);
    }

    let mut args: Vec<OsString> = vec!["/nologo".into(), format!("/out:{}", out.display()).into()];
    args.extend(objects.iter().map(|o| o.as_os_str().to_os_string()));
    run(&lib_exe, &args, "lib.exe combining the Win64 runtime");
}

/// The glcc binary to compile the soft-float subset with: `$GLCC`, else
/// the one a previous `cargo build` left in `target/<profile>/`.
/// Compiling it is a bootstrap knot — glcc is needed to build itself a
/// runtime — so the first build passes over with a warning and the next
/// one stages.
fn find_glcc(root: &Path) -> Option<PathBuf> {
    if let Some(v) = std::env::var_os("GLCC") {
        if !v.is_empty() {
            return Some(PathBuf::from(v));
        }
    }
    let target = std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from);
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let exe = target.join(profile).join(format!("glcc{}", std::env::consts::EXE_SUFFIX));
    exe.is_file().then_some(exe)
}

/// Run a tool to completion; build-script staging fails loudly.
fn run(prog: &Path, args: &[OsString], what: &str) {
    let status = Command::new(prog)
        .args(args)
        .spawn()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", prog.display()))
        .wait()
        .expect("tool exited abnormally");
    if !status.success() {
        panic!("{what} failed");
    }
}

/// Whether this host's native objects use ELF (Linux, FreeBSD, ...), the
/// only format `rt/glccrt.c` currently targets. PE/COFF and Mach-O hosts
/// need their own runtime; the check keeps that a missing runtime rather
/// than a broken build.
fn host_uses_elf() -> bool {
    cfg!(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly",
    ))
}

/// `$key` when set and non-empty, else `name` if it runs on PATH.
fn find_tool(key: &str, name: &str) -> Option<OsString> {
    if !key.is_empty() {
        if let Some(v) = std::env::var_os(key) {
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    Command::new(name)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
        .then(|| OsString::from(name))
}

/// The user's home directory: `HOME`, then `USERPROFILE` (Windows), so
/// the staged runtime lands in the same place the driver looks for it.
fn home_dir() -> PathBuf {
    for key in ["HOME", "USERPROFILE"] {
        if let Some(v) = std::env::var_os(key) {
            if !v.is_empty() {
                return PathBuf::from(v);
            }
        }
    }
    PathBuf::from(".")
}

/// Whether the staged object is at least as new as every source it is
/// built from, so `cargo build` does not shell out on every run.
fn is_fresh(out: &Path, root: &Path) -> bool {
    let Ok(out_time) = std::fs::metadata(out).and_then(|m| m.modified()) else {
        return false;
    };
    let runtime = std::fs::metadata(root.join("rt/glccrt.c")).and_then(|m| m.modified());
    if !runtime.is_ok_and(|t| t <= out_time) {
        return false;
    }
    if !cfg!(target_arch = "x86_64") {
        return true;
    }
    let Ok(entries) = std::fs::read_dir(root.join("rt/softfloat")) else {
        return false;
    };
    entries
        .flatten()
        .all(|e| std::fs::metadata(e.path()).and_then(|m| m.modified()).is_ok_and(|t| t <= out_time))
}

fn compile(cc: &OsString, src: &Path, out: &Path) {
    let status = Command::new(cc)
        .args(CC_FLAGS)
        .arg(src)
        .arg("-o")
        .arg(out)
        .spawn()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", cc.to_string_lossy()))
        .wait()
        .expect("cc exited abnormally");
    if !status.success() {
        panic!("cc failed to produce {}", out.display());
    }
}

/// Vendored soft-float, one object per TU and relocatably linked behind
/// the asm runtime; entries are renamed `__glcc_*` so they never
/// collide with (or depend on) toolchain runtimes.
fn combine_softfloat(cc: &OsString, ld: &OsString, root: &Path, out: &Path) {
    const TUS: [&str; 12] = [
        "addtf3",
        "subtf3",
        "multf3",
        "divtf3",
        "extendxftf2",
        "trunctfxf2",
        "floattixf",
        "floatuntixf",
        "fixxfti",
        "fixunsxfti",
        "clzti2",
        "glcc-fe",
    ];
    let dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("softfloat");
    std::fs::create_dir_all(&dir).expect("cannot create softfloat objdir");
    let mut objects = Vec::new();
    for tu in TUS {
        let obj = dir.join(format!("{tu}.o"));
        let status = Command::new(cc)
            .args(CC_FLAGS)
            .arg(format!("-I{}", root.join("rt/softfloat").display()))
            .args(RENAMES)
            .arg(root.join("rt/softfloat").join(format!("{tu}.c")))
            .arg("-o")
            .arg(&obj)
            .spawn()
            .unwrap_or_else(|e| panic!("cannot run {}: {e}", cc.to_string_lossy()))
            .wait()
            .expect("cc exited abnormally");
        if !status.success() {
            panic!("cc failed to produce {}", obj.display());
        }
        objects.push(obj);
    }
    let combined = out.with_file_name("glccrt-combined.o");
    let status = Command::new(ld)
        .arg("-r")
        .arg("-o")
        .arg(&combined)
        .arg(out)
        .args(&objects)
        .spawn()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", ld.to_string_lossy()))
        .wait()
        .expect("ld exited abnormally");
    if !status.success() {
        panic!("ld failed to combine {}", out.display());
    }
    if let Err(e) = std::fs::rename(&combined, out) {
        panic!("cannot move {} into place: {e}", combined.display());
    }
}
