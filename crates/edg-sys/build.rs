// build.rs (edg-sys) -- Build EDG (C-gen backend) as a static
// library plus a small C++ bridge exposing a stable C ABI, and stage
// EDG's C++ runtime (`libC.a`) for glcpp's link step.
//
//  1. Resolve EDG macro defines with `edg-cmakedef` for `linux-gcc/base` +
//     `linux-gcc/cpfe`, overriding only MAKE_FRONT_END_CALLABLE so the front
//     end can be invoked in-process (edg_main returns instead of exit()ing).
//  2. Generate `cmake_defines.h` + `err_codes.h`/`err_data.h` into OUT_DIR.
//  3. Compile all `edg/src/*.c` as C++14 (as EDG's own build does) plus the
//     bridge into `libedg_frontend.a` via the `cc` crate.
//  4. Configure EDG's cmake once and build `edg-cpp-rt-linux_x86_64`
//     (cpfe compiles its own runtime), staging `libC.a` into OUT_DIR and
//     publishing `cargo:libc_dir` metadata for glcpp.
//
// Generated files are written only when their content changes, and the
// cmake build reruns only when `edg/lib_src` is newer than the staged
// archive, so warm `cargo build` runs relink in seconds.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(cmd: &mut Command) {
    let status = cmd.status().expect("failed to spawn command");
    assert!(status.success(), "command failed: {cmd:?}");
}

fn write_if_changed(path: &Path, content: &[u8]) -> bool {
    match fs::read(path) {
        Ok(existing) if existing == content => false,
        _ => {
            fs::write(path, content).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
            true
        }
    }
}

fn mtime(path: &Path) -> Option<std::time::SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn cmakedef_values(macro_root: &Path, tool: &str, target: &str) -> BTreeMap<String, Option<String>> {
    let out = Command::new("python3")
        .arg(tool)
        .arg(macro_root)
        .arg(target)
        .arg("print-values")
        .output()
        .expect("failed to run edg-cmakedef");
    assert!(out.status.success(), "edg-cmakedef failed for {target}: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).expect("non-utf8 cmakedef output");
    let mut map = BTreeMap::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.to_string(), Some(v.to_string()));
        } else {
            map.insert(line.to_string(), None);
        }
    }
    map
}

fn main() {
    // Nothing links the front end without `cxx`: skip the heavy steps
    // (EDG's 70+ TUs, the runtime cmake build) instead of compiling them
    // into a build tree no one uses. Enabling the feature changes this
    // script's fingerprint, so the real build still happens on demand.
    if env::var_os("CARGO_FEATURE_CXX").is_none() {
        println!("cargo:rerun-if-changed=build.rs");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Workspace layout: glcc/crates/edg-sys -> glcc/ holds `edg/`.
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("edg-sys must live in crates/ at the glcc root")
        .to_path_buf();
    let compiler_src = workspace_root.join("edg/src");
    let macro_root = workspace_root.join("edg/cmake/macro-conf");
    let bridge_dir = manifest_dir.join("bridge");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=bridge/edg.cpp");
    println!("cargo:rerun-if-changed=bridge/mir_gen_be.cpp");
    println!("cargo:rerun-if-changed={}", compiler_src.display());
    println!("cargo:rerun-if-changed={}", macro_root.display());
    println!("cargo:rerun-if-changed={}", workspace_root.join("edg/util/mk_errinfo.c").display());

    // ---- 1. Macro defines -------------------------------------------------
    let tool = workspace_root.join("edg/dev_tools/bin/edg-cmakedef").to_string_lossy().into_owned();
    let mut defs = BTreeMap::<String, Option<String>>::new();
    defs.extend(cmakedef_values(&macro_root, &tool, "linux-gcc/base.cmakedef"));
    defs.extend(cmakedef_values(&macro_root, &tool, "linux-gcc/cpfe.cmakedef"));

    // libquadmath only ships with gcc on x86. Everywhere else turn it off
    // and emulate the float128 conversions instead (APPROXIMATE_QUADMATH;
    // a_host_fp_value is `_Float128` there per targ_def.h, which gcc
    // supports on aarch64). defines_linux.h would auto-enable the
    // emulation, but only under CONFIG_FOR_GPP_HEADER_COMPATIBILITY,
    // which defines.h forces to 0 in the cmake build -- hence explicit.
    // Long double is x87 80-bit extended only on x86; on aarch64 (and
    // riscv64) it is IEEE binary128, and the x87 configuration miscompiles
    // every long double value.
    if !matches!(env::var("CARGO_CFG_TARGET_ARCH").as_deref(), Ok("x86_64") | Ok("x86")) {
        defs.insert("USE_QUADMATH_LIBRARY".to_string(), Some("0".to_string()));
        defs.insert("APPROXIMATE_QUADMATH".to_string(), Some("1".to_string()));
        defs.insert("FP_LONG_DOUBLE_IS_80BIT_EXTENDED".to_string(), Some("0".to_string()));
        defs.insert("FP_LONG_DOUBLE_IS_BINARY128".to_string(), Some("1".to_string()));
        // The linux-gcc cmakedef chain bakes x86-64 values into the
        // UNSUFFIXED TARG_ defaults, which beat defines_linux.h's
        // per-arch auto-detection (#ifndef guards). On aarch64 the setjmp
        // buffer EDG's EH lowering allocates must match glibc's 312-byte
        // jmp_buf -- 25 x86-64 elements overflow it and corrupt the
        // stack. (wchar_t is also unsigned here unlike the baked-in
        // ik_int; fixing that needs a patched $EDG_BASE predefined-macro
        // file, since __WCHAR_TYPE__ comes from there, so it is left
        // divergent from native g++ for now.)
        if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
            defs.insert("TARG_JMP_BUF_NUM_ELEMENTS".to_string(), Some("39".to_string()));
            defs.insert("TARG_JMP_BUF_ELEMENT_INT_KIND".to_string(), Some("((an_integer_kind)ik_long_long)".to_string()));
        }
    }

    //In-process front end: EDG_MAIN returns the status and cleans up IL
    //instead of calling exit(). The C-gen back end, IL lowering, and
    //fixed-point/ifunc lowering defaults from the import chain stay as-is.
    defs.insert("MAKE_FRONT_END_CALLABLE".to_string(), Some("1".to_string()));

    //The `linux-gcc` flavor above imports EDG's debug configuration, which
    //enables EXPENSIVE_CHECKING: cpfe re-validates scope members and pragma
    //lists on nearly every scope pop (~75% of its run time on large C++ TUs
    //like mrustc's codegen_mmir.cpp). The checks are validation-only and do
    //not affect the generated C, so match EDG's release configuration here.
    defs.insert("EXPENSIVE_CHECKING".to_string(), Some("0".to_string()));

    // ---- 2a. cmake_defines.h ----------------------------------------------
    let mut cmake_defines = String::from("// Generated by edg-sys build.rs; do not edit.\n#pragma once\n");
    for (k, v) in &defs {
        cmake_defines.push_str(&format!("#ifndef {k}\n"));
        match v {
            Some(val) => cmake_defines.push_str(&format!("#define {k} {val}\n")),
            None => cmake_defines.push_str(&format!("#define {k} 1\n")),
        }
        cmake_defines.push_str(&format!("#endif /* {k} */\n"));
    }
    write_if_changed(&out_dir.join("cmake_defines.h"), cmake_defines.as_bytes());

    // ---- 2b. err_codes.h / err_data.h via mk_errinfo ------------------------
    let err_inputs = [
        compiler_src.join("error_msg.txt"),
        compiler_src.join("error_tag.txt"),
        workspace_root.join("edg/util/mk_errinfo.c"),
    ];
    let err_outputs = [out_dir.join("err_codes.h"), out_dir.join("err_data.h")];
    let regen_needed = err_outputs.iter().any(|o| !o.exists()) || {
        let newest_input = err_inputs.iter().filter_map(|p| mtime(p)).max();
        let oldest_output = err_outputs.iter().filter_map(|p| mtime(p)).min();
        match (newest_input, oldest_output) {
            (Some(i), Some(o)) => i > o,
            _ => true,
        }
    };
    if regen_needed {
        let bin_dir = out_dir.join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let mk_errinfo = bin_dir.join("mk_errinfo");
        run(Command::new("g++")
            .args(["-std=c++14", "-x", "c++", "-O1", "-o"])
            .arg(&mk_errinfo)
            .arg(workspace_root.join("edg/util/mk_errinfo.c"))
            .arg("-I")
            .arg(&compiler_src));
        let tmp_codes = out_dir.join("err_codes.h.tmp");
        let tmp_data = out_dir.join("err_data.h.tmp");
        run(Command::new(&mk_errinfo)
            .arg(compiler_src.join("error_msg.txt"))
            .arg(compiler_src.join("error_tag.txt"))
            .arg(&tmp_codes)
            .arg(&tmp_data));
        for (tmp, dst) in [(&tmp_codes, &err_outputs[0]), (&tmp_data, &err_outputs[1])] {
            let content = fs::read(tmp).unwrap();
            write_if_changed(dst, &content);
            let _ = fs::remove_file(tmp);
        }
    }

    // ---- 3. Compile EDG front end into libedg.a ------------------------------
    let core: &[&str] = &[
        "attribute.c",
        "cfe.c",
        "class_decl.c",
        "cmd_line.c",
        "const_ints.c",
        "cp_gen_be.c",
        "debug.c",
        "decl_inits.c",
        "decl_spec.c",
        "declarator.c",
        "decls.c",
        "def_arg.c",
        "disambig.c",
        "error.c",
        "expr.c",
        "exprutil.c",
        "extasm.c",
        "fe_init.c",
        "fe_wrapup.c",
        "fixed_pt.c",
        "float_pt.c",
        "floating.c",
        "folding.c",
        "func_def.c",
        "host_envir.c",
        "ifc_modules.c",
        "il.c",
        "il_alloc.c",
        "il_display.c",
        "il_read.c",
        "il_to_str.c",
        "il_walk.c",
        "il_write.c",
        "inline.c",
        "interpret.c",
        "layout.c",
        "lexical.c",
        "literals.c",
        "lookup.c",
        "lower_c99.c",
        "lower_eh.c",
        "lower_il.c",
        "lower_init.c",
        "lower_name.c",
        "macro.c",
        "mem_manage.c",
        "modules.c",
        "ms_attrib.c",
        "overload.c",
        "pch.c",
        "pragma.c",
        "preproc.c",
        "scope_stk.c",
        "src_seq.c",
        "statements.c",
        "symbol_ref.c",
        "symbol_tbl.c",
        "sys_predef.c",
        "target.c",
        "templates.c",
        "trans_copy.c",
        "trans_corresp.c",
        "trans_unit.c",
        "types.c",
    ];
    let optional: &[&str] = &[
        "cfe_daemon.c",
        "ifc_map_functions.c",
        "ifc_map_functions_acc.c",
        "ifc_map_functions_dbg.c",
        "ifc_map_functions_mut.c",
        "ifc_map_functions_val.c",
        "ifc_modules_read.c",
        "ifc_modules_templ.c",
        "ifc_modules_write.c",
        "unicode_name_fsm.c",
    ];

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++14")
        // Always optimized without debuginfo, regardless of the cargo
        // profile: this is third-party code (nobody debugs EDG internals
        // through our build), it runs on every glcpp invocation so -O2
        // pays off, and `-g` on 70+ TUs is what ballooned the binary
        // (~190MB, almost all `.debug_*` sections).
        .opt_level(2)
        .debug(false)
        .warnings(false)
        .flag("-fno-exceptions")
        .flag("-fno-rtti")
        .flag("-fno-unwind-tables")
        .include(&compiler_src)
        .include(&out_dir)
        .define("USE_CMAKE_DEFINES", None);
    for (k, v) in &defs {
        match v {
            Some(val) => {
                build.define(k, Some(val.as_str()));
            }
            None => {
                build.define(k, Some("1"));
            }
        }
    }
    for f in core {
        let p = compiler_src.join(f);
        assert!(p.exists(), "missing EDG source: {}", p.display());
        build.file(p);
    }
    for f in optional {
        let p = compiler_src.join(f);
        if p.exists() {
            build.file(p);
        }
    }
    build.file(bridge_dir.join("edg.cpp"));
    build.file(bridge_dir.join("mir_gen_be.cpp"));
    build.compile("edg_frontend");

    // The legacy C-generating back end compiles separately with its
    // `back_end` symbol renamed, so both back ends can link into one
    // binary and a dispatcher (in mir_gen_be.cpp) picks between them
    // at run time via GLCPP_EDG_BACKEND.
    let mut cgen_build = cc::Build::new();
    cgen_build
        .cpp(true)
        .std("c++14")
        .opt_level(2)
        .debug(false)
        .warnings(false)
        .flag("-fno-exceptions")
        .flag("-fno-rtti")
        .flag("-fno-unwind-tables")
        .include(&compiler_src)
        .include(&out_dir)
        .define("USE_CMAKE_DEFINES", None)
        .define("back_end", Some("c_gen_be_back_end"));
    for (k, v) in &defs {
        match v {
            Some(val) => {
                cgen_build.define(k, Some(val.as_str()));
            }
            None => {
                cgen_build.define(k, Some("1"));
            }
        }
    }
    cgen_build.file(compiler_src.join("c_gen_be.c"));
    cgen_build.compile("edg_c_gen_be");

    // ---- 4. EDG C++ runtime (libC.a) --------------------------------------
    // Exception handling, RTTI, `new`/`delete`, and friends live here
    // (`edg/lib_src`). The archive is staged into OUT_DIR and its path
    // published as `cargo:libc_dir` metadata for glcpp's build script
    // (which bakes it into the driver); `rustc-link-search` propagates
    // to every downstream link automatically.
    let lib_src = workspace_root.join("edg/lib_src");
    println!("cargo:rerun-if-changed={}", lib_src.display());
    if matches!(env::var("CARGO_CFG_TARGET_ARCH").as_deref(), Ok("x86_64") | Ok("x86")) {
        // cpfe self-hosts its runtime (EDG compiles lib_src with itself),
        // so drive EDG's cmake target: configure once with the same preset
        // the checkout documents, then build only the host runtime.
        let edg_build = workspace_root.join("edg/build/gcc");
        if !edg_build.join("CMakeCache.txt").exists() {
            // EDG_BASE explicitly: EDG's host-defaults detection only knows
            // x86_64/amd64 Linux hosts, but the base itself (edg_eccp_config
            // plus per-target include/lib trees) is arch-independent.
            let edg_base = workspace_root.join("edg/bases/docker/dev-env/gcc");
            run(Command::new("cmake")
                .arg("--preset")
                .arg("linux-gcc-debug")
                .arg(format!("-DEDG_BASE={}", edg_base.display()))
                .current_dir(workspace_root.join("edg")));
        }
        let built = edg_build.join("lib_linux_x86_64/libC.a");
        let staged = out_dir.join("libC.a");
        let newest_input = newest_mtime(&lib_src);
        let staged_fresh =
            staged.exists() && mtime(&staged).is_some_and(|o| newest_input.is_none_or(|i| o >= i)) && built.exists();
        if !staged_fresh {
            let jobs = std::thread::available_parallelism().map(|n| n.get().to_string()).unwrap_or_else(|_| "4".to_string());
            run(Command::new("cmake")
                .arg("--build")
                .arg(&edg_build)
                .arg("--target")
                .arg("edg-cpp-rt-linux_x86_64")
                .arg("-j")
                .arg(jobs));
            fs::copy(&built, &staged).unwrap_or_else(|e| panic!("copy {} -> {}: {e}", built.display(), staged.display()));
        }
    } else {
        // Outside x86, cpfe's own runtime build emits generated C with the
        // GNU `__float80`/`__float128` keywords (gcc has those only on
        // x86), so compile the same lib_src sources with the host C++
        // compiler instead. lib_src is ordinary C++ -- the pieces that
        // clash with the libstdc++ headers glcpp links anyway (placement
        // new/delete, std::exception/-bad_alloc/-type_info) are skipped;
        // c99_complex.c needs the complex f80/f128 helpers those keywords
        // buy and is skipped for the same reason.
        build_runtime_with_host_cc(&out_dir, &lib_src, &workspace_root);
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:libc_dir={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=edg_frontend");
    println!("cargo:rustc-link-lib=static=edg_c_gen_be");
    println!("cargo:rustc-link-lib=stdc++");
}

/// Compile `edg/lib_src` with the host C++ compiler into `libC.a`.
/// The `-D` set mirrors what cpfe predefines for a `--building_runtime`
/// compile (captured via `cpfe --list_macros` against the cmake_defines
/// this script generates); keep in sync with the EDG checkout.
fn build_runtime_with_host_cc(out_dir: &Path, lib_src: &Path, workspace_root: &Path) {
    const SOURCES: &[&str] = &[
        "array_del", "array_del_aligned", "array_new", "array_new_aligned",
        "array_nodel", "array_nodel_aligned", "array_nonew", "array_nonew_aligned",
        "del_virt", "delete", "delete_aligned", "delnothrow", "delnothrow_aligned",
        "dtor_list", "edg_exit", "eh_util", "error", "exit", "guard", "main",
        "memzero", "munch_ctors", "munch_dtors", "new", "new_aligned",
        "newhandler", "newnothrow", "newnothrow_aligned", "pure_virt", "rtti",
        "set_new", "sized_array_del", "sized_array_del_aligned", "sized_del",
        "sized_del_aligned", "static_init", "stdexcept", "thread_dtor", "throw",
        "vars", "vec_cctor", "vec_newdel", "vla_alloc",
    ];
    const PREDEFINES: &[(&str, &str)] = &[
        ("__EDG_RUNTIME_USES_NAMESPACES", "1"),
        ("__EDG_IA64_ABI", "1"),
        ("__EDG_IA64_ABI_VARIANT_CTORS_AND_DTORS_RETURN_THIS", "1"),
        ("__EDG_IA64_ABI_USE_INT_STATIC_INIT_GUARD", "1"),
        ("__EDG_TYPE_TRAITS_ENABLED", "1"),
        ("__EDG_CONSTEXPR_ENABLED__", "1"),
        ("__EDG__", "1"),
        ("__EDG_VERSION__", "700"),
        ("__EDG_SIZE_TYPE__", "unsigned long"),
        ("__EDG_PTRDIFF_TYPE__", "long"),
        // aarch64 glibc jmp_buf is 312 bytes (see the TARG_ overrides in
        // main()); 25 x86-64 elements would overflow it.
        ("__EDG_JMP_BUF_ELEMENT_TYPE", "long long"),
        ("__EDG_JMP_BUF_NUM_ELEMENTS", "39"),
        ("__EDG_DELTA_TYPE", "long"),
        ("__EDG_IA64_VTABLE_ENTRY_TYPE", "long"),
        ("__EDG_VIRTUAL_FUNCTION_INDEX_TYPE", "short"),
        ("__EDG_VAR_HANDLE_TYPE", "unsigned short"),
        ("__EDG_REGION_NUMBER_TYPE", "unsigned short"),
        ("__EDG_ETS_FLAG_TYPE", "unsigned int"),
        ("__EDG_NULL_EH_REGION_NUMBER", "65535"),
        ("__EDG_LOWER_VARIABLE_LENGTH_ARRAYS", "0"),
        ("__EDG_IA64_ABI_USE_VARIANT_ARRAY_COOKIES", "1"),
        ("__EDG_ABI_COMPATIBILITY_VERSION", "99999"),
        ("__EDG_ABI_CHANGES_FOR_RTTI", "1"),
        ("__EDG_ABI_CHANGES_FOR_ARRAY_NEW_AND_DELETE", "1"),
        ("__EDG_ABI_CHANGES_FOR_PLACEMENT_DELETE", "1"),
        ("__EDG_BSD", "0"),
        ("__EDG_SYSV", "0"),
        ("__EDG_ANSIC", "1"),
        ("__EDG_CPP11_IL_EXTENSIONS_SUPPORTED", "1"),
        ("__EDG_HOST_HAS_FLOAT16_TYPE", "1"),
        ("__EDG_FLOAT80_ENABLING_POSSIBLE", "1"),
        ("__EDG_FLOAT128_ENABLING_POSSIBLE", "1"),
        ("__EDG_INT128_EXTENSIONS_ALLOWED", "1"),
        ("__EDG_BFLOAT16_ENABLING_POSSIBLE", "0"),
    ];
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++14")
        .opt_level(2)
        .debug(false)
        .warnings(false)
        .flag("-fPIC")
        .include(lib_src)
        .include(workspace_root.join("edg/include_c++"))
        .include(workspace_root.join("edg/include_c99"))
        .define("CHECKING", Some("1"))
        .define("DEBUG", Some("1"));
    for (k, v) in PREDEFINES {
        build.define(k, Some(*v));
    }
    // eh_util.c defines std::uncaught_exception with a different return
    // type than the libstdc++ header declaration; the libstdc++ one is
    // what glcpp programs see, so compile this file without the copy.
    let scratch = out_dir.join("edg-rt");
    fs::create_dir_all(&scratch).unwrap();
    let eh_util = fs::read_to_string(lib_src.join("eh_util.c")).unwrap();
    let patched = eh_util.replace(
        "__bool uncaught_exception() THROW_NOTHING()\n/*\nReturn TRUE if an exception is in the process of being thrown.\n*/\n{\n  return uncaught_exceptions() > 0;\n}  /* uncaught_exception */",
        "",
    );
    assert_ne!(eh_util, patched, "eh_util.c changed upstream; update the uncaught_exception patch");
    let eh_util_path = scratch.join("eh_util.c");
    write_if_changed(&eh_util_path, patched.as_bytes());
    for s in SOURCES {
        build.file(if *s == "eh_util" {
            eh_util_path.clone()
        } else {
            lib_src.join(format!("{s}.c"))
        });
    }
    build.compile("C");
}

/// Newest mtime under `dir` (for staleness checks), or `None` if the tree
/// cannot be walked.
fn newest_mtime(dir: &Path) -> Option<std::time::SystemTime> {
    let mut newest = mtime(dir)?;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Some(t) = mtime(&path) {
                if t > newest {
                    newest = t;
                }
            }
        }
    }
    Some(newest)
}
