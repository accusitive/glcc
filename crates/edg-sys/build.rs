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
    println!("cargo:rerun-if-changed={}", compiler_src.display());
    println!("cargo:rerun-if-changed={}", macro_root.display());
    println!("cargo:rerun-if-changed={}", workspace_root.join("edg/util/mk_errinfo.c").display());

    // ---- 1. Macro defines -------------------------------------------------
    let tool = workspace_root.join("edg/dev_tools/bin/edg-cmakedef").to_string_lossy().into_owned();
    let mut defs = BTreeMap::<String, Option<String>>::new();
    defs.extend(cmakedef_values(&macro_root, &tool, "linux-gcc/base.cmakedef"));
    defs.extend(cmakedef_values(&macro_root, &tool, "linux-gcc/cpfe.cmakedef"));

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
        "c_gen_be.c",
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
    build.compile("edg_frontend");

    // ---- 4. EDG C++ runtime (libC.a) --------------------------------------
    // Exception handling, RTTI, `new`/`delete`, and friends live here
    // (`edg/lib_src`). It must be built by cpfe itself (EDG compiles its
    // own runtime), so drive EDG's cmake target rather than cc: configure
    // once with the same preset the checkout documents, then build only
    // the host runtime. The archive is staged into OUT_DIR and its path
    // published as `cargo:libc_dir` metadata for glcpp's build script
    // (which bakes it into the driver); `rustc-link-search` propagates
    // to every downstream link automatically.
    println!("cargo:rerun-if-changed={}", workspace_root.join("edg/lib_src").display());
    let edg_build = workspace_root.join("edg/build/gcc");
    if !edg_build.join("CMakeCache.txt").exists() {
        run(Command::new("cmake").arg("--preset").arg("linux-gcc-debug").current_dir(workspace_root.join("edg")));
    }
    let staged = out_dir.join("libC.a");
    let built = edg_build.join("lib_linux_x86_64/libC.a");
    let newest_input = newest_mtime(&workspace_root.join("edg/lib_src"));
    let staged_fresh = staged.exists() && mtime(&staged).is_some_and(|o| newest_input.is_none_or(|i| o >= i)) && built.exists();
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

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:libc_dir={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=edg_frontend");
    println!("cargo:rustc-link-lib=stdc++");
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
