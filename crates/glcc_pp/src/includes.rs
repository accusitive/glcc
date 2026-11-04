//! Default system include search, per target.
//!
//! glcc computes its own `#include <...>` list instead of replaying a
//! clang/gcc search: a toolchain resource directory (for `stddef.h`
//! and friends), then the host's standard libc directories on POSIX,
//! or the MSVC/Windows SDK layout on Windows. Cross targets read the
//! binutils sysroot layout (`/usr/<triple>/include`); layouts that
//! differ (vendor toolchains) need explicit `-I` flags.

use std::path::{Path, PathBuf};

use glcc_frontend::target::{Env, Os, Target};
use glcc_toolchain::msvc_dirs;

/// Default system include directories for `target`, in search order.
/// Callers prepend their own `-I` paths and skip this under
/// `-nostdinc` ([`crate::PpOptions::no_std_includes`]).
pub fn include_paths(target: &Target) -> Vec<PathBuf> {
    // mingw-w64 cross sysroot: the headers are tied to the Windows
    // target, not the host, so the binutils layout `/usr/<triple>/
    // include` is the search path on every host. (Windows hosts with
    // a native mingw install pass its include directory with `-I`.)
    if target.os == Os::Windows && target.env == Env::Gnu {
        let triple = target.triple();
        let mut out = vec![PathBuf::from(format!("/usr/{triple}/include"))];
        // The cross gcc's private headers back what the sysroot
        // expects the compiler to provide (`<malloc.h>` includes the
        // compiler-owned `mm_malloc.h`, and `<stdarg.h>` belongs to
        // the compiler too). Sysroot copies win where both exist.
        if let Some(dir) = newest_versioned_dir(&Path::new("/usr/lib/gcc").join(triple), "include") {
            out.push(dir);
        }
        out.retain(|d| d.is_dir());
        return out;
    }
    let mut out = Vec::new();
    if cfg!(windows) {
        // No fixed POSIX layout: honor the toolchain's `INCLUDE` when a
        // Developer Command Prompt set it, else use the MSVC + Windows
        // SDK layout discovered from the Visual Studio install.
        match std::env::var_os("INCLUDE") {
            Some(value) => out.extend(std::env::split_paths(&value)),
            None => {
                if let Some(msvc) = msvc_dirs() {
                    out.extend(msvc.includes.iter().cloned());
                }
            }
        }
    } else {
        if let Some(dir) = newest_versioned_dir(Path::new("/usr/lib/clang"), "include") {
            out.push(dir);
        } else if let Some(dir) = newest_versioned_dir(Path::new("/usr/lib/gcc"), "include") {
            out.push(dir);
        }
        out.push(PathBuf::from("/usr/local/include"));
        let multiarch = PathBuf::from(format!("/usr/include/{}", target.triple()));
        if multiarch.is_dir() {
            out.push(multiarch);
        }
        out.push(PathBuf::from("/usr/include"));
    }
    out.retain(|d| d.is_dir());
    out
}

/// Newest `<parent>/*/<leaf>` or `<parent>/*/*/<leaf>` (clang keeps
/// versions directly, gcc nests a host triple in between), chosen by
/// numeric-leading directory name.
fn newest_versioned_dir(parent: &Path, leaf: &str) -> Option<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in std::fs::read_dir(parent).ok()?.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let direct = dir.join(leaf);
        if direct.is_dir() {
            consider_versioned(&mut best, &dir, direct);
        } else if let Ok(sub) = std::fs::read_dir(&dir) {
            for sub in sub.flatten() {
                let sub = sub.path();
                let cand = sub.join(leaf);
                if cand.is_dir() {
                    consider_versioned(&mut best, &sub, cand);
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

fn consider_versioned(best: &mut Option<(u64, PathBuf)>, name_dir: &Path, candidate: PathBuf) {
    let key = name_dir
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.split(['.', '-', '_']).next())
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(0);
    if best.as_ref().is_none_or(|(k, _)| key > *k) {
        *best = Some((key, candidate));
    }
}
