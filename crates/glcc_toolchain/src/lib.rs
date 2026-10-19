//! Host toolchain discovery: where clang and the Visual Studio / Windows
//! SDK layout live on this machine.
//!
//! The preprocessor seeds its predefines and include paths from clang, the
//! driver needs `link.exe`, `lib.exe`, and the MSVC library directories for
//! Windows-target links, and the build script stages the Win64 runtime with
//! them. None is on `PATH` outside a Developer Command Prompt, so callers
//! look here instead of relying on the environment. Nothing here runs on
//! ELF hosts beyond the `clang` probe.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

/// The clang binary: `clang` on `PATH` when it responds, else the usual
/// install locations (Visual Studio's LLVM component, standalone LLVM).
/// Callers layer `GLCC_PP` on top; this is only the default.
pub fn find_clang() -> Option<PathBuf> {
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE.get_or_init(find_clang_uncached).clone()
}

fn runs(prog: &Path) -> bool {
    Command::new(prog)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn find_clang_uncached() -> Option<PathBuf> {
    if runs(Path::new("clang")) {
        return Some(PathBuf::from("clang"));
    }
    if cfg!(windows) {
        for vs in visual_studio_roots() {
            for suffix in ["VC\\Tools\\Llvm\\x64\\bin\\clang.exe", "VC\\Tools\\Llvm\\bin\\clang.exe"] {
                let cand = vs.join(suffix);
                if cand.is_file() {
                    return Some(cand);
                }
            }
        }
        for root in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Some(dir) = std::env::var_os(root) {
                let cand = PathBuf::from(dir).join("LLVM\\bin\\clang.exe");
                if cand.is_file() {
                    return Some(cand);
                }
            }
        }
    }
    None
}

/// The MSVC include directories, library directories, and host `bin`
/// directory (where `link.exe`/`lib.exe` live), in compiler search order.
pub struct MsvcDirs {
    pub bin: Option<PathBuf>,
    pub includes: Vec<PathBuf>,
    pub libs: Vec<PathBuf>,
}

/// Discover the MSVC toolchain and Windows SDK layout. `None` when the
/// host has neither (non-Windows hosts always yield `None`).
pub fn msvc_dirs() -> Option<&'static MsvcDirs> {
    static CACHE: OnceLock<Option<MsvcDirs>> = OnceLock::new();
    CACHE.get_or_init(find_msvc_dirs).as_ref()
}

#[cfg(not(windows))]
fn find_msvc_dirs() -> Option<MsvcDirs> {
    None
}

#[cfg(windows)]
fn find_msvc_dirs() -> Option<MsvcDirs> {
    let (vc_tools, sdk) = (find_vc_tools(), find_windows_sdk());
    if vc_tools.is_none() && sdk.is_none() {
        return None;
    }
    let mut includes = Vec::new();
    let mut libs = Vec::new();
    let mut bin = None;
    if let Some(vc) = &vc_tools {
        includes.push(vc.join("include"));
        libs.push(vc.join("lib").join("x64"));
        bin = host_bin_dir(vc);
    }
    if let Some(sdk) = &sdk {
        for part in ["ucrt", "shared", "um"] {
            includes.push(sdk.includes.join(part));
        }
        for part in ["ucrt", "um"] {
            libs.push(sdk.libs.join(part).join("x64"));
        }
    }
    includes.retain(|d| d.is_dir());
    libs.retain(|d| d.is_dir());
    Some(MsvcDirs { bin, includes, libs })
}

/// `VC\Tools\MSVC\<version>`: `VCToolsInstallDir` when a Developer Command
/// Prompt set it, else the newest version under the newest Visual Studio
/// install.
#[cfg(windows)]
fn find_vc_tools() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("VCToolsInstallDir") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Some(dir);
        }
    }
    let mut best: Option<(Vec<u32>, PathBuf)> = None;
    for vs in visual_studio_roots() {
        let Ok(entries) = std::fs::read_dir(vs.join("VC\\Tools\\MSVC")) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(version) = parse_version(&entry.file_name().to_string_lossy()) else {
                continue;
            };
            if best.as_ref().is_none_or(|(b, _)| version > *b) {
                best = Some((version, path));
            }
        }
    }
    best.map(|(_, path)| path)
}

/// The newest Windows 10/11 SDK, or the one `WindowsSdkDir` +
/// `WindowsSDKVersion` name (Developer Command Prompt).
#[cfg(windows)]
struct WindowsSdk {
    includes: PathBuf,
    libs: PathBuf,
}

#[cfg(windows)]
fn find_windows_sdk() -> Option<WindowsSdk> {
    if let Some(dir) = std::env::var_os("WindowsSdkDir") {
        let dir = PathBuf::from(dir);
        if let Some(version) = std::env::var_os("WindowsSDKVersion") {
            let version = version.to_string_lossy();
            let version = version.trim_end_matches('\\');
            let (includes, libs) = (dir.join("Include").join(version), dir.join("Lib").join(version));
            if includes.is_dir() && libs.is_dir() {
                return Some(WindowsSdk { includes, libs });
            }
        }
    }
    for root in ["ProgramFiles(x86)", "ProgramFiles"] {
        let Some(pf) = std::env::var_os(root) else {
            continue;
        };
        let root = PathBuf::from(pf).join("Windows Kits\\10");
        let mut best: Option<(Vec<u32>, PathBuf)> = None;
        let Ok(entries) = std::fs::read_dir(root.join("Include")) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() || !root.join("Lib").join(entry.file_name()).is_dir() {
                continue;
            }
            let Some(version) = parse_version(&entry.file_name().to_string_lossy()) else {
                continue;
            };
            if best.as_ref().is_none_or(|(b, _)| version > *b) {
                best = Some((version, path));
            }
        }
        if let Some((_, includes)) = best {
            let libs = root.join("Lib").join(includes.file_name()?);
            return Some(WindowsSdk { includes, libs });
        }
    }
    None
}

/// The `bin` directory holding the x64-build host tools for a `VC\Tools\
/// MSVC\<version>` directory, e.g. `bin\Hostx64\x64` for a 64-bit host.
#[cfg(windows)]
fn host_bin_dir(vc: &Path) -> Option<PathBuf> {
    let host = match std::env::consts::ARCH {
        "x86" => "Hostx86",
        "aarch64" => "Hostarm64",
        _ => "Hostx64",
    };
    let cand = vc.join("bin").join(host).join("x64");
    cand.is_dir().then_some(cand)
}

/// Visual Studio install roots: `vswhere` (any product, any edition),
/// then the well-known install directories.
fn visual_studio_roots() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for root in ["ProgramFiles(x86)", "ProgramFiles"] {
        let Some(pf) = std::env::var_os(root) else {
            continue;
        };
        let vswhere = PathBuf::from(&pf).join("Microsoft Visual Studio\\Installer\\vswhere.exe");
        let Ok(outp) = Command::new(&vswhere).args(["-latest", "-products", "*", "-property", "installationPath"]).output() else {
            continue;
        };
        if outp.status.success() {
            let path = String::from_utf8_lossy(&outp.stdout).trim().to_string();
            if !path.is_empty() {
                out.push(PathBuf::from(path));
            }
        }
    }
    if let Some(pf) = std::env::var_os("ProgramFiles") {
        let pf = PathBuf::from(pf);
        for year in ["2022", "2019"] {
            for edition in ["Enterprise", "Professional", "Community", "BuildTools", "Preview"] {
                out.push(pf.join("Microsoft Visual Studio").join(year).join(edition));
            }
        }
    }
    out
}

/// Parse a dotted version directory name (`14.34.31933`, `10.0.22621.0`)
/// for newest-wins selection; `None` for names that are not versions.
#[cfg(any(windows, test))]
fn parse_version(name: &str) -> Option<Vec<u32>> {
    if name.is_empty() {
        return None;
    }
    name.split('.').map(|part| part.parse().ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_order_numerically_not_lexically() {
        assert!(parse_version("10.0.22621.0") > parse_version("10.0.19041.0"));
        assert!(parse_version("14.34.31933") > parse_version("14.9.31933"));
        assert!(parse_version("2022") > parse_version("2019"));
        assert_eq!(parse_version("not-a-version"), None);
        assert_eq!(parse_version(""), None);
    }
}
