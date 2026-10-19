<#
.SYNOPSIS
  Build SQLite on Windows with glcc and run its "veryquick" Tcl suite.

.DESCRIPTION
  The Windows counterpart of download-sqlite.sh. It builds a Tcl 8.6
  shell, generates SQLite's amalgamation, builds the testfixture harness
  with the MSVC toolchain, recompiles sqlite3.c with glcc, relinks
  testfixture against that object, and runs test/veryquick.test.

  Requires an MSVC environment: run from a Developer PowerShell, after
  ilammy/msvc-dev-cmd on CI, or let the script locate vcvars64.bat.

  Environment:
    GLCC              path to glcc (default: target/debug/glcc.exe)
    SQLITE_WORKDIR    scratch/download dir (default: %TEMP%\glcc-sqlite)
    TCLSH_CMD         an existing tclsh86t.exe to reuse (skips the build)
    TCL_LIBRARY       Tcl script library matching TCLSH_CMD
#>
[CmdletBinding()]
param(
  [string]$Version = "3540100",
  [string]$Url = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
  param([scriptblock]$Command, [string]$What)
  & $Command
  if ($LASTEXITCODE -ne 0) { throw "$What failed (exit $LASTEXITCODE)" }
}

function Import-MsvcEnvironment {
  if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }

  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) {
    throw "cl.exe is not on PATH and vswhere.exe was not found; run from a Developer prompt"
  }
  $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $install) { throw "no MSVC installation found" }
  $vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
  if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

  foreach ($line in (cmd /c "`"$vcvars`" >nul 2>&1 && set")) {
    if ($line -match '^([^=]+)=(.*)$') {
      Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
  }
  if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "failed to import the MSVC environment from $vcvars"
  }
}

# The harness compile line carries SQLite's own -D/-I set. Reusing it
# keeps the glcc build in step with the pinned source instead of copying
# a flag list that upstream can change.
function Get-GlccFlags {
  param([string]$Log)

  $line = Get-Content $Log | Where-Object { $_ -match 'sqlite3\.c' -and $_ -match '(^|\s)cl(\.exe)? ' } |
    Select-Object -First 1
  if (-not $line) { throw "could not find the testfixture compile line in $Log" }

  $flags = New-Object System.Collections.Generic.List[string]
  foreach ($token in ($line.Trim() -split '\s+')) {
    if ($token -like '-DSQLITE_PRIVATE=*') { $flags.Add("-DSQLITE_PRIVATE="); continue }
    if ($token -like '-D*' -or $token -like '-I*' -or $token -match '^-O[0-9sgz]?$') {
      $flags.Add($token)
    }
    # Everything else is cl-only (-W4, -MT, -Zi, -wd*, -fp:*, -Fe*, /link)
    # or a source file; glcc does not understand those.
  }
  return $flags.ToArray()
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$glcc = if ($env:GLCC) { $env:GLCC } else { Join-Path $repo "target\debug\glcc.exe" }
if (-not (Test-Path $glcc)) { throw "glcc not found at $glcc (run 'cargo build' or set GLCC=...)" }

$work = if ($env:SQLITE_WORKDIR) { $env:SQLITE_WORKDIR } else {
  Join-Path ([System.IO.Path]::GetTempPath()) "glcc-sqlite"
}
New-Item -ItemType Directory -Force -Path $work | Out-Null

if (-not $Url) { $Url = "https://sqlite.org/2026/sqlite-src-$Version.zip" }
$src = Join-Path $work "sqlite-src-$Version"

Import-MsvcEnvironment

# --- Tcl 8.6 -------------------------------------------------------------
# SQLite's suite does not work with Tcl 9 (its regexp expectations start
# with '-'), and the Windows runner ships no Tcl at all.
$tclsh = $env:TCLSH_CMD
if (-not $tclsh) {
  $tclTag = "core-8-6-16"
  $tclRoot = Join-Path $work "tcl-$tclTag"
  $tclsh = Get-ChildItem $tclRoot -Recurse -Filter "tclsh86t.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName

  if (-not $tclsh) {
    $tgz = Join-Path $work "$tclTag.tar.gz"
    if (-not (Test-Path $tclRoot)) {
      Write-Host "Downloading Tcl $tclTag"
      Invoke-WebRequest -UseBasicParsing "https://github.com/tcltk/tcl/archive/refs/tags/$tclTag.tar.gz" -OutFile $tgz
      Invoke-Checked { tar -xzf $tgz -C $work } "extracting Tcl"
    }
    # win/makefile.vc stamps tclUuid.h from manifest.uuid via `git rev-parse`;
    # a fixed uuid keeps the tarball build hermetic and git-free.
    $uuid = Join-Path $tclRoot "manifest.uuid"
    if (-not (Test-Path $uuid)) {
      Set-Content -Path $uuid -Value "git-0000000000000000000000000000000000000000"
    }
    Write-Host "Building Tcl $tclTag"
    Push-Location (Join-Path $tclRoot "win")
    try { Invoke-Checked { nmake -nologo -f makefile.vc release } "building Tcl" }
    finally { Pop-Location }

    $tclsh = Get-ChildItem $tclRoot -Recurse -Filter "tclsh86t.exe" |
      Select-Object -First 1 -ExpandProperty FullName
    if (-not $tclsh) { throw "the Tcl build produced no tclsh86t.exe" }
  }
}

$tclBin = Split-Path $tclsh -Parent
$tclRoot = Split-Path (Split-Path $tclBin -Parent) -Parent
$tclLibrary = if ($env:TCL_LIBRARY) { $env:TCL_LIBRARY } else { Join-Path $tclRoot "library" }
if (-not (Test-Path $tclLibrary)) { throw "Tcl script library not found at $tclLibrary" }

# --- SQLite source -------------------------------------------------------
if (-not (Test-Path (Join-Path $src "Makefile.msc"))) {
  $zip = Join-Path $work "sqlite-src-$Version.zip"
  Write-Host "Downloading $Url"
  Invoke-WebRequest -UseBasicParsing $Url -OutFile $zip
  Invoke-Checked { Expand-Archive -Path $zip -DestinationPath $work -Force } "extracting SQLite"
}

# nmake macros for a Tcl-enabled testfixture.
$env:TCLVERSION = "86"
$env:TCLSUFFIX = "t"
$env:TCLINCDIR = Join-Path $tclRoot "generic"
$env:TCLLIBDIR = $tclBin
$env:LIBTCL = "tcl86t.lib"
$env:LIBTCLPATH = $tclBin
$env:TCLSH_CMD = $tclsh
$env:TCL_LIBRARY = $tclLibrary
$env:PATH = "$tclBin;$env:PATH"

Push-Location $src
try {
  Write-Host "Generating the SQLite amalgamation"
  Invoke-Checked { nmake -nologo -f Makefile.msc sqlite3.c } "generating sqlite3.c"

  # A dry run of the harness build prints the cl invocation (sources and
  # flags) without compiling anything.
  $dryRun = Join-Path $work "testfixture-cl.txt"
  Invoke-Checked { nmake -nologo -n -f Makefile.msc testfixture.exe *> $dryRun } "collecting compile flags"
  $flags = Get-GlccFlags -Log $dryRun

  # `__try`/`__except` is not implemented yet.
  $flags += "-DSQLITE_OMIT_SEH"
  Write-Host "Compiling sqlite3.c with $glcc"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  Invoke-Checked { & $glcc @flags sqlite3.c -c -o sqlite3-glcc.obj } "compiling sqlite3.c with glcc"
  Write-Host ("glcc compiled sqlite3.c in {0:n0}s" -f $sw.Elapsed.TotalSeconds)

  Remove-Item testfixture.exe -ErrorAction SilentlyContinue
  Write-Host "Building and linking testfixture against the glcc object"
  Invoke-Checked {
    nmake -nologo -f Makefile.msc testfixture.exe SQLITE3C=sqlite3-glcc.obj TLIBS=user32.lib
  } "building testfixture"

  Remove-Item -Recurse -Force testdir -ErrorAction SilentlyContinue
  $log = Join-Path $work "veryquick.log"
  Write-Host "Running the veryquick suite"
  & .\testfixture.exe test\veryquick.test *> $log
  $code = $LASTEXITCODE

  $summary = Select-String -Path $log -Pattern '^(\d+) errors out of (\d+) tests' | Select-Object -Last 1
  if ($code -eq 0 -and $summary -and $summary.Matches[0].Groups[1].Value -eq "0") {
    Write-Host $summary.Line.Trim()
    Write-Host "SQLite veryquick suite passed."
    exit 0
  }

  Write-Host "SQLite veryquick suite FAILED (exit $code):" -ForegroundColor Red
  if ($summary) { Write-Host $summary.Line.Trim() }
  Select-String -Path $log -Pattern '^! ' | Select-Object -First 40 | ForEach-Object { $_.Line.Trim() }
  if (-not $summary) {
    Write-Host "No result summary; last lines of ${log}:"
    Get-Content $log | Select-Object -Last 20
  }
  exit 1
}
finally {
  Pop-Location
}
