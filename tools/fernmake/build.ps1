# Build fernmake with fernc and glcc. No dependency tracking: every
# invocation rebuilds from scratch.
#
#   .\build.ps1           build build\fernmake.exe
#   .\build.ps1 run       build, then run the demo
#   .\build.ps1 clean     remove build\
param([string]$Target = "build")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# $env: assignments survive this script into the caller (and the CI step
# that runs the examples), so store absolute paths: a tools/fernmake-relative
# path would resolve against whatever directory the next recipe runs in.
# build.sh picks plain shell variables and does not have this problem.
if (-not $env:FERNC) {
  $fernc = @("..\..\target\debug\fernc.exe", "..\..\target\release\fernc.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
  $env:FERNC = if ($fernc) { (Resolve-Path $fernc).Path } else { "fernc.exe" }
}
if (-not $env:GLCC) {
  $glcc = @("..\..\target\debug\glcc.exe", "..\..\target\release\glcc.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
  $env:GLCC = if ($glcc) { (Resolve-Path $glcc).Path } else { "glcc.exe" }
}

switch ($Target) {
  "build" {
    New-Item -ItemType Directory -Force -Path build | Out-Null
    & $env:FERNC -O1 make.fern -o build\fernmake.obj
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $env:GLCC driver.c build\fernmake.obj -o build\fernmake.exe
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  "run" {
    & $PSScriptRoot\build.ps1 build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & .\build\fernmake.exe demo.win.mk run
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  "clean" {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
  }
  default {
    Write-Error "usage: build.ps1 [build|run|clean]"
  }
}
