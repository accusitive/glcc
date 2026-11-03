# Build fernmake with fernc and glcc. No dependency tracking: every
# invocation rebuilds from scratch.
#
#   .\build.ps1           build build\fernmake.exe
#   .\build.ps1 run       build, then run the demo
#   .\build.ps1 clean     remove build\
param([string]$Target = "build")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:FERNC) {
  if (Test-Path "..\..\target\debug\fernc.exe") { $env:FERNC = "..\..\target\debug\fernc.exe" }
  else { $env:FERNC = "fernc.exe" }
}
if (-not $env:GLCC) {
  if (Test-Path "..\..\target\debug\glcc.exe") { $env:GLCC = "..\..\target\debug\glcc.exe" }
  else { $env:GLCC = "glcc.exe" }
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
