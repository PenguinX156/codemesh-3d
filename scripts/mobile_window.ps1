# Opens CodeMesh 3D in a phone-sized app window (iPhone 17 Pro, 402x874 CSS px) for demos and recordings.
# Usage: .\scripts\mobile_window.ps1            (hosted Modal app)
#        .\scripts\mobile_window.ps1 -Local     (local demo frontend on :5180)
param([switch]$Local, [int]$Width = 402, [int]$Height = 874)

$url = if ($Local) { "http://localhost:5180" } else { "https://fjziemski--codemesh-web.us-east.modal.run" }
$edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
if (-not (Test-Path $edge)) { $edge = "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe" }
# A separate profile makes Edge honour --window-size instead of reusing the last window size.
$profileDir = Join-Path $env:LOCALAPPDATA "CodeMesh3D\mobile-window"
# +16/+40 px for the window frame and title bar so the page itself is Width x Height.
& $edge "--app=$url" "--window-size=$($Width + 16),$($Height + 40)" "--user-data-dir=$profileDir" "--no-first-run"
