#Requires -Version 5.1
<#
Builds a fully self-contained, portable Windows x64 AutoAnatomy bundle:
a real Python embeddable distribution + CPU-only torch + all dependencies
+ pre-downloaded model weights, packaged so a user can extract anywhere and
run AutoAnatomy-GUI.bat / AutoAnatomy-CLI.bat with no Python, no pip, and
no internet connection required.

This is a scripted replacement for the manual zip-and-upload process used
for the v0.1.0 release -- run this instead of rebuilding the bundle by hand.

Output is .tar.gz, not .zip: PowerShell's Compress-Archive/Expand-Archive
silently drop the vast majority of torch's files (it ships ~14,000 files,
many with long nested paths under Lib\site-packages\torch\include\ATen\...)
-- confirmed by testing, a real build went from 13,759 files to 1 after a
zip round-trip, with no error, just a broken bundle. tar.exe (built into
Windows 10 1803+ and Windows 11) round-trips every file correctly. Extract
with: tar -xzf AutoAnatomy-Portable-win64.tar.gz (Windows 11's Explorer can
also extract .tar.gz directly via right-click).

Usage:
    .\scripts\build_portable.ps1
    .\scripts\build_portable.ps1 -PythonVersion 3.13.1 -OutputArchive AutoAnatomy-Portable-win64.tar.gz
#>

param(
    [string]$PythonVersion = "3.13.1",
    [string]$OutputArchive = "AutoAnatomy-Portable-win64.tar.gz"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Deliberately NOT under $root: torch ships license files nested extremely
# deep (e.g. .../torch-*.dist-info/licenses/third_party/kineto/libkineto/
# third_party/dynolog/third_party/prometheus-cpp/3rdparty/googletest/
# googlemock/scripts/generator), and combined with $root's own path length
# that reliably exceeds Windows' 260-character MAX_PATH during pip install
# ("[WinError 206] The filename or extension is too long" -- confirmed by
# actually hitting it). Building at the drive root leaves enough headroom.
$buildDir = Join-Path "$env:SystemDrive\" "aa_build"
$stageDir = Join-Path $buildDir "AutoAnatomy"
$pythonDir = Join-Path $stageDir "python"
$weightsDir = Join-Path $stageDir ".autoanatomy"

function Get-FreeDiskGB {
    $drive = Get-PSDrive -Name ($root.Substring(0, 1))
    return [math]::Round($drive.Free / 1GB, 1)
}

Write-Host "== AutoAnatomy portable build ==" -ForegroundColor Cyan
Write-Host "Free disk before build: $(Get-FreeDiskGB) GB"
if ((Get-FreeDiskGB) -lt 5) {
    throw "Less than 5GB free disk space -- the build needs real headroom (downloads + weights + zip). Aborting."
}

# --- 1. Clean previous build ---
if (Test-Path $buildDir) {
    Write-Host "Removing previous build directory..."
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $stageDir -Force | Out-Null

# --- 2. Download + extract the Python embeddable distribution ---
Write-Host "Downloading Python $PythonVersion embeddable distribution..." -ForegroundColor Cyan
$embedZipUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$embedZipPath = Join-Path $buildDir "python-embed.zip"
Invoke-WebRequest -Uri $embedZipUrl -OutFile $embedZipPath
Expand-Archive -Path $embedZipPath -DestinationPath $pythonDir
Remove-Item $embedZipPath

# --- 3. Enable site-packages (embeddable distros ship isolated by default) ---
$pthFile = Get-ChildItem $pythonDir -Filter "python*._pth" | Select-Object -First 1
if (-not $pthFile) { throw "Could not find the embeddable distribution's ._pth file." }
(Get-Content $pthFile.FullName) -replace '^#import site$', 'import site' | Set-Content $pthFile.FullName
Add-Content $pthFile.FullName "`nLib\site-packages"

# --- 4. Bootstrap pip inside the embedded interpreter ---
Write-Host "Bootstrapping pip..." -ForegroundColor Cyan
$getPipPath = Join-Path $buildDir "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath
$py = Join-Path $pythonDir "python.exe"
& $py $getPipPath --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "Failed to bootstrap pip in the embedded Python." }

# --- 5. Install CPU-only torch, then autoanatomy + its remaining dependencies ---
# CPU-only build deliberately: keeps the bundle portable to any Windows machine
# regardless of GPU/driver, at the cost of no GPU acceleration in this build.
Write-Host "Installing CPU-only torch..." -ForegroundColor Cyan
# Pinned, not "latest": 2.13.0+cpu was found to ship without the torch._strobelight
# submodule (torch imports it unconditionally), breaking the app entirely. 2.11.0 is
# the exact version already verified working in the regular (CUDA) dev environment.
& $py -m pip install --no-warn-script-location "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Failed to install torch." }

Write-Host "Installing AutoAnatomy and its remaining dependencies..." -ForegroundColor Cyan
& $py -m pip install --no-warn-script-location "$root"
if ($LASTEXITCODE -ne 0) { throw "Failed to install autoanatomy." }

# nnunetv2's own torchvision pin silently replaces our pinned CPU torch build
# with a plain (non-CPU-index) torch during the install above -- same issue
# already seen and fixed for the CUDA dev environment in setup_env.ps1. Re-pin.
Write-Host "Re-pinning torch==2.11.0+cpu and matching torchvision (nnunetv2's install can override both)..." -ForegroundColor Cyan
# Must re-pin torch AND torchvision together: torchvision ships native C++
# extensions built against a specific torch ABI, so re-pinning torch alone
# (as an earlier version of this script did) leaves a mismatched torchvision
# behind and fails at runtime with "operator torchvision::nms does not
# exist" the moment anything touches it -- caught by testing a real
# segmentation run through the packaged build, not just "check".
& $py -m pip install --force-reinstall --no-deps --no-warn-script-location "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Failed to re-pin torch/torchvision." }

& $py -m pip cache purge | Out-Null

# The embeddable distribution's pip-installed setuptools ends up missing the
# _distutils_hack submodule that distutils-precedence.pth expects, which prints
# a scary (but non-fatal) traceback on every single interpreter startup. We
# don't use distutils directly at runtime, so just drop the .pth file.
$precedencePth = Join-Path $stageDir "python\Lib\site-packages\distutils-precedence.pth"
if (Test-Path $precedencePth) { Remove-Item $precedencePth }

# --- 5b. Self-check: fail the build loudly rather than ship a broken bundle ---
# (This exact class of bug shipped once already: an unpinned torch version was
# silently missing a submodule, and "check" reported "torch: NOT INSTALLED"
# only when a human happened to run it after extracting the zip.)
Write-Host "Verifying core imports work..." -ForegroundColor Cyan
# Actually exercises torchvision's native ops (nms), not just "does it
# import" -- plain `import torch` succeeded even when torch/torchvision
# were ABI-mismatched; the failure only surfaced when nnU-Net's real
# prediction path touched a native op at runtime.
$selfCheckScript = @"
import torch, torchvision
import autoanatomy.engine.api, autoanatomy.tui.app
boxes = torch.tensor([[0.,0.,1.,1.]])
scores = torch.tensor([0.9])
torchvision.ops.nms(boxes, scores, 0.5)
print('torch', torch.__version__, 'torchvision', torchvision.__version__, '- imports and native ops OK')
"@
& $py -c $selfCheckScript
if ($LASTEXITCODE -ne 0) { throw "Build self-check failed: core imports or native ops are broken in the bundled Python. Do not ship this build." }

# --- 6. Download real model weights straight into the bundle ---
Write-Host "Downloading model weights into the bundle..." -ForegroundColor Cyan
$env:AUTOANATOMY_HOME_DIR = $weightsDir
& $py -m autoanatomy.cli.main download-weights
$downloadExitCode = $LASTEXITCODE
Remove-Item Env:\AUTOANATOMY_HOME_DIR
if ($downloadExitCode -ne 0) { throw "Failed to download model weights." }

# --- 7. Write portable launcher scripts ---
Write-Host "Writing launcher scripts..." -ForegroundColor Cyan
@"
@echo off
set AUTOANATOMY_HOME_DIR=%~dp0.autoanatomy
"%~dp0python\python.exe" -m autoanatomy.tui.app
"@ | Set-Content -Encoding ASCII (Join-Path $stageDir "AutoAnatomy-GUI.bat")

@"
@echo off
set AUTOANATOMY_HOME_DIR=%~dp0.autoanatomy
"%~dp0python\python.exe" -m autoanatomy.cli.main %*
"@ | Set-Content -Encoding ASCII (Join-Path $stageDir "AutoAnatomy-CLI.bat")

@"
AutoAnatomy - Portable Windows x64 build
=========================================

No install required. This folder is fully self-contained (Python + all
dependencies + model weights are bundled).

Run:
  AutoAnatomy-GUI.bat        launches the terminal GUI
  AutoAnatomy-CLI.bat --help launches the headless CLI

This build is CPU-only -- no GPU acceleration even on machines with an
NVIDIA GPU. Segmentation will work, just slower than a GPU build.

You can copy/move this entire folder anywhere; nothing here depends on
its original location.
"@ | Set-Content -Encoding ASCII (Join-Path $stageDir "README.txt")

# --- 8. Archive it up (tar, not Compress-Archive -- see header comment) ---
# Full path, not bare "tar": if Git for Windows is installed, its MSYS tar
# can shadow the native one on PATH, and it mangles plain Windows-style
# paths (with a drive-letter colon) that PowerShell passes it as arguments.
$tarExe = "$env:SystemRoot\System32\tar.exe"
if (-not (Test-Path $tarExe)) {
    throw "Could not find $tarExe (built into Windows 10 1803+ / Windows 11). Cannot package the build."
}

Write-Host "Compressing bundle..." -ForegroundColor Cyan
$outputPath = Join-Path $root $OutputArchive
if (Test-Path $outputPath) { Remove-Item $outputPath }
Push-Location $buildDir
& $tarExe -czf $outputPath "AutoAnatomy"
$tarExitCode = $LASTEXITCODE
Pop-Location
if ($tarExitCode -ne 0) { throw "tar failed to create the archive." }

# Self-check #2: confirm the archive itself round-trips every file, not just
# that the staged directory worked (that's exactly how the zip bug shipped).
# tar -tzf also lists directory entries (paths ending in "/"), so exclude
# those to compare file-to-file against Get-ChildItem -File.
$stagedFileCount = (Get-ChildItem $stageDir -Recurse -File).Count
$archivedFileCount = (& $tarExe -tzf $outputPath | Where-Object { $_ -notmatch '/$' } | Measure-Object).Count
if ($stagedFileCount -ne $archivedFileCount) {
    throw "Archive file count mismatch: staged $stagedFileCount files but archive contains $archivedFileCount. Do not ship this build."
}
Write-Host "Archive file count verified: $archivedFileCount files." -ForegroundColor Green

$archiveSizeMB = [math]::Round((Get-Item $outputPath).Length / 1MB, 1)
Write-Host "== Build complete ==" -ForegroundColor Green
Write-Host "Output: $outputPath ($archiveSizeMB MB)"
Write-Host "Free disk after build: $(Get-FreeDiskGB) GB"
Write-Host ""
Write-Host "Verify before shipping: extract (tar -xzf $OutputArchive) to a DIFFERENT path than $stageDir and run AutoAnatomy-CLI.bat check" -ForegroundColor Yellow
