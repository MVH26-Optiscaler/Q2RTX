<#
.SYNOPSIS
    Launches q2rtx.exe with GFXReconstruct capture enabled.

.DESCRIPTION
    Sets the Vulkan loader env vars needed to inject the GFXReconstruct capture
    layer (VK_LAYER_LUNARG_gfxreconstruct), launches q2rtx.exe, and reports the
    resulting .gfxr trace file. Requires the LunarG Vulkan SDK (or a standalone
    gfxreconstruct release) installed, with the layer registered or reachable
    via -LayerPath.

.PARAMETER ExePath
    Path to q2rtx.exe. Defaults to '.\q2rtx.exe' relative to the repo root.

.PARAMETER OutputDir
    Directory the capture file is written to. Defaults to '.\captures'.

.PARAMETER GameArgs
    Extra command-line args passed through to q2rtx.exe, e.g. '+map q2dm1'.

.PARAMETER TimeoutSeconds
    If set, the script force-closes the game after this many seconds instead
    of waiting for you to quit it yourself. Useful for unattended/CI captures.
    If omitted (default), the script waits for you to play and exit normally
    so the whole session is captured.

.PARAMETER CaptureFrames
    Optional frame range to limit the capture, e.g. '1-500'. Maps to
    GFXRECON_CAPTURE_FRAMES. Omit to capture the whole run.

.PARAMETER LayerPath
    Optional path to the directory containing VkLayer_gfxreconstruct.json/.dll,
    for setups where the layer isn't registered system-wide (e.g. a standalone
    gfxreconstruct release rather than the full Vulkan SDK).

.PARAMETER Info
    After capture, run gfxrecon-info.exe on the resulting file and print the
    summary (frame count, pipelines, etc.).

.EXAMPLE
    .\scripts\gfxrecon_capture.ps1
    Launch the game, play normally, quit when done - capture saved to .\captures.

.EXAMPLE
    .\scripts\gfxrecon_capture.ps1 -TimeoutSeconds 20 -Info
    Unattended 20-second capture of the main menu, then print a summary.

.EXAMPLE
    .\scripts\gfxrecon_capture.ps1 -GameArgs '+map q2dm1' -CaptureFrames '1-300'
#>
[CmdletBinding()]
param(
    [string]$ExePath = ".\q2rtx.exe",
    [string]$OutputDir = ".\captures",
    [string]$GameArgs = "",
    [int]$TimeoutSeconds = 0,
    [string]$CaptureFrames = "",
    [string]$LayerPath = "",
    [switch]$Info
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) {
    throw "q2rtx.exe not found at '$ExePath'. Build it first, or pass -ExePath."
}
$ExePath = (Resolve-Path $ExePath).Path
$WorkingDirectory = Split-Path $ExePath -Parent

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$capFile = Join-Path $OutputDir "q2rtx_$stamp.gfxr"

# Locate the layer, so we can warn early instead of silently capturing nothing.
$layerFound = $false
foreach ($hive in @("HKLM:\SOFTWARE\Khronos\Vulkan\ExplicitLayers", "HKLM:\SOFTWARE\WOW6432Node\Khronos\Vulkan\ExplicitLayers")) {
    if (Test-Path $hive) {
        $props = Get-ItemProperty -Path $hive -ErrorAction SilentlyContinue
        if ($props -and ($props.PSObject.Properties.Name | Where-Object { $_ -match "gfxreconstruct" })) {
            $layerFound = $true
        }
    }
}
if ($LayerPath) {
    if (-not (Test-Path (Join-Path $LayerPath "VkLayer_gfxreconstruct.json"))) {
        throw "VkLayer_gfxreconstruct.json not found under -LayerPath '$LayerPath'."
    }
    $layerFound = $true
}
if (-not $layerFound) {
    Write-Warning "VK_LAYER_LUNARG_gfxreconstruct doesn't appear to be registered system-wide. Install the Vulkan SDK (includes gfxreconstruct), or pass -LayerPath pointing at a standalone gfxreconstruct release's layer directory."
}

$env:VK_INSTANCE_LAYERS = "VK_LAYER_LUNARG_gfxreconstruct"
$env:GFXRECON_CAPTURE_FILE = $capFile
$env:GFXRECON_LOG_LEVEL = "info"
if ($LayerPath) { $env:VK_LAYER_PATH = $LayerPath }
if ($CaptureFrames) { $env:GFXRECON_CAPTURE_FRAMES = $CaptureFrames } else { Remove-Item Env:\GFXRECON_CAPTURE_FRAMES -ErrorAction SilentlyContinue }

$outLog = Join-Path $OutputDir "q2rtx_$stamp.stdout.log"
$errLog = Join-Path $OutputDir "q2rtx_$stamp.stderr.log"

Write-Host "Launching $ExePath (capture -> $capFile) ..."
$startArgs = @{
    FilePath               = $ExePath
    WorkingDirectory        = $WorkingDirectory
    PassThru                = $true
    RedirectStandardOutput  = $outLog
    RedirectStandardError   = $errLog
}
if ($GameArgs) { $startArgs["ArgumentList"] = $GameArgs }
$proc = Start-Process @startArgs

if ($TimeoutSeconds -gt 0) {
    Write-Host "Capturing for $TimeoutSeconds second(s) ..."
    Start-Sleep -Seconds $TimeoutSeconds
    $proc.CloseMainWindow() | Out-Null
    if (-not $proc.WaitForExit(8000)) {
        Write-Warning "Game didn't close gracefully, forcing termination."
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Play the game, then quit normally (Alt+F4 or the in-game quit) to finish the capture."
    $proc.WaitForExit()
}

# gfxreconstruct appends a timestamp to the configured filename by default.
Start-Sleep -Seconds 1
$result = Get-ChildItem -Path $OutputDir -Filter "q2rtx_$stamp*.gfxr" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $result) {
    Write-Warning "No .gfxr file was produced. Check $errLog / $outLog for details."
    return
}

Write-Host "Capture saved: $($result.FullName) ($([math]::Round($result.Length / 1MB, 1)) MB)"

if ($Info) {
    $gfxreconInfo = Get-Command "gfxrecon-info.exe" -ErrorAction SilentlyContinue
    if (-not $gfxreconInfo -and $env:VULKAN_SDK) {
        $candidate = Join-Path $env:VULKAN_SDK "Bin\gfxrecon-info.exe"
        if (Test-Path $candidate) { $gfxreconInfo = Get-Item $candidate }
    }
    if ($gfxreconInfo) {
        & $gfxreconInfo.Source $result.FullName
    } else {
        Write-Warning "gfxrecon-info.exe not found on PATH or under VULKAN_SDK\Bin - skipping summary."
    }
}
