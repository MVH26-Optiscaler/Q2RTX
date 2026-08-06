<#
.SYNOPSIS
    Stages Quake II RTX game assets into .\baseq2 and validates the result.

.DESCRIPTION
    Windows/ARM64 counterpart to deploy-assets.sh. Assets are never stored in git;
    they come from an installed copy of the game.

    Beyond staging, this script runs a validation pass covering the failure modes
    that actually bite on this branch - each check exists because it caught a real
    problem, and each is documented at its call site:

      * A loose file under baseq2\ shadowing a pak0.pak entry. The game directory
        takes priority over pak files (src/common/files.c:2630), so a stray copy
        silently wins. A corrupt pics\colormap.pcx this way makes every palettized
        asset render as an opaque black silhouette - black HUD icons, ammo/health
        digits and crosshairs - while PNG/TGA-overridden assets look fine.

      * An ONNX upscaler model whose external-data file is misnamed. The .onnx
        references its weights by filename - quicksrnetsmall.data - so a model
        re-fetched or hand-placed outside the committed set easily ends up with its
        weights saved as quicksrnetsmall-w8a8.data instead. The model then loads but
        its weights do not.

      * Missing ONNX Runtime / QNN runtime DLLs next to q2rtx.exe, or an exe built
        for the wrong architecture.

    Staging never overwrites a file that already exists in baseq2\, matching
    deploy-assets.sh. Locally built artifacts (game*.dll/pdb, shaders.pkz) are skipped.

.PARAMETER GameDir
    Installed Quake II RTX folder. Resolution order:
    -GameDir  ->  $env:Q2RTX_GAME_DIRECTORY  ->  the Steam default under Program Files (x86).

.PARAMETER Mode
    Link (default) uses hardlinks for files and junctions for directories - both work
    without administrator rights, unlike symlinks, and avoid duplicating ~1 GB of media.
    Copy duplicates instead, giving a self-contained baseq2\ that survives uninstalling
    the source game. Link automatically falls back to Copy across volumes.

.PARAMETER UpscalerModel
    Path to an NPU upscaler model - either the .onnx itself or a directory holding it.
    Staged into baseq2\models along with the external-data file it references, under
    whatever name the model asks for.

    The models are NOT part of the game and are never taken from the game install. All
    four ship in the repo under baseq2\models, so you normally need neither this
    parameter nor -WithUpscalerModel; use this one to try a model built elsewhere.
    Models already in baseq2\models are validated either way, and an absent one is
    reported as a warning rather than an error - everything except the matching
    upscaler setting works without it.

.PARAMETER WithUpscalerModel
    Re-download the two Qualcomm AI Hub models - QuickSRNetSmall (flt_upscaling 2) and
    QuickSRNetLarge (flt_upscaling 3), onnx-w8a8 - into baseq2\models, verifying a
    pinned sha256 for each. Since both are committed to the repo this is only a repair
    path; `git checkout -- baseq2/models` does the same thing faster. Counterpart to
    deploy-assets.sh --with-upscaler-model, from the same pinned qai-hub-models release;
    that script fetches the TFLite variants because it targets USE_LITE_RT, while
    upscaler.c on this branch loads ONNX.

    The Q2RTX-tuned QuickSRNetLarge (flt_upscaling 4) is skipped: it has no upstream
    zip to fetch (it is fine-tuned locally), so the repo is its only source.

    Note the rename asymmetry, which is what makes this fiddly to do by hand: each zip
    ships e.g. quicksrnetsmall.onnx + quicksrnetsmall.data, and the model must be
    renamed to quicksrnetsmall-w8a8.onnx (what upscaler.c loads) while the .data must
    keep its name (the .onnx references it by that name). Renaming both by variant
    produces a model that loads but has no weights.

.PARAMETER Fix
    Apply repairs instead of only reporting: park a shadowing/corrupt pics\colormap.pcx
    as colormap.disabled.pcx, and correct a misnamed ONNX external-data file.

.PARAMETER VerifyOnly
    Run the validation pass only; stage nothing.

.EXAMPLE
    .\scripts\deploy-assets.ps1
    Stage from the default Steam install and validate.

.EXAMPLE
    .\scripts\deploy-assets.ps1 -VerifyOnly -Fix
    Don't stage anything; just check the current baseq2\ and repair what's repairable.

.EXAMPLE
    .\scripts\deploy-assets.ps1 -WithUpscalerModel
    Stage assets and re-download the AI Hub upscaler models, if they were deleted from
    baseq2\models.

.EXAMPLE
    .\scripts\deploy-assets.ps1 -GameDir "D:\Games\Quake II RTX" -Mode Copy -UpscalerModel .\models
    Self-contained staging from a non-default install, taking the model from a local path.
#>
[CmdletBinding()]
param(
    [string]$GameDir = "",
    [ValidateSet("Link", "Copy")]
    [string]$Mode = "Link",
    [string]$UpscalerModel = "",
    [switch]$WithUpscalerModel,
    [switch]$Fix,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path (Split-Path $MyInvocation.MyCommand.Path -Parent) -Parent
$Baseq2   = Join-Path $RepoRoot "baseq2"

# The NPU upscaler models. All three are committed under baseq2\models, so a
# fresh clone already has them; the download path below only exists to restore
# the two AI Hub ones if they get deleted.
#
# The AI Hub entries are pinned to the same qai-hub-models release
# deploy-assets.sh uses. That script fetches the TFLite variants; upscaler.c
# loads ONNX, so we take the onnx-w8a8 zips from the same base URL. The
# checksums guard against a mismatched or compromised download, exactly as in
# deploy-assets.sh. Bundled entries have no upstream to fetch from: the
# Q2RTX-tuned model is produced locally, not published, so the repo is its only
# source.
#
# Onnx is the filename upscaler.c loads relative to the game dir and Selector is the
# console setting that picks it; both come from upscaler_models[] in
# src/refresh/vkpt/upscaler.c and must stay in step with it.
$QaiHubBaseUrl  = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/{0}/releases/v0.57.3"
$UpscalerModels = @(
    @{
        Id       = "quicksrnetsmall"
        Zip      = "quicksrnetsmall-onnx-w8a8.zip"
        Sha      = "22a62639f0523dee0b1e492f86785a9d27a70f1225894b096941d6a662d5968f"
        Onnx     = "quicksrnetsmall-w8a8.onnx"
        Selector = "flt_upscaling 2"
    },
    @{
        Id       = "quicksrnetlarge"
        Zip      = "quicksrnetlarge-onnx-w8a8.zip"
        Sha      = "7aafb97849a90844028fd862537f8c8ff27aafef89a034939daa0e4d5f656f42"
        Onnx     = "quicksrnetlarge-w8a8.onnx"
        Selector = "flt_upscaling 3"
    },
    @{
        Id       = "quicksrnetlarge-q2rtx"
        Onnx     = "quicksrnetlarge-q2rtx-w8a8.onnx"
        Selector = "flt_upscaling 4"
        Bundled  = $true
    }
)

# Built by this repo, so never staged from the installed game.
$Excluded = @("gamex86.dll", "gamex86.pdb", "gamex86_64.dll", "gamex86_64.pdb", "shaders.pkz")

# Not stock game content. Quake II RTX does not ship an NPU upscaler model - ours come
# from QAI Hub or from local fine-tuning, and are committed to this repo. Staging
# models\ from a game install would silently import whatever happens to be sitting
# there on top of them, so that directory is never taken from the install.
$NotStock = @("models")

# Copied next to q2rtx.exe by the client POST_BUILD step (src/CMakeLists.txt:532)
# when USE_ORT_QNN_UPSCALER is ON. Listed here so a broken build is reported as a
# missing-runtime problem rather than a silent fallback to no upscaling.
$RuntimeDlls = @(
    "onnxruntime.dll", "onnxruntime_providers_qnn.dll", "onnxruntime_providers_shared.dll",
    "QnnHtp.dll", "QnnHtpPrepare.dll", "QnnSystem.dll"
)

$script:Problems = @()
$script:Warnings = @()
$script:Repairs  = @()

function Write-Head([string]$Text) { Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Item([string]$Text) { Write-Host "    $Text" }
function Add-Problem([string]$Text) { $script:Problems += $Text; Write-Host "    FAIL  $Text" -ForegroundColor Red }
function Add-Warning([string]$Text) { $script:Warnings += $Text; Write-Host "    WARN  $Text" -ForegroundColor Yellow }
function Add-Repair([string]$Text)  { $script:Repairs  += $Text; Write-Host "    FIXED $Text" -ForegroundColor Green }
# A problem that -Fix resolved must stop counting against the exit code, so that
# callers can tell "repaired" apart from "still broken".
function Resolve-Problem([string]$Text) { $script:Problems = @($script:Problems | Where-Object { $_ -ne $Text }) }
function Write-Ok([string]$Text)    { Write-Host "    ok    $Text" -ForegroundColor DarkGray }

# ---------------------------------------------------------------------------
# Game directory resolution - mirrors deploy-assets.sh, Windows paths.
# ---------------------------------------------------------------------------
function Resolve-GameDir([string]$Explicit) {
    if ($Explicit) { return $Explicit }
    if ($env:Q2RTX_GAME_DIRECTORY) { return $env:Q2RTX_GAME_DIRECTORY }
    return (Join-Path ${env:ProgramFiles(x86)} "Steam\steamapps\common\Quake II RTX")
}

# ---------------------------------------------------------------------------
# pak0.pak directory index. Format: "PACK", int dirofs, int dirlen; then
# 64-byte entries of char name[56] + int filepos + int filelen.
# ---------------------------------------------------------------------------
function Get-PakIndex([string]$PakPath) {
    $index = @{}
    $fs = [System.IO.File]::OpenRead($PakPath)
    try {
        $br = New-Object System.IO.BinaryReader($fs)
        if ([System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4)) -ne "PACK") { return $null }
        $dirofs = $br.ReadInt32(); $dirlen = $br.ReadInt32()
        $fs.Position = $dirofs
        for ($i = 0; $i -lt ($dirlen / 64); $i++) {
            $nameBytes = $br.ReadBytes(56)
            $z = [Array]::IndexOf($nameBytes, [byte]0)
            if ($z -lt 0) { $z = 56 }
            $name = [System.Text.Encoding]::ASCII.GetString($nameBytes, 0, $z)
            $pos = $br.ReadInt32(); $len = $br.ReadInt32()
            $index[$name.Replace('\', '/').ToLowerInvariant()] = @{ Pos = $pos; Len = $len }
        }
    } finally { $fs.Dispose() }
    return $index
}

# ---------------------------------------------------------------------------
# PCX palette validation, matching what IMG_DecodePCX / IMG_GetPalette accept
# after the hardening in src/refresh/images.c. A 768-byte palette must be
# preceded by a 0x0C marker byte, and an all-zero palette is never legitimate.
# ---------------------------------------------------------------------------
function Test-PcxPalette([byte[]]$Bytes) {
    $r = [pscustomobject]@{ Ok = $false; Reason = ""; Width = 0; Height = 0; NonZero = 0 }
    if ($Bytes.Length -lt 769) { $r.Reason = "file too small ($($Bytes.Length) bytes)"; return $r }
    if ($Bytes[0] -ne 10 -or $Bytes[1] -ne 5) { $r.Reason = "not a version-5 PCX"; return $r }
    $r.Width  = [BitConverter]::ToUInt16($Bytes, 8)  - [BitConverter]::ToUInt16($Bytes, 4) + 1
    $r.Height = [BitConverter]::ToUInt16($Bytes, 10) - [BitConverter]::ToUInt16($Bytes, 6) + 1
    if ($Bytes[$Bytes.Length - 769] -ne 0x0C) {
        $r.Reason = "missing 256-color palette marker (0x{0:X2} at offset len-769)" -f $Bytes[$Bytes.Length - 769]
        return $r
    }
    for ($i = $Bytes.Length - 768; $i -lt $Bytes.Length; $i++) { if ($Bytes[$i] -ne 0) { $r.NonZero++ } }
    if ($r.NonZero -eq 0) { $r.Reason = "palette is entirely zeros (every palettized asset would render black)"; return $r }
    $r.Ok = $true
    return $r
}

# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------
function Invoke-Stage([string]$SourceBaseq2) {
    Write-Head "Staging assets from $SourceBaseq2  (mode: $Mode)"
    if (-not (Test-Path -LiteralPath $Baseq2)) { $null = New-Item -ItemType Directory -Path $Baseq2 }

    $sameVolume = (Split-Path $SourceBaseq2 -Qualifier) -eq (Split-Path $Baseq2 -Qualifier)
    if ($Mode -eq "Link" -and -not $sameVolume) {
        Write-Item "source is on a different volume; falling back to Copy"
    }

    foreach ($src in Get-ChildItem -LiteralPath $SourceBaseq2 -Force) {
        if ($Excluded -contains $src.Name) { Write-Item "skip  $($src.Name) (built locally)"; continue }
        if ($NotStock -contains $src.Name) { Write-Item "skip  $($src.Name) (not stock game content; use -UpscalerModel)"; continue }
        $dest = Join-Path $Baseq2 $src.Name
        if (Test-Path -LiteralPath $dest) { Write-Item "keep  $($src.Name) (already present, not overwriting)"; continue }

        $useLink = ($Mode -eq "Link") -and $sameVolume
        if ($useLink) {
            # Hardlinks for files, junctions for directories: both work without
            # administrator rights, unlike symlinks.
            $type = "HardLink"
            if ($src.PSIsContainer) { $type = "Junction" }
            try {
                $null = New-Item -ItemType $type -Path $dest -Target $src.FullName -ErrorAction Stop
                Write-Item "link  $($src.Name)  [$type]"
                continue
            } catch {
                Write-Item "link  $($src.Name) failed ($($_.Exception.Message.Trim())); copying instead"
            }
        }
        Copy-Item -LiteralPath $src.FullName -Destination $dest -Recurse -Force
        Write-Item "copy  $($src.Name)"
    }
}

# Mirrors fetch_model_variant() in deploy-assets.sh: download, verify the pinned
# sha256, extract into baseq2\models. Runs for every entry in $UpscalerModels;
# the two models reference differently named .data files, so they coexist.
function Invoke-FetchModel {
    Write-Head "Fetching upscaler models from Qualcomm AI Hub"
    $modelsDir = Join-Path $Baseq2 "models"
    if (-not (Test-Path -LiteralPath $modelsDir)) { $null = New-Item -ItemType Directory -Path $modelsDir }

    foreach ($m in $UpscalerModels) {
        if ($m.Bundled) { Write-Item "skip  $($m.Onnx) (ships in the repo; nothing to download)"; continue }

        $dest = Join-Path $modelsDir $m.Onnx
        if (Test-Path -LiteralPath $dest) { Write-Item "keep  $($m.Onnx) (already present, not re-downloading)"; continue }

        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("q2rtx-model-" + [System.Guid]::NewGuid().ToString("N"))
        $null = New-Item -ItemType Directory -Path $tmp
        try {
            $zip = Join-Path $tmp $m.Zip
            Write-Item "fetch $($m.Zip)"
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri "$($QaiHubBaseUrl -f $m.Id)/$($m.Zip)" -OutFile $zip -UseBasicParsing

            $got = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($got -ne $m.Sha) {
                Add-Problem "checksum mismatch for $($m.Zip) (expected $($m.Sha), got $got)"
                continue
            }
            Write-Item "verify sha256 ok"

            Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $tmp "x") -Force
            $onnx = Get-ChildItem -LiteralPath (Join-Path $tmp "x") -Recurse -Filter "*.onnx" -File | Select-Object -First 1
            if (-not $onnx) { Add-Problem "no .onnx inside $($m.Zip)"; continue }

            # The zip ships the model as e.g. quicksrnetsmall.onnx; upscaler.c loads it
            # under a variant-qualified name, so rename on the way in - the same thing
            # deploy-assets.sh does for the TFLite variants.
            Copy-Item -LiteralPath $onnx.FullName -Destination $dest -Force
            Write-Item "stage $($onnx.Name) -> $($m.Onnx)"

            # The weights are referenced BY NAME from inside the .onnx, so unlike the model
            # itself they must keep the name the zip ships them under. Renaming these by
            # variant is exactly what leaves the model loadable but its weights missing.
            foreach ($ref in Get-OnnxExternalData $dest) {
                $src = Get-ChildItem -LiteralPath (Join-Path $tmp "x") -Recurse -Filter $ref -File | Select-Object -First 1
                if (-not $src) { Add-Problem "$($m.Onnx) references '$ref' but the zip does not contain it"; continue }
                Copy-Item -LiteralPath $src.FullName -Destination (Join-Path $modelsDir $ref) -Force
                Write-Item "stage $ref (name preserved - the model references it by this name)"
            }

            $meta = Get-ChildItem -LiteralPath (Join-Path $tmp "x") -Recurse -Filter "metadata.json" -File | Select-Object -First 1
            if ($meta) {
                $metaName = [System.IO.Path]::GetFileNameWithoutExtension($m.Onnx) + ".metadata.json"
                Copy-Item -LiteralPath $meta.FullName -Destination (Join-Path $modelsDir $metaName) -Force
                Write-Item "stage metadata.json -> $metaName"
            }
        } finally {
            if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
        }
    }
}

function Invoke-StageModel([string]$ModelPath) {
    Write-Head "Staging upscaler model"
    $onnx = $null
    if (Test-Path -LiteralPath $ModelPath -PathType Container) {
        $onnx = Get-ChildItem -LiteralPath $ModelPath -Filter "*.onnx" -File | Select-Object -First 1
    } elseif (Test-Path -LiteralPath $ModelPath) {
        $onnx = Get-Item -LiteralPath $ModelPath
    }
    if (-not $onnx) { Add-Problem "no .onnx found at '$ModelPath'"; return }

    $modelsDir = Join-Path $Baseq2 "models"
    if (-not (Test-Path -LiteralPath $modelsDir)) { $null = New-Item -ItemType Directory -Path $modelsDir }

    Copy-Item -LiteralPath $onnx.FullName -Destination (Join-Path $modelsDir $onnx.Name) -Force
    Write-Item "copy  $($onnx.Name)"

    # The .onnx names its weights file internally; stage it under the name the
    # model actually asks for, regardless of what it is called at the source.
    foreach ($ref in Get-OnnxExternalData $onnx.FullName) {
        $cand = Get-ChildItem -LiteralPath $onnx.DirectoryName -Filter "*.data" -File
        if (-not $cand) { Add-Problem "$($onnx.Name) references '$ref' but no .data file exists beside it"; continue }
        $pick = $cand | Where-Object { $_.Name -eq $ref } | Select-Object -First 1
        if (-not $pick) { $pick = $cand | Select-Object -First 1 }
        Copy-Item -LiteralPath $pick.FullName -Destination (Join-Path $modelsDir $ref) -Force
        if ($pick.Name -ne $ref) { Write-Item "copy  $($pick.Name) -> $ref (renamed to the name the model references)" }
        else { Write-Item "copy  $ref" }
    }
}

# Returns the external-data filenames referenced inside an .onnx protobuf.
function Get-OnnxExternalData([string]$OnnxPath) {
    $txt = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($OnnxPath))
    return [regex]::Matches($txt, '[A-Za-z0-9_\-\.]+\.data') | ForEach-Object { $_.Value } | Sort-Object -Unique
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
function Test-CoreAssets {
    Write-Head "Core assets"
    foreach ($f in @("pak0.pak", "q2rtx_media.pkz", "blue_noise.pkz")) {
        $p = Join-Path $Baseq2 $f
        if (Test-Path -LiteralPath $p) { Write-Ok "$f ($([math]::Round((Get-Item -LiteralPath $p).Length / 1MB, 1)) MB)" }
        else { Add-Problem "$f is missing from baseq2\" }
    }
}

function Test-PakShadowing {
    Write-Head "Loose files shadowing pak0.pak"
    $pak = Join-Path $Baseq2 "pak0.pak"
    if (-not (Test-Path -LiteralPath $pak)) { Add-Warning "pak0.pak absent; skipping shadow check"; return }

    $index = Get-PakIndex $pak
    if ($null -eq $index) { Add-Warning "pak0.pak has a bad header; skipping shadow check"; return }

    $prefix = (Resolve-Path -LiteralPath $Baseq2).Path.TrimEnd('\') + '\'
    $shadowing = @()
    foreach ($f in Get-ChildItem -LiteralPath $Baseq2 -Recurse -File -Force) {
        $rel = $f.FullName.Substring($prefix.Length).Replace('\', '/').ToLowerInvariant()
        if ($index.ContainsKey($rel)) { $shadowing += [pscustomobject]@{ Rel = $rel; File = $f; PakLen = $index[$rel].Len } }
    }

    if (-not $shadowing) { Write-Ok "no loose file shadows a pak0.pak entry" }

    foreach ($s in $shadowing) {
        # The game directory wins over pak files (src/common/files.c:2630), so a
        # loose copy silently replaces the packed one. Intentional overrides are
        # legitimate; a corrupt one is not, so validate rather than blanket-warn.
        if ($s.Rel -eq "pics/colormap.pcx") {
            $res = Test-PcxPalette ([System.IO.File]::ReadAllBytes($s.File.FullName))
            if ($res.Ok) {
                Add-Warning "pics/colormap.pcx overrides pak0.pak but looks valid ($($res.Width)x$($res.Height), $($res.NonZero)/768 non-zero)"
            } else {
                $msg = "pics/colormap.pcx overrides pak0.pak and is CORRUPT: $($res.Reason)"
                Add-Problem $msg
                Write-Item "      -> this is what makes HUD icons, ammo/health digits and crosshairs render solid black"
                if ($Fix) {
                    # .disabled.pcx keeps the *.pcx gitignore rule applying, so the
                    # parked file does not show up in git status.
                    $parked = Join-Path $s.File.DirectoryName "colormap.disabled.pcx"
                    if (Test-Path -LiteralPath $parked) { Remove-Item -LiteralPath $parked -Force }
                    Rename-Item -LiteralPath $s.File.FullName -NewName "colormap.disabled.pcx"
                    Add-Repair "parked as colormap.disabled.pcx; the good 256x320 palette in pak0.pak now loads"
                    Resolve-Problem $msg
                } else {
                    Write-Item "      -> re-run with -Fix to park it, or delete it by hand"
                }
            }
        } else {
            Add-Warning "$($s.Rel) overrides a pak0.pak entry (loose $($s.File.Length) B vs packed $($s.PakLen) B)"
        }
    }
}

function Test-UpscalerModel {
    Write-Head "NPU upscaler models"
    $modelsDir = Join-Path $Baseq2 "models"
    if (-not (Test-Path -LiteralPath $modelsDir -PathType Container)) {
        Add-Warning "baseq2\models is absent; no NPU upscaler will be available (re-run with -WithUpscalerModel)"
        return
    }

    # Every weights name any present model asks for. A .data file outside this
    # set is an orphan and is the only thing safe to rename onto a missing ref -
    # with two models staged, grabbing "the first .data" could hand one model the
    # other one's weights.
    $claimed = @{}
    foreach ($m in $UpscalerModels) {
        $p = Join-Path $modelsDir $m.Onnx
        if (Test-Path -LiteralPath $p) { foreach ($r in Get-OnnxExternalData $p) { $claimed[$r] = $true } }
    }

    foreach ($m in $UpscalerModels) {
        $onnx = Join-Path $modelsDir $m.Onnx
        if (-not (Test-Path -LiteralPath $onnx)) {
            # The repo is the only source for a bundled model, so pointing at
            # -WithUpscalerModel would send you looking for a download that does
            # not exist.
            $how = "re-run with -WithUpscalerModel"
            if ($m.Bundled) { $how = "restore it with: git checkout -- baseq2/models" }
            Add-Warning "baseq2\models\$($m.Onnx) is absent; $($m.Selector) will be unavailable ($how)"
            continue
        }
        Write-Ok "$($m.Onnx) present"

        foreach ($ref in Get-OnnxExternalData $onnx) {
            $refPath = Join-Path $modelsDir $ref
            if (Test-Path -LiteralPath $refPath) { Write-Ok "external data '$ref' present"; continue }

            # The model asks for its weights by name. Installs that ship the file under
            # a different name leave the model unable to load them.
            $alt = Get-ChildItem -LiteralPath $modelsDir -Filter "*.data" -File |
                Where-Object { -not $claimed.ContainsKey($_.Name) } | Select-Object -First 1
            if (-not $alt) {
                Add-Problem "$($m.Onnx) references external data '$ref' but no unclaimed .data file is present"
                continue
            }

            $msg = "$($m.Onnx) references '$ref' but the file present is named '$($alt.Name)'"
            Add-Problem $msg
            if ($Fix) {
                Copy-Item -LiteralPath $alt.FullName -Destination $refPath -Force
                Add-Repair "copied $($alt.Name) -> $ref"
                Resolve-Problem $msg
            } else {
                Write-Item "      -> re-run with -Fix to copy it to the expected name"
            }
        }
    }
}

function Test-Runtime {
    Write-Head "Client binary and runtime"
    $exe = Join-Path $RepoRoot "q2rtx.exe"
    if (-not (Test-Path -LiteralPath $exe)) {
        Add-Problem "q2rtx.exe not found in the repo root - build the 'client' target first"
        return
    }

    # PE header: e_lfanew at 0x3C, machine word 4 bytes into the PE signature.
    $b = [System.IO.File]::ReadAllBytes($exe)
    $machine = [BitConverter]::ToUInt16($b, [BitConverter]::ToInt32($b, 0x3C) + 4)
    $arch = "0x{0:X4}" -f $machine
    if ($machine -eq 0xAA64) { $arch = "ARM64" } elseif ($machine -eq 0x8664) { $arch = "x64" }
    Write-Ok "q2rtx.exe ($arch, $([math]::Round($b.Length / 1MB, 1)) MB, built $((Get-Item -LiteralPath $exe).LastWriteTime))"

    $missing = @()
    foreach ($d in $RuntimeDlls) { if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $d))) { $missing += $d } }
    if ($missing.Count -eq 0) {
        Write-Ok "ONNX Runtime + QNN DLLs present beside q2rtx.exe"
    } else {
        Add-Warning "missing beside q2rtx.exe: $($missing -join ', ') - rebuild the 'client' target with USE_ORT_QNN_UPSCALER=ON"
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Q2RTX asset deployment - $RepoRoot" -ForegroundColor White

if (-not $VerifyOnly) {
    $gd = Resolve-GameDir $GameDir
    $srcBaseq2 = Join-Path $gd "baseq2"
    if (-not (Test-Path -LiteralPath $srcBaseq2 -PathType Container)) {
        Write-Host ""
        Write-Host "error: no baseq2\ under game dir: $gd" -ForegroundColor Red
        Write-Host "       point -GameDir or `$env:Q2RTX_GAME_DIRECTORY at your Quake II RTX install," -ForegroundColor Red
        Write-Host "       or pass -VerifyOnly to just check what is already staged." -ForegroundColor Red
        exit 1
    }
    Invoke-Stage $srcBaseq2
    if ($UpscalerModel) { Invoke-StageModel $UpscalerModel }
    if ($WithUpscalerModel) { Invoke-FetchModel }
} else {
    Write-Head "Verify only - nothing will be staged"
}

Test-CoreAssets
Test-PakShadowing
Test-UpscalerModel
Test-Runtime

Write-Host ""
if ($script:Repairs.Count -gt 0) { Write-Host "Repaired $($script:Repairs.Count) item(s)." -ForegroundColor Green }

if ($script:Problems.Count -eq 0) {
    if ($script:Warnings.Count -gt 0) { Write-Host "$($script:Warnings.Count) warning(s), nothing blocking." -ForegroundColor Yellow }
    Write-Host "Setup looks good. Launch with:  .\q2rtx.exe   (from the repo root)" -ForegroundColor Green
    Write-Host ""
    exit 0
}

Write-Host "$($script:Problems.Count) problem(s) found:" -ForegroundColor Red
foreach ($p in $script:Problems) { Write-Host "  - $p" -ForegroundColor Red }
if (-not $Fix) { Write-Host "Re-run with -Fix to repair what is repairable." -ForegroundColor Yellow }
Write-Host ""
exit 1
