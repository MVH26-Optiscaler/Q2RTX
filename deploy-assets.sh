#!/usr/bin/env bash
# Stage Quake II RTX game assets from an installed copy into ./baseq2 so a locally
# built ./q2rtx can find them. Mirrors Amnesia64's asset strategy: assets are never
# stored in git; they come from your install.
#
# Game dir resolution:  --game-dir <path>  ->  $Q2RTX_GAME_DIRECTORY  ->  Steam default.
# Symlinks by default (media is ~1 GB); pass --copy to duplicate instead.
#
# --with-upscaler-model additionally downloads the QuickSRNetSmall TFLite model
# (float + w8a8) into ./baseq2/models for the experimental LiteRT-based upscaler.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_GAME_DIR="$HOME/.local/share/Steam/steamapps/common/Quake II RTX"
GAME_DIR=""
MODE="link"   # link | copy
WITH_UPSCALER_MODEL=0

# QuickSRNetSmall (4x TFLite super-resolution), pinned to the qai-hub-models v0.57.3
# release. Checksums guard against a compromised/mismatched download.
QAI_HUB_MODEL_BASE_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/quicksrnetsmall/releases/v0.57.3"
MODELS_DIR="$ROOT/baseq2/models"

TMP_DIR=""
cleanup() { [[ -n "$TMP_DIR" ]] && rm -rf "$TMP_DIR"; return 0; }
trap cleanup EXIT

usage() { cat <<'EOF'
Usage: ./deploy-assets.sh [--game-dir <path>] [--copy] [--with-upscaler-model]
  --game-dir <path>       Installed Quake II RTX folder (default: $Q2RTX_GAME_DIRECTORY
                          or ~/.local/share/Steam/steamapps/common/Quake II RTX)
  --copy                  Copy assets instead of symlinking (~1 GB, self-contained)
  --with-upscaler-model   Also fetch the QuickSRNetSmall TFLite model into baseq2/models
  -h, --help              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --game-dir)             GAME_DIR="${2:-}"; shift 2 ;;
    --copy)                 MODE="copy"; shift ;;
    --link)                 MODE="link"; shift ;;
    --with-upscaler-model)  WITH_UPSCALER_MODEL=1; shift ;;
    -h|--help)              usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

# Fetch one QuickSRNetSmall precision variant: download its zip, verify its
# pinned sha256, then extract the .tflite + metadata.json into baseq2/models,
# renamed by variant since both zips contain a file named "quicksrnetsmall.tflite".
fetch_model_variant() {
  local variant="$1" sha256="$2"
  local dest_tflite="$MODELS_DIR/quicksrnetsmall-$variant.tflite"

  if [[ -e "$dest_tflite" ]]; then
    echo "    keep  quicksrnetsmall-$variant.tflite (already present, not re-downloading)"
    return 0
  fi

  local zip="$TMP_DIR/quicksrnetsmall-tflite-$variant.zip"
  echo "    fetch quicksrnetsmall-tflite-$variant.zip"
  curl -fsSL --retry 3 -o "$zip" "$QAI_HUB_MODEL_BASE_URL/quicksrnetsmall-tflite-$variant.zip"

  echo "$sha256  $zip" | sha256sum -c - >/dev/null || {
    echo "error: checksum mismatch for quicksrnetsmall-tflite-$variant.zip" >&2
    exit 1
  }

  local extract_dir="$TMP_DIR/extract-$variant"
  mkdir -p "$extract_dir"
  unzip -qo "$zip" -d "$extract_dir"

  local src_dir="$extract_dir/quicksrnetsmall-tflite-$variant"
  cp "$src_dir/quicksrnetsmall.tflite" "$dest_tflite"
  cp "$src_dir/metadata.json" "$MODELS_DIR/quicksrnetsmall-$variant.metadata.json"
  echo "    done  quicksrnetsmall-$variant.tflite"
}

stage_upscaler_model() {
  echo "==> Staging QuickSRNetSmall upscaler model into $MODELS_DIR"
  mkdir -p "$MODELS_DIR"
  TMP_DIR="$(mktemp -d)"
  fetch_model_variant float 3869381f771d201b02522167f440ab2f1be5e3629d1bba5e8044b72ba8be57a2
  fetch_model_variant w8a8 0e1453839fa2fe96917f6713be40fba438970c49d9d09664431ba3c9aea5aed2
}

if [[ "$WITH_UPSCALER_MODEL" == "1" ]]; then
  stage_upscaler_model
fi

GAME_DIR="${GAME_DIR:-${Q2RTX_GAME_DIRECTORY:-$DEFAULT_GAME_DIR}}"
if [[ ! -d "$GAME_DIR/baseq2" ]]; then
  echo "error: no baseq2/ under game dir: $GAME_DIR" >&2
  echo "       point --game-dir or \$Q2RTX_GAME_DIRECTORY at your Quake II RTX install." >&2
  exit 1
fi
GAME_DIR_REAL="$(realpath "$GAME_DIR")"   # resolve Steam-library symlinks

is_excluded() { case "$1" in game*.so|shaders.pkz) return 0 ;; *) return 1 ;; esac; }

mkdir -p "$ROOT/baseq2"
echo "==> Staging assets from $GAME_DIR_REAL/baseq2  (mode: $MODE)"
shopt -s dotglob nullglob
for src in "$GAME_DIR_REAL"/baseq2/*; do
  name="$(basename "$src")"
  is_excluded "$name" && { echo "    skip  $name (built locally)"; continue; }
  dest="$ROOT/baseq2/$name"
  if [[ -e "$dest" && ! -L "$dest" ]]; then
    echo "    keep  $name (real file already present, not overwriting)"; continue
  fi
  if [[ "$MODE" == "link" ]]; then
    ln -sfn "$src" "$dest"; echo "    link  $name -> $src"
  else
    cp -ru "$src" "$ROOT/baseq2/"; echo "    copy  $name"
  fi
done
echo "==> Done. Launch with: ./q2rtx   (run from the repo root)"
