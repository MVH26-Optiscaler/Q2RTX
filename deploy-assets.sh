#!/usr/bin/env bash
# Stage Quake II RTX game assets from an installed copy into ./baseq2 so a locally
# built ./q2rtx can find them. Mirrors Amnesia64's asset strategy: assets are never
# stored in git; they come from your install.
#
# Game dir resolution:  --game-dir <path>  ->  $Q2RTX_GAME_DIRECTORY  ->  Steam default.
# Symlinks by default (media is ~1 GB); pass --copy to duplicate instead.
#
# There is no upscaler-model download here any more: the NPU upscaler is
# Windows-ARM64 only (ONNX Runtime + the QNN execution provider), and its two
# models are committed under baseq2/models rather than fetched. The TFLite
# download this script used to carry targeted a LiteRT backend that no longer
# exists in the tree.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_GAME_DIR="$HOME/.local/share/Steam/steamapps/common/Quake II RTX"
GAME_DIR=""
MODE="link"   # link | copy

usage() { cat <<'EOF'
Usage: ./deploy-assets.sh [--game-dir <path>] [--copy]
  --game-dir <path>       Installed Quake II RTX folder (default: $Q2RTX_GAME_DIRECTORY
                          or ~/.local/share/Steam/steamapps/common/Quake II RTX)
  --copy                  Copy assets instead of symlinking (~1 GB, self-contained)
  -h, --help              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --game-dir)             GAME_DIR="${2:-}"; shift 2 ;;
    --copy)                 MODE="copy"; shift ;;
    --link)                 MODE="link"; shift ;;
    -h|--help)              usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

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
