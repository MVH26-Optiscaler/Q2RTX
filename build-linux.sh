#!/usr/bin/env bash
# Linux build wrapper for Quake II RTX.
#
# Native build by default. Pass --docker to build inside the Steam Runtime
# sniper SDK image (matches .github/workflows/build.yml exactly).
set -euo pipefail

BUILD_TYPE="Release"
USE_DOCKER=0
CLEAN=0
NO_DEPLOY=0
GAME_DIR=""
ASSET_MODE=()
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./build-linux.sh [Release|Debug] [options] [-- <extra cmake args>]

Options:
    --docker            Build inside the q2rtx-builder Docker image (reproducible, matches CI)
    --clean             Remove build/ before configuring
    --game-dir <path>   Installed Quake II RTX folder to stage assets from
                        (default: $Q2RTX_GAME_DIRECTORY or the Steam install)
    --copy              Copy staged assets instead of symlinking (~1 GB)
    --no-deploy         Skip staging game assets into baseq2/ after the build
    -h, --help          Show this help

After building, game assets are staged into baseq2/ via deploy-assets.sh (see BUILD.md).

Examples:
    ./build-linux.sh                     # native Release + stage assets
    ./build-linux.sh Debug               # native Debug
    ./build-linux.sh Release --docker    # reproducible Docker build + stage assets
    ./build-linux.sh --no-deploy         # build only, don't touch baseq2/
    ./build-linux.sh Release -- -DCONFIG_VKPT_ENABLE_IMAGE_DUMPS=ON
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        Release|release) BUILD_TYPE="Release"; shift ;;
        Debug|debug)     BUILD_TYPE="Debug"; shift ;;
        --docker)        USE_DOCKER=1; shift ;;
        --clean)         CLEAN=1; shift ;;
        --game-dir)      GAME_DIR="${2:-}"; shift 2 ;;
        --copy)          ASSET_MODE=(--copy); shift ;;
        --no-deploy)     NO_DEPLOY=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; EXTRA_ARGS=("$@"); break ;;
        *)               echo "error: unknown argument '$1'" >&2; usage; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Stage game assets from an installed copy into baseq2/ (see deploy-assets.sh).
# Never fails the build — a missing game dir just leaves baseq2/ as-is.
run_deploy() {
    [[ "$NO_DEPLOY" == "1" ]] && return 0
    local gd_args=(); [[ -n "$GAME_DIR" ]] && gd_args=(--game-dir "$GAME_DIR")
    "$ROOT/deploy-assets.sh" "${gd_args[@]}" "${ASSET_MODE[@]}" "$@" || \
        echo "warning: asset staging failed (build still succeeded); see deploy-assets.sh" >&2
}

# Q2RTX needs its submodules; CMake also shells out to git at configure time.
if [[ ! -f extern/glslang/CMakeLists.txt ]]; then
    echo "==> Initialising git submodules"
    git submodule update --init --recursive
fi

if [[ "$USE_DOCKER" == "1" ]]; then
    command -v docker >/dev/null || { echo "error: docker not found in PATH" >&2; exit 1; }
    if [[ "$CLEAN" == "1" && -d build-docker ]]; then
        echo "==> Cleaning build-docker/"
        rm -rf build-docker
    fi
    if ! docker image inspect q2rtx-builder >/dev/null 2>&1; then
        echo "==> Building q2rtx-builder image"
        docker build -t q2rtx-builder .
    fi
    echo "==> Building Q2RTX ($BUILD_TYPE) via Docker"
    # Run as the host user so build artifacts (binaries, build-docker/, baseq2/
    # shaders) stay host-owned instead of root-owned. HOME=/tmp because the host
    # uid can't write the image's /root. (git safe.directory is set in the image.)
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$ROOT:/root/q2rtx" q2rtx-builder "$BUILD_TYPE" "${EXTRA_ARGS[@]}"
    echo "==> Build complete: ./q2rtx and ./q2rtxded"
    run_deploy
    exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: build-linux.sh runs on Linux hosts (or use --docker)" >&2
    exit 1
fi

if [[ "$CLEAN" == "1" && -d build ]]; then
    echo "==> Cleaning build/"
    rm -rf build
fi

JOBS="$(nproc 2>/dev/null || echo 4)"
# Prefer Ninja, but only pick a generator when configuring a fresh build dir — forcing
# one onto an existing cache with a different generator makes CMake error out.
GEN_ARGS=()
if [[ ! -f build/CMakeCache.txt ]] && command -v ninja >/dev/null 2>&1; then
    GEN_ARGS=(-GNinja)
fi

echo "==> Configuring ($BUILD_TYPE)"
cmake -B ./build "${GEN_ARGS[@]}" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCONFIG_BUILD_GLSLANG=ON "${EXTRA_ARGS[@]}"
cmake --build ./build -j"$JOBS"
echo "==> Build complete: ./q2rtx and ./q2rtxded"
run_deploy
