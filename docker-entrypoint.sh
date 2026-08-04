#!/bin/bash
set -e

# First positional arg selects the CMake build type (Release/Debug). Default Release.
# Any further args pass straight through to the configure step.
BUILD_TYPE="${1:-Release}"
shift || true
CMAKE_ARGS="$@"

export CC="${CC:-gcc-14}" CXX="${CXX:-g++-14}"

# Build into a dedicated directory so the container toolchain (gcc-14 + Ninja) never
# clashes with a host-side native build/ (different compiler paths & generator).
# CMake shells out to `git rev-parse`; the repo (with .git) is bind-mounted at WORKDIR.
# Binaries still land at the repo root regardless of the build dir.
cmake -B ./build-docker -GNinja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCONFIG_BUILD_GLSLANG=ON \
    ${CMAKE_ARGS}

cmake --build ./build-docker -j"$(nproc)"
