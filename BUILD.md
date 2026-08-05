# Building Quake II RTX (Linux / Docker)

Q2RTX builds with CMake. On Linux you can either build natively or build inside a
Docker image based on the Steam Runtime "sniper" SDK — the same environment
`.github/workflows/build.yml` uses for its Linux job, so a Docker build reproduces CI
exactly.

Windows and macOS builds are covered in the main [`readme.md`](readme.md).

## Quick start

```bash
./build-linux.sh            # native Release build
./build-linux.sh --docker   # reproducible Docker build (matches CI)
```

Both init the git submodules on first run, configure a `build/` directory at the repo
root, build, and drop the executables in the repo root.

Output:

- `./q2rtx` — the client
- `./q2rtxded` — the dedicated server
- `baseq2/game*.so` — the game module(s)

The build produces only the engine. To *run* the game you also need the RTX media
(`q2rtx_media.pkz`, `blue_noise.pkz`) and base-game `pak*.pak` files in `baseq2/`, plus a
Vulkan-capable GPU/driver. After each build the wrapper stages those assets from an
installed copy of Quake II RTX — see [Assets / running the game](#assets--running-the-game).

## `build-linux.sh`

```
./build-linux.sh [Release|Debug] [options] [-- <extra cmake args>]

  --docker            Build inside the q2rtx-builder Docker image (reproducible, matches CI)
  --clean             Remove build/ before configuring
  --game-dir <path>   Installed Quake II RTX folder to stage assets from
  --copy              Copy staged assets instead of symlinking (~1 GB)
  --no-deploy         Skip staging assets into baseq2/ after the build
  -h, --help          Show help
```

Examples:

```bash
./build-linux.sh                     # native Release + stage assets
./build-linux.sh Debug               # native Debug
./build-linux.sh Release --docker    # reproducible Docker build + stage assets
./build-linux.sh Release --clean     # wipe build/ and rebuild
./build-linux.sh --no-deploy         # build only, don't touch baseq2/
./build-linux.sh Release -- -DCONFIG_VKPT_ENABLE_IMAGE_DUMPS=ON
```

The native path uses your system compiler (and Ninja if it's installed). The `--docker`
path builds — and, if needed, first builds — the `q2rtx-builder` image, then runs the
build inside it with the pinned toolchain (gcc-14/g++-14, Ninja, CMake 4.1.2).

## Docker (raw flow)

`./build-linux.sh --docker` does all of this for you, but the underlying commands are:

```bash
docker build -t q2rtx-builder .

# Release, in-place (artifacts land in your working tree)
docker run --rm -v "$(pwd):/root/q2rtx" q2rtx-builder

# Debug + extra CMake args
docker run --rm -v "$(pwd):/root/q2rtx" q2rtx-builder Debug -DCONFIG_VKPT_ENABLE_IMAGE_DUMPS=ON
```

The image only holds the toolchain; your repo is bind-mounted at `/root/q2rtx` and built
in place. The container runs [`docker-entrypoint.sh`](docker-entrypoint.sh), which
configures a `build-docker/` directory with
`-DCMAKE_BUILD_TYPE=<Release|Debug> -DCONFIG_BUILD_GLSLANG=ON` and builds with Ninja.

The Docker build uses its own `build-docker/` directory (separate from the native
`build/`) so the container's gcc-14/Ninja toolchain never clashes with a host-side
native build. Both are gitignored, and the binaries land at the repo root either way.

**Submodules for raw `docker run`:** the entrypoint does not initialise submodules (to
avoid git "dubious ownership" issues on the bind-mounted, host-owned tree), and CMake
also needs the `.git` directory present. So before a raw `docker run`, make sure the
tree is complete:

```bash
git submodule update --init --recursive
```

`build-linux.sh` does this automatically.

**Ownership:** `build-linux.sh --docker` runs the container as your host user
(`--user $(id -u):$(id -g)`, with `HOME=/tmp`), so everything it writes — the binaries,
`build-docker/`, and staged shaders — stays owned by you. This matters because a native
build and the Docker build both write into the same tree; root-owned artifacts from a
root container would otherwise block a subsequent native build. (A raw `docker run`
without `--user` writes root-owned files; prefer the wrapper.)

## Native build (without the wrapper)

```bash
git submodule update --init --recursive
cmake -B ./build -DCMAKE_BUILD_TYPE=Release -DCONFIG_BUILD_GLSLANG=ON
cmake --build ./build -j"$(nproc)"
```

All C/C++ dependencies (zlib, curl, SDL2, glslang, openal-soft, volk) build statically
from the submodules under `extern/`, so the only external requirement at build time is a
compiler toolchain, CMake (>= 3.20), and the X11 / Wayland dev libraries that SDL2 links
against. The sniper SDK Docker image already includes all of these.

The Vulkan loader is *not* needed at build time: both volk and SDL2 open
`libvulkan.so.1` at runtime, so no `libvulkan-dev` / import library is linked. Only the
runtime loader (`libvulkan1`, already in `CPACK_DEBIAN_PACKAGE_DEPENDS`) has to be
present on the machine that runs the game.

## Assets / running the game

The build produces the engine (`q2rtx`, `q2rtxded`), the game module (`baseq2/game*.so`),
and the compiled shaders (`baseq2/shader_vkpt/`). The RTX **media** and base-game **paks**
are copyrighted/large and are **not** stored in git — they come from an installed copy of
Quake II RTX (e.g. the Steam version).

`build-linux.sh` stages those assets into `baseq2/` after each build via
[`deploy-assets.sh`](deploy-assets.sh), which **symlinks** (default) the installed
`baseq2/` contents into your working tree — `q2rtx_media.pkz`, `blue_noise.pkz`,
`pak0.pak`, `players/` — while skipping the pieces you build yourself
(`game*.so`, `shaders.pkz`). Because `q2rtx` runs from the repo root and reads `./baseq2`,
that's all it needs to launch:

```bash
./q2rtx        # run from the repo root
```

### Choosing the game directory

The installed-game path is resolved in this order:

1. `--game-dir <path>` on `build-linux.sh` or `deploy-assets.sh`
2. the `Q2RTX_GAME_DIRECTORY` environment variable
3. the Steam default: `~/.local/share/Steam/steamapps/common/Quake II RTX`

### Staging assets without rebuilding

`deploy-assets.sh` is standalone — run it any time to (re)stage assets:

```bash
./deploy-assets.sh                                   # symlink from the default/Steam install
./deploy-assets.sh --game-dir "/path/to/Quake II RTX"
./deploy-assets.sh --copy                            # duplicate ~1 GB instead of symlinking
```

Symlinks point into the installed copy, so they break if you move or uninstall it; use
`--copy` for a self-contained `baseq2/`. To build without staging, pass `--no-deploy`.

### Staging on Windows

[`scripts/deploy-assets.ps1`](scripts/deploy-assets.ps1) is the Windows counterpart. It
uses hardlinks for files and junctions for directories — both work without administrator
rights, unlike symlinks — and falls back to copying across volumes.

```powershell
.\scripts\deploy-assets.ps1                                    # stage from the default Steam install
.\scripts\deploy-assets.ps1 -GameDir "D:\Games\Quake II RTX"
.\scripts\deploy-assets.ps1 -Mode Copy                         # duplicate ~1 GB instead of linking
.\scripts\deploy-assets.ps1 -VerifyOnly                        # check what is staged, change nothing
.\scripts\deploy-assets.ps1 -VerifyOnly -Fix                   # ...and repair what is repairable
```

Beyond staging it validates the result: core assets present, no loose file shadowing a
`pak0.pak` entry (a corrupt `pics\colormap.pcx` staged this way renders every palettized
asset — HUD icons, ammo/health digits, crosshairs — solid black), the NSS model and its
metadata present, and the client binary's architecture plus its ONNX Runtime/QNN runtime
DLLs. It exits non-zero when something is wrong, so it works in a build script.

There is no model download path. `baseq2/models/nss-temporal-high-int8.onnx` is committed,
so the way to restore a deleted one is `git checkout -- baseq2/models`. Use
`-UpscalerModel <path>` only to stage a model exported elsewhere. The script never takes
`models/` from the game install — Quake II RTX does not ship one, and staging that
directory would silently import whatever happened to be sitting there.

### Full game vs demo

The staged `pak0.pak` from the Steam install is the **demo** data. For the full game, add
your retail `pak*.pak` to `baseq2/` (`setup/find-retail-paks.sh` can locate and copy them
from a retail Quake II install). The `q2rtx` client needs a Vulkan-capable GPU + display;
the `q2rtxded` dedicated server runs headless.
