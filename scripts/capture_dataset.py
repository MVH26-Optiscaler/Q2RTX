#!/usr/bin/env python3
"""Capture a Q2RTX upscaler training set, start to finish.

Runs the capture tool for the number of rounds you ask for and converts
everything it collected into the HR/LR pairs QuickSRNetSmall is fine-tuned on.
Build it first -- the game library carries the 'setpos' cheat the capture bot
teleports with, so a stale one makes every teleport silently do nothing:

    cmake --build build --target capture game -j8
    python scripts/capture_dataset.py --rounds 84 --out dataset

Each round is one self-driving pass over every map the game can see. A round
re-seeds from the clock, so it stops in different rooms and looks in different
directions than the last one, and writes under its own filename prefix so rounds
accumulate instead of overwriting each other.

Measured on an RX 7800 XT over the three demo maps, one round at 3840x2160 takes
about 2.7 minutes and yields ~1200 tiles at ~1.4 GB. 84 rounds is therefore
roughly 100,000 tiles, 3.8 hours and 117 GB. A full forty-map install yields far
more per round.

Nothing appears on screen: the capture tool renders headless, off a
VK_EXT_headless_surface with no window and no display connection, so a run needs
no DISPLAY or WAYLAND_DISPLAY and does not fight the desktop for the machine.
That is also why --geometry is the real render resolution rather than a size a
compositor is free to clamp to the monitor. Pass --windowed to watch a round.
Each round drives itself and exits when it is done.

Output:

    <out>/hr/*.png    512x512 uint8, encoded pow(1/2.2)
    <out>/lr/*.png    128x128 uint8, the training input

Pairs are matched by filename. There is no train/validation split: hold out
whatever you want at training time.

Needs numpy and pillow: pip install numpy pillow
"""

import argparse
import pathlib
import shutil
import struct
import subprocess
import sys
import time
import zlib

try:
    import numpy as np
    from PIL import Image
except ImportError:
    sys.exit("this needs numpy and pillow: pip install numpy pillow")

REPO = pathlib.Path(__file__).resolve().parent.parent

# Where the engine writes tiles: fs_gamedir, which on Linux is the XDG data
# directory rather than the source tree.
DEFAULT_RAW_DIRS = [
    pathlib.Path.home() / ".local/share/quake2rtx/baseq2/screenshots/dataset",
    REPO / "baseq2/screenshots/dataset",
]

# Must match upscaler_linear_to_encoded() in
# src/refresh/vkpt/shader/upscaler_shared.h. Raw tiles hold tone-mapped *linear*
# values; this is the encode the model sees at runtime, applied here so the
# transfer curve can be retuned without recapturing anything.
GAMMA = 2.2

# Downscale factor: the model's tile_out / tile_in.
SCALE = 4

# Hamming distance between 8x8 average hashes below which two tiles count as the
# same view. 64 bits total, so 5 is a fairly tight match.
DUPLICATE_DISTANCE = 5


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def find_raw_dir(explicit):
    if explicit:
        return pathlib.Path(explicit).expanduser()

    for candidate in DEFAULT_RAW_DIRS:
        if candidate.exists():
            return candidate

    # Nothing captured yet: the first candidate whose install root exists wins,
    # and the engine creates the rest.
    for candidate in DEFAULT_RAW_DIRS:
        if candidate.parent.parent.exists():
            return candidate

    return DEFAULT_RAW_DIRS[0]


def run(command, **kwargs):
    print(f"+ {' '.join(str(c) for c in command)}", flush=True)
    return subprocess.run(command, **kwargs)


def write_user_cfg(path, args, round_index):
    """Per-round overrides, read after the tracked capture_dataset.cfg."""
    lines = ["// Written by scripts/capture_dataset.py; edits here are lost."]

    # Tile names are built from the map and the frame counter, and the frame
    # counter restarts every run. Without a per-round prefix, round two would
    # overwrite round one rather than adding to it.
    lines.append(f"capture_prefix p{round_index}_")

    for cvar, value in (("capture_samples", args.samples),
                        ("capture_looks", args.looks),
                        ("capture_settle", args.settle),
                        ("capture_tile", args.tile),
                        ("capture_tiles_per_frame", args.tiles_per_frame),
                        ("vid_geometry", args.geometry)):
        if value is not None:
            lines.append(f"{cvar} {value}")

    # An explicit scan started here wins: the startup sequence only falls back to
    # scanning everything when nothing is running yet.
    if args.maps:
        lines.append("capture_scan " + " ".join(args.maps))

    path.write_text("\n".join(lines) + "\n")


def count_tiles(raw_dir):
    return sum(1 for _ in raw_dir.glob("*.png")) if raw_dir.exists() else 0


def bytes_on_disk(raw_dir):
    return sum(f.stat().st_size for f in raw_dir.glob("*.png")) if raw_dir.exists() else 0


def capture(args, raw_dir):
    binary = REPO / "q2rtx_capture"
    if not binary.exists():
        sys.exit(f"{binary} does not exist -- build it first:\n"
                 f"    cmake --build build --target capture game -j8")

    if args.fresh and raw_dir.exists():
        print(f"clearing {raw_dir}")
        shutil.rmtree(raw_dir)

    root = raw_dir if raw_dir.exists() else raw_dir.parent.parent
    free_gb = shutil.disk_usage(root).free / 1024 ** 3

    # ~1.2 MB per 512x512 16-bit tile, ~1200 tiles per round at 4K. Say what the
    # run is going to cost before spending hours on it.
    estimate_gb = args.rounds * 1.4
    print(f"{args.rounds} rounds, roughly {args.rounds * 1200} tiles and "
          f"{estimate_gb:.0f} GB; {free_gb:.0f} GB free")

    if estimate_gb > free_gb and not args.force:
        sys.exit("that will not fit -- lower --rounds, try --tile 256, or pass --force")

    user_cfg = REPO / "baseq2/capture_user.cfg"
    started = time.monotonic()

    for index in range(1, args.rounds + 1):
        have = count_tiles(raw_dir)
        print(f"\n=== round {index} of {args.rounds} ({have} tiles so far) ===", flush=True)

        write_user_cfg(user_cfg, args, index)

        # The video driver is chosen in CL_InitRefresh(), before capture_user.cfg
        # is ever read, so a window has to be asked for on the command line.
        command = [str(binary)]
        if args.windowed:
            command += ["+set", "vid_driver", "sdl"]

        try:
            result = run(command, cwd=REPO, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            sys.exit(f"round {index} did not finish within {args.timeout}s")
        finally:
            user_cfg.unlink(missing_ok=True)

        if result.returncode:
            sys.exit(f"round {index} exited with status {result.returncode}")

        if count_tiles(raw_dir) <= have:
            sys.exit(f"round {index} produced no tiles; stopping rather than "
                     f"burning time on rounds that collect nothing")

    print(f"\ncapture finished in {(time.monotonic() - started) / 60:.1f} min: "
          f"{count_tiles(raw_dir)} tiles, {bytes_on_disk(raw_dir) / 1024 ** 3:.1f} GB")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def encode(linear):
    """Linear [0,1] -> display-referred, the same curve the shader uses."""
    return np.clip(linear, 0.0, 1.0) ** (1.0 / GAMMA)


def decode(encoded):
    return np.clip(encoded, 0.0, 1.0) ** GAMMA


def read_png16(path):
    """Decodes a 16-bit RGB PNG to a uint16 HxWx3 array.

    Pillow silently truncates 16-bit RGB PNGs to 8 bits on load, which would
    throw away exactly the shadow precision the capture stores them at 16 bits
    for. These files come from write_png16() in src/refresh/vkpt/capture.c, which
    always emits non-interlaced truecolor with filter 0.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")

    offset = 8
    header = None
    idat = bytearray()

    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset:offset + 4])
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        offset += 12 + length

        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", payload)
            if (depth, color, interlace) != (16, 2, 0):
                raise ValueError(f"{path}: expected 16-bit non-interlaced RGB, got "
                                 f"depth {depth}, color type {color}, interlace {interlace}")
            header = (width, height)
        elif kind == b"IDAT":
            idat += payload
        elif kind == b"IEND":
            break

    if header is None:
        raise ValueError(f"{path}: no IHDR")

    width, height = header
    stride = width * 3 * 2

    raw = zlib.decompress(bytes(idat))
    expected = (stride + 1) * height
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} decompressed bytes, got {len(raw)}")

    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride + 1)
    if rows[:, 0].any():
        raise ValueError(f"{path}: uses PNG row filters, which this reader does not implement")

    samples = rows[:, 1:].reshape(height, width, 3, 2).astype(np.uint16)
    return (samples[..., 0] << 8) | samples[..., 1]


def load_linear(path):
    return read_png16(path).astype(np.float32) / 65535.0


def average_hash(encoded):
    """64-bit average hash of the encoded image, for duplicate rejection."""
    small = np.asarray(
        Image.fromarray((encoded * 255).astype(np.uint8)).convert("L").resize((8, 8), Image.BOX),
        dtype=np.float32,
    )
    bits = (small > small.mean()).flatten()
    return np.uint64(int("".join("1" if bit else "0" for bit in bits), 2))


# Population count over a uint64 array, so a new hash can be compared against
# every hash seen so far in one vectorised pass. A per-tile Python loop is fine
# for a few hundred tiles and hopeless for the tens of thousands a long run makes.
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_to_all(digest, seen):
    if not len(seen):
        return np.empty(0, dtype=np.uint16)

    xor = np.bitwise_xor(seen, digest)
    counts = np.zeros(len(xor), dtype=np.uint16)
    for shift in range(0, 64, 8):
        counts += _POPCOUNT8[((xor >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)]
    return counts


def downscale(hr_encoded, filter_name):
    """Produces the low-resolution input from the high-resolution target.

    'bicubic-encoded' is the DIV2K convention QuickSRNet was pretrained on, so
    fine-tuning starts close to the weights' existing prior. 'area-linear' is the
    physically correct alternative -- averaging light rather than gamma-encoded
    numbers -- which is closer to what a lower render resolution would actually
    produce, at the cost of moving further from the pretrained behaviour.
    """
    size = hr_encoded.shape[0] // SCALE

    if filter_name == "area-linear":
        blocks = decode(hr_encoded).reshape(size, SCALE, size, SCALE, 3)
        return encode(blocks.mean(axis=(1, 3)))

    source = Image.fromarray((hr_encoded * 255).astype(np.uint8))
    return np.asarray(source.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0


def convert(args, raw_dir, out_dir):
    tiles = sorted(raw_dir.glob("*.png"))
    if not tiles:
        sys.exit(f"no tiles in {raw_dir}")

    for subdir in ("hr", "lr"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f"\nconverting {len(tiles)} tiles")

    hashes = np.empty(0, dtype=np.uint64)
    written = 0
    skipped = 0

    for index, tile in enumerate(tiles, 1):
        if index % 1000 == 0:
            print(f"  {index}/{len(tiles)} tiles", flush=True)

        hr = encode(load_linear(tile))

        if hr.shape[0] != hr.shape[1] or hr.shape[0] % SCALE:
            print(f"skipping {tile.name}: {hr.shape[1]}x{hr.shape[0]} is not "
                  f"a square multiple of {SCALE}")
            skipped += 1
            continue

        if not args.no_dedup:
            digest = average_hash(hr)
            if np.any(hamming_to_all(digest, hashes) <= DUPLICATE_DISTANCE):
                skipped += 1
                continue
            hashes = np.append(hashes, digest)

        lr = downscale(hr, args.filter)

        Image.fromarray((hr * 255).astype(np.uint8)).save(out_dir / "hr" / tile.name)
        Image.fromarray((lr * 255).astype(np.uint8)).save(out_dir / "lr" / tile.name)

        written += 1

    print(f"wrote {written} pairs ({skipped} skipped as duplicates or malformed) "
          f"using {args.filter}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=1,
                        help="how many self-driving passes over the maps to run. "
                             "At 4K on the demo maps a round is ~1200 tiles and "
                             "~2.7 min, so 84 rounds is about 100,000 tiles")
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="folder to write the finished hr/lr pairs into")

    parser.add_argument("--geometry", default="3840x2160",
                        help="render resolution. Tiles are crops at render scale, "
                             "so this decides how many a frame can yield: 4K holds "
                             "7x4 disjoint 512 tiles, 1080p only 3x2")
    parser.add_argument("--maps", nargs="+", help="limit the scan to these maps")
    parser.add_argument("--samples", type=int, help="shots per map (default 60)")
    parser.add_argument("--looks", type=int, help="look directions per stop (default 6)")
    parser.add_argument("--settle", type=int,
                        help="rendered frames held still before each shot (default 16). "
                             "Lowering this much captures half-converged frames")
    parser.add_argument("--tile", type=int,
                        help="tile edge in pixels (default 512). 256 yields far more "
                             "tiles per frame at a quarter the size, and is a normal "
                             "SR training crop -- the model is fully convolutional")
    parser.add_argument("--tiles-per-frame", type=int, help="tiles per shot (default 24)")

    parser.add_argument("--filter", choices=("bicubic-encoded", "area-linear"),
                        default="bicubic-encoded", help="how to produce the LR input")
    parser.add_argument("--no-dedup", action="store_true", help="keep near-duplicate tiles")

    parser.add_argument("--windowed", action="store_true",
                        help="render into a window instead of headless. Needs a "
                             "display, and the compositor may clamp --geometry "
                             "to the size of the monitor")

    parser.add_argument("--raw-dir", help="where the engine writes tiles (default: auto-detect)")
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60,
                        help="seconds to allow each round (default 6h)")
    parser.add_argument("--fresh", action="store_true",
                        help="delete previously captured tiles before starting")
    parser.add_argument("--force", action="store_true",
                        help="capture even if the estimate will not fit on disk")
    parser.add_argument("--skip-capture", action="store_true",
                        help="convert what was captured before, without running the game")
    args = parser.parse_args()

    raw_dir = find_raw_dir(args.raw_dir)

    if not args.skip_capture:
        capture(args, raw_dir)

    convert(args, raw_dir, args.out)

    pairs = sum(1 for _ in (args.out / "hr").glob("*.png"))
    print(f"\ndone: {pairs} pairs under {args.out.resolve()}")
    print(f"  {args.out}/hr  512x512 targets")
    print(f"  {args.out}/lr  128x128 inputs, matched by filename")


if __name__ == "__main__":
    main()
