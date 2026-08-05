# Capturing an upscaler training set

This branch turns Q2RTX into a data-collection tool. It drives itself around the
game's levels, photographs the rendered frame in 512x512 tiles, and converts
those into the HR/LR pairs QuickSRNetSmall is fine-tuned on.

The point is domain match. The stock Qualcomm weights are trained on DIV2K
photographs; the model is asked at runtime to upscale path-traced,
ASVGF-denoised, TAA'd Quake 2 frames, which look nothing like photographs. These
tiles are read at exactly the point `vkpt_upscaler_do()` reads the frame, so what
comes out of the capture is pixel-for-pixel the domain the model is deployed in.

---

## Before you start

**Game data.** The capture visits every `.bsp` the filesystem can see. If
`baseq2/pak0.pak` points at the free Q2RTX demo, that is three maps (`demo1`,
`demo2`, `demo3`); a full Quake II install is roughly forty. Check with:

```
ls -l baseq2/pak0.pak
```

Three maps is enough to prove the pipeline out, but a set built from them will
hit heavy duplicate rejection long before it gets large — see
[Getting to 100,000 tiles](#getting-to-100000-tiles).

**Python.** `numpy` and `pillow`:

```
pip install numpy pillow
```

**Disk.** A 512x512 16-bit tile is about 1.2 MB, and a round at 4K collects
~1200 of them. The script estimates the total up front and refuses to start if it
will not fit.

---

## The short version

Build once:

```
cmake --build build --target capture game -j8
```

The `game` target matters as much as the tool: the capture bot teleports with a
`setpos` cheat that lives in the game library, and a stale one makes every
teleport silently do nothing.

Then capture:

```
python scripts/capture_dataset.py --rounds 1 --out dataset
```

That runs one round over every map and writes:

```
dataset/hr/*.png    512x512 uint8, encoded pow(1/2.2)
dataset/lr/*.png    128x128 uint8, the training input
```

Pairs are matched by filename. There is no train/validation split -- hold out
whatever you want at training time.

A window opens and drives itself. It ignores the keyboard and mouse entirely, so
nothing you type can disturb it, and it closes itself when the pass is done.
Leave it alone and do something else — but note it does hold the GPU.

**Capture at 4K if the machine can take it.** Tiles are crops at render scale,
not resizes of the frame, so resolution decides how many distinct tiles a frame
can yield: 3840x2160 holds 7x4 = 28 disjoint 512 tiles, 1920x1080 holds only
3x2 = 6. It is the single highest-leverage setting here.

---

## Getting to 100,000 tiles

One pass at 3840x2160 with the shipped settings produces **~1200 tiles in
~2.7 minutes** (measured on an RX 7800 XT, three demo maps). So:

```
python scripts/capture_dataset.py --rounds 84 --out dataset
```

Expect roughly **3.8 hours and ~117 GB** on three maps. On a full forty-map
install each round yields far more, so far fewer rounds get you there.

Every round re-seeds from the clock and so stops in different rooms and looks in
different directions, and each writes under its own filename prefix, so rounds
accumulate rather than overwrite. The run is resumable: leave off `--fresh` and
it adds to what is already there. The script prints the estimated size up front
and refuses to start if it will not fit on disk.

### The duplicate problem

On three maps, a long run keeps revisiting the same rooms. The conversion step
drops near-duplicates by perceptual hash, and the fraction it drops climbs as the
set grows — a single pass already loses about 25%. Ways out, best first:

1. **Install the full game.** Forty maps instead of three is the real fix.
2. **`--tile 256`.** QuickSRNet is fully convolutional, so the training crop size
   does not have to equal the 512 inference tile; 256 is a normal SR crop. A 4K
   frame then holds 15x8 = 120 disjoint tiles instead of 28, and each file is a
   quarter the size — 100,000 tiles becomes ~8 GB in far fewer rounds. Pair it
   with `--tiles-per-frame 96`.
3. **`--no-dedup`.** Keeps everything. Only sensible if you intend to dedup at
   training time instead.

---

## What the knobs do

| flag | meaning |
|---|---|
| `--rounds` | how many self-driving passes over the maps to run |
| `--out` | where the finished pairs go (required) |
| `--geometry` | render resolution, default `3840x2160`. Decides tiles per frame |
| `--maps` | limit the scan to named maps, e.g. `--maps demo1 demo2` |
| `--samples` | stops-worth of shots per map (default 60) |
| `--looks` | look directions per stop (default 6) |
| `--settle` | rendered frames held still before each shot (default 16) |
| `--tile` | tile edge in pixels (default 512) |
| `--tiles-per-frame` | tiles taken from each shot (default 24) |
| `--filter` | `bicubic-encoded` (default) or `area-linear` |
| `--fresh` | delete previously captured tiles first |
| `--skip-capture` | convert an existing capture without running the game |

`--settle` is the one not to lower much. The bot holds the camera still for that
many *rendered* frames before each shot so the path tracer and TAA converge; a
half-converged tile teaches the model to reproduce denoiser trails.

To re-convert an existing capture — a different downscale filter, say — without
running the game again:

```
python scripts/capture_dataset.py --rounds 0 --out dataset2 \
    --skip-capture --filter area-linear
```

---

## Where things are

| what | where |
|---|---|
| raw 16-bit tiles + `manifest.jsonl` | `~/.local/share/quake2rtx/baseq2/screenshots/dataset` |
| finished pairs | whatever you passed to `--out` |
| capture preset | `baseq2/capture_dataset.cfg` |
| per-round overrides | `baseq2/capture_user.cfg` (written and deleted by the script) |

Raw tiles land in `fs_gamedir`, which on Linux is the XDG data directory, not the
source tree. `manifest.jsonl` records the map, camera origin and angles for every
tile -- the conversion step does not need it, but it is there if you want to
group or filter the set by where the shot was taken.

Raw tiles hold **tone-mapped linear** values at 16 bits. The `pow(1/2.2)` encode
that `upscaler_pack.comp` applies at runtime is deliberately left to the
conversion step, so the transfer curve can be retuned later without recapturing
anything.

---

## Doing it by hand

The script is a wrapper. To drive the tool directly:

```
./q2rtx_capture
```

It execs `capture_dataset.cfg`, then `capture_user.cfg` if present, then scans.
Inside a normal `q2rtx` session the same commands exist:

- `capture_scan [map ...]` — scan the listed maps, or all of them
- `capture_scan_stop` — stop and report
- `capture_shot` — write tiles from the next frame, wherever you are standing

Then convert whatever was collected:

```
python scripts/capture_dataset.py --rounds 0 --out dataset --skip-capture
```

---

## When it goes wrong

**`teleports are not landing -- is this build's game library current?`**
The bot moves via a `setpos` cheat added to the game library, and the build does
not rebuild it for you. `cmake --build build --target game`.

**`no maps to scan` or only three maps found.** The filesystem is seeing the demo
data. Point `baseq2/pak0.pak` at a full install.

**`<map> did not come up, skipping it`.** That level refused to load; the scan
moves on rather than stalling. Some maps in a full install are not loadable
standalone.

**Very few tiles per shot.** Everything in view was flat enough to be rejected as
uninteresting — sky, an unlit wall. Raise the resolution, or lower
`capture_min_stddev` in `baseq2/capture_dataset.cfg` if you want flat tiles kept.

**Note on this branch:** the capture behaviour is unconditional, so `q2rtx` itself
also boots straight into a scan and ignores input. This is a throwaway
data-collection branch, not a playable build.
