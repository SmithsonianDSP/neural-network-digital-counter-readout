"""Inspect and calibrate the LCD photometric augmenter against real captures.

Usage:
    python tools/augment_preview.py --stats [--n 2000] [--data joes-samples]
    python tools/augment_preview.py --sheet [--out work/augment_sheet.png]
    python tools/augment_preview.py --probe [--model dig-class11_2000_s2.tflite]

``--stats`` compares the contrast distribution (dig_data.contrast = p95-p5 of
luma) of the real corpus against augmented samples drawn from it. The goal is
for the augmented distribution to *bracket* reality: p5 at or a little below
the real p5, median not far below the real median. Too soft and the model
never sees the hazy case it fails on; too aggressive and we train on noise.

``--sheet`` renders contact sheets (model-resolution and native-resolution) so
a human can confirm the segments are still legible after degradation.

``--probe`` feeds augmented hazy ``7`` crops to a shipped .tflite model. The
shipped model is *expected* to do badly on these -- that is the whole reason
for the retrain. This is only a smoke test that augmented 7s have not become
literally unreadable.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augment_lcd import DEFAULT, augment_lcd, make_idg_preprocessing_fn  # noqa: E402,F401
from dig_data import CLASS_NAMES, TARGET_H, TARGET_W, contrast, load_dirs  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCTS = (5, 25, 50, 75, 95)


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def cmd_stats(args) -> int:
    x, y, meta = load_dirs(args.data, resize="nearest")
    if len(x) == 0:
        print(f"no images found under {args.data!r}")
        return 1

    real = np.array([contrast(a) for a in x], dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(x), size=args.n)
    aug = np.empty(args.n, dtype=np.float64)
    for i, j in enumerate(idx):
        aug[i] = contrast(augment_lcd(x[j].astype(np.float32), rng, DEFAULT))

    r = np.percentile(real, PCTS)
    a = np.percentile(aug, PCTS)
    dark_frac = float((aug < r[0]).mean())

    print(f"data      : {args.data}   real n={len(real)}   augmented n={args.n}")
    print(f"config    : veil_k={DEFAULT.veil_k} veil_level={DEFAULT.veil_level} "
          f"p={DEFAULT.veil_p} | contrast={DEFAULT.contrast} exposure={DEFAULT.exposure}")
    print()
    print("contrast (p95-p5 luma) on 20x32 NEAREST crops")
    print(f"  {'':10s} {'p5':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p95':>8s}")
    print(f"  {'real':10s} " + " ".join(f"{v:8.1f}" for v in r))
    print(f"  {'augmented':10s} " + " ".join(f"{v:8.1f}" for v in a))
    print(f"  {'ratio':10s} " + " ".join(f"{av / rv:8.2f}" if rv else f"{'-':>8s}"
                                         for av, rv in zip(a, r)))
    print()
    print(f"augmented below real p5 ({r[0]:.1f}) : {dark_frac * 100:.1f}%")
    med_ratio = a[2] / r[2] if r[2] else float("nan")
    print(f"augmented median / real median      : {med_ratio:.2f}")

    ok_med = med_ratio >= 0.60
    ok_dark = dark_frac <= 0.25
    ok_p5 = a[0] <= r[0] * 1.05
    print()
    print(f"  [{'ok' if ok_p5 else 'FAIL'}] augmented p5 at or below real p5")
    print(f"  [{'ok' if ok_med else 'FAIL'}] augmented median >= 60% of real median")
    print(f"  [{'ok' if ok_dark else 'FAIL'}] <=25% of augmented below real p5")

    # per-screen split is informative: kwh frames are the hazy ones
    by_screen: dict = {}
    for c, m in zip(real, meta):
        by_screen.setdefault(m.get("screen") or "?", []).append(c)
    print()
    print("real contrast by screen type")
    for k in sorted(by_screen):
        v = np.array(by_screen[k])
        print(f"  {k:8s} n={len(v):4d}  p5={np.percentile(v, 5):6.1f} "
              f"p50={np.percentile(v, 50):6.1f} p95={np.percentile(v, 95):6.1f}")

    return 0 if (ok_med and ok_dark and ok_p5) else 2


# --------------------------------------------------------------------------
# contact sheet
# --------------------------------------------------------------------------

# Rows we want a human to eyeball: the hazy dig3 sevens that the shipped model
# reads as ones, the dig6 ones they get confused with, plus an 8 and a blank.
ROW_SPECS = [
    ("7", 3, 4),
    ("1", 6, 2),
    ("8", 6, 2),
    ("N", 3, 2),
]


def _pick_rows(paths, labels, positions, contrasts):
    """Select source images per ROW_SPECS.

    Candidates are sorted by contrast and sampled evenly across the hazier
    (lower-contrast) 70%, so a row set shows the range of source haze rather
    than only the single worst frame -- which would make every variant look
    hopeless and tell us nothing about the typical case.
    """
    chosen = []
    for lab, pos, count in ROW_SPECS:
        cand = [i for i in range(len(paths))
                if labels[i] == lab and positions[i] == pos]
        if not cand:
            continue
        cand.sort(key=lambda i: contrasts[i])
        pool = cand[:max(count, int(round(len(cand) * 0.7)))]
        take = min(count, len(pool))
        for k in np.linspace(0, len(pool) - 1, take):
            chosen.append(pool[int(round(k))])
    return chosen


def _tile(img: np.ndarray, zoom: int) -> np.ndarray:
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    return np.kron(u8, np.ones((zoom, zoom, 1), dtype=np.uint8))


def _build_sheet(sources, label_texts, n_variants, zoom, seed, cfg=DEFAULT):
    """Grid: col 0 = original, cols 1.. = augmented variants."""
    from PIL import ImageDraw

    rng = np.random.default_rng(seed)
    sep = 3
    pad_left = 74
    th, tw = sources[0].shape[:2]
    cw, ch = tw * zoom, th * zoom
    cols = n_variants + 1
    width = pad_left + cols * cw + (cols + 1) * sep
    height = len(sources) * ch + (len(sources) + 1) * sep

    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    for r, src in enumerate(sources):
        y0 = sep + r * (ch + sep)
        cells = [src.astype(np.float32)]
        cells += [augment_lcd(src.astype(np.float32), rng, cfg) for _ in range(n_variants)]
        for c, cell in enumerate(cells):
            x0 = pad_left + sep + c * (cw + sep)
            canvas[y0:y0 + ch, x0:x0 + cw] = _tile(cell, zoom)
        # a brighter separator right after the original marks the boundary
        xb = pad_left + sep + cw
        canvas[y0:y0 + ch, xb:xb + sep] = 200

    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    for r, text in enumerate(label_texts):
        y0 = sep + r * (ch + sep)
        draw.text((4, y0 + max(0, ch // 2 - 6)), text, fill=(230, 230, 230))
    return im


def cmd_sheet(args) -> int:
    x, y, meta = load_dirs(args.data, resize="nearest")
    if len(x) == 0:
        print(f"no images found under {args.data!r}")
        return 1

    labels = [CLASS_NAMES[int(v)] for v in y]
    positions = [m.get("position") for m in meta]
    paths = [m["path"] for m in meta]
    contrasts = [m["contrast"] for m in meta]

    picks = _pick_rows(paths, labels, positions, contrasts)
    if not picks:
        print("no rows matched ROW_SPECS")
        return 1

    texts = [f"{labels[i]} d{positions[i]}\nC={contrasts[i]:.0f}" for i in picks]

    out = args.out or os.path.join(REPO, "work", "augment_sheet.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    # sheet 1: model resolution, 6x nearest upscale
    small = [x[i] for i in picks]
    im = _build_sheet(small, texts, args.variants, args.zoom, args.seed)
    im.save(out)
    print(f"wrote {out}  ({im.size[0]}x{im.size[1]}, {len(picks)} rows x "
          f"{args.variants + 1} cols, 20x32 crops at {args.zoom}x)")

    # sheet 2: native capture resolution, for eyeballing real image quality
    native = []
    for i in picks:
        native.append(np.array(Image.open(paths[i]).convert("RGB"), dtype=np.uint8))
    root, ext = os.path.splitext(out)
    out2 = f"{root}_native{ext}"
    im2 = _build_sheet(native, texts, args.variants, 1, args.seed)
    im2.save(out2)
    print(f"wrote {out2}  ({im2.size[0]}x{im2.size[1]}, native "
          f"{native[0].shape[1]}x{native[0].shape[0]})")
    return 0


# --------------------------------------------------------------------------
# model probe
# --------------------------------------------------------------------------


def cmd_probe(args) -> int:
    from ai_edge_litert.interpreter import Interpreter

    x, y, meta = load_dirs(args.data, resize="nearest")
    sel = [i for i in range(len(x))
           if int(y[i]) == 7 and meta[i].get("position") == 3]
    if not sel:
        print("no dig3 '7' crops found")
        return 1

    itp = Interpreter(model_path=args.model)
    itp.allocate_tensors()
    ii = itp.get_input_details()[0]["index"]
    oi = itp.get_output_details()[0]["index"]

    def predict(arr: np.ndarray) -> int:
        itp.set_tensor(ii, arr.reshape(1, TARGET_H, TARGET_W, 3).astype(np.float32))
        itp.invoke()
        return int(np.argmax(itp.get_tensor(oi)[0]))

    plain = Counter(CLASS_NAMES[predict(x[i].astype(np.float32))] for i in sel)

    rng = np.random.default_rng(args.seed)
    augc: Counter = Counter()
    for i in sel:
        for _ in range(args.reps):
            augc[CLASS_NAMES[predict(augment_lcd(x[i].astype(np.float32), rng, DEFAULT))]] += 1

    n_plain = sum(plain.values())
    n_aug = sum(augc.values())
    print(f"model : {args.model}")
    print(f"source: {len(sel)} hazy dig3 '7' crops, {args.reps} augmentations each")
    print()
    print(f"  {'pred':>5s} {'unaug %':>9s} {'aug %':>9s}")
    for k in sorted(set(plain) | set(augc), key=lambda s: CLASS_NAMES.index(s)):
        print(f"  {k:>5s} {100 * plain[k] / n_plain:9.1f} {100 * augc[k] / n_aug:9.1f}")
    print()
    print(f"correct ('7'): unaug {100 * plain['7'] / n_plain:.1f}%  "
          f"aug {100 * augc['7'] / n_aug:.1f}%")
    return 0


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", action="store_true")
    p.add_argument("--sheet", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--data", default=os.path.join(REPO, "joes-samples"))
    p.add_argument("--n", type=int, default=2000, help="augmented samples for --stats")
    p.add_argument("--out", default=None, help="--sheet output png")
    p.add_argument("--variants", type=int, default=11, help="augmented cols per row")
    p.add_argument("--zoom", type=int, default=6, help="nearest upscale for --sheet")
    p.add_argument("--model", default=os.path.join(REPO, "dig-class11_2000_s2.tflite"))
    p.add_argument("--reps", type=int, default=20, help="augmentations per crop for --probe")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not (args.stats or args.sheet or args.probe):
        p.error("choose at least one of --stats / --sheet / --probe")

    rc = 0
    if args.stats:
        rc |= cmd_stats(args)
    if args.sheet:
        rc |= cmd_sheet(args)
    if args.probe:
        rc |= cmd_probe(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
