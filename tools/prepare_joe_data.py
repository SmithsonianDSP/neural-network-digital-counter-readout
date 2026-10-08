"""Audit and build a 20x32 training crop directory from Joe's raw dig samples.

Usage:
    python tools/prepare_joe_data.py --audit
    python tools/prepare_joe_data.py --build [--force]

--audit prints a summary of joes-samples/*.jpg (per-label counts, per-position
x per-label matrix, per-frame label table) and writes a contact sheet PNG
(work/audit_sheet.png, split into work/audit_sheet_NN.png if it would be too
tall) so the captures can be eyeballed for labeling mistakes.

--build resizes every joes-samples/*.jpg to the model's native 20x32 input
size and writes it to 04_joe_lcd_20x32/<same basename>, JPEG quality=100.
Refuses to run if the output dir already exists unless --force is given (in
which case it is cleared first). Never touches
03_data_resize_all-use_for_training/ or anything else.

    --resize nearest   (default) PIL NEAREST + PIL's default 4:2:0 chroma
                       subsampling -- byte-for-byte the historical build.
    --resize stb       dig_data.stb_resize, the firmware's Mitchell downscale
                       (also bilinear / area / lanczos). Any non-nearest build is
                       saved 4:4:4 (subsampling=0) so the 20x32 chroma survives.
    --src DIR / --out DIR override joes-samples/ and 04_joe_lcd_20x32/ (relative
    paths resolve against the repo root). Filenames stay *.jpg either way.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from glob import glob

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dig_data import (  # noqa: E402
    CLASS_NAMES,
    RESAMPLE,
    parse_frame,
    parse_label,
    parse_position,
    resize_image,
    screen_type,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "joes-samples")
BUILD_DIR = os.path.join(REPO_ROOT, "04_joe_lcd_20x32")
SHEET_PATH = os.path.join(REPO_ROOT, "work", "audit_sheet.png")

POSITIONS = [2, 3, 4, 5, 6]
TARGET_W, TARGET_H = 20, 32

ROW_H = 64  # display height for each digit crop in the contact sheet
MAX_SHEET_H = 8000


# --------------------------------------------------------------------------
# common: gather + parse every sample, loudly reporting bad names
# --------------------------------------------------------------------------


def gather_samples():
    """Return (records, bad) where records is a list of dicts for every file
    that parsed cleanly, and bad is a list of (path, error) for files that
    didn't."""
    files = sorted(glob(os.path.join(SRC_DIR, "*.jpg")))
    records = []
    bad = []
    for f in files:
        try:
            label = parse_label(f)
            pos = parse_position(f)
            frame = parse_frame(f)
            if pos is None:
                raise ValueError(f"{os.path.basename(f)!r}: no _dig<N>_ position token found")
            if frame is None:
                raise ValueError(f"{os.path.basename(f)!r}: no YYYYmmdd-HHMMSS timestamp token found")
            records.append({"path": f, "label": label, "position": pos, "frame": frame})
        except Exception as exc:  # noqa: BLE001 - we want to catch & report *any* parse failure
            bad.append((f, str(exc)))
    return records, bad


def report_bad(bad):
    if not bad:
        return
    print()
    print("!" * 70)
    print(f"!! {len(bad)} FILE(S) FAILED TO PARSE -- these are NOT included below:")
    for f, err in bad:
        print(f"!!   {f}")
        print(f"!!     -> {err}")
    print("!" * 70)


# --------------------------------------------------------------------------
# --audit
# --------------------------------------------------------------------------


def audit():
    records, bad = gather_samples()
    report_bad(bad)

    total = len(records)
    print(f"\nTotal parsed samples: {total}" + (f"  ({len(bad)} failed to parse)" if bad else ""))
    if not records and not bad:
        print(f"No .jpg files found under {SRC_DIR}")
        return

    # --- per-label counts ---
    print("\nPer-label counts")
    print("-" * 60)
    label_counts = {c: 0 for c in CLASS_NAMES}
    for r in records:
        label_counts[CLASS_NAMES[r["label"]]] += 1
    for c in CLASS_NAMES:
        print(f"  {c:>3}: {label_counts[c]:>4}")

    # --- per-position x per-label matrix ---
    print("\nPer-position x per-label matrix")
    print("-" * 60)
    positions = sorted({r["position"] for r in records})
    header = "pos   " + "".join(f"{c:>5}" for c in CLASS_NAMES) + f"{'tot':>7}"
    print(header)
    for p in positions:
        row = [r for r in records if r["position"] == p]
        counts = {c: 0 for c in CLASS_NAMES}
        for r in row:
            counts[CLASS_NAMES[r["label"]]] += 1
        print(f"dig{p:<3}" + "".join(f"{counts[c]:>5}" for c in CLASS_NAMES) + f"{len(row):>7}")
    counts = {c: 0 for c in CLASS_NAMES}
    for r in records:
        counts[CLASS_NAMES[r["label"]]] += 1
    print(f"{'tot':<6}" + "".join(f"{counts[c]:>5}" for c in CLASS_NAMES) + f"{total:>7}")

    # --- group into frames ---
    by_frame: dict = {}
    for r in records:
        by_frame.setdefault(r["frame"], {})[r["position"]] = r
    frames = sorted(by_frame.keys())

    print(f"\nPer-frame table ({len(frames)} frames)")
    print("-" * 90)
    header = f"{'timestamp':<16} {'screen':<9} " + "  ".join(f"dig{p}" for p in POSITIONS)
    print(header)
    screen_counts: dict = {}
    for frame in frames:
        recs = by_frame[frame]
        frame_labels = {pos: r["label"] for pos, r in recs.items()}
        screen = screen_type(frame_labels)
        screen_counts[screen] = screen_counts.get(screen, 0) + 1
        cells = []
        for p in POSITIONS:
            if p in recs:
                cells.append(f"{CLASS_NAMES[recs[p]['label']]:>4}")
            else:
                cells.append(f"{'-':>4}")
        print(f"{frame:<16} {screen:<9} " + "  ".join(cells))

    print("\nScreen-type counts across frames")
    print("-" * 60)
    for s in sorted(screen_counts, key=lambda k: -screen_counts[k]):
        print(f"  {s:<10} {screen_counts[s]:>4}")

    build_contact_sheets(frames, by_frame)


def _load_thumb(path):
    """Load a digit crop and resize (preserving aspect) to display height ROW_H."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_w = max(1, round(w * ROW_H / h))
    return img.resize((new_w, ROW_H), Image.Resampling.NEAREST)


def build_contact_sheets(frames, by_frame):
    os.makedirs(os.path.dirname(SHEET_PATH), exist_ok=True)

    text_h = 18
    pad = 6
    row_h = ROW_H + text_h + pad
    label_col_w = 260

    # figure out a common crop width so columns line up
    thumb_w = None
    for frame in frames:
        for p in POSITIONS:
            r = by_frame[frame].get(p)
            if r is not None:
                thumb = _load_thumb(r["path"])
                thumb_w = thumb.width
                break
        if thumb_w is not None:
            break
    if thumb_w is None:
        thumb_w = TARGET_W * (ROW_H // TARGET_H)

    sheet_w = label_col_w + len(POSITIONS) * (thumb_w + pad) + pad

    # split frames into pages so no single PNG exceeds MAX_SHEET_H
    rows_per_page = max(1, MAX_SHEET_H // row_h)
    pages = [frames[i : i + rows_per_page] for i in range(0, len(frames), rows_per_page)]

    written = []
    for page_idx, page_frames in enumerate(pages):
        sheet_h = pad + len(page_frames) * row_h
        sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)

        y = pad
        for frame in page_frames:
            recs = by_frame[frame]
            frame_labels = {pos: r["label"] for pos, r in recs.items()}
            screen = screen_type(frame_labels)
            label_bits = ",".join(
                f"d{p}={CLASS_NAMES[recs[p]['label']] if p in recs else '-'}" for p in POSITIONS
            )
            text = f"{frame}  [{screen}]  {label_bits}"
            draw.text((pad, y), text, fill=(255, 255, 0))

            x = label_col_w
            for p in POSITIONS:
                r = recs.get(p)
                if r is not None:
                    thumb = _load_thumb(r["path"])
                    sheet.paste(thumb, (x, y + text_h))
                    draw.rectangle(
                        [x, y + text_h, x + thumb.width - 1, y + text_h + thumb.height - 1],
                        outline=(80, 80, 80),
                    )
                else:
                    draw.rectangle(
                        [x, y + text_h, x + thumb_w - 1, y + text_h + ROW_H - 1],
                        outline=(120, 0, 0),
                    )
                    draw.text((x + 2, y + text_h + ROW_H // 2 - 6), "missing", fill=(255, 80, 80))
                x += thumb_w + pad

            y += row_h

        if len(pages) == 1:
            path = SHEET_PATH
        else:
            root, ext = os.path.splitext(SHEET_PATH)
            path = f"{root}_{page_idx + 1:02d}{ext}"
        sheet.save(path)
        written.append((path, sheet.size))

    print("\nContact sheet(s) written")
    print("-" * 60)
    for path, size in written:
        print(f"  {path}  ({size[0]}x{size[1]})")


# --------------------------------------------------------------------------
# --build
# --------------------------------------------------------------------------


def build(force: bool, resize: str = "nearest"):
    records, bad = gather_samples()
    report_bad(bad)
    if bad:
        print("\nERROR: refusing to build with unparsed filenames present. Fix them first.")
        return 1

    if os.path.normcase(os.path.abspath(BUILD_DIR)) == os.path.normcase(os.path.abspath(SRC_DIR)):
        print("ERROR: --out is the source directory; refusing (--force would delete it).")
        return 1

    if os.path.exists(BUILD_DIR):
        if not force:
            print(
                f"ERROR: {BUILD_DIR} already exists. Re-run with --force to clear and rebuild it."
            )
            return 1
        print(f"--force given: clearing existing {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)

    os.makedirs(BUILD_DIR, exist_ok=True)
    print(f"source {SRC_DIR}\noutput {BUILD_DIR}\nresize {resize}"
          + ("" if resize == "nearest" else "  (JPEG q100, 4:4:4)"))

    written = 0
    label_counts = {c: 0 for c in CLASS_NAMES}
    for r in records:
        src = r["path"]
        img = Image.open(src).convert("RGB")
        dst = os.path.join(BUILD_DIR, os.path.basename(src))
        if resize == "nearest":
            # The historical build, kept bit-identical for reproducibility.
            img = img.resize((TARGET_W, TARGET_H), Image.Resampling.NEAREST)
            img.save(dst, "JPEG", quality=100)
        else:
            img = resize_image(img, TARGET_W, TARGET_H, resize)
            img.save(dst, "JPEG", quality=100, subsampling=0)
        written += 1
        label_counts[CLASS_NAMES[r["label"]]] += 1

    print(f"\nWrote {written} files to {BUILD_DIR}")
    print("\nPer-label counts")
    print("-" * 60)
    for c in CLASS_NAMES:
        print(f"  {c:>3}: {label_counts[c]:>4}")

    src_count = len(records)
    out_count = len(glob(os.path.join(BUILD_DIR, "*.jpg")))
    print(f"\nVerify: source samples = {src_count}, files written = {written}, files on disk = {out_count}")
    if not (src_count == written == out_count):
        print("MISMATCH -- something went wrong!")
        return 1
    print("OK: counts match.")
    return 0


# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="audit joes-samples/ and write a contact sheet")
    mode.add_argument("--build", action="store_true", help="resize joes-samples/ into 04_joe_lcd_20x32/")
    ap.add_argument("--force", action="store_true", help="with --build: clear an existing output dir first")
    ap.add_argument("--resize", default="nearest", choices=sorted(RESAMPLE),
                    help="with --build: 20x32 resize filter (default nearest = historical; "
                         "stb = the firmware's Mitchell downscale)")
    ap.add_argument("--src", default=None, help="source dir (default joes-samples/)")
    ap.add_argument("--out", default=None, help="with --build: output dir (default 04_joe_lcd_20x32/)")
    args = ap.parse_args(argv)

    global SRC_DIR, BUILD_DIR
    if args.src:
        SRC_DIR = os.path.join(REPO_ROOT, args.src) if not os.path.isabs(args.src) else args.src
    if args.out:
        BUILD_DIR = os.path.join(REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out

    if args.audit:
        audit()
        return 0
    return build(force=args.force, resize=args.resize)


if __name__ == "__main__":
    raise SystemExit(main())
