"""Shared data loading helpers for the 11-class digit (7-segment) readout models.

Model I/O contract (matches notebook 03 and the ESP32 device):
  * input  : float32 (1, 32, 20, 3), RAW 0-255 pixel values -- NO normalisation
             (the first layer of the network is a BatchNormalization layer).
  * resize : 20 wide x 32 tall, PIL NEAREST.
  * output : softmax over 11 classes, class 10 == NaN / blank ("N").

Dependency-light on purpose: PIL + numpy only.
"""

from __future__ import annotations

import os
import re
from glob import glob

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

LABELS = {**{str(d): d for d in range(10)}, "N": 10}
CLASS_NAMES = [str(d) for d in range(10)] + ["N"]
NAN_CLASS = 10

TARGET_W = 20
TARGET_H = 32

RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "area": Image.Resampling.BOX,  # PIL's BOX == area averaging
    "lanczos": Image.Resampling.LANCZOS,
    "stb": None,  # the firmware's resize -- see stb_resize(); not a PIL filter
}

# What the device does (jomjol/AI-on-the-Edge-Device, verified 2026-10-08):
# ClassFlowCNNGeneral cuts the ROI at full resolution (the image saved to SD), then
# CImageBasis::Resize calls stbir_resize_uint8(...) with library defaults -- for a
# downscale that is stb_image_resize v1's Mitchell filter (B = C = 1/3), clamped edges,
# linear (no sRGB) arithmetic -- and CTfLiteClass feeds the raw 0-255 RGB as floats.
# Training/eval used PIL NEAREST until then, a train/deploy mismatch.
DEVICE_RESIZE = "stb"


def _mitchell(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return np.where(x < 1, (16 + x * x * (21 * x - 36)) / 18,
                    np.where(x < 2, (32 + x * (-60 + x * (36 - 7 * x))) / 18, 0.0))


def _stb_axis_weights(n_in: int, n_out: int) -> np.ndarray:
    """[n_out, n_in] normalised Mitchell weights for one axis (downscale or upscale)."""
    scale = n_out / n_in
    support = 2.0 / scale if scale < 1 else 2.0      # filter widens when shrinking
    w = np.zeros((n_out, n_in), dtype=np.float64)
    for j in range(n_out):
        c = (j + 0.5) / scale - 0.5                  # output centre in input coords
        lo, hi = int(np.floor(c - support)), int(np.ceil(c + support))
        for i in range(lo, hi + 1):
            d = (i - c) * scale if scale < 1 else (i - c)
            v = float(_mitchell(np.array(d)))
            if v:
                w[j, min(max(i, 0), n_in - 1)] += v  # clamp edges
        w[j] /= w[j].sum()
    return w


_STB_CACHE: dict = {}


def stb_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    """Emulate the firmware's stbir_resize_uint8 (Mitchell, clamp, linear)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    key = (a.shape[1], a.shape[0], w, h)
    if key not in _STB_CACHE:
        _STB_CACHE[key] = (_stb_axis_weights(a.shape[1], w), _stb_axis_weights(a.shape[0], h))
    wx, wy = _STB_CACHE[key]
    out = np.einsum("yi,ixc->yxc", wy, np.einsum("xj,ijc->ixc", wx, a))
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), "RGB")


def resize_image(img: Image.Image, w: int, h: int, mode: str = DEVICE_RESIZE) -> Image.Image:
    if mode == "stb":
        return stb_resize(img, w, h)
    return img.resize((w, h), RESAMPLE[mode])

_POS_RE = re.compile(r"_dig(\d+)_")
_FRAME_RE = re.compile(r"(\d{8}-\d{6})")


# --------------------------------------------------------------------------
# filename parsing
# --------------------------------------------------------------------------


def parse_label(name: str) -> int:
    """Return the class index (0..10) encoded in a file name.

    User-capture style names look like ``5_main_dig3_20260730-163930.jpg`` or
    ``N_main_dig2_...`` -- the label is the token before the first underscore.
    Upstream corpus names are messier (``0.0_ROI0_...``, ``0 - 0_1_...``,
    ``0-1.jpg``) and the upstream convention is simply ``basename[0]``.

    A leading token of exactly "10" is a hard error: the upstream
    ``basename[0]`` rule would silently read such a file as class 1.
    """
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    token = stem.split("_", 1)[0]

    if token == "10":
        raise ValueError(
            f"{base!r}: leading token '10' is ambiguous -- upstream label parsing "
            f"uses basename[0] and would read this as class 1. Use 'N_' for the "
            f"NaN/blank class (class 10)."
        )

    if token in LABELS:
        return LABELS[token]

    # upstream fallback: first character of the basename
    first = base[0:1]
    if first == "N":
        return NAN_CLASS
    if first.isdigit():
        return int(first)
    raise ValueError(f"{base!r}: cannot parse a label from the file name")


def parse_position(name: str) -> int | None:
    """Digit position from a ``_dig<N>_`` token, else None."""
    m = _POS_RE.search(os.path.basename(name))
    return int(m.group(1)) if m else None


def parse_frame(name: str) -> str | None:
    """``YYYYmmdd-HHMMSS`` capture timestamp token, else None."""
    m = _FRAME_RE.search(os.path.basename(name))
    return m.group(1) if m else None


SCREEN_POSITIONS = ("dig2", "dig3", "dig4", "dig5", "dig6")


def _norm_frame_labels(labels: dict) -> dict:
    """{2|'dig2'|'2': 5|'5'|10|'10'|'N'} -> {'dig2': '5', ...} (str labels, N for NaN)."""
    out = {}
    for k, v in labels.items():
        ks = str(k).lower()
        pos = ks if ks.startswith("dig") else f"dig{int(ks)}"
        if isinstance(v, (int, np.integer)):
            v = "N" if int(v) == NAN_CLASS else str(int(v))
        v = str(v).upper()
        out[pos] = "N" if v in ("10", "N") else v
    return out


def classify_screen(labels: dict) -> str:
    """Type one frame from its FULL label set, primarily by dig2.

    The display cycles reading `5 d3 d4 d5 d6`, `00000`, `88888` and the dash screen
    (`- - - # #`). dig2 is the most discriminating position (5 / 0 / 8 / blank), and
    the other digits corroborate:

      dig2 5 -> "reading"  if dig3 in {7,8,9}           (meter range 57000-59999)
      dig2 0 -> "test0"    if >=2 of dig3..dig6 read 0
      dig2 8 -> "test8"    if >=2 of dig3..dig6 read 8
      dig2 N -> "dash"     if dig3 and dig4 are also N

    If dig2's call is not corroborated, fall back on the remaining digits
    (dig3=dig4=N -> dash; >=3 zeros -> test0; >=3 eights -> test8), else
    "uncertain". Labels may be ints (10 = N) or strings; keys int or "digN".
    """
    lab = _norm_frame_labels(labels)
    d2 = lab.get("dig2")
    rest = [lab.get(p) for p in SCREEN_POSITIONS[1:]]
    n0 = sum(1 for v in rest if v == "0")
    n8 = sum(1 for v in rest if v == "8")
    dashlike = lab.get("dig3") == "N" and lab.get("dig4") == "N"

    if d2 == "5" and lab.get("dig3") in ("7", "8", "9"):
        return "reading"
    if d2 == "0" and n0 >= 2:
        return "test0"
    if d2 == "8" and n8 >= 2:
        return "test8"
    if d2 == "N" and dashlike:
        return "dash"
    # dig2 not corroborated -> majority of the others
    if dashlike:
        return "dash"
    if n0 >= 3:
        return "test0"
    if n8 >= 3:
        return "test8"
    return "uncertain"


_LEGACY_SCREEN = {"reading": "kwh", "test0": "zeros", "test8": "test8", "dash": "blank09"}


def screen_type(frame_labels: dict, full_labels: dict | None = None) -> str:
    """Classify one captured frame from the labels of its digit positions.

    Returns the legacy vocabulary: "kwh" | "test8" | "zeros" | "blank09".

    full_labels (optional): the frame's COMPLETE label set (all five positions, as
    the model / review assigned them), independent of which crops survive in the
    directory being loaded. When given, typing is dig2-primary via
    classify_screen(); an "uncertain" result falls back to the legacy rule over the
    full set. Without it, behaviour is exactly the legacy rule over frame_labels:

    "test8"   -> every position reads 8 (segment self-test screen)
    "zeros"   -> every position reads 0
    "blank09" -> any position is blank / NaN
    "kwh"     -> anything else (the normal meter reading)
    """
    if full_labels:
        kind = classify_screen(full_labels)
        if kind in _LEGACY_SCREEN:
            return _LEGACY_SCREEN[kind]
        frame_labels = {
            k: (NAN_CLASS if v == "N" else int(v))
            for k, v in _norm_frame_labels(full_labels).items()
        }
    vals = list(frame_labels.values())
    if not vals:
        return "kwh"
    if any(v == NAN_CLASS for v in vals):
        return "blank09"
    if all(v == 8 for v in vals):
        return "test8"
    if all(v == 0 for v in vals):
        return "zeros"
    return "kwh"


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------


def load_image(path: str, resize: str = "nearest") -> np.ndarray:
    """Load one image as uint8 [32, 20, 3] RGB, resizing only if needed.

    Pass resize=DEVICE_RESIZE ("stb") to match what the firmware feeds the model."""
    img = Image.open(path).convert("RGB")
    if img.size != (TARGET_W, TARGET_H):
        img = resize_image(img, TARGET_W, TARGET_H, resize)
    return np.array(img, dtype=np.uint8)


def contrast(img) -> float:
    """p95 - p5 of luma over a (already cropped/resized) image."""
    a = np.asarray(img, dtype=np.float32)
    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return float(np.percentile(luma, 95) - np.percentile(luma, 5))


def luma(img) -> np.ndarray:
    """Rec.601 luma of an RGB array (float32)."""
    a = np.asarray(img, dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def crop_light_stats(path: str, resize: str = "nearest") -> dict:
    """Per-crop light measures, from one decode.

    mean / median : luma over the NATIVE crop (94x202). The median is essentially
                    the LCD background level -- segments are a minority of pixels
                    even on 88888 -- so it is insensitive to which screen is shown.
    contrast      : the existing p95-p5 helper, on the 20x32 model view (same
                    definition the corpus / eval tooling uses).
    """
    img = Image.open(path).convert("RGB")
    y = luma(img)
    small = img if img.size == (TARGET_W, TARGET_H) else resize_image(
        img, TARGET_W, TARGET_H, resize)
    return {
        "mean": float(y.mean()),
        "median": float(np.median(y)),
        "contrast": contrast(small),
    }


def frame_brightness(crop_medians) -> float:
    """Frame brightness = median over the frame's crops of each crop's median luma.

    Median-of-medians: robust to one crop carrying glare or a neighbour's segment,
    and to the screen shown (background dominates every crop).
    """
    vals = [float(v) for v in crop_medians if v is not None]
    if not vals:
        return float("nan")
    return float(np.median(vals))


# --------------------------------------------------------------------------
# light buckets
# --------------------------------------------------------------------------

# Legacy capture-hour rule (batch 1/2, early August). Stale by late September:
# hour 18 is flash-only and hour 07 is dim by then. Kept as fallback/for comparison.
HOUR_TRANSITION = frozenset({6, 19})
HOUR_FLASH = frozenset({20, 21, 22, 23, 0, 1, 2, 3, 4, 5})

# Brightness thresholds on frame_brightness(), calibrated on batch 3
# (2026-08-15..09-30, 16,559 frames; `select_review_set.py --derive --stats-cache`
# prints the calibration):
#   hours 01-04 (known flash-only, n=2758): p1 73.8  p50 77.1  p99.5 80.1  max 80.8
#   hours 11-14 (known day,        n=2759): min 82.0  p1 87.9  p5 97.2  p50 137.1
#   deployed-9002 error rate on pinned crops by brightness: <80 ~14-16%,
#   80-96 falls 6.0% -> 1.5%, >=96 ~0.1-0.4% (with a ~1% bump at >=140: bright
#   morning sun and the dig6 reflection streak).
# So: flash <= 81 (just above every known-night frame), day >= 96 (where the error
# rate reaches its daytime floor; ~day-hours p5), transition in between.
LIGHT_FLASH_MAX = 81.0  # brightness <= this -> "flash"
LIGHT_DAY_MIN = 96.0    # brightness >= this -> "day"; in between -> "transition"


def bucket_for_hour(h: int) -> str:
    """Legacy hour-of-day light bucket (06/19 transition, 20-05 flash, 07-18 day)."""
    h = int(h)
    if h in HOUR_TRANSITION:
        return "transition"
    if h in HOUR_FLASH:
        return "flash"
    return "day"


def light_bucket(brightness: float | None, hour: int | None = None) -> str:
    """Light bucket from measured frame brightness; falls back to the hour rule
    when brightness is unavailable (NaN/None) and an hour is given."""
    if brightness is None or brightness != brightness:  # None or NaN
        if hour is None:
            raise ValueError("light_bucket: no brightness and no hour fallback")
        return bucket_for_hour(hour)
    if brightness <= LIGHT_FLASH_MAX:
        return "flash"
    if brightness >= LIGHT_DAY_MIN:
        return "day"
    return "transition"


def read_exclude(path: str | None) -> set:
    """Read a text file of basenames to skip (blank lines and '#' ignored)."""
    if not path:
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(os.path.basename(line))
    return out


# --------------------------------------------------------------------------
# dataset loading
# --------------------------------------------------------------------------


def list_files(directory: str) -> list:
    files = sorted(glob(os.path.join(directory, "*.jpg")))
    files += sorted(glob(os.path.join(directory, "*.jpeg")))
    files += sorted(glob(os.path.join(directory, "*.png")))
    return sorted(set(files))


def load_dirs(dirs, weights=None, exclude=None, resize: str = "nearest",
              frame_labels: dict | None = None):
    """Load one or more directories of digit crops.

    Returns ``(x, y, meta)``:
      x    : uint8 [N, 32, 20, 3]
      y    : int64 [N]
      meta : list of dicts with keys path, label, position, frame, screen,
             source_dir, contrast

    ``weights`` is an optional per-directory integer replication factor
    (default 1 for every directory). ``exclude`` is an optional path to a text
    file listing basenames to skip.

    ``frame_labels`` is an optional ``{frame_stamp: {pos: label}}`` map giving each
    frame's FULL label set; when a frame is present in it, its screen type comes
    from that set (dig2-primary) instead of from whichever of its crops happen to
    survive in the directory.
    """
    if isinstance(dirs, str):
        dirs = [dirs]
    dirs = list(dirs)
    if weights is None:
        weights = [1] * len(dirs)
    if len(weights) != len(dirs):
        raise ValueError("weights must have one entry per directory")

    skip = read_exclude(exclude) if isinstance(exclude, (str, type(None))) else set(exclude)

    x_list, y_list, meta = [], [], []

    for directory, weight in zip(dirs, weights):
        files = [f for f in list_files(directory) if os.path.basename(f) not in skip]
        if not files:
            continue

        # screen type is a per-frame property -> group this dir's files by frame
        by_frame: dict = {}
        for f in files:
            frame = parse_frame(f)
            pos = parse_position(f)
            if frame is None or pos is None:
                continue
            by_frame.setdefault(frame, {})[pos] = parse_label(f)
        frame_screen = {
            fr: screen_type(d, (frame_labels or {}).get(fr)) for fr, d in by_frame.items()
        }

        for f in files:
            label = parse_label(f)
            arr = load_image(f, resize=resize)
            frame = parse_frame(f)
            rec = {
                "path": os.path.abspath(f),
                "label": label,
                "position": parse_position(f),
                "frame": frame,
                "screen": frame_screen.get(frame),
                "source_dir": directory,
                "contrast": contrast(arr),
            }
            for _ in range(int(weight)):
                x_list.append(arr)
                y_list.append(label)
                meta.append(dict(rec))

    if not x_list:
        return (
            np.zeros((0, TARGET_H, TARGET_W, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            [],
        )

    x = np.stack(x_list).astype(np.uint8)
    y = np.asarray(y_list, dtype=np.int64)
    return x, y, meta
