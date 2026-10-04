"""Detect ROI drift in the raw ESP32-CAM digit crops (pure numpy FFT; no new deps).

Raw tree: <root>/<YYYYmmdd>/<HH>/<label>_main_dig<pos>_<YYYYmmdd-HHMMSS>.jpg
Crops are 94 x 202 (w x h) RGB. <label> is the deployed model's prediction (`10_` == `N_`).

The firmware ROIs are frozen. If the camera/meter creeps, glyphs get clipped or neighbours
intrude. This tool measures, for every crop, the (dx, dy) translation of the glyph versus a
reference template built from the first days of data, and flags suspicious crops for a human.
Flags are ADVISORY -- nothing here excludes anything.

Method
  1. band-pass each crop (difference of Gaussians, done in the Fourier domain on a
     reflect-padded luma image) so slow lighting / reflections are removed and every crop is
     contrast-normalised.
  2. Templates (median of per-crop-normalised, band-passed crops from daytime hours of the
     first --ref-days days):
       t8   : per position, from frames whose dig2/dig3/dig4 labels are all `8` (the 88888
              test screen -- identical glyph every time, so a content-independent alignment
              reference).
       cond : per (position, model label), same selection by label. dig2 is `5`, dig3 `7` on
              every reading screen, so these give a drift signal on ~40% of all frames, not
              just the 88888 screen. Only kept if the template is self-consistent.
  3. Normalised cross-correlation of the template's central patch (10 px margin) against the
     crop over a +-SEARCH px window (FFT correlation + integral-image denominators), with a
     separable parabolic sub-pixel refinement. Score = NCC peak (0..1).
  4. Frame joint estimate: the five crops of one frame move together, so the score-weighted
     median of the reliable per-crop shifts is the frame shift.

Sign convention: (dx, dy) = how far the glyph content has moved RIGHT / DOWN inside the
crop relative to the template (so the ROI has moved the opposite way relative to the glyph).

Usage:
    python tools/roi_drift.py                      # full batch, uses/refreshes the cache
    python tools/roi_drift.py --limit 2000         # quick test on a deterministic subsample
    python tools/roi_drift.py --days 20260820 20260905
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dig_data import LABELS, parse_frame, parse_label, parse_position  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_ROOT = Path(r"C:\Users\josep\source\repos\AIOTED-digital-rawdigits")
POSITIONS = (2, 3, 4, 5, 6)
CROP_H, CROP_W = 202, 94
SEARCH = 16  # +- px search window (spec asked +-10; the real drift saturated it)
MARGIN = 10  # template patch margin inside the 94x202 crop
SIGMA_LO, SIGMA_HI = 1.2, 8.0  # difference-of-Gaussians band-pass (px)
PAD = 24  # reflect padding for the Fourier-domain filter
MIN_COND = 8  # min reference crops for a (pos, label) template
MIN_SELF = 0.35  # min median self-score for a cond template to be trusted
PARAMS = f"v2|s{SEARCH}|m{MARGIN}|g{SIGMA_LO},{SIGMA_HI}|p{PAD}"

NAME_RE = re.compile(r"^(?P<label>10|[0-9]|N)_main_dig(?P<pos>\d)_(?P<stamp>\d{8}-\d{6})\.jpg$", re.I)

# result columns produced per crop by the worker
COLS = ("dx8", "dy8", "s8", "e8", "dxc", "dyc", "sc", "ec", "mean", "contrast")
NCOL = len(COLS)


# --------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------


def index_crops(root: Path, days=None):
    """Return a list of dicts (path, rel, stamp, pos, label) sorted by stamp, pos."""
    out = []
    for day_dir in sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)):
        if days and not (days[0] <= day_dir.name <= days[-1]):
            continue
        for hour_dir in sorted(day_dir.iterdir()):
            if not hour_dir.is_dir():
                continue
            for f in hour_dir.iterdir():
                m = NAME_RE.match(f.name)
                if not m:
                    continue
                tok = "N" if m.group("label") == "10" else m.group("label").upper()
                # dig_data.parse_label rejects a leading '10'; normalise to N first.
                label = parse_label(f"{tok}_{f.name.split('_', 1)[1]}")
                assert parse_position(f.name) == int(m.group("pos")) and parse_frame(f.name) == m.group("stamp")
                out.append(
                    dict(
                        path=str(f),
                        rel=f"{day_dir.name}/{hour_dir.name}/{f.name}",
                        stamp=m.group("stamp"),
                        pos=int(m.group("pos")),
                        label=label,
                    )
                )
    out.sort(key=lambda r: (r["stamp"], r["pos"]))
    return out


def screen_guess(labels: dict) -> str:
    """Guess the display screen of one frame from the model labels {pos: class}."""
    l2, l3, l4 = labels.get(2), labels.get(3), labels.get(4)
    if l2 == l3 == l4 == 8:
        return "88888"
    if all(labels.get(p) == 0 for p in POSITIONS):
        return "00000"
    if l2 == l3 == l4 == LABELS["N"]:
        return "dash"
    if l2 == 5:
        return "reading"
    return "other"


# --------------------------------------------------------------------------
# preprocessing + matching (numpy FFT only)
# --------------------------------------------------------------------------

_PH, _PW = CROP_H + 2 * PAD, CROP_W + 2 * PAD
_fy = np.fft.fftfreq(_PH)[:, None]
_fx = np.fft.rfftfreq(_PW)[None, :]
_f2 = _fx**2 + _fy**2
_BANDPASS = (np.exp(-2 * np.pi**2 * SIGMA_LO**2 * _f2) - np.exp(-2 * np.pi**2 * SIGMA_HI**2 * _f2)).astype(np.float32)
_PATCH = (slice(MARGIN, CROP_H - MARGIN), slice(MARGIN, CROP_W - MARGIN))
_PH_, _PW_ = CROP_H - 2 * MARGIN, CROP_W - 2 * MARGIN
_NWIN = 2 * SEARCH + 1
_EXTRA = SEARCH - MARGIN  # zero padding so the patch may slide beyond the crop edge
# canvas = crop + zero border, rounded up to FFT-friendly (7-smooth) sizes
CAN_H, CAN_W = 216, 108
assert CAN_H >= CROP_H + 2 * _EXTRA and CAN_W >= CROP_W + 2 * _EXTRA
assert CAN_H >= 2 * SEARCH + _PH_ and CAN_W >= 2 * SEARCH + _PW_  # no circular wrap


def load_luma(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (CROP_W, CROP_H):
        raise ValueError(f"{path}: unexpected size {img.size}")
    a = np.asarray(img, dtype=np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def bandpass(luma: np.ndarray) -> np.ndarray:
    """Band-passed, unit-std version of a luma crop (lighting-invariant)."""
    p = np.pad(luma, PAD, mode="reflect")
    f = np.fft.irfft2(np.fft.rfft2(p) * _BANDPASS, s=p.shape)[PAD:-PAD, PAD:-PAD]
    return (f / (f.std() + 1e-6)).astype(np.float32)


def to_canvas(f: np.ndarray) -> np.ndarray:
    c = np.zeros((CAN_H, CAN_W), np.float32)
    c[_EXTRA : _EXTRA + CROP_H, _EXTRA : _EXTRA + CROP_W] = f
    return c


def fourier_shift(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Move the content of f right by dx and down by dy (sub-pixel, circular)."""
    ky = np.fft.fftfreq(f.shape[0])[:, None]
    kx = np.fft.rfftfreq(f.shape[1])[None, :]
    return np.fft.irfft2(np.fft.rfft2(f) * np.exp(-2j * np.pi * (kx * dx + ky * dy)), s=f.shape).astype(np.float32)


def prep_template(med: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm central patch, zero-padded to canvas size, as conj rfft2."""
    patch = med[_PATCH].astype(np.float64)
    patch = patch - patch.mean()
    patch /= np.linalg.norm(patch) + 1e-9
    full = np.zeros((CAN_H, CAN_W), np.float64)
    full[:_PH_, :_PW_] = patch
    return np.conj(np.fft.rfft2(full)).astype(np.complex64)


def _window_stats(c: np.ndarray):
    """Sum and sum-of-squares of the canvas over every candidate patch window."""
    out = []
    for a in (c.astype(np.float64), c.astype(np.float64) ** 2):
        ii = np.zeros((CAN_H + 1, CAN_W + 1))
        ii[1:, 1:] = a.cumsum(0).cumsum(1)
        s = (ii[_PH_ : _PH_ + _NWIN, _PW_ : _PW_ + _NWIN] - ii[:_NWIN, _PW_ : _PW_ + _NWIN]
             - ii[_PH_ : _PH_ + _NWIN, :_NWIN] + ii[:_NWIN, :_NWIN])
        out.append(s)
    return out


def _parabola(a, b, c):
    den = a - 2 * b + c
    if den >= -1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def prepare(f: np.ndarray):
    """Per-crop quantities shared by every template match: canvas FFT + window sums."""
    c = to_canvas(f)
    return np.fft.rfft2(c), _window_stats(c)


def match(prep, tmpl_conj):
    """NCC of a template patch against a prepared crop over +-SEARCH px.

    Returns (dx, dy, score, edge); edge bit0 = y peak on the window boundary, bit1 = x.
    """
    F, (wsum, wsq) = prep
    corr = np.fft.irfft2(F * tmpl_conj, s=(CAN_H, CAN_W))[:_NWIN, :_NWIN]
    var = np.maximum(wsq - wsum**2 / (_PH_ * _PW_), 1e-6)
    ncc = corr / np.sqrt(var)
    iy, ix = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    edge = int(iy in (0, _NWIN - 1)) | (2 * int(ix in (0, _NWIN - 1)))
    sy = sx = 0.0
    if 0 < iy < _NWIN - 1:
        sy = _parabola(ncc[iy - 1, ix], ncc[iy, ix], ncc[iy + 1, ix])
    if 0 < ix < _NWIN - 1:
        sx = _parabola(ncc[iy, ix - 1], ncc[iy, ix], ncc[iy, ix + 1])
    return (ix + sx - SEARCH, iy + sy - SEARCH, float(ncc[iy, ix]), edge)


# worker state (set per process by the pool initialiser)
_T8: dict = {}
_TC: dict = {}


def _init_worker(t8, tc):
    global _T8, _TC
    _T8, _TC = t8, tc


def _process(rec):
    """Worker: one crop -> float vector in COLS order."""
    pos, label = rec["pos"], rec["label"]
    out = np.full(NCOL, np.nan, np.float32)
    try:
        luma = load_luma(rec["path"])
    except Exception:  # unreadable/truncated jpeg -> leave NaN, flagged later
        return out
    prep = prepare(bandpass(luma))
    dx, dy, s, e = match(prep, _T8[pos])
    out[0:4] = (dx, dy, s, e)
    key = (pos, label)
    if key in _TC:
        dx, dy, s, e = match(prep, _TC[key])
        out[4:8] = (dx, dy, s, e)
    out[8] = luma.mean()
    out[9] = np.percentile(luma, 95) - np.percentile(luma, 5)
    return out


# --------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------


def _load_norm(rec):
    return bandpass(load_luma(rec["path"]))


def build_templates(recs, ref_days: int, cond_days: int, cache_dir: Path, workers: int):
    """Return (t8 {pos: conj-FT}, cond {(pos,label): conj-FT}, raw display templates, info)."""
    days = sorted({r["stamp"][:8] for r in recs})
    first = days[0] if days else ""
    # first --ref-days *calendar days with data* ending at the earliest full capture day;
    # the 2 stray frames on 08-07 are ignored because the batch proper starts 08-15.
    main_days = [d for d in days if d >= "20260815"] or days
    ref_set = set(main_days[:ref_days])  # t8 reference days
    cond_set = set(main_days[:max(cond_days, ref_days)])  # per-label templates: more days, more coverage
    by_frame = defaultdict(dict)
    for r in recs:
        by_frame[r["stamp"]][r["pos"]] = r
    sel8 = defaultdict(list)  # pos -> recs
    selc = defaultdict(list)  # (pos, label) -> recs
    for stamp, d in by_frame.items():
        if stamp[:8] not in cond_set or not (10 <= int(stamp[9:11]) <= 14):
            continue
        is88 = all(p in d and d[p]["label"] == 8 for p in (2, 3, 4))
        for p, r in d.items():
            if is88 and stamp[:8] in ref_set:
                sel8[p].append(r)
            selc[(p, r["label"])].append(r)
    sig = hashlib.md5((PARAMS + str(sorted(ref_set)) + str(sorted(cond_set)) + str(sorted(r["rel"] for v in sel8.values() for r in v))
                       + str(sorted(r["rel"] for v in selc.values() for r in v))).encode()).hexdigest()[:12]
    cache = cache_dir / f"templates_{sig}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        meds = z["meds"].item()
        info = z["info"].item()
    else:
        print(f"building templates from {sum(len(v) for v in sel8.values())} t8 crops "
              f"(t8 ref days {sorted(ref_set)[0]}..{sorted(ref_set)[-1]}, cond ref days ..{sorted(cond_set)[-1]}) ...")
        meds, info = {}, {}
        jobs = [(("t8", p), rs) for p, rs in sel8.items()] + [(("c", k), rs) for k, rs in selc.items()
                                                              if len(rs) >= MIN_COND and k[1] != LABELS["N"]]
        with Pool(workers) as pool:
            for (kind, key), rs in jobs:
                stack = np.stack(pool.map(_load_norm, rs, chunksize=16))
                raw = np.median(np.stack(pool.map(load_luma_rec, rs, chunksize=16)), axis=0).astype(np.float32)
                med, scores, shifts = align_median(stack)
                meds[(kind, key)] = (med, raw)
                info[(kind, key)] = dict(n=len(rs), self_score=float(np.median(scores)),
                                         shift_sd=tuple(np.std(shifts, axis=0)),
                                         shift_med=tuple(np.median(shifts, axis=0)))
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, meds=np.array(meds, dtype=object), info=np.array(info, dtype=object))
    t8, tc, raws = {}, {}, {}
    for (kind, key), (med, raw) in meds.items():
        if kind == "t8":
            t8[key] = prep_template(med)
            raws[("t8", key)] = raw
        elif info[(kind, key)]["self_score"] >= MIN_SELF:
            tc[key] = prep_template(med)
            raws[("c", key)] = raw
    missing = [p for p in POSITIONS if p not in t8]
    if missing:
        raise SystemExit(f"no 88888 reference frames found for positions {missing}")
    return t8, tc, raws, info, sig


def align_median(stack: np.ndarray, iters: int = 2):
    """Median template of band-passed crops, with iterative sub-pixel alignment.

    The camera wobbles by a px or two even in the reference days, so a plain median would be
    blurred. Each pass registers every source crop to the current template, shifts it onto it
    and re-medians. The mean applied shift is removed, so the template stays anchored to the
    median raw crop position. Returns (template, source self-scores, residual raw shifts).
    """
    med = np.median(stack, axis=0).astype(np.float32)
    for _ in range(iters):
        tc = prep_template(med)
        sh = np.array([match(prepare(f), tc)[:2] for f in stack])
        sh -= np.median(sh, axis=0)
        med = np.median(np.stack([fourier_shift(f, -dx, -dy) for f, (dx, dy) in zip(stack, sh)]), axis=0).astype(np.float32)
    tc = prep_template(med)
    res = [match(prepare(f), tc) for f in stack]
    return med, [r[2] for r in res], [(r[0], r[1]) for r in res]


def load_luma_rec(rec):
    return load_luma(rec["path"])


# --------------------------------------------------------------------------
# batch run with cache
# --------------------------------------------------------------------------


def run_batch(recs, t8, tc, sig, cache_dir: Path, workers: int):
    cache = cache_dir / f"results_{sig}.npz"
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = dict(zip(z["rel"].tolist(), z["res"]))
    todo = [r for r in recs if r["rel"] not in have]
    print(f"{len(recs)} crops, {len(recs) - len(todo)} cached, {len(todo)} to process "
          f"on {workers} workers")
    if todo:
        t0 = time.time()
        res = []
        with Pool(workers, initializer=_init_worker, initargs=(t8, tc)) as pool:
            for i, v in enumerate(pool.imap(_process, todo, chunksize=64), 1):
                res.append(v)
                if i % 10000 == 0:
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(todo)}  {rate:.0f} crops/s  eta {(len(todo) - i) / rate / 60:.1f} min",
                          flush=True)
        for r, v in zip(todo, res):
            have[r["rel"]] = v
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, rel=np.array(list(have.keys())), res=np.array(list(have.values())))
        print(f"processed in {time.time() - t0:.0f}s")
    return np.stack([have[r["rel"]] for r in recs])


# --------------------------------------------------------------------------
# per-crop / per-frame analysis + flags
# --------------------------------------------------------------------------

REASONS = ("shift", "frame_shift", "disagree", "low_score", "edge", "unreadable")
EPOCH = datetime(2026, 8, 15)


def wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def _ddays(a: str, b: str) -> int:
    return (datetime.strptime(a, "%Y%m%d") - datetime.strptime(b, "%Y%m%d")).days


def analyse(recs, res, baseline_days: int):
    """Turn raw match results into per-crop estimates, frame joint estimates and flags."""
    n = len(recs)
    stamp = np.array([r["stamp"] for r in recs])
    day = np.array([s[:8] for s in stamp])
    hour = np.array([int(s[9:11]) for s in stamp])
    pos = np.array([r["pos"] for r in recs])
    label = np.array([r["label"] for r in recs])

    frames = defaultdict(dict)
    for r in recs:
        frames[r["stamp"]][r["pos"]] = r["label"]
    screen = np.array([screen_guess(frames[s]) for s in stamp])

    # own estimate: the 88888 screen is known, so always use the content-matched t8 template
    # there (the model's dig5/dig6 label can be wrong); elsewhere the label-conditioned one.
    has_c = ~np.isnan(res[:, 6])
    use_c = has_c & (screen != "88888")
    sel = np.where(use_c[:, None], res[:, 4:8], res[:, 0:4])
    dx, dy, score, edge = sel.T.copy()
    method = np.where(use_c, "cond", "t8")
    unread = np.isnan(res[:, 0])
    # a t8 match against some other glyph (no label template; dash screen) is not meaningful
    is88 = screen == "88888"
    meaningful = (is88 | use_c) & ~unread

    day_list = sorted(set(day))
    base_days = [d for d in day_list if d >= "20260815"][:baseline_days]

    # ---- thresholds chosen from data -------------------------------------
    s88 = score[is88 & meaningful]
    s_low = float(np.floor(np.percentile(s88, 1) * 20) / 20) if len(s88) else 0.5
    base = is88 & meaningful & np.isin(day, base_days)
    env = {}
    for name, v in (("dx", dx), ("dy", dy)):
        vals = [np.percentile(np.abs(v[base & (pos == p)]), 99.5) for p in POSITIONS if (base & (pos == p)).any()]
        env[name] = float(np.ceil(2 * max(vals + [1.5])) / 2) if vals else 5.0
    rel = meaningful & (score >= s_low)

    # ---- per-position offsets relative to the frame (rotation/scale) ------
    f88 = sorted(set(stamp[is88]))
    fidx = {s: i for i, s in enumerate(f88)}
    DX = np.full((len(f88), 5), np.nan)
    DY = DX.copy()
    for i in np.where(is88 & rel)[0]:
        DX[fidx[stamp[i]], pos[i] - 2] = dx[i]
        DY[fidx[stamp[i]], pos[i] - 2] = dy[i]
    f88day = np.array([s[:8] for s in f88])
    off = {d: np.zeros((5, 2)) for d in day_list}
    if len(f88):
        with np.errstate(all="ignore"):
            relx = DX - np.nanmedian(DX, axis=1, keepdims=True)
            rely = DY - np.nanmedian(DY, axis=1, keepdims=True)
            perday = {}
            for d in sorted(set(f88day)):
                k = f88day == d
                perday[d] = np.stack([np.nanmedian(relx[k], 0), np.nanmedian(rely[k], 0)], axis=1)
            pd_days = sorted(perday)
            for d in day_list:
                near = sorted(pd_days, key=lambda q: abs(_ddays(q, d)))[:3]
                off[d] = np.nan_to_num(np.nanmedian(np.stack([perday[q] for q in near]), axis=0))

    # ---- frame joint estimate -------------------------------------------
    by_frame = defaultdict(list)
    for i in range(n):
        by_frame[stamp[i]].append(i)
    fdx = np.full(n, np.nan)
    fdy = np.full(n, np.nan)
    fn = np.zeros(n, int)
    for s, idxs in by_frame.items():
        ok = [i for i in idxs if rel[i]]
        if len(ok) < 3:
            continue
        w = np.array([score[i] ** 2 for i in ok])
        ex = np.array([dx[i] - off[s[:8]][pos[i] - 2, 0] for i in ok])
        ey = np.array([dy[i] - off[s[:8]][pos[i] - 2, 1] for i in ok])
        mx, my = wmedian(ex, w), wmedian(ey, w)
        for i in idxs:
            fdx[i], fdy[i], fn[i] = mx, my, len(ok)
    exp_x = np.array([fdx[i] + off[day[i]][pos[i] - 2, 0] for i in range(n)])
    exp_y = np.array([fdy[i] + off[day[i]][pos[i] - 2, 1] for i in range(n)])
    dev = np.hypot(dx - exp_x, dy - exp_y)
    dv = dev[rel & ~np.isnan(dev)]
    sig_dev = 1.4826 * np.median(np.abs(dv - np.median(dv))) if len(dv) else 0.5
    t_dev = float(max(2.0, np.ceil(2 * (np.median(dv) + 6 * sig_dev)) / 2)) if len(dv) else 2.0

    # ---- flags -----------------------------------------------------------
    tx, ty = env["dx"], env["dy"]
    ef = ~np.isnan(exp_x)
    r_shift = rel & ((np.abs(dx) > tx) | (np.abs(dy) > ty))
    # crops without a usable own estimate inherit the frame's shift (never double-count)
    r_fshift = ~rel & ef & ((np.abs(exp_x) > tx) | (np.abs(exp_y) > ty))
    r_dis = rel & ef & (dev > t_dev)
    r_low = meaningful & (score < s_low)
    r_edge = meaningful & (edge > 0)
    reasons = [r_shift, r_fshift, r_dis, r_low, r_edge, unread]
    flag_reason = np.array(["|".join(nm for nm, m in zip(REASONS, (m[i] for m in reasons)) if m) for i in range(n)])
    flag = flag_reason != ""
    thr = dict(s_low=s_low, tx=tx, ty=ty, t_dev=t_dev, sig_dev=float(sig_dev), base_days=base_days,
               reliable=int(rel.sum()), meaningful=int(meaningful.sum()))
    return dict(stamp=stamp, day=day, hour=hour, pos=pos, label=label, screen=screen, dx=dx, dy=dy, score=score,
                edge=edge, method=method, rel=rel, meaningful=meaningful, fdx=fdx, fdy=fdy, fn=fn, exp_x=exp_x,
                exp_y=exp_y, dev=dev, flag=flag, reason=flag_reason, mean=res[:, 8], contrast=res[:, 9],
                thr=thr, off=off, is88=is88)


def write_csv(path: Path, A):
    path.parent.mkdir(parents=True, exist_ok=True)

    def f(x, d=2):
        return "" if np.isnan(x) else f"{x:.{d}f}"

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stamp", "pos", "screen_guess", "dx", "dy", "score", "flag", "flag_reason",
                    "label", "method", "frame_dx", "frame_dy", "frame_n", "mean_luma", "contrast"])
        for i in range(len(A["stamp"])):
            lab = "N" if A["label"][i] == LABELS["N"] else int(A["label"][i])
            w.writerow([A["stamp"][i], f"dig{A['pos'][i]}", A["screen"][i], f(A["dx"][i]), f(A["dy"][i]),
                        f(A["score"][i], 3), int(A["flag"][i]), A["reason"][i], lab, A["method"][i],
                        f(A["fdx"][i]), f(A["fdy"][i]), A["fn"][i], f(A["mean"][i], 1), f(A["contrast"][i], 1)])
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def _med(v):
    return np.nan if len(v) == 0 else float(np.median(v))


def _week(d: str) -> int:
    return _ddays(d, "20260815") // 7


def summarise(A):
    thr = A["thr"]
    day, pos, hour, is88, rel = A["day"], A["pos"], A["hour"], A["is88"], A["rel"]
    print("\n=== thresholds (chosen from the data) ===")
    print(f"baseline period for the shift envelope: {thr['base_days'][0]}..{thr['base_days'][-1]} (88888 crops)")
    print(f"low-score threshold : score < {thr['s_low']:.2f}  (1st percentile of all 88888 crop scores, rounded down)")
    print(f"shift threshold     : |dx| > {thr['tx']} px or |dy| > {thr['ty']} px  (99.5th percentile of |shift| of the "
          f"88888 crops in the baseline period, worst position, rounded up to 0.5 px)")
    print(f"disagree threshold  : crop vs frame-expected shift > {thr['t_dev']} px  (median + 6 robust sigma "
          f"[sigma={thr['sig_dev']:.2f}] of the deviation over reliable crops, floor 2 px)")
    print(f"reliable crops (own estimate usable): {thr['reliable']}/{len(day)}")
    print("note: a MAD z-score on the 88888 series was NOT used for the shift flag -- the baseline wobble is bimodal "
          "(day vs flash-lit night, a few px apart), so a MAD sigma of ~1 px would flag normal daytime wobble.")

    wkarr = np.array([_week(d) if d >= "20260815" else -1 for d in day])
    weeks = sorted(set(wkarr[wkarr >= 0]))
    for title, hmask in (("all hours", np.ones_like(hour, bool)), ("night, hours 0-5 (stable flash lighting)", hour <= 5),
                         ("day, hours 9-16", (hour >= 9) & (hour <= 16))):
        print(f"\n=== 88888 frames: median dx/dy (px) per week -- {title} ===")
        print("week from  " + "".join(f"   dig{p} dx/dy " for p in POSITIONS) + "   n(frames)")
        for w in weeks:
            d0 = (EPOCH + timedelta(days=7 * int(w))).strftime("%m-%d")
            cells = []
            for p in POSITIONS:
                k = is88 & rel & (pos == p) & (wkarr == w) & hmask
                cells.append(f"{_med(A['dx'][k]):6.1f}/{_med(A['dy'][k]):5.1f}" if k.any() else "      -/-    ")
            nn = int((is88 & (pos == 2) & (wkarr == w) & hmask).sum())
            print(f"  {d0}    " + " ".join(cells) + f"   {nn:4d}")

    print("\n=== step changes / trends in the nightly (hours 0-5) 88888 series (pooled median of the 5 positions) ===")
    nights = sorted({d for d in day if d >= "20260815"})
    ser = {}
    for d in nights:
        k = is88 & rel & (day == d) & (hour <= 5)
        if k.sum() >= 5:
            ser[d] = (np.array([_med(A["dx"][k & (pos == p)]) for p in POSITIONS]),
                      np.array([_med(A["dy"][k & (pos == p)]) for p in POSITIONS]))
    nd = sorted(ser)
    if len(nd) < 3:
        print("not enough nights with 88888 frames for step detection")
    else:
        with np.errstate(all="ignore"):
            mx = np.array([np.nanmedian(ser[d][0]) for d in nd])
            my = np.array([np.nanmedian(ser[d][1]) for d in nd])
        print(" night       dx     dy    d(dx)  d(dy)   [consecutive-night changes > 1 px]")
        for i in range(1, len(nd)):
            ddx, ddy = mx[i] - mx[i - 1], my[i] - my[i - 1]
            if abs(ddx) > 1 or abs(ddy) > 1:
                print(f" {nd[i]}  {mx[i]:6.1f} {my[i]:6.1f}   {ddx:+5.1f}  {ddy:+5.1f}")
        b = min(7, len(nd))
        print(f" first-{b}-night level dx {np.median(mx[:b]):.1f} dy {np.median(my[:b]):.1f};  "
              f"last-{b}-night level dx {np.median(mx[-b:]):.1f} dy {np.median(my[-b:]):.1f}")
        for j, p in enumerate(POSITIONS):
            px = np.array([ser[d][0][j] for d in nd])
            py = np.array([ser[d][1][j] for d in nd])
            print(f"  dig{p}: night dx/dy first-{b} {np.nanmedian(px[:b]):6.1f}/{np.nanmedian(py[:b]):5.1f} -> last-{b} "
                  f"{np.nanmedian(px[-b:]):6.1f}/{np.nanmedian(py[-b:]):5.1f}   net "
                  f"{np.nanmedian(px[-b:]) - np.nanmedian(px[:b]):+5.1f}/{np.nanmedian(py[-b:]) - np.nanmedian(py[:b]):+5.1f} px")
        for v, nm in ((mx, "dx"), (my, "dy")):
            lvl = np.median(v[:b])
            out = np.abs(v - lvl) > 1
            first = next((nd[i] for i in range(len(nd) - 2) if out[i : i + 3].all()), None)
            big = np.abs(v - lvl) > max(thr["tx"], thr["ty"])
            fbig = next((nd[i] for i in range(len(nd) - 2) if big[i : i + 3].all()), None)
            print(f"  {nm}: first night with a >1 px departure from the first-week level (3 nights sustained): {first};"
                  f" beyond the flag envelope: {fbig}")
        tail = [i for i, d in enumerate(nd) if d >= "20260906"]
        if len(tail) >= 5:
            t = np.arange(len(tail))
            sx = np.polyfit(t, mx[tail], 1)[0] * 7
            sy = np.polyfit(t, my[tail], 1)[0] * 7
            print(f"  linear trend of nightly medians from 09-06 on: dx {sx:+.2f} px/week, dy {sy:+.2f} px/week")

    print("\n=== flagged crops by day x position (all reasons; advisory) ===")
    print("day        " + "".join(f"  dig{p}" for p in POSITIONS) + "   total  /crops")
    for d in sorted(set(day)):
        row = [int((A["flag"] & (day == d) & (pos == p)).sum()) for p in POSITIONS]
        print(f"{d}  " + "".join(f"{c:6d}" for c in row) + f"  {sum(row):6d}  /{int((day == d).sum())}")
    tot = int(A["flag"].sum())
    print(f"total flagged: {tot} / {len(day)} crops ({100 * tot / max(1, len(day)):.1f}%)")
    print("by reason (a crop can carry several): " + ", ".join(
        f"{r}={int(sum(r in x.split('|') for x in A['reason']))}" for r in REASONS))


# --------------------------------------------------------------------------
# plots + contact sheet
# --------------------------------------------------------------------------

POS_COLORS = {2: "#0072B2", 3: "#E69F00", 4: "#009E73", 5: "#CC79A7", 6: "#D55E00"}  # Okabe-Ito


def _times(stamps):
    return np.array([datetime.strptime(s, "%Y%m%d-%H%M%S") for s in stamps])


def _rolling_median(v, w):
    out = np.empty(len(v))
    h = w // 2
    for i in range(len(v)):
        out[i] = np.median(v[max(0, i - h) : i + h + 1])
    return out


def plot_88888(A, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr = A["thr"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, key, T, ylab in ((axes[0], "dx", thr["tx"], "dx (px, +right)"), (axes[1], "dy", thr["ty"], "dy (px, +down)")):
        for p in POSITIONS:
            k = np.where(A["is88"] & A["rel"] & (A["pos"] == p))[0]
            if not len(k):
                continue
            t = _times(A["stamp"][k])
            v = A[key][k]
            ax.scatter(t, v, s=3, alpha=0.25, color=POS_COLORS[p], linewidths=0)
            ax.plot(t, _rolling_median(v, 25), color=POS_COLORS[p], lw=1.6, label=f"dig{p}")
        ax.axhline(T, color="0.4", ls="--", lw=1)
        ax.axhline(-T, color="0.4", ls="--", lw=1)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=5, loc="upper left")
    axes[0].set_title(f"ROI drift from 88888 test-screen frames (dots = frames, lines = rolling median of 25; dashed = "
                      f"flag envelope +-{thr['tx']}/{thr['ty']} px)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def plot_all(A, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = sorted(set(A["day"]))
    dd = [datetime.strptime(d, "%Y%m%d") for d in days]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, key, T, ylab in ((axes[0], "dx", A["thr"]["tx"], "dx (px)"), (axes[1], "dy", A["thr"]["ty"], "dy (px)")):
        for p in POSITIONS:
            med = []
            for d in days:
                k = A["rel"] & (A["pos"] == p) & (A["day"] == d)
                med.append(np.median(A[key][k]) if k.sum() >= 5 else np.nan)
            ax.plot(dd, med, color=POS_COLORS[p], lw=1.6, marker="o", ms=3, label=f"dig{p}")
        ax.axhline(T, color="0.4", ls="--", lw=1)
        ax.axhline(-T, color="0.4", ls="--", lw=1)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_ylabel(f"daily median {ylab}")
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=5, loc="upper left")
    axes[0].set_title("Per-crop shift, all screens (daily median of reliable crops) and flagged crops per day")
    bottom = np.zeros(len(days))
    groups = (("shift / frame_shift", ("shift", "frame_shift"), "#D55E00"), ("disagree", ("disagree",), "#0072B2"),
              ("low_score / edge / unreadable", ("low_score", "edge", "unreadable"), "#999999"))
    seen = np.zeros(len(A["stamp"]), bool)
    for name, rs, col in groups:
        m = np.array([any(r in x.split("|") for r in rs) for x in A["reason"]]) & ~seen
        seen |= m
        cnt = np.array([int((m & (A["day"] == d)).sum()) for d in days])
        axes[2].bar(dd, cnt, bottom=bottom, color=col, label=name, width=0.8)
        bottom += cnt
    axes[2].set_ylabel("flagged crops / day")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def _outline(raw: np.ndarray) -> np.ndarray:
    """Boundary of the template's dark glyph segments (bool mask, crop sized)."""
    f = bandpass(raw)
    m = f < -0.6
    er = m.copy()
    er[1:] &= m[:-1]
    er[:-1] &= m[1:]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    return m & ~er


def _shift_mask(m: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(m)
    ys, xs = slice(max(dy, 0), CROP_H + min(dy, 0)), slice(max(dx, 0), CROP_W + min(dx, 0))
    yd, xd = slice(max(-dy, 0), CROP_H + min(-dy, 0)), slice(max(-dx, 0), CROP_W + min(-dx, 0))
    out[ys, xs] = m[yd, xd]
    return out


def _overlay(raw: np.ndarray, rgb: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Template glyph outline on the crop: red at zero shift (where the ROI expects the glyph),
    green moved by the estimated shift (where the glyph actually is)."""
    m = _outline(raw)
    out = rgb.copy()
    out[m] = (255, 40, 40)
    if np.isfinite(dx) and np.isfinite(dy):
        out[_shift_mask(m, int(round(dx)), int(round(dy)))] = (40, 255, 40)
    return out


def contact_sheet(A, recs, raws, path: Path, n_total: int = 20):
    from PIL import ImageDraw

    thr = A["thr"]
    sev = np.maximum(np.abs(A["dx"]) / thr["tx"], np.abs(A["dy"]) / thr["ty"])
    has = lambda r: np.array([r in x.split("|") for x in A["reason"]])  # noqa: E731

    def pick(mask, order, k, taken, per_day=1, per_pos=2):
        out, perday, perpos = [], defaultdict(int), defaultdict(int)
        for i in order:
            if not mask[i] or i in taken:
                continue
            if perday[A["day"][i]] >= per_day or perpos[A["pos"][i]] >= per_pos:
                continue
            perday[A["day"][i]] += 1
            perpos[A["pos"][i]] += 1
            out.append(i)
            if len(out) >= k:
                break
        return out

    taken: set = set()
    picks = []
    shift = has("shift") & A["rel"]
    sev_nan = np.nan_to_num(sev, nan=-1)
    for name, mask, order, k in (
        ("largest shift", shift, np.argsort(-sev_nan), 7),
        ("shift, before 09-01", shift & (A["day"] < "20260901"), np.argsort(-sev_nan), 5),
        ("disagree with frame", has("disagree"), np.argsort(-np.nan_to_num(A["dev"], nan=-1)), 4),
        ("low score", has("low_score"), np.argsort(A["score"]), 4),
    ):
        got = pick(mask, order, k, taken)
        taken.update(got)
        picks += [(name, i) for i in got]
    picks = picks[:n_total]
    if not picks:
        print("no flagged crops to show")
        return
    cw, ch, cols = 2 * CROP_W + 6, CROP_H + 44, 5
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cw + 8) + 8, rows * (ch + 8) + 8), "white")
    dr = ImageDraw.Draw(sheet)
    for j, (name, i) in enumerate(picks):
        r = recs[i]
        key = ("c", (r["pos"], r["label"])) if A["method"][i] == "cond" else ("t8", r["pos"])
        raw = raws.get(key, raws[("t8", r["pos"])])
        rgb = np.asarray(Image.open(r["path"]).convert("RGB"))
        tmpl = np.clip((raw - np.percentile(raw, 1)) / (np.percentile(raw, 99) - np.percentile(raw, 1) + 1e-6), 0, 1)
        tmpl = np.repeat((tmpl * 255).astype(np.uint8)[..., None], 3, axis=2)
        x0, y0 = 8 + (j % cols) * (cw + 8), 8 + (j // cols) * (ch + 8)
        sheet.paste(Image.fromarray(_overlay(raw, rgb, A['dx'][i], A['dy'][i])), (x0, y0 + 44))
        sheet.paste(Image.fromarray(tmpl), (x0 + CROP_W + 6, y0 + 44))
        lab = "N" if r["label"] == LABELS["N"] else r["label"]
        dr.text((x0, y0), f"{name}", fill=(0, 0, 0))
        dr.text((x0, y0 + 11), f"{r['stamp']} dig{r['pos']} lbl={lab} {A['screen'][i]}", fill=(0, 0, 0))
        dr.text((x0, y0 + 22), f"dx={A['dx'][i]:+.1f} dy={A['dy'][i]:+.1f} sc={A['score'][i]:.2f}", fill=(0, 0, 0))
        dr.text((x0, y0 + 33), f"{A['reason'][i]}", fill=(160, 0, 0))
    sheet.save(path)
    print(f"wrote {path} ({len(picks)} crops; left = crop with template glyph outline: red at zero shift, green at estimated shift; "
          f"right = template)")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(RAW_ROOT), help="raw digits root (default: %(default)s)")
    ap.add_argument("--out-csv", default=str(REPO_ROOT / "work" / "roi_drift.csv"))
    ap.add_argument("--plot-dir", default=str(REPO_ROOT / "work"))
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "work" / "roi_drift_cache"))
    ap.add_argument("--days", nargs="+", metavar="YYYYmmdd",
                    help="only process these days (one value = that day, two = inclusive range)")
    ap.add_argument("--limit", type=int, default=0, help="quick test: process ~N crops (whole frames, evenly spaced)")
    ap.add_argument("--ref-days", type=int, default=5, help="days of 88888 crops in the t8 template (default 5)")
    ap.add_argument("--cond-days", type=int, default=10, help="days of crops in the per-label templates (default 10)")
    ap.add_argument("--baseline-days", type=int, default=14, help="days defining the no-drift shift envelope (default 14)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    root = Path(args.root)
    all_recs = index_crops(root)
    if not all_recs:
        raise SystemExit(f"no crops found under {root}")
    print(f"indexed {len(all_recs)} crops, {len({r['stamp'] for r in all_recs})} frames in {time.time() - t0:.1f}s")
    cache_dir = Path(args.cache_dir)
    t8, tc, raws, info, sig = build_templates(all_recs, args.ref_days, args.cond_days, cache_dir, args.workers)
    print(f"templates: 5 x t8, {len(tc)} per-(pos,label) templates "
          f"(dropped {sum(1 for k in info if k[0] == 'c') - len(tc)} with self-score < {MIN_SELF}); "
          f"reference jitter (sd of dx/dy within the reference crops after alignment): "
          + ", ".join(f"dig{p}={info[('t8', p)]['shift_sd'][0]:.1f}/{info[('t8', p)]['shift_sd'][1]:.1f}" for p in POSITIONS))

    recs = all_recs
    if args.days:
        lo, hi = args.days[0], args.days[-1]
        recs = [r for r in recs if lo <= r["stamp"][:8] <= hi]
    if args.limit:
        stamps = sorted({r["stamp"] for r in recs})
        nfr = max(1, args.limit // 5)
        pick = set(stamps[i] for i in np.linspace(0, len(stamps) - 1, min(nfr, len(stamps))).astype(int))
        recs = [r for r in recs if r["stamp"] in pick]
    print(f"processing {len(recs)} crops ({len({r['stamp'] for r in recs})} frames)")
    res = run_batch(recs, t8, tc, sig, cache_dir, args.workers)

    A = analyse(recs, res, args.baseline_days)
    write_csv(Path(args.out_csv), A)
    summarise(A)
    if not args.no_plots:
        pdir = Path(args.plot_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        plot_88888(A, pdir / "roi_drift_88888.png")
        plot_all(A, pdir / "roi_drift_all.png")
        contact_sheet(A, recs, raws, pdir / "roi_drift_flagged_sample.png")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
