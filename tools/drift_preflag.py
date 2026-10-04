"""Can per-frame ROI drift be pre-flagged automatically? (analysis + optional preflag)

Joseph marked 17 of the 175 holdout frames `drift` in grid_review (whole frames: the
per-image alignment step failed for that capture). This tool asks whether those
frames stand out in work/roi_drift.csv:

  feature `dev`     |frame shift - local rolling median of frame shift|, the median
                    taken over the other frames within +-WINDOW hours in the same
                    light bucket (frame shift = roi_drift's frame_dx / frame_dy);
  feature `medcrop` median over the frame's crops of |crop shift - that local median|
                    (works when roi_drift had no reliable frame estimate);
  feature `dxday`   horizontal frame shift minus the median frame_dx of the same
                    capture day (all buckets); positive = glyphs sit further right,
                    i.e. the left neighbour intrudes at each crop's left edge;
  plus the NCC match scores (`medscore`, low = template matched poorly).

It prints precision / recall over the 175 holdout frames at a few thresholds, and
writes a contact sheet of the 17 drift frames next to 17 random ok frames of the
same days (work/drift_holdout_sheet.png).

With --apply-preflag FEATURE THRESH it adds the token `drift?` to the preflag column
of screen-mode queue items whose frame exceeds the threshold (column update only:
items and order unchanged, verified after writing).

Usage (from repo root):
    python tools/drift_preflag.py                       # analysis + contact sheet
    python tools/drift_preflag.py --apply-preflag dev 2.5
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import grid_review as gr  # noqa: E402

ROOT = TOOLS.parent
DRIFT_CSV = ROOT / "work" / "roi_drift.csv"
DERIVE = ROOT / "work" / "b3_derive.csv"
LEDGER = ROOT / "work" / "review_ledger.csv"
QDIR = ROOT / "work" / "queues"
SHEET = ROOT / "work" / "drift_holdout_sheet.png"
SCREEN_QUEUES = ("legacy", "day_err", "night_err", "day_fill", "night_fill")
WINDOW_H = 2.0
FLAG = "drift?"


def hours(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").timestamp() / 3600.0


def load_drift(path=DRIFT_CSV):
    """-> (frames {stamp: (fdx|None, fdy|None, n)}, crops {stamp: {pos: (dx, dy, score)}})"""
    frames, crops = {}, defaultdict(dict)
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s = r["stamp"]
            crops[s][int(r["pos"][3:])] = (float(r["dx"]), float(r["dy"]), float(r["score"]))
            if s not in frames:
                frames[s] = (float(r["frame_dx"]) if r["frame_dx"] else None,
                             float(r["frame_dy"]) if r["frame_dy"] else None,
                             int(r["frame_n"] or 0))
    return frames, crops


def load_buckets(path=DERIVE) -> dict[str, str]:
    out = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["stamp"], r["bucket"])
    return out


class Features:
    def __init__(self, frames, crops, buckets, window_h: float = WINDOW_H):
        self.frames, self.crops, self.buckets, self.win = frames, crops, buckets, window_h
        self.stamps = sorted(frames)
        self.t = [hours(s) for s in self.stamps]
        byday = defaultdict(list)
        for s_, f in frames.items():
            if f[0] is not None:
                byday[s_[:8]].append(f[0])
        self.day_dx = {d: st.median(v) for d, v in byday.items()}

    def frame_dx(self, s):
        f = self.frames.get(s)
        if f and f[0] is not None:
            return f[0]
        cs = self.crops.get(s, {})
        return st.median(v[0] for v in cs.values()) if cs else None

    def bucket(self, s):
        return self.buckets.get(s) or gr.bucket_for_hour(int(s[9:11]))

    def local(self, s):
        """Rolling median (dx, dy) of the reliable frame shifts within +-window hours,
        same bucket, excluding s. Falls back to any bucket if < 3 neighbours."""
        ts = hours(s)
        i0 = bisect.bisect_left(self.t, ts - self.win)
        i1 = bisect.bisect_right(self.t, ts + self.win)
        for same in (True, False):
            dx, dy = [], []
            for j in range(i0, i1):
                o = self.stamps[j]
                if o == s or self.frames[o][0] is None:
                    continue
                if same and self.bucket(o) != self.bucket(s):
                    continue
                dx.append(self.frames[o][0])
                dy.append(self.frames[o][1])
            if len(dx) >= 3:
                return st.median(dx), st.median(dy)
        return None

    def of(self, s) -> dict:
        f = self.frames.get(s)
        out = {"dev": None, "medcrop": None, "medscore": None, "dxday": None, "frame_n": 0}
        if f is None:
            return out
        fx = self.frame_dx(s)
        if fx is not None and s[:8] in self.day_dx:
            out["dxday"] = fx - self.day_dx[s[:8]]
        out["frame_n"] = f[2]
        loc = self.local(s)
        cs = self.crops.get(s, {})
        if cs:
            out["medscore"] = st.median(v[2] for v in cs.values())
        if loc is None:
            return out
        if f[0] is not None:
            out["dev"] = math.hypot(f[0] - loc[0], f[1] - loc[1])
        if cs:
            out["medcrop"] = st.median(math.hypot(v[0] - loc[0], v[1] - loc[1])
                                       for v in cs.values())
        return out


def holdout_frames(ledger=LEDGER):
    eff = gr.load_effective(ledger)
    frames = defaultdict(set)
    for iid, e in eff.items():
        if (e.get("queue") or "") == "holdout":
            st_, _ = gr.split_item_id(iid, e)
            frames[st_].add(e["screen_verdict"])
    drift = sorted(s for s, v in frames.items() if "drift" in v)
    ok = sorted(s for s, v in frames.items() if v and v <= set(gr.SCREEN_OK))
    return drift, ok, eff


def pr_table(feat: dict[str, dict], pos: set, neg: set, name: str, thresholds,
             higher_is_drift=True) -> list[tuple]:
    rows = []
    for th in thresholds:
        tp = fp = 0
        for s in pos | neg:
            v = feat[s][name]
            hit = v is not None and (v > th if higher_is_drift else v < th)
            if hit and s in pos:
                tp += 1
            elif hit:
                fp += 1
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rows.append((th, tp, fp, prec, tp / len(pos)))
    return rows


def contact_sheet(drift, ok_pool, src_of, path=SHEET, seed=20261003, scale=0.5):
    rng = random.Random(seed)
    by_day = defaultdict(list)
    for s in ok_pool:
        by_day[s[:8]].append(s)
    chosen_ok, used = [], set()
    for s in drift:
        cands = [x for x in by_day[s[:8]] if x not in used]
        pick = rng.choice(cands) if cands else None
        if pick:
            used.add(pick)
        chosen_ok.append(pick)
    cw, ch = round(94 * scale), round(202 * scale)
    gap, label_h, mid = 4, 14, 24
    block_w = 5 * (cw + 2)
    W = 2 * block_w + mid + 2 * gap
    H = label_h + len(drift) * (ch + label_h + gap) + gap
    sheet = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("consola.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    d.text((gap, 1), "DRIFT (Joseph)", fill=(255, 154, 31), font=font)
    d.text((gap + block_w + mid, 1), "ok, same day (random)", fill=(140, 220, 140), font=font)
    for i, (sd, so) in enumerate(zip(drift, chosen_ok)):
        y = label_h + i * (ch + label_h + gap)
        for col, s in ((0, sd), (1, so)):
            if s is None:
                continue
            x0 = gap + col * (block_w + mid)
            d.text((x0, y), f"{s[:8]} {s[9:11]}:{s[11:13]}", fill=(220, 220, 220), font=font)
            for k, p in enumerate(gr.POSITIONS):
                src = src_of(s, p)
                if not src:
                    continue
                with Image.open(src) as im:
                    tile = im.convert("RGB").resize((cw, ch), Image.Resampling.LANCZOS)
                sheet.paste(tile, (x0 + k * (cw + 2), y + label_h))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return chosen_ok


def apply_preflag(flagged_stamps: set[str], qdir=QDIR, queues=SCREEN_QUEUES) -> dict:
    out = {}
    for name in queues:
        p = Path(qdir) / f"{name}.csv"
        if not p.is_file():
            continue
        with p.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            fields, rows = list(rd.fieldnames or []), list(rd)
        order = [r["item_id"] for r in rows]
        n_items, frames = 0, set()
        for r in rows:
            toks = [t for t in (r.get("preflag") or "").split("|") if t and t != FLAG]
            if r["stamp"] in flagged_stamps:
                toks.append(FLAG)
                n_items += 1
                frames.add(r["stamp"])
            r["preflag"] = "|".join(toks)
        tmp = p.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        with tmp.open(newline="", encoding="utf-8") as fh:
            back = [r["item_id"] for r in csv.DictReader(fh)]
        if back != order:
            raise SystemExit(f"{p}: order changed; left {tmp}")
        os.replace(tmp, p)
        out[name] = (len(frames), n_items, len(rows))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", type=float, default=WINDOW_H, help="+- hours")
    ap.add_argument("--sheet", type=Path, default=SHEET)
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--apply-preflag", nargs=2, metavar=("FEATURE", "THRESH"))
    args = ap.parse_args(argv)

    frames, crops = load_drift()
    buckets = load_buckets()
    F = Features(frames, crops, buckets, args.window)
    drift, ok, eff = holdout_frames()
    pos, neg = set(drift), set(ok)
    feat = {s: F.of(s) for s in pos | neg}
    print(f"holdout frames: {len(drift)} drift, {len(ok)} ok; window +-{args.window} h, same bucket")
    print("\nper drift frame:")
    for s in drift:
        f = feat[s]
        print(f"  {s} {F.bucket(s):<10} frame_n {f['frame_n']}  dev "
              f"{'-' if f['dev'] is None else f'{f['dev']:.2f}'}  medcrop "
              f"{'-' if f['medcrop'] is None else f'{f['medcrop']:.2f}'}  medscore "
              f"{'-' if f['medscore'] is None else f'{f['medscore']:.2f}'}  dxday "
              f"{'-' if f['dxday'] is None else f'{f['dxday']:+.2f}'}")
    for name, ths, hi in (("dev", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0), True),
                          ("medcrop", (1.0, 1.5, 2.0, 2.5, 3.0, 4.0), True),
                          ("dxday", (0.5, 1.0, 1.5, 2.0, 3.0), True),
                          ("medscore", (0.5, 0.7, 0.8, 0.85, 0.9), False)):
        print(f"\nfeature {name} ({'>' if hi else '<'} threshold flags drift):")
        print(f"  {'thresh':>7} {'TP':>4} {'FP':>4} {'precision':>10} {'recall':>7}")
        for th, tp, fp, prec, rec in pr_table(feat, pos, neg, name, ths, hi):
            print(f"  {th:>7} {tp:>4} {fp:>4} {prec:>10.2f} {rec:>7.2f}")
        miss = sum(1 for s in pos if feat[s][name] is None)
        if miss:
            print(f"  ({miss} drift frame(s) have no value for {name})")

    if not args.no_sheet:
        src = {}
        for iid, e in eff.items():
            s_, p_ = gr.split_item_id(iid, e)
            src[(s_, p_)] = e.get("src", "")
        chosen = contact_sheet(drift, ok, lambda s, p: src.get((s, p)), args.sheet)
        print(f"\ncontact sheet -> {args.sheet} (ok frames: {', '.join(c or '-' for c in chosen)})")

    if args.apply_preflag:
        name, th = args.apply_preflag[0], float(args.apply_preflag[1])
        hi = name != "medscore"
        stamps = set()
        for qn in SCREEN_QUEUES:
            p = QDIR / f"{qn}.csv"
            if p.is_file():
                with p.open(newline="", encoding="utf-8") as fh:
                    stamps |= {r["stamp"] for r in csv.DictReader(fh)}
        flagged = set()
        for s in stamps:
            v = F.of(s)[name] if s in frames else None
            if v is not None and (v > th if hi else v < th):
                flagged.add(s)
        res = apply_preflag(flagged)
        for qn, (nf, ni, n) in res.items():
            print(f"preflag {FLAG}: {qn:<11} {nf:>4} frames / {ni:>5} of {n} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
