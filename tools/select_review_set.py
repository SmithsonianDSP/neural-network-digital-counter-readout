"""Derive ground truth from meter physics, then emit a validation set + review queue.

Raw tree: <root>/<YYYYmmdd>/<HH>/<label>_main_dig<pos>_<YYYYmmdd-HHMMSS>.jpg
<label> is the *model's* automatic label (unverified). Upstream writes `10_` for the
NaN class; everything downstream expects `N_` (HANDOFF.md s3).

Four screen types appear (dig2-primary typing, see dig_data.classify_screen):
    5 d3 d4 d5 d6   the meter reading; full reading = 50000 + 1000*d3 + 100*d4
                    + 10*d5 + d6, d3 in {7,8,9} (meter range 57000..59999)
    0 0 0 0 0       test screen
    8 8 8 8 8       test screen (also lights the `.` and `deg` annunciators)
    - - - # #       dash screen; dig2/3/4 blank. dig5/dig6 were believed constant
                    `0 9`; batch 3's model labels put that in question, so they are
                    reported as status "dash_unverified" with no derived truth.

The reading is monotone non-decreasing, so most labels are derivable without human
review: fit a monotone step function through the model's own full readings (and,
optionally, AIOTE's accepted values from the history CSVs), robust to a minority of
frames being wrong, then use it to *corroborate* each frame's screen type and pin
each digit. The fit is on the full 5-digit reading, so 57999 -> 58000 -> ... ->
59000 rollovers need no special casing.

Per-crop status (tiering semantics unchanged from batch 2):
    agree / agree_soft      model == derived truth (soft: fit can't pin the digit)
    disagree                model != derived truth, digit pinned  ("conflict")
    ambiguous               model != derived truth, fit can't pin the digit
    uncertain               screen type or reading unresolved
    underivable             no anchors to bracket the frame
    dash_unverified         dash-screen dig5/dig6 (model label recorded only)

Usage:
    python tools/select_review_set.py --derive \
        --root C:/Users/josep/source/repos/AIOTED-digital-rawdigits \
        --stats-cache work/b3_frame_stats.csv --out-csv work/b3_derive.csv \
        --rollover-sheet work/b3_rollover_check.png
    python tools/select_review_set.py --emit --out work/review
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob as globmod
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dig_data  # noqa: E402
from dig_data import bucket_for_hour, classify_screen, frame_brightness  # noqa: E402,F401

RAW_ROOT = Path(r"C:\Users\josep\source\repos\AIOTED-digital-rawdigits")
MISTAKES = RAW_ROOT / "mistakes"

NAME_RE = re.compile(
    r"^(?P<label>10|[0-9]|N)_main_(?P<pos>dig\d)_(?P<stamp>\d{8}-\d{6})\.jpg$", re.I
)
STRIP_RE = re.compile(r"^(?:10|[0-9]|N)_", re.I)

POSITIONS = ("dig2", "dig3", "dig4", "dig5", "dig6")
BASE = 50000  # full reading = BASE + 1000*d3 + 100*d4 + 10*d5 + d6 (dig2 == 5)
DIG3_OK = ("7", "8", "9")  # meter range 57000..59999
DIV = {"dig2": 10000, "dig3": 1000, "dig4": 100, "dig5": 10, "dig6": 1}
# Reading screens appear only ~40% of the time (the display cycles through four
# screens), so anchors can be 10-20 min apart and the fit's fine resolution is
# limited. Corroborate the slow digits loosely and dig6 tightly.
FIT_TOL_COARSE = 5  # gates screen type + dig2..dig5
FIT_TOL_FINE = 1  # additionally required before dig6 is treated as pinned
CSV_MATCH_TOL_S = 30  # CSV stamp is 0-9 s after the crop stamp (measured, batch 3)

# legacy hour rule, kept for reference; the live rule is dig_data.bucket_for_hour
TRANSITION = set(dig_data.HOUR_TRANSITION)
FLASH = set(dig_data.HOUR_FLASH)
BUCKETS = ("day", "transition", "flash")

# Review-queue tiers, most informative first.
TIER_CONFLICT = "A_conflict"      # model vs derivation disagree
TIER_AMBIGUOUS = "B_ambiguous"    # derivation cannot pin the digit
TIER_UNCERTAIN = "C_uncertain"    # screen type itself unresolved
TIER_DASHDIGIT = "D_dashdigit"    # dash-screen dig5/dig6: not verified
TIER_CONFIRM = "E_confirm"        # both agree; fast skim only


def norm_label(raw: str) -> str:
    raw = raw.upper()
    return "N" if raw in ("10", "N") else raw


@dataclass
class Crop:
    path: Path
    model: str
    pos: str
    stamp: str
    hour: int
    truth: str | None = None
    status: str = "unknown"
    light: str | None = None  # brightness bucket, when frame stats are available
    contrast: float | None = None

    @property
    def bucket(self) -> str:
        return self.light or bucket_for_hour(self.hour)

    @property
    def bucket_hour(self) -> str:
        return bucket_for_hour(self.hour)

    @property
    def day(self) -> str:
        return self.stamp[:8]


@dataclass
class Frame:
    stamp: str
    crops: dict[str, Crop] = field(default_factory=dict)
    reading: int | None = None  # model's full reading (reading screens only)
    fitted: int | None = None  # fitted / bracketed meter reading at this frame
    kind: str = "uncertain"
    corroborated: bool = False
    anchor_source: str = ""  # "", "frame", "csv", "frame+csv"
    csv_row: dict | None = None
    brightness: float | None = None
    light: str | None = None

    @property
    def dt(self) -> datetime:
        return datetime.strptime(self.stamp, "%Y%m%d-%H%M%S")

    @property
    def ts(self) -> float:
        return self.dt.timestamp()

    @property
    def hour(self) -> int:
        return int(self.stamp[9:11])

    @property
    def bucket(self) -> str:
        return self.light or bucket_for_hour(self.hour)

    @property
    def day(self) -> str:
        return self.stamp[:8]

    def labels(self) -> dict[str, str]:
        return {p: c.model for p, c in self.crops.items()}

    def label_str(self) -> str:
        return "".join(self.crops[p].model if p in self.crops else "-" for p in POSITIONS)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _day_ok(day: str, days: tuple[str, str] | set | None) -> bool:
    if not days:
        return True
    if isinstance(days, tuple):
        return days[0] <= day <= days[1]
    return day in days


def load_index(root: Path | list[Path] = RAW_ROOT, days=None) -> tuple[list[Crop], int]:
    """Index one or more raw roots. `days` = (first, last) YYYYmmdd tuple or a set."""
    roots = [Path(r) for r in (root if isinstance(root, (list, tuple)) else [root])]
    corrections: dict[str, str] = {}
    for r in roots:
        mistakes = r / "mistakes"
        if mistakes.is_dir():
            for p in mistakes.iterdir():
                m = NAME_RE.match(p.name)
                if m:
                    corrections[STRIP_RE.sub("", p.name)] = norm_label(m.group("label"))

    crops: list[Crop] = []
    for r in roots:
        for daydir in sorted(r.iterdir()):
            if not (daydir.is_dir() and re.fullmatch(r"\d{8}", daydir.name)):
                continue
            if not _day_ok(daydir.name, days):
                continue
            for hourdir in sorted(daydir.iterdir()):
                if not hourdir.is_dir():
                    continue
                for p in sorted(hourdir.iterdir()):
                    m = NAME_RE.match(p.name)
                    if not m:
                        continue
                    key = STRIP_RE.sub("", p.name)
                    crops.append(
                        Crop(
                            path=p.resolve(),
                            model=corrections.get(key, norm_label(m.group("label"))),
                            pos=m.group("pos").lower(),
                            stamp=m.group("stamp"),
                            hour=int(hourdir.name),
                        )
                    )
    return crops, len(corrections)


def build_frames(crops: list[Crop]) -> list[Frame]:
    by_stamp: dict[str, Frame] = {}
    for c in crops:
        by_stamp.setdefault(c.stamp, Frame(stamp=c.stamp)).crops[c.pos] = c
    return [by_stamp[k] for k in sorted(by_stamp)]


def load_history_csv(paths: list[str]) -> list[dict]:
    """AIOTE history CSVs (no header, 13 cols). Returns rows sorted by time."""
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf8") as fh:
            for r in csv.reader(fh):
                if len(r) < 13:
                    continue
                try:
                    t = datetime.fromisoformat(r[0]).replace(tzinfo=None)
                except ValueError:
                    continue
                acc = None
                if r[3].strip():
                    try:
                        acc = int(round(float(r[3])))
                    except ValueError:
                        acc = None
                rows.append({
                    "t": t.timestamp(),
                    "time": r[0],
                    "raw": r[2],
                    "accepted": acc,
                    "error": r[7],
                    "labels": "".join(norm_label(x) if x else "N" for x in r[8:13]),
                })
    rows.sort(key=lambda x: x["t"])
    return rows


def filter_csv_anchors(rows: list[dict], confirm: int = 5) -> tuple[list[dict], Counter]:
    """CSV accepted values that are safe to use as anchors.

    Drops: non-numeric / out-of-range (outside 57000..59999 -- catches the accepted
    `88888`s of 2026-09-02), all-identical-digit values, and values below the running
    max (the early-morning dips). The running max only advances to a value that the
    next `confirm` accepted values support (their median >= it), so one bad accepted
    jump cannot poison everything after it.
    """
    why = Counter()
    cand = []
    for r in rows:
        v = r["accepted"]
        if v is None:
            continue
        s = str(v)
        if not (57000 <= v <= 59999):
            why["out_of_range"] += 1
            continue
        if len(set(s)) == 1:
            why["repdigit"] += 1
            continue
        cand.append(r)
    out = []
    run = None
    vals = [r["accepted"] for r in cand]
    for i, r in enumerate(cand):
        v = r["accepted"]
        if run is not None and v < run:
            why["below_running_max"] += 1
            continue
        nxt = vals[i + 1:i + 1 + confirm]
        if nxt and median(nxt) < v:
            why["unconfirmed_jump"] += 1
            continue
        run = v
        out.append(r)
    why["kept"] = len(out)
    return out, why


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


def provisional_kind(f: Frame) -> str:
    """First-pass screen typing from the frame's full label set (dig2-primary)."""
    return classify_screen(f.labels())


def full_reading(lab: dict[str, str]) -> int | None:
    """50000 + 1000*d3 + 100*d4 + 10*d5 + d6, or None if any of d3..d6 isn't a digit."""
    try:
        return BASE + sum(int(lab[p]) * DIV[p] for p in POSITIONS[1:])
    except (KeyError, ValueError):
        return None


def implied_reading(f: Frame) -> int | None:
    """Full reading from the model's dig3..dig6 (dig2 assumed 5)."""
    return full_reading(f.labels())


def digit_of(reading: int, pos: str) -> str:
    return str((reading // DIV[pos]) % 10)


def pava(values: list[float]) -> list[float]:
    """Pool-adjacent-violators: least-squares monotone non-decreasing fit."""
    if not values:
        return []
    lvl, wt, cnt = list(values), [1.0] * len(values), [1] * len(values)
    i = 0
    while i < len(lvl) - 1:
        if lvl[i] <= lvl[i + 1]:
            i += 1
            continue
        tot = wt[i] + wt[i + 1]
        lvl[i] = (lvl[i] * wt[i] + lvl[i + 1] * wt[i + 1]) / tot
        wt[i] = tot
        cnt[i] += cnt[i + 1]
        del lvl[i + 1], wt[i + 1], cnt[i + 1]
        if i > 0:
            i -= 1
    out: list[float] = []
    for v, c in zip(lvl, cnt):
        out.extend([v] * c)
    return out


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


@dataclass
class Point:
    t: float
    value: int
    source: str
    frame: Frame | None = None
    fitted: int | None = None
    weight: float = 1.0


# Anchor weights for the monotone selection. Daylight readings are near-perfect
# (batch 2: zero daytime errors in 1,271 unseen crops); flash-only dig6 is not --
# batch 3's night model leans hard on `9` (e.g. 58021 read as 58029 for hours).
LIGHT_WEIGHT = {"day": 3, "transition": 2, "flash": 1}


def _fit(points: list[Point], win: int = 0, tol: float = 0.0) -> list[Point]:
    """Select the maximum-weight non-decreasing subsequence of anchor values.

    Replaces batch 2's local-median filter + PAVA. Least squares is the wrong tool
    here: night dig6 misreads are biased (towards `9`) and can be the local
    majority for hours, which dragged the PAVA level up and the running max then
    pinned it high. The max-weight monotone subsequence instead keeps whichever
    story the *whole* series supports: a run of 58029s at 02:00 is dropped because
    the 58023..58028 readings of 05:00-09:00 contradict it. Sparse misreads in
    either direction cost one point and are skipped. The kept points are monotone
    by construction, so fitted == value. O(n log n) via a Fenwick prefix-max over
    value ranks. (`win`/`tol` are accepted for backward compatibility, unused.)
    """
    points = sorted(points, key=lambda p: p.t)
    if not points:
        return []
    ranks = {v: i + 1 for i, v in enumerate(sorted({p.value for p in points}))}
    m = len(ranks)
    tree_s = [0.0] * (m + 1)
    tree_i = [-1] * (m + 1)
    score = [0.0] * len(points)
    prev = [-1] * len(points)
    for i, p in enumerate(points):
        r = ranks[p.value]
        best, bi, k = 0.0, -1, r
        while k > 0:  # prefix max over values <= p.value
            if tree_s[k] > best:
                best, bi = tree_s[k], tree_i[k]
            k -= k & -k
        score[i] = best + p.weight
        prev[i] = bi
        k = r
        while k <= m:
            if score[i] > tree_s[k]:
                tree_s[k], tree_i[k] = score[i], i
            k += k & -k
    i = max(range(len(points)), key=lambda j: score[j])
    chain = []
    while i >= 0:
        chain.append(points[i])
        i = prev[i]
    kept = chain[::-1]
    for p in kept:
        p.fitted = p.value
    return kept


class Bracket:
    """O(log n) lookup of the kept anchors either side of a time."""

    def __init__(self, kept: list[Point]):
        self.kept = kept
        self.times = [p.t for p in kept]

    def __call__(self, t: float) -> tuple[int | None, int | None]:
        # prev = last anchor with time <= t (an anchor brackets itself), next = first after
        i = bisect.bisect_right(self.times, t)
        lo = self.kept[i - 1].fitted if i > 0 else None
        hi = self.kept[i].fitted if i < len(self.kept) else None
        return lo, hi


def _est(lo: int | None, hi: int | None) -> int | None:
    if lo is not None and hi is not None:
        return lo if lo == hi else (lo + hi) // 2
    return lo if lo is not None else hi


def _nearest_in_bracket(lab: dict[str, str], lo: int, hi: int, default: int,
                        span_max: int = 200) -> int:
    """Best estimate of a reading frame's value inside its anchor bracket: the value
    in [lo, hi] sharing the most digits with the model's labels (ties -> closest to
    the midpoint). E.g. bracket 58999..59001 with model 58949 -> 58999, not the
    midpoint 59000. Digits that differ between lo and hi stay unpinned regardless."""
    if hi - lo > span_max:
        return default
    want = [lab.get(p) for p in POSITIONS[1:]]
    best, key = default, None
    for v in range(lo, hi + 1):
        s = str(v)[1:]
        score = sum(1 for a, b in zip(s, want) if a == b)
        k = (-score, abs(v - default))
        if key is None or k < key:
            best, key = v, k
    return best


def _hamming_fit(lab: dict[str, str], fitted: int, lo, hi) -> tuple[int, list[str]]:
    """Positions (dig2..dig6) where the model's label differs from the fitted
    reading's digit, ignoring digits the fit cannot pin (lo/hi differ there)."""
    diff = []
    for p in POSITIONS:
        want = "5" if p == "dig2" else digit_of(fitted, p)
        if lo is not None and hi is not None and lo // DIV[p] != hi // DIV[p]:
            continue  # fit can't pin this digit
        if lab.get(p) != want:
            diff.append(p)
    return len(diff), diff


def attach_csv(frames: list[Frame], rows: list[dict], tol_s: float = CSV_MATCH_TOL_S):
    """Nearest-timestamp join CSV rows -> frames. Returns (matched, offsets, label_match)."""
    if not rows:
        return 0, Counter(), Counter()
    ts = [r["t"] for r in rows]
    offs, lm = Counter(), Counter()
    matched = 0
    for f in frames:
        t = f.ts
        i = bisect.bisect_left(ts, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(rows) and abs(ts[j] - t) <= tol_s:
                if best is None or abs(ts[j] - t) < abs(ts[best] - t):
                    best = j
        if best is None:
            continue
        f.csv_row = rows[best]
        matched += 1
        offs[int(ts[best] - t)] += 1
        lm[rows[best]["labels"] == f.label_str()] += 1
    return matched, offs, lm


def derive(frames: list[Frame], csv_rows: list[dict] | None = None,
           win: int = 7, tol: float = 4.0) -> dict:
    meta: dict = {}
    for f in frames:
        f.kind = provisional_kind(f)
        f.reading = implied_reading(f) if f.kind == "reading" else None
        f.fitted, f.corroborated, f.anchor_source = None, False, ""

    # 0. CSV anchors (AIOTE accepted values), joined to frames by nearest stamp.
    csv_rows = csv_rows or []
    csv_ok, why = filter_csv_anchors(csv_rows) if csv_rows else ([], Counter())
    matched, offs, lm = attach_csv(frames, csv_rows)
    meta.update(csv_rows=len(csv_rows), csv_filter=why, csv_matched=matched,
                csv_offsets=offs, csv_label_match=lm)
    csv_ok_ids = {id(r) for r in csv_ok}
    frame_of_row = {id(f.csv_row): f for f in frames if f.csv_row is not None}

    def build_points(reading_frames: list[Frame]) -> list[Point]:
        pts: list[Point] = []
        used = set()
        for f in reading_frames:
            src = "frame"
            if f.csv_row is not None and id(f.csv_row) in csv_ok_ids \
                    and f.csv_row["accepted"] == f.reading:
                src = "frame+csv"
                used.add(id(f.csv_row))
            pts.append(Point(f.ts, f.reading, src, f, weight=LIGHT_WEIGHT[f.bucket]))
        for r in csv_ok:
            if id(r) in used:
                continue
            fr = frame_of_row.get(id(r))
            if fr is not None and fr.kind == "reading" and fr.reading is not None \
                    and fr.reading != r["accepted"]:
                continue  # frame's own labels already say otherwise; frame wins
            pts.append(Point(r["t"], r["accepted"], "csv", fr))
        return pts

    # 1. first fit, from frames typed reading by dig2/dig3 (+ CSV).
    strict = [f for f in frames if f.kind == "reading" and f.reading is not None]
    kept = _fit(build_points(strict), win, tol)
    meta["strict"] = len(strict)
    meta["kept1"] = len(kept)

    # 2. recover reading screens the model mislabelled at dig2/dig3: a frame whose
    #    labels sit within one digit of where the fit says the meter was is a reading
    #    screen -- an external constraint that owes nothing to the misread digit.
    br = Bracket(kept)
    recovered = 0
    for f in frames:
        if f.kind != "uncertain" or len(f.crops) < 5:
            continue
        lo, hi = br(f.ts)
        est = _est(lo, hi)
        if est is None:
            continue
        lab = f.labels()
        imp = implied_reading(f)
        nd, _ = _hamming_fit(lab, est, lo, hi)
        close = imp is not None and lab.get("dig2") == "5" \
            and abs(imp - est) <= FIT_TOL_COARSE
        if close or nd <= 1:
            f.kind = "reading"
            # only frames whose own digits land on the fit become anchors
            f.reading = imp if (imp is not None and abs(imp - est) <= FIT_TOL_COARSE) else None
            recovered += 1
    meta["recovered"] = recovered

    # 3. refit with the recovered frames folded in.
    anchors_f = [f for f in frames if f.kind == "reading" and f.reading is not None]
    points = build_points(anchors_f)
    kept = _fit(points, win, tol)
    meta["points"] = len(points)
    meta["points_by_source"] = Counter(p.source for p in points)
    meta["kept"] = len(kept)
    meta["kept_by_source"] = Counter(p.source for p in kept)
    for p in kept:
        if p.frame is not None:
            p.frame.anchor_source = p.source
            p.frame.fitted = p.fitted

    # 4. bracket every frame between its nearest kept anchors.
    br = Bracket(kept)
    brackets: dict[str, tuple[int | None, int | None]] = {}
    for f in frames:
        lo, hi = br(f.ts)
        brackets[f.stamp] = (lo, hi)
        if f.fitted is None:
            f.fitted = _est(lo, hi)
            if f.kind == "reading" and lo is not None and hi is not None and lo != hi:
                f.fitted = _nearest_in_bracket(f.labels(), lo, hi, f.fitted)
    # bracket estimates are chosen independently; keep the series monotone. Reading
    # frames first (their estimates use their own digits), then clamp the other
    # screens' midpoint estimates between the neighbouring reading-frame values so
    # a midpoint can never drag a later reading frame upward.
    run = None
    rd = [f for f in frames if f.kind == "reading" and f.fitted is not None]
    for f in rd:
        if run is not None and f.fitted < run:
            f.fitted = run
        run = f.fitted
    rd_ts = [f.ts for f in rd]
    for f in frames:
        if f.kind == "reading" or f.fitted is None or not rd:
            continue
        i = bisect.bisect_right(rd_ts, f.ts)
        if i > 0 and f.fitted < rd[i - 1].fitted:
            f.fitted = rd[i - 1].fitted
        if i < len(rd) and f.fitted > rd[i].fitted:
            f.fitted = rd[i].fitted

    # 4b. skip guard. A value the anchor chain never shows (198 -> 200) is
    #     usually not a fast tick but a CONSISTENT misread that monotonicity
    #     cannot see: from 2026-09-05 the deployed model reads dig6 5 as 6 and 9
    #     as 8 in daylight (~0-2% 5/9 vs ~19% 6/8), so 199 hides at the end of the
    #     198 run and 205 at the start of the 206 run. Night 8->9 / 5->6 does the
    #     same. Widen the bracket of every frame in the two runs either side of a
    #     skip to the skip's endpoints, so the affected digits are not pinned.
    runs: list[list] = []  # [value, t_start, t_end]
    for p in kept:
        if runs and runs[-1][0] == p.fitted:
            runs[-1][2] = p.t
        else:
            runs.append([p.fitted, p.t, p.t])
    skips = []
    for a, b in zip(runs, runs[1:]):
        if b[0] - a[0] > 1:
            if skips and skips[-1][1] >= a[1]:  # chain overlapping skips
                skips[-1][1], skips[-1][3] = b[2], b[0]
            else:
                skips.append([a[1], b[2], a[0], b[0]])
    meta["skips"] = len(skips)
    starts = [s[0] for s in skips]
    widened = 0
    for f in frames:
        i = bisect.bisect_right(starts, f.ts) - 1
        if i >= 0 and skips[i][0] <= f.ts <= skips[i][1]:
            lo, hi = brackets[f.stamp]
            lo2 = skips[i][2] if lo is None else min(lo, skips[i][2])
            hi2 = skips[i][3] if hi is None else max(hi, skips[i][3])
            if (lo2, hi2) != (lo, hi):
                brackets[f.stamp] = (lo2, hi2)
                widened += 1
    meta["skip_widened_frames"] = widened

    # 5. corroborate reading frames against the refit: within COARSE of the fit, or
    #    exactly one pinned digit off (a single misread digit, incl. dig3 9->8 that
    #    puts the raw reading 1000 away).
    for f in frames:
        if f.kind != "reading" or f.fitted is None:
            continue
        lo, hi = brackets[f.stamp]
        if f.reading is not None and abs(f.reading - f.fitted) <= FIT_TOL_COARSE:
            f.corroborated = True
        else:
            nd, _ = _hamming_fit(f.labels(), f.fitted, lo, hi)
            f.corroborated = nd <= 1

    def digit_certain(f: Frame, pos: str, lo: int | None, hi: int | None) -> bool:
        """A digit is pinned when the kept anchors either side share every digit
        down to and including it: the series is monotone, so every value between
        them does too. (Batch 2 additionally required |reading - fit| <= 1 for
        dig6 because its PAVA level could sit between anchors; the monotone
        subsequence fit has no such in-between level.)"""
        if pos == "dig2":
            return True
        if lo is None or hi is None:
            return False
        return lo // DIV[pos] == hi // DIV[pos]

    # 6. per-crop truth.
    for f in frames:
        lo, hi = brackets[f.stamp]
        for pos, c in f.crops.items():
            c.truth = None
            if f.kind in ("test0", "test8"):
                c.truth = f.kind[-1]
                c.status = "agree" if c.model == c.truth else "disagree"
            elif f.kind == "dash":
                if pos in ("dig2", "dig3", "dig4"):
                    c.truth = "N"
                    c.status = "agree" if c.model == "N" else "disagree"
                else:
                    # Believed constant `0 9` in batch 2; batch 3's labels say
                    # NNN08/05/06 often enough that it is unverified. Record only.
                    c.status = "dash_unverified"
            elif f.kind == "uncertain" or (f.kind == "reading" and not f.corroborated
                                           and f.fitted is not None):
                c.status = "uncertain"
            elif f.fitted is None:
                c.status = "underivable"
            else:
                c.truth = "5" if pos == "dig2" else digit_of(f.fitted, pos)
                certain = digit_certain(f, pos, lo, hi)
                if c.model == c.truth:
                    c.status = "agree" if certain else "agree_soft"
                else:
                    c.status = "disagree" if certain else "ambiguous"

    meta.update(
        frames=len(frames),
        anchors=len(anchors_f),
        corroborated=sum(1 for f in frames if f.corroborated),
        lo=kept[0].fitted if kept else None,
        hi=kept[-1].fitted if kept else None,
    )

    # CSV accepted vs frame-derived reading, on frames typed reading.
    agree = Counter()
    bad = []
    for f in frames:
        r = f.csv_row
        if r is None or r["accepted"] is None or f.fitted is None:
            continue
        key = "reading" if f.kind == "reading" else "other_screen"
        ok = r["accepted"] == f.fitted
        agree[(key, ok)] += 1
        if not ok:
            bad.append((f.stamp, f.kind, f.label_str(), r["accepted"], f.fitted,
                        id(r) in csv_ok_ids))
    meta["csv_agree"] = agree
    meta["csv_disagree_rows"] = bad
    return meta


def tier_for(c: Crop) -> str:
    return {
        "disagree": TIER_CONFLICT,
        "ambiguous": TIER_AMBIGUOUS,
        "uncertain": TIER_UNCERTAIN,
        "underivable": TIER_UNCERTAIN,
        "dash_unverified": TIER_DASHDIGIT,
        "dashdigit": TIER_DASHDIGIT,
    }.get(c.status, TIER_CONFIRM)


# --------------------------------------------------------------------------
# frame light stats (brightness / contrast), cached
# --------------------------------------------------------------------------

STATS_FIELDS = (["stamp", "n", "brightness", "mean_luma"]
                + [f"med_{p}" for p in POSITIONS]
                + [f"mean_{p}" for p in POSITIONS]
                + [f"con_{p}" for p in POSITIONS])


def _crop_stats_job(path: str) -> tuple[str, dict]:
    return path, dig_data.crop_light_stats(path)


def load_frame_stats(frames: list[Frame], cache: Path | None, jobs: int = 8) -> dict:
    """Per-frame brightness / per-crop contrast, cached in `cache` (CSV, per frame)."""
    have: dict[str, dict] = {}
    if cache is not None and cache.exists():
        with cache.open(newline="", encoding="utf8") as fh:
            for r in csv.DictReader(fh):
                have[r["stamp"]] = r
    todo = [f for f in frames if f.stamp not in have or int(have[f.stamp]["n"]) != len(f.crops)]
    if todo:
        paths = [str(c.path) for f in todo for c in f.crops.values()]
        print(f"computing light stats for {len(todo)} frames / {len(paths)} crops "
              f"({jobs} workers)...", flush=True)
        res: dict[str, dict] = {}
        if jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                for i, (p, st) in enumerate(ex.map(_crop_stats_job, paths, chunksize=256)):
                    res[p] = st
                    if (i + 1) % 10000 == 0:
                        print(f"  {i + 1}/{len(paths)}", flush=True)
        else:
            for p in paths:
                res[p] = dig_data.crop_light_stats(p)
        for f in todo:
            row = {"stamp": f.stamp, "n": len(f.crops)}
            meds, means = [], []
            for p in POSITIONS:
                c = f.crops.get(p)
                st = res.get(str(c.path)) if c else None
                row[f"med_{p}"] = f"{st['median']:.2f}" if st else ""
                row[f"mean_{p}"] = f"{st['mean']:.2f}" if st else ""
                row[f"con_{p}"] = f"{st['contrast']:.2f}" if st else ""
                if st:
                    meds.append(st["median"])
                    means.append(st["mean"])
            row["brightness"] = f"{frame_brightness(meds):.2f}"
            row["mean_luma"] = f"{(sum(means) / len(means)) if means else float('nan'):.2f}"
            have[f.stamp] = row
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("w", newline="", encoding="utf8") as fh:
                w = csv.DictWriter(fh, fieldnames=STATS_FIELDS)
                w.writeheader()
                for k in sorted(have):
                    w.writerow({k2: have[k].get(k2, "") for k2 in STATS_FIELDS})
            print(f"light stats cache -> {cache}")
    return have


def apply_light(frames: list[Frame], stats: dict) -> None:
    for f in frames:
        r = stats.get(f.stamp)
        if r is None:
            continue
        f.brightness = float(r["brightness"])
        f.light = dig_data.light_bucket(f.brightness, f.hour)
        for p, c in f.crops.items():
            c.light = f.light
            v = r.get(f"con_{p}", "")
            c.contrast = float(v) if v not in ("", None) else None


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    i = int(math.floor(k))
    j = min(i + 1, len(s) - 1)
    return s[i] + (s[j] - s[i]) * (k - i)


def calibrate_light(frames: list[Frame]) -> tuple[float, float]:
    """Print brightness distributions for known-night (01-04) vs known-day (11-14)
    frames, the model's error rate on pinned crops per brightness bin, and the
    thresholds they imply. Returns (flash_max, day_min) suggestions."""
    night = [f.brightness for f in frames if f.brightness is not None and 1 <= f.hour <= 4]
    day = [f.brightness for f in frames if f.brightness is not None and 11 <= f.hour <= 14]
    qs = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 1.0)
    print("\n=== light calibration (frame brightness = median of per-crop median luma) ===")
    print("  " + f"{'':<12}{'n':>6}" + "".join(f"{'p'+format(q*100, 'g'):>8}" for q in qs))
    for name, xs in (("night 01-04", night), ("day 11-14", day)):
        print("  " + f"{name:<12}{len(xs):>6}" + "".join(f"{_pct(xs, q):>8.1f}" for q in qs))
    W = 4
    hn, hd, ha, ne, nn = Counter(), Counter(), Counter(), Counter(), Counter()
    for f in frames:
        if f.brightness is None:
            continue
        b = int(f.brightness // W) * W
        ha[b] += 1
        if 1 <= f.hour <= 4:
            hn[b] += 1
        if 11 <= f.hour <= 14:
            hd[b] += 1
        for c in f.crops.values():
            if c.status in ("agree", "disagree") and f.kind in ("reading", "test0", "test8"):
                nn[b] += 1
                ne[b] += c.status == "disagree"
    print(f"  {'bin':>5} {'night':>6} {'day':>6} {'all':>6} {'pinned':>7} {'err%':>6}")
    for b in sorted(ha):
        er = f"{100 * ne[b] / nn[b]:.2f}" if nn[b] else ""
        print(f"  {b:>5} {hn[b]:>6} {hd[b]:>6} {ha[b]:>6} {nn[b]:>7} {er:>6}")
    flash_max = max(night) if night else float("nan")
    day_min = _pct(day, 0.05)
    print(f"  suggested: flash <= {flash_max:.1f} (max of night 01-04);  day >= {day_min:.1f} "
          f"(day 11-14 p5; cross-check against the err% floor above)")
    print(f"  in use   : flash <= {dig_data.LIGHT_FLASH_MAX};  day >= {dig_data.LIGHT_DAY_MIN}")
    return flash_max, day_min


def print_hour_buckets(frames: list[Frame], first: str, last: str) -> None:
    sel = [f for f in frames if first <= f.day <= last and f.light]
    print(f"\n--- frames per hour x brightness bucket, {first}..{last} "
          f"({len(sel)} frames)  [day/transition/flash | old hour rule]")
    tab: dict[int, Counter] = defaultdict(Counter)
    for f in sel:
        tab[f.hour][f.light] += 1
    for h in range(24):
        t = tab[h]
        if not t:
            continue
        print(f"  {h:02d}  {t['day']:>4} {t['transition']:>4} {t['flash']:>4}   "
              f"| {bucket_for_hour(h)}")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def report(crops: list[Crop], frames: list[Frame], meta: dict) -> None:
    print(
        f"frames {meta['frames']}  strict reading frames {meta['strict']}  "
        f"+recovered {meta['recovered']}  -> anchor points {meta.get('points')} "
        f"kept {meta['kept']}  corroborated {meta['corroborated']}"
    )
    print(f"fitted reading span: {meta['lo']} .. {meta['hi']}")
    print(f"skip guard: {meta.get('skips', 0)} skipped-value spans in the anchor chain; "
          f"{meta.get('skip_widened_frames', 0)} frames had their bracket widened")
    if meta.get("csv_rows"):
        print(f"\n=== history CSV ===\n  rows {meta['csv_rows']}  matched to frames "
              f"{meta['csv_matched']}  (tolerance {CSV_MATCH_TOL_S}s)")
        print("  CSV-minus-crop stamp offsets (s):",
              dict(sorted(meta["csv_offsets"].items())))
        print("  CSV digit columns == frame model labels:", dict(meta["csv_label_match"]))
        print("  accepted-value anchor filter:", dict(meta["csv_filter"]))
        print("  anchor points by source:", dict(meta["points_by_source"]))
        print("  kept anchors by source  :", dict(meta["kept_by_source"]))
        ag = meta["csv_agree"]
        for key in ("reading", "other_screen"):
            n_ok, n_bad = ag[(key, True)], ag[(key, False)]
            if n_ok + n_bad:
                print(f"  CSV accepted == fitted reading, frame kind {key}: "
                      f"{n_ok}/{n_ok + n_bad} ({100 * n_ok / (n_ok + n_bad):.2f}%)")
        bad = meta["csv_disagree_rows"]
        if bad:
            print(f"  disagreements ({len(bad)}; stamp, kind, labels, accepted, fitted, "
                  f"passed-anchor-filter):")
            for b in bad[:25]:
                print("   ", *b)

    print("\nframe kinds:", dict(Counter(f.kind for f in frames)))
    kb: dict[str, Counter] = defaultdict(Counter)
    for f in frames:
        kb[f.kind][f.bucket] += 1
    for k in sorted(kb):
        print(f"  {k:<10}" + "".join(f"{b}={kb[k][b]:<6} " for b in BUCKETS))

    print("\n=== crop status x position ===")
    order = ("agree", "agree_soft", "disagree", "ambiguous", "uncertain", "underivable",
             "dash_unverified")
    tab: dict[str, Counter] = defaultdict(Counter)
    for c in crops:
        tab[c.pos][c.status] += 1
    print(f"  {'pos':<6}" + "".join(f"{h[:12]:>13}" for h in order))
    for pos in POSITIONS:
        print(f"  {pos:<6}" + "".join(f"{tab[pos][h]:>13}" for h in order))
    tot = Counter()
    for r in tab.values():
        tot.update(r)
    print(f"  {'ALL':<6}" + "".join(f"{tot[h]:>13}" for h in order))

    need = sum(v for k, v in tot.items() if k not in ("agree", "agree_soft"))
    print(f"\nneeds human eyes: {need} of {len(crops)} ({100*need/len(crops):.1f}%)")

    print("\n=== review tiers ===")
    for t, n in sorted(Counter(tier_for(c) for c in crops).items()):
        print(f"  {t}: {n}")


def supply_and_errors(frames: list[Frame]) -> None:
    def kind_group(f: Frame) -> str | None:
        if f.kind == "reading":
            return "reading"
        if f.kind in ("test0", "test8", "dash"):
            return "test"
        return None

    for grp in ("reading", "test"):
        cell: dict[tuple, Counter] = defaultdict(Counter)
        for f in frames:
            if kind_group(f) != grp:
                continue
            for c in f.crops.values():
                if c.truth is None:
                    continue
                if c.status in ("agree", "disagree"):
                    cell[(c.pos, c.truth)][c.bucket] += 1
                else:
                    cell[(c.pos, c.truth)]["soft"] += 1
        print(f"\n=== supply: PINNED derived-truth crops on {grp} screens, (pos, truth) x "
              f"bucket  (soft = agree_soft/ambiguous: truth not pinned) ===")
        print(f"  {'pos':<6}{'truth':>6}" + "".join(f"{b:>12}" for b in BUCKETS)
              + f"{'pinned':>8}{'soft':>7}")
        for pos in POSITIONS:
            for t in [str(d) for d in range(10)] + ["N"]:
                if (pos, t) not in cell:
                    continue
                r = cell[(pos, t)]
                print(f"  {pos:<6}{t:>6}" + "".join(f"{r[b]:>12}" for b in BUCKETS)
                      + f"{sum(r[b] for b in BUCKETS):>8}{r['soft']:>7}")

    wrong = Counter()
    by_grp = Counter()
    for f in frames:
        for c in f.crops.values():
            if c.truth is not None and c.model != c.truth:
                wrong[(c.pos, f"{c.truth}->{c.model}", c.bucket, c.status)] += 1
                by_grp[(kind_group(f), c.status, c.bucket)] += 1
    print(f"\n=== model label != derived truth: {sum(wrong.values())} crops ===")
    for (g, s, b), n in sorted(by_grp.items(), key=lambda x: -x[1]):
        print(f"  {g:<8} {s:<10} {b:<11} {n}")
    agg = Counter()
    for (pos, tm, b, s), n in wrong.items():
        agg[(pos, tm, b)] += n
    print("  top 30 cells (pos, truth->model, bucket): n  [disagree/ambiguous]")
    for (pos, tm, b), n in agg.most_common(30):
        d = wrong[(pos, tm, b, "disagree")]
        a = wrong[(pos, tm, b, "ambiguous")]
        print(f"    {pos} {tm:<6} {b:<11} {n:>5}   [{d}/{a}]")

    print("\n=== dash screen: dig5/dig6 model labels by bucket ===")
    for pos in ("dig5", "dig6"):
        t: dict[str, Counter] = defaultdict(Counter)
        for f in frames:
            if f.kind == "dash" and pos in f.crops:
                t[f.bucket][f.crops[pos].model] += 1
        for b in BUCKETS:
            print(f"  {pos} {b:<11} " + " ".join(
                f"{k}={v}" for k, v in sorted(t[b].items(), key=lambda x: -x[1])))
    pair: dict[str, Counter] = defaultdict(Counter)
    for f in frames:
        if f.kind == "dash":
            pair[f.bucket][f.label_str()[3:]] += 1
    for b in BUCKETS:
        print(f"  dig5dig6 {b:<11} " + " ".join(
            f"{k}={v}" for k, v in pair[b].most_common(8)))

    print("\n=== test8 screen: dig5/dig6 model labels by bucket ===")
    pair = defaultdict(Counter)
    for f in frames:
        if f.kind == "test8":
            pair[f.bucket][f.label_str()] += 1
    for b in BUCKETS:
        print(f"  {b:<11} " + " ".join(f"{k}={v}" for k, v in pair[b].most_common(8)))


def write_derive_csv(frames: list[Frame], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["stamp", "day", "hour", "pos", "path", "model_label", "truth",
                    "status", "screen", "reading_est", "bucket", "bucket_hour",
                    "frame_brightness", "contrast", "anchor_source"])
        for f in frames:
            for p in POSITIONS:
                c = f.crops.get(p)
                if c is None:
                    continue
                w.writerow([
                    f.stamp, f.day, c.hour, p, str(c.path), c.model,
                    c.truth if c.truth is not None else "?", c.status, f.kind,
                    f.fitted if f.fitted is not None else "", c.bucket, c.bucket_hour,
                    f"{f.brightness:.2f}" if f.brightness is not None else "",
                    f"{c.contrast:.2f}" if c.contrast is not None else "",
                    f.anchor_source,
                ])
    print(f"\nper-crop derivation -> {out}")


def rollover_sheet(frames: list[Frame], out: Path, n_each: int = 26,
                   scale: float = 0.45) -> None:
    """Contact sheet of frames around each thousand rollover of the fitted reading."""
    from PIL import Image, ImageDraw

    marks = []
    for i in range(1, len(frames)):
        a, b = frames[i - 1].fitted, frames[i].fitted
        if a is not None and b is not None and a // 1000 != b // 1000:
            marks.append(i)
    if not marks:
        print("rollover sheet: no thousand rollover in the fitted series")
        return
    tw, th = int(94 * scale), int(202 * scale)
    pad, text_w = 3, 330
    col_w = text_w + 5 * (tw + pad) + 12
    rows_n = n_each
    sheet = Image.new("RGB", (col_w * len(marks), (th + pad) * rows_n + 24), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for k, m in enumerate(marks):
        x0 = k * col_w
        sel = frames[max(0, m - n_each // 2): m + n_each - n_each // 2]
        draw.text((x0 + 4, 4), f"rollover {frames[m - 1].fitted} -> {frames[m].fitted} "
                  f"at {frames[m].stamp}", fill=(255, 255, 255))
        for r, f in enumerate(sel):
            y = 24 + r * (th + pad)
            x = x0 + 4
            for p in POSITIONS:
                c = f.crops.get(p)
                if c is not None:
                    im = Image.open(c.path).convert("RGB").resize((tw, th))
                    sheet.paste(im, (x, y))
                x += tw + pad
            stat = "".join(
                {"agree": ".", "agree_soft": ",", "disagree": "X", "ambiguous": "?",
                 "uncertain": "u", "underivable": "-", "dash_unverified": "d"}.get(
                    f.crops[p].status, "!") if p in f.crops else " " for p in POSITIONS)
            truth = "".join((f.crops[p].truth or "?") if p in f.crops else " "
                            for p in POSITIONS)
            col = (255, 90, 90) if "X" in stat or "?" in stat else (230, 230, 120)
            if f.kind != "reading":
                col = (150, 150, 150)
            draw.text((x + 4, y + 2), f"{f.stamp[4:8]} {f.stamp[9:]}  {f.kind}", fill=col)
            draw.text((x + 4, y + 16), f"model {f.label_str()}  truth {truth}", fill=col)
            draw.text((x + 4, y + 30), f"fit {f.fitted}  st {stat}  {f.anchor_source}",
                      fill=col)
            draw.text((x + 4, y + 44), f"bri {f.brightness or 0:.0f} {f.bucket}",
                      fill=col)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"rollover sheet ({len(marks)} rollovers) -> {out}")


def emit(crops, frames, out: Path, seed: int, n_val_normal: int, n_val_test: int,
         cap_day: int, cap_trans: int, cap_flash: int, target: int) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)

    # --- validation set: complete frames, drawn FIRST, stratified, never trained.
    usable = [
        f for f in frames
        if len(f.crops) == 5 and f.kind == "reading" and f.corroborated
    ]
    by_bucket: dict[str, list[Frame]] = defaultdict(list)
    for f in usable:
        by_bucket[f.bucket].append(f)
    quota = {
        "flash": int(n_val_normal * 0.45),
        "transition": int(n_val_normal * 0.20),
        "day": n_val_normal - int(n_val_normal * 0.45) - int(n_val_normal * 0.20),
    }
    val_frames: list[Frame] = []
    for b, q in quota.items():
        pool = by_bucket.get(b, [])
        # spread across days so one day's weather cannot dominate the holdout
        per_day: dict[str, list[Frame]] = defaultdict(list)
        for f in pool:
            per_day[f.day].append(f)
        picked, days = [], sorted(per_day)
        i = 0
        while len(picked) < q and any(per_day.values()):
            d = days[i % len(days)]
            if per_day[d]:
                picked.append(per_day[d].pop(rng.randrange(len(per_day[d]))))
            i += 1
        val_frames.extend(picked)

    tests = [f for f in frames if f.kind in ("test0", "test8") and len(f.crops) == 5]
    rng.shuffle(tests)
    val_frames.extend(tests[:n_val_test])
    val_stamps = {f.stamp for f in val_frames}

    # --- review queue. Tiers A-D are emitted whole (they are cheap to scroll and
    #     each resolved item unlocks data); only tier E, the fast-skim training
    #     pool, gets the cell caps.
    pool = [c for c in crops if c.stamp not in val_stamps]
    tiers: dict[str, list[Crop]] = defaultdict(list)
    for c in pool:
        tiers[tier_for(c)].append(c)

    queue: dict[str, list[Crop]] = {
        TIER_CONFLICT: tiers[TIER_CONFLICT],
        TIER_AMBIGUOUS: tiers[TIER_AMBIGUOUS],
        TIER_UNCERTAIN: tiers[TIER_UNCERTAIN],
        # One question -- is the dash screen's `# #` constant? -- so a sample does.
        TIER_DASHDIGIT: rng.sample(tiers[TIER_DASHDIGIT],
                                   min(60, len(tiers[TIER_DASHDIGIT]))),
    }

    caps = {"day": cap_day, "transition": cap_trans, "flash": cap_flash}
    cells: dict[tuple, list[Crop]] = defaultdict(list)
    for c in tiers[TIER_CONFIRM]:
        cells[(c.pos, c.truth or "?", c.bucket)].append(c)
    selected: list[Crop] = []
    for key, members in cells.items():
        members.sort(key=lambda c: c.stamp)
        step = max(1, len(members) // caps[key[2]])  # spread across the window
        selected.extend(members[::step][: caps[key[2]]])

    # global caps on the two classes that would otherwise flood the corpus
    def trim(pred, limit):
        hits = [c for c in selected if pred(c)]
        if len(hits) <= limit:
            return
        keep = set(id(c) for c in rng.sample(hits, limit))
        for c in hits:
            if id(c) not in keep:
                selected.remove(c)

    test_stamps = {f.stamp for f in frames if f.kind in ("test0", "test8")}
    trim(lambda c: c.stamp in test_stamps, 45)
    trim(lambda c: c.truth == "N", 55)

    if len(selected) > target:
        selected = rng.sample(selected, target)
    queue[TIER_CONFIRM] = selected

    # --- write it out. Filenames carry the DERIVED label, so accepting a crop is
    #     a no-op and correcting one is a single-character rename.
    manifest = []
    valdir = out / "validation"
    if valdir.exists():
        shutil.rmtree(valdir)
    for f in val_frames:
        for pos, c in f.crops.items():
            lbl = c.truth if c.truth else "REVIEW"
            d = valdir / f.bucket
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c.path, d / f"{lbl}_main_{pos}_{c.stamp}.jpg")
            manifest.append(("validation", f.bucket, c.pos, c.stamp, c.model,
                             c.truth or "", c.status, f.kind))

    qdir = out / "queue"
    if qdir.exists():
        shutil.rmtree(qdir)
    for t, members in queue.items():
        for c in members:
            d = qdir / t / c.bucket
            d.mkdir(parents=True, exist_ok=True)
            if t in (TIER_UNCERTAIN, TIER_DASHDIGIT):
                # These are per-FRAME calls ("which screen is this?"), so name
                # them stamp-first: all five crops of a frame sort together.
                name = f"{c.stamp}_{c.pos}_saw-{c.model}.jpg"
            else:
                # Derived label leads, so accepting is a no-op and correcting is
                # a one-character rename -- same convention as the corpus.
                name = f"{c.truth or 'REVIEW'}_main_{c.pos}_{c.stamp}.jpg"
            shutil.copy2(c.path, d / name)
            manifest.append(("queue/" + t, c.bucket, c.pos, c.stamp, c.model,
                             c.truth or "", c.status, ""))

    with (out / "MANIFEST.csv").open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "bucket", "pos", "stamp", "model_label",
                    "derived_label", "status", "frame_kind"])
        w.writerows(manifest)

    print(f"validation: {len(val_frames)} frames / "
          f"{sum(len(f.crops) for f in val_frames)} crops -> {valdir}")
    print("  by bucket:", dict(Counter(f.bucket for f in val_frames)))
    total_q = sum(len(v) for v in queue.values())
    print(f"\nreview queue: {total_q} crops -> {qdir}")
    for t in sorted(queue):
        members = queue[t]
        bb = dict(Counter(c.bucket for c in members))
        print(f"  {t:<14} {len(members):>5}   {bb}")
    print("\n  tier E label mix:",
          dict(sorted(Counter(c.truth or "?" for c in selected).items())))
    print("  tier E bucket mix:", dict(Counter(c.bucket for c in selected)))
    print(f"\nmanifest: {out / 'MANIFEST.csv'}")


def parse_days(s: str | None):
    """'20260815:20260930' -> inclusive range tuple; '20260815,20260816' -> set."""
    if not s:
        return None
    if ":" in s:
        a, b = s.split(":", 1)
        return (a or "00000000", b or "99999999")
    return set(x.strip() for x in s.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--root", type=Path, nargs="+", default=[RAW_ROOT],
                    help="one or more raw roots (<root>/<YYYYmmdd>/<HH>/*.jpg)")
    ap.add_argument("--csv-glob", nargs="*", default=None,
                    help="AIOTE history CSV glob(s); default <root>/data_*.csv for each "
                         "root. Pass with no value to disable.")
    ap.add_argument("--days", default=None,
                    help="date filter: FIRST:LAST (inclusive, YYYYmmdd) or a comma list")
    ap.add_argument("--stats-cache", type=Path, default=None,
                    help="per-frame light-stats cache CSV; enables brightness buckets")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="write one row per crop with derived truth / status / bucket")
    ap.add_argument("--rollover-sheet", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("work/review"))
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--val-frames", type=int, default=45)
    ap.add_argument("--val-test", type=int, default=5)
    ap.add_argument("--cap-day", type=int, default=5)
    ap.add_argument("--cap-transition", type=int, default=8)
    ap.add_argument("--cap-flash", type=int, default=12)
    ap.add_argument("--target", type=int, default=900)
    args = ap.parse_args()

    days = parse_days(args.days)
    crops, ncorr = load_index(list(args.root), days)
    frames = build_frames(crops)
    print(f"crops {len(crops)}  frames {len(frames)}  corrections applied {ncorr}")
    print("crops per frame:", dict(Counter(len(f.crops) for f in frames)))

    if args.csv_glob is None:
        csv_paths = sorted(p for r in args.root for p in globmod.glob(str(r / "data_*.csv")))
    else:
        csv_paths = sorted(p for g in args.csv_glob for p in globmod.glob(g))
    rows = load_history_csv(csv_paths) if csv_paths else []
    if rows and frames:
        lo_t, hi_t = frames[0].ts - 86400, frames[-1].ts + 86400
        rows = [r for r in rows if lo_t <= r["t"] <= hi_t]
    print(f"history CSVs {len(csv_paths)}  rows in window {len(rows)}\n")

    # light first: anchor weights in the fit depend on the frame's light bucket
    if args.stats_cache is not None:
        stats = load_frame_stats(frames, args.stats_cache, args.jobs)
        apply_light(frames, stats)

    meta = derive(frames, rows)

    if args.derive:
        report(crops, frames, meta)
        if args.stats_cache is not None:
            calibrate_light(frames)
            print_hour_buckets(frames, "20260816", "20260820")
            print_hour_buckets(frames, "20260925", "20260929")
            cross = Counter((f.light, bucket_for_hour(f.hour)) for f in frames if f.light)
            print("\n  brightness bucket x old hour bucket:")
            for b in BUCKETS:
                print(f"    {b:<11}" + "".join(f"{h}={cross[(b, h)]:<6} " for h in BUCKETS))
        supply_and_errors(frames)
    if args.out_csv is not None:
        write_derive_csv(frames, args.out_csv)
    if args.rollover_sheet is not None:
        rollover_sheet(frames, args.rollover_sheet)
    if args.emit:
        emit(crops, frames, args.out, args.seed, args.val_frames, args.val_test,
             args.cap_day, args.cap_transition, args.cap_flash, args.target)


if __name__ == "__main__":
    main()
