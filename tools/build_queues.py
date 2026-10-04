"""Build the batch-3 human-review queues for tools/grid_review.py, in priority order.

Every crop that enters training or the holdout must be human-reviewed; this script
decides *which* crops Joseph looks at, and in what order, so he can stop after any
queue and still have a usable corpus.

    0 holdout.csv     complete frames from 7 held-out days (frame mode, group=stamp)
    1 legacy.csv      derived-only corpus crops + staged culls + high-confidence
                      9002-vs-label disagreements from the Aug 2-7 pool
    2 night_err.csv   flash/transition crops the deployed model got (likely) wrong
    3 day_err.csv     day crops the deployed model got (likely) wrong
    4 night_fill.csv  pinned-truth night crops filling each (pos, class, bucket) cell
    5 day_fill.csv    pinned-truth day crops so day >= flash per (pos, class)

Inputs (read-only): work/b3_derive.csv, work/roi_drift.csv, joes-samples/*.jpg,
work/labeled/**, joes-samples/batch 1/, work/derived_only_review/,
work/eval_9002_labeledpool.csv, work/validation_labeled/.
Outputs: work/queues/*.csv, work/queues/holdout_days.txt, work/queues/SUMMARY.md.

Proposed labels for b3 crops:
    test0/test8 screen  -> 0 / 8
    dash screen         -> N at dig2-4, 0 at dig5, 9 at dig6 (dig5/6 unverified)
    reading screen      -> digit of reading_est at that position, except dig6 crops
                           in a "skip host" run (below), which get the inferred digit.

Skip inference. From ~09-05 the deployed model reads dig6 9 as 8 and 5 as 6 (and
pre-09-05 at night, 8 as 9) consistently, so the monotone fit never sees the hidden
value: the reading-frame anchors jump v -> v+2. The hidden v+1 sits on whichever
side's run lasts ~2x the usual ~40 min (9: the lower run; 5 and 8: the upper run).
We take the longer side as host (needs >= 1.3x the other), split its time span in
half, and propose v+1 for dig6 crops in the half adjacent to the skip. A +-10% guard
band around the split is not queued (except in the holdout, where frames are whole).
Every dig6 crop in a host run is kept out of the fill queues.

needs_verify (0/1 column): whether the proposed label is genuinely uncertain and needs
a human label check (grid_review --mode verify) before the crop can be used.
b3 crops start from: 1 if the b3_derive status is not agree/agree_soft, or the
deployed model's label differs from the proposal, or the proposal came from skip
inference. Labels pinned by screen structure are then exempt (0; they only get the
screen pass), unless the status is `uncertain`:
    test0 / test8 screens with status disagree (the other digits pin 00000 / 88888);
    reading-screen dig3 with status disagree (thousands digit pinned by the series);
    dig2 on any screen, and dash-screen dig2/3/4 (N).
What stays 1: reading dig4/5/6 ambiguous/disagree/uncertain or skip-inferred, dash
dig5/dig6 (dash_unverified: the 0/9 constant is unconfirmed), any `uncertain`.
Legacy crops: 1 only for the high-confidence 9002 disagreements ("9002 says ...").

Usage (from repo root):
    python tools/build_queues.py
    python tools/build_queues.py --annotate-needs-verify   # add/refresh the column
                                  # in the existing work/queues/*.csv (items untouched)
"""
from __future__ import annotations

import bisect
import csv
import glob
import math
import os
import random
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import dig_data  # noqa: E402
import grid_review  # noqa: E402

ROOT = TOOLS.parent
WORK = ROOT / "work"
QDIR = WORK / "queues"
CORPUS = ROOT / "joes-samples"
BATCH1 = CORPUS / "batch 1"
LABELED = WORK / "labeled"
VALIDATION = WORK / "validation_labeled"
STAGED = WORK / "derived_only_review"
DERIVE = WORK / "b3_derive.csv"
DRIFT = WORK / "roi_drift.csv"
EVALPOOL = WORK / "eval_9002_labeledpool.csv"

B3_START = "20260815"
HOLDOUT_DAYS = ["20260818", "20260825", "20260901", "20260909",
                "20260916", "20260923", "20260929"]
FULL_DAY_FRAMES = 360          # one frame every 4 min
SEED = 20261003

FRAMES_PER_HOLDOUT_DAY = 25
HOLDOUT_BUCKET_MIX = {"flash": 0.50, "transition": 0.15, "day": 0.35}
HOLDOUT_SCREEN_MIX = {"reading": 0.60, "test0": 0.40 / 3, "test8": 0.40 / 3, "dash": 0.40 / 3}

OVERAGE = 1.15                 # queue 15% extra for expected rejects
YIELD = 1.0 / OVERAGE          # expected accepted fraction, used in projections
ERR_QUOTA_NIGHT = 60
ERR_QUOTA_DAY = 40
TARGET = {"flash": 80, "transition": 40}
TARGET_N = 40                  # N cells (dig2-4, from the dash screen), any bucket
DAY_MAX = 60
DAY_MAX_N = 40
TEST_SHARE = 0.30              # max test-screen share of a class-0/8 cell's target
TEST_ONLY_FRAC = 0.05          # (pos, class) with < 5% non-test supply is test-only: exempt
DASH_ERR_CAP = 15              # dash dig5/dig6 crops per err cell
DASH_FILL_SHARE = 0.20         # dash dig5/dig6 share of a fill cell's queue
HIGH_CONF = 0.85               # eval_9002 disagreement threshold for the legacy queue
SKIP_HOST_RATIO = 1.3
SKIP_GUARD = 0.10
# known consistent misreads: 9 read as 8 (hidden 9 sits in the lower run); 5 read as 6
# and (night, pre-09-05) 8 read as 9 (hidden value sits in the upper run)
SKIP_PRIOR = {9: "lower", 5: "upper", 8: "upper"}

NIGHT = ("flash", "transition")
BUCKETS = ("flash", "transition", "day")
CLASSES = tuple("0123456789") + ("N",)
VALID = {2: set("058") | {"N"}, 3: set("0789") | {"N"},
         4: set("0123456789") | {"N"}, 5: set("0123456789"), 6: set("0123456789")}
NIGHT_FRONT = [(6, "1"), (3, "9"), (5, "8"), (6, "8"), (3, "0")]
DAY_FRONT = [(6, "9"), (6, "5")]
DIV = {2: 10000, 3: 1000, 4: 100, 5: 10, 6: 1}
NAME_RE = re.compile(r"^(10|[0-9]|N)_main_dig(\d)_(\d{8}-\d{6})\.jpg$", re.I)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass
class B3:
    stamp: str
    day: str
    hour: int
    pos: int
    path: str
    model: str
    truth: str
    status: str
    screen: str
    est: int | None
    bucket: str
    bright: float
    contrast: float
    drift: bool = False
    proposed: str = ""
    kind: str = ""             # pinned | dashfill | err | none
    src_kind: str = ""         # reading | test | dash | other
    context: str = ""
    skip_note: str = ""
    skip_host: bool = False

    @property
    def iid(self) -> str:
        return f"{self.stamp}_dig{self.pos}"

    @property
    def cell(self) -> tuple:
        return (self.pos, self.proposed, self.bucket)


def norm(lab: str) -> str:
    lab = lab.strip().upper()
    return "N" if lab in ("10", "N") else lab


def index_dir(pattern: str, recursive: bool = False) -> dict:
    """(stamp, pos) -> (label, path) for every well-named crop matching the glob."""
    out = {}
    for f in glob.glob(pattern, recursive=recursive):
        m = NAME_RE.match(os.path.basename(f))
        if m:
            out[(m.group(3), int(m.group(2)))] = (norm(m.group(1)), os.path.abspath(f))
    return out


def load_b3() -> list[B3]:
    edge = set()
    with DRIFT.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if "edge" in (r["flag_reason"] or ""):
                edge.add((r["stamp"], int(r["pos"][3:])))
    out = []
    with DERIVE.open(newline="") as fh:
        for r in csv.DictReader(fh):
            pos = int(r["pos"][3:])
            out.append(B3(
                stamp=r["stamp"], day=r["day"], hour=int(r["hour"]), pos=pos,
                path=r["path"], model=norm(r["model_label"]), truth=r["truth"],
                status=r["status"], screen=r["screen"],
                est=int(r["reading_est"]) if r["reading_est"] else None,
                bucket=r["bucket"], bright=float(r["frame_brightness"]),
                contrast=float(r["contrast"]), drift=(r["stamp"], pos) in edge))
    return out


def ts(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").timestamp() / 60.0


def skip_inference(crops: list[B3]) -> tuple[dict, set, set, list]:
    """-> (inferred {stamp: (digit, note)}, host stamps, guard stamps, log)."""
    rd = {}
    for c in crops:
        if c.screen == "reading" and c.est is not None and c.day >= B3_START:
            rd.setdefault(c.stamp, c.est)
    all_stamps = sorted({c.stamp for c in crops if c.day >= B3_START})
    st = sorted(rd)
    runs = []  # [value, first_stamp, last_stamp]
    for s in st:
        if runs and runs[-1][0] == rd[s]:
            runs[-1][2] = s
        else:
            runs.append([rd[s], s, s])
    inferred, host, guard, log = {}, set(), set(), []
    for i in range(len(runs) - 2):
        a, b, c = runs[i], runs[i + 1], runs[i + 2]
        if b[0] - a[0] != 2:
            continue
        v = a[0]
        h = (v + 1) % 10
        if h == 0:
            continue  # hidden value would change dig5 too; not handled
        d_lo = ts(b[1]) - ts(a[1])
        d_up = ts(c[1]) - ts(b[1])
        if d_lo >= SKIP_HOST_RATIO * d_up:
            side, t0, t1 = "lower", ts(a[1]), ts(b[1])
        elif d_up >= SKIP_HOST_RATIO * d_lo:
            side, t0, t1 = "upper", ts(b[1]), ts(c[1])
        else:
            side = "unresolved"
        if side != "unresolved" and SKIP_PRIOR.get(h, side) != side:
            side = "contradicts-prior"
        if side not in ("lower", "upper"):
            # no inference; keep both runs' dig6 crops out of the fill queues
            lo_i = bisect.bisect_left(all_stamps, a[1])
            hi_i = bisect.bisect_left(all_stamps, c[1])
            host.update(all_stamps[lo_i:hi_i])
            log.append((v, h, a[1], side, d_lo, d_up))
            continue
        mid = (t0 + t1) / 2.0
        band = SKIP_GUARD * (t1 - t0)
        lo_i = bisect.bisect_left(all_stamps, a[1] if side == "lower" else b[1])
        hi_i = bisect.bisect_left(all_stamps, b[1] if side == "lower" else c[1])
        for s in all_stamps[lo_i:hi_i]:
            t = ts(s)
            host.add(s)
            if abs(t - mid) < band:
                guard.add(s)
            hidden = (t > mid) if side == "lower" else (t < mid)
            if hidden:
                inferred[s] = (str(h), f"skip {v}->{v + 2}: inferred {v + 1} ({side} run)")
        log.append((v, h, a[1], side, d_lo, d_up))
    return inferred, host, guard, log


def annotate(crops: list[B3], inferred: dict, host: set, guard: set) -> None:
    for c in crops:
        m = f"model saw {c.model}"
        if c.screen in ("test0", "test8"):
            c.src_kind = "test"
            c.proposed = c.screen[-1]
            c.context = f"screen {c.screen}; {m}; {c.status}"
            c.kind = "pinned" if c.status == "agree" else "err"
        elif c.screen == "dash":
            c.src_kind = "dash"
            c.proposed = {5: "0", 6: "9"}.get(c.pos, "N")
            if c.pos in (5, 6):
                c.context = f"dash screen (dig5/6 believed 0 9, unverified); {m}"
                c.kind = "dashfill" if c.model == c.proposed else "err"
            else:
                c.context = f"dash screen; {m}; {c.status}"
                c.kind = "pinned" if c.status == "agree" else "err"
        elif c.screen == "reading" and c.est is not None:
            c.src_kind = "reading"
            c.proposed = "5" if c.pos == 2 else str(c.est // DIV[c.pos] % 10)
            parts = [f"reading_est {c.est}"]
            if c.pos == 6 and c.stamp in host:
                c.skip_host = True
                if c.stamp in inferred:
                    c.proposed, c.skip_note = inferred[c.stamp]
                    parts.append(c.skip_note)
                else:
                    parts.append("skip-host run")
            parts += [m, c.status]
            c.context = "; ".join(parts)
            if c.status == "uncertain":
                c.kind = "none"
            elif c.model != c.proposed:
                c.kind = "none" if c.stamp in guard and c.skip_host else "err"
            elif c.skip_host:
                c.kind = "none"   # unpinned in a skip run: keep out of fill
            elif c.status in ("agree", "agree_soft"):
                c.kind = "pinned"
            else:
                c.kind = "none"
        else:
            c.src_kind = "other"
            c.proposed = c.model
            c.context = f"screen {c.screen}; {m}; {c.status}"
            c.kind = "none"


# --------------------------------------------------------------------------
# existing corpus
# --------------------------------------------------------------------------


def corpus_counts():
    corp = index_dir(str(CORPUS / "*.jpg"))
    known = {}
    for src in (index_dir(str(LABELED / "**" / "*.jpg"), True), index_dir(str(BATCH1 / "*.jpg")),
                index_dir(str(VALIDATION / "**" / "*.jpg"), True), corp):
        for (st, p), (lab, _) in src.items():
            known.setdefault(st, {})[f"dig{p}"] = lab

    def screen_of(st: str) -> str:
        labs = known.get(st, {})
        kind = dig_data.classify_screen(labs)
        if kind != "uncertain":
            return kind
        if labs.get("dig2") in ("0", "8"):
            return "test" + labs["dig2"]
        if labs.get("dig3") == "0":
            return "test0"
        return kind

    # screen of the 9002 corpus crops as eval_dig_model typed them (legacy vocabulary)
    ev = {}
    with (WORK / "eval_9002_corpus.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            ev[os.path.basename(r["path"])] = r["screen"]
    n, ntest, info = Counter(), Counter(), Counter()
    for (st, p), (lab, path) in corp.items():
        b = dig_data.bucket_for_hour(int(st[9:11]))
        n[(p, lab, b)] += 1
        scr = ev.get(os.path.basename(path))
        if scr is not None:
            is_test = scr in ("zeros", "test8")
            info["from_eval_9002_corpus"] += 1
        else:
            kind = screen_of(st)
            if kind != "uncertain":
                is_test = kind in ("test0", "test8")
                info["from_frame_labels"] += 1
            else:
                # derived-only 0/8 night crops whose frame is otherwise unknown were
                # pinned from the 00000 / 88888 screens (HANDOFF s5d): count as test
                is_test = lab in ("0", "8")
                info["assumed_test" if is_test else "unknown_nontest"] += 1
        if is_test:
            ntest[(p, lab, b)] += 1
    return corp, n, ntest, info


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def _spread(cands: list[B3], n: int, rng: random.Random) -> list[B3]:
    """Round-robin across days, and within each day across contrast quantiles."""
    if n <= 0:
        return []
    if len(cands) <= n:
        return list(cands)
    nb = 4 if len(cands) >= 16 else (2 if len(cands) >= 4 else 1)
    vals = sorted(c.contrast for c in cands)
    edges = [vals[int(len(vals) * k / nb)] for k in range(1, nb)]
    by_day: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for c in sorted(cands, key=lambda c: (c.stamp, c.pos)):
        by_day[c.day][bisect.bisect_right(edges, c.contrast)].append(c)
    days = sorted(by_day)
    rng.shuffle(days)
    seqs = []
    for k, d in enumerate(days):
        bins = by_day[d]
        for lst in bins.values():
            rng.shuffle(lst)
        order = [(k + j) % nb for j in range(nb)]
        seq = []
        while any(bins.get(q) for q in order):
            for q in order:
                if bins.get(q):
                    seq.append(bins[q].pop())
        seqs.append(seq)
    out = []
    r = 0
    while len(out) < n:
        progressed = False
        for seq in seqs:
            if r < len(seq):
                out.append(seq[r])
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            break
        r += 1
    return out


def spread_pick(cands: list[B3], n: int, rng: random.Random) -> list[B3]:
    clean = [c for c in cands if not c.drift]
    out = _spread(clean, n, rng)
    if len(out) < n:
        out += _spread([c for c in cands if c.drift], n - len(out), rng)
    return out


def pick_sources(cands: list[B3], n: int, plan: list[tuple[str, float]],
                 rng: random.Random) -> list[B3]:
    """Fill n from source kinds in plan order, each up to its cap."""
    out: list[B3] = []
    for kind, cap in plan:
        room = min(n - len(out), cap)
        if room <= 0:
            continue
        pool = [c for c in cands if c.src_kind == kind]
        out += spread_pick(pool, int(room), rng)
    return out


# --------------------------------------------------------------------------
# queue rows
# --------------------------------------------------------------------------


def preflag(drift: bool, disagrees: bool) -> str:
    return "|".join(x for x, on in (("drift", drift), ("model_disagrees", disagrees)) if on)


def needs_verify_b3(status: str, model: str, proposed: str, skip_inferred: bool,
                    screen: str = "", pos: int | None = None) -> int:
    """1 = the proposed label is genuinely uncertain (see module docstring)."""
    if not (status not in ("agree", "agree_soft") or norm(model) != norm(proposed)
            or bool(skip_inferred)):
        return 0
    if status == "uncertain" or skip_inferred:
        return 1
    if screen in ("test0", "test8") and status == "disagree":
        return 0                      # pinned by the 00000 / 88888 screen
    if screen == "reading" and pos == 3 and status == "disagree":
        return 0                      # thousands digit pinned by the monotone series
    if pos == 2 or (screen == "dash" and pos in (2, 3, 4)):
        return 0                      # dig2 is structural; dash dig2-4 are N
    return 1


def needs_verify_legacy(context: str) -> int:
    return int("9002 says" in (context or ""))


def b3_row(queue: str, c: B3, group: str = "") -> dict:
    return {"queue": queue, "item_id": c.iid, "stamp": c.stamp, "pos": c.pos,
            "src": c.path, "proposed": c.proposed, "bucket": c.bucket,
            "brightness": f"{c.bright:.2f}", "context": c.context,
            "preflag": preflag(c.drift, c.model != c.proposed), "group": group,
            "needs_verify": needs_verify_b3(c.status, c.model, c.proposed, bool(c.skip_note),
                                            c.screen, c.pos)}


def write_queue(name: str, rows: list[dict]) -> Path:
    QDIR.mkdir(parents=True, exist_ok=True)
    p = QDIR / f"{name}.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=grid_review.QUEUE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return p


# --------------------------------------------------------------------------
# queue 0: holdout
# --------------------------------------------------------------------------


def largest_remainder(total: int, weights: dict, rot: int = 0) -> dict:
    keys = list(weights)
    raw = {k: total * weights[k] / sum(weights.values()) for k in keys}
    out = {k: int(math.floor(raw[k])) for k in keys}
    left = total - sum(out.values())
    order = sorted(keys, key=lambda k: (-(raw[k] - out[k]), (keys.index(k) + rot) % len(keys)))
    for k in order[:left]:
        out[k] += 1
    return out


def build_holdout(crops: list[B3]) -> tuple[list[dict], list[str], dict]:
    frames: dict[str, dict[int, B3]] = defaultdict(dict)
    for c in crops:
        frames[c.stamp][c.pos] = c
    nday = Counter(s[:8] for s in frames)
    days = []
    for d in HOLDOUT_DAYS:
        if nday[d] >= FULL_DAY_FRAMES:
            days.append(d)
            continue
        # substitute the nearest full non-holdout neighbour day
        dt = datetime.strptime(d, "%Y%m%d").toordinal()
        cands = sorted((abs(datetime.strptime(x, "%Y%m%d").toordinal() - dt), x)
                       for x in nday if nday[x] >= FULL_DAY_FRAMES
                       and x not in HOLDOUT_DAYS and x not in days)
        print(f"holdout: {d} has {nday[d]} frames -> substituting {cands[0][1]}")
        days.append(cands[0][1])

    stats = {"by_bucket": Counter(), "by_screen": Counter(), "hours": Counter()}
    per_day_frames: dict[str, list[str]] = {}
    for di, d in enumerate(days):
        rng = random.Random(f"{SEED}|holdout|{d}")
        ok = {s: f for s, f in frames.items() if s[:8] == d and len(f) == 5
              and next(iter(f.values())).screen in HOLDOUT_SCREEN_MIX
              and not any(c.drift for c in f.values())}
        strata: dict[tuple, list[str]] = defaultdict(list)
        for s, f in ok.items():
            c = f[2]
            strata[(c.bucket, c.screen)].append(s)
        nb = largest_remainder(FRAMES_PER_HOLDOUT_DAY, HOLDOUT_BUCKET_MIX, di)
        chosen: list[str] = []
        short = 0
        for b in BUCKETS:
            ns = largest_remainder(nb[b], HOLDOUT_SCREEN_MIX, di + BUCKETS.index(b))
            for scr, k in ns.items():
                pool = strata.get((b, scr), [])
                got = hour_spread(pool, k, rng)
                short += k - len(got)
                chosen += got
        if short:  # top up from any unused frame of the day, reading screens first
            rest = [s for s in ok if s not in chosen]
            rest.sort(key=lambda s: (ok[s][2].screen != "reading", s))
            chosen += hour_spread(rest, short, rng)
        rng.shuffle(chosen)
        per_day_frames[d] = chosen
        for s in chosen:
            stats["by_bucket"][ok[s][2].bucket] += 1
            stats["by_screen"][ok[s][2].screen] += 1
            stats["hours"][int(s[9:11])] += 1

    rows = []
    r = 0
    while any(r < len(v) for v in per_day_frames.values()):
        for d in days:
            lst = per_day_frames[d]
            if r < len(lst):
                s = lst[r]
                for p in (2, 3, 4, 5, 6):
                    rows.append(b3_row("holdout", frames[s][p], group=s))
        r += 1
    return rows, days, stats


def hour_spread(stamps: list[str], k: int, rng: random.Random) -> list[str]:
    if k <= 0 or not stamps:
        return []
    by_h: dict[int, list[str]] = defaultdict(list)
    for s in stamps:
        by_h[int(s[9:11])].append(s)
    hours = sorted(by_h)
    rng.shuffle(hours)
    for h in hours:
        rng.shuffle(by_h[h])
    out = []
    while len(out) < k and any(by_h[h] for h in hours):
        for h in hours:
            if by_h[h] and len(out) < k:
                out.append(by_h[h].pop())
    return out


# --------------------------------------------------------------------------
# queue 1: legacy
# --------------------------------------------------------------------------


def build_legacy(corp: dict) -> tuple[list[dict], dict]:
    labeled = index_dir(str(LABELED / "**" / "*.jpg"), True)
    batch1 = index_dir(str(BATCH1 / "*.jpg"))
    staged = index_dir(str(STAGED / "*.jpg"))
    items: OrderedDict[tuple, dict] = OrderedDict()
    info = {"derived_only": 0, "staged": len(staged), "staged_not_in_corpus": 0,
            "eval_hi": 0, "eval_resolved": [], "eval_in_corpus": 0}

    def bucket(st: str) -> str:
        return dig_data.bucket_for_hour(int(st[9:11]))

    for key in sorted(k for k in corp if k not in labeled and k not in batch1):
        lab, path = corp[key]
        ctx = "derived-only in corpus"
        if key in staged:
            ctx += "; staged cull (work/derived_only_review)"
        if key[0].startswith("1970"):
            ctx += "; 1970 sentinel (provenance lost)"
        items[key] = {"src": path, "proposed": lab, "ctx": ctx, "flag": ""}
        info["derived_only"] += 1
    for key in staged:
        if key not in items:
            lab, path = staged[key]
            items[key] = {"src": path, "proposed": lab, "flag": "",
                          "ctx": "staged cull (work/derived_only_review); not in corpus"}
            info["staged_not_in_corpus"] += 1

    with EVALPOOL.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["label"] != r["pred"] and float(r["conf"]) > HIGH_CONF]
    rows.sort(key=lambda r: -float(r["conf"]))
    info["eval_hi"] = len(rows)
    for r in rows:
        key = (r["frame"], int(r["position"]))
        pred = "N" if r["pred"] in ("10", "N") else r["pred"]
        # the CSV's path/label can be stale: find the file as it is now
        if key in labeled:
            lab, path = labeled[key]
        elif os.path.isfile(r["path"]):
            lab, path = norm(r["label"]), r["path"]
        elif key in corp:
            lab, path = corp[key]
        else:
            raise SystemExit(f"legacy: cannot locate {key}")
        if lab == pred:
            info["eval_resolved"].append(f"{key[0]} dig{key[1]}: csv label {r['label']}, "
                                         f"file now {os.path.relpath(path, ROOT)} (= 9002's {pred})")
            continue
        ctx = f"9002 says {pred} @{float(r['conf']):.2f}; label {lab}; screen {r['screen']}"
        if key in corp:
            ctx += f"; in corpus as {corp[key][0]}"
            info["eval_in_corpus"] += 1
        if key in items:
            items[key]["ctx"] += "; " + ctx
        else:
            items[key] = {"src": path, "proposed": lab, "ctx": ctx, "flag": "model_disagrees"}

    out = []
    for (st, p), it in items.items():
        out.append({"queue": "legacy", "item_id": f"{st}_dig{p}", "stamp": st, "pos": p,
                    "src": it["src"], "proposed": it["proposed"], "bucket": bucket(st),
                    "brightness": f"{grid_review.image_brightness(it['src']):.2f}",
                    "context": it["ctx"], "preflag": it["flag"], "group": "",
                    "needs_verify": needs_verify_legacy(it["ctx"])})
    return out, info


# --------------------------------------------------------------------------
# in-place needs_verify annotation of existing queues
# --------------------------------------------------------------------------


QUEUE_NAMES = ("holdout", "legacy", "night_err", "day_err", "night_fill", "day_fill")


def read_queue(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or []), list(rd)


def rewrite_queue(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Rewrite a queue CSV with the same rows in the same order (columns may be
    added); written to a temp file and swapped in, then re-read to check the
    item_id order is unchanged."""
    before = [r["item_id"] for r in rows]
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    _, back = read_queue(tmp)
    if [r["item_id"] for r in back] != before:
        raise SystemExit(f"{path}: item order changed while rewriting; left {tmp}")
    os.replace(tmp, path)


def annotate_needs_verify(qdir: Path = QDIR) -> dict:
    """Add/refresh needs_verify in every queue CSV of qdir. Items and order are
    unchanged; the b3 proposal must still match what annotate() derives (else the
    queue is stale and we stop). -> {queue: (n, n_needs_verify)}"""
    crops = load_b3()
    inferred, host, guard, _ = skip_inference(crops)
    annotate(crops, inferred, host, guard)
    by = {c.iid: c for c in crops}
    out = {}
    for name in QUEUE_NAMES:
        path = qdir / f"{name}.csv"
        if not path.is_file():
            continue
        fields, rows = read_queue(path)
        mism = []
        for r in rows:
            if name == "legacy":
                r["needs_verify"] = needs_verify_legacy(r["context"])
                continue
            c = by.get(r["item_id"])
            if c is None:
                raise SystemExit(f"{name}: {r['item_id']} not in {DERIVE}")
            if norm(c.proposed) != norm(r["proposed"]):
                mism.append(f"{r['item_id']}: queue {r['proposed']} vs derived {c.proposed}")
            skip = bool(c.skip_note) or "inferred" in r["context"]
            r["needs_verify"] = needs_verify_b3(c.status, c.model, r["proposed"], skip,
                                                c.screen, c.pos)
        if mism:
            raise SystemExit(f"{name}: {len(mism)} proposal(s) differ from the current "
                             f"derivation, e.g. {mism[:3]}")
        new_fields = fields + [f for f in grid_review.QUEUE_FIELDS if f not in fields]
        rewrite_queue(path, rows, new_fields)
        out[name] = (len(rows), sum(int(r["needs_verify"]) for r in rows))
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    if "--annotate-needs-verify" in sys.argv[1:]:
        res = annotate_needs_verify()
        for name, (n, nv) in res.items():
            print(f"{name:<11} {n:>5} items, needs_verify=1: {nv}")
        return 0
    crops_all = load_b3()
    validation = index_dir(str(VALIDATION / "**" / "*.jpg"), True)
    inferred, host, guard, skiplog = skip_inference(crops_all)
    annotate(crops_all, inferred, host, guard)
    corp, cnt, cnt_test, cinfo = corpus_counts()

    # ---------------- queue 0
    hold_rows, hold_days, hstats = build_holdout(
        [c for c in crops_all if c.day >= B3_START])
    hold_set = set(hold_days)

    pool = [c for c in crops_all if c.day >= B3_START and c.day not in hold_set
            and (c.stamp, c.pos) not in validation]
    used: set[str] = set()

    # test-only (pos, class): <5% non-test supply in b3 -> share cap exempt
    supply_scr = defaultdict(Counter)
    for c in pool:
        if c.kind in ("pinned", "err", "dashfill"):
            supply_scr[(c.pos, c.proposed)][c.src_kind] += 1
    test_only = {k for k, v in supply_scr.items()
                 if k[1] in ("0", "8") and v["test"]
                 and (sum(v.values()) - v["test"]) < TEST_ONLY_FRAC * sum(v.values())}

    def target(pos: int, cls: str, b: str) -> int:
        if cls == "N":
            return TARGET_N
        return TARGET[b]

    test_budget: dict[tuple, float] = {}

    def budget(cell: tuple, T: int) -> float:
        pos, cls, b = cell
        if cls not in ("0", "8") or (pos, cls) in test_only:
            return float("inf")
        if cell not in test_budget:
            test_budget[cell] = max(0, round(TEST_SHARE * T * OVERAGE) - cnt_test[cell])
        return test_budget[cell]

    def take(cell, picked):
        n = sum(1 for c in picked if c.src_kind == "test")
        if cell in test_budget and test_budget[cell] != float("inf"):
            test_budget[cell] -= n
        for c in picked:
            used.add(c.iid)

    def night_order(pairs):
        front = [p for p in NIGHT_FRONT if p in pairs]
        rest = sorted((p for p in pairs if p not in front),
                      key=lambda p: (cnt[(p[0], p[1], "flash")] + cnt[(p[0], p[1], "transition")],
                                     p[0], CLASSES.index(p[1])))
        return front + rest

    queued = defaultdict(Counter)   # queue -> cell -> n

    # ---------------- queue 2: night_err
    errs = defaultdict(list)
    for c in pool:
        if c.kind == "err" and c.bucket in NIGHT and c.model != c.proposed:
            errs[c.cell].append(c)
    pairs = {(p, l) for p, l, _ in errs}
    q2 = []
    for pos, cls in night_order(pairs):
        for b in NIGHT:
            cell = (pos, cls, b)
            cands = errs.get(cell, [])
            if not cands:
                continue
            rng = random.Random(f"{SEED}|night_err|{cell}")
            plan = [("dash", DASH_ERR_CAP), ("reading", 1e9),
                    ("test", budget(cell, target(pos, cls, b))), ("other", 0)]
            got = pick_sources(cands, ERR_QUOTA_NIGHT, plan, rng)
            take(cell, got)
            queued["night_err"][cell] += len(got)
            q2 += [b3_row("night_err", c) for c in got]

    # ---------------- queue 4: night_fill (computed before q3 so day targets can use it)
    pinned = defaultdict(list)
    for c in pool:
        if c.kind in ("pinned", "dashfill") and c.iid not in used:
            pinned[c.cell].append(c)
    pairs = {(p, l) for p, l, _ in pinned if l in VALID[p]}
    q4 = []
    fill_short = []
    for pos, cls in night_order(pairs):
        for b in NIGHT:
            cell = (pos, cls, b)
            T = target(pos, cls, b)
            have = cnt[cell] + YIELD * queued["night_err"][cell]
            need = T - have
            if need <= 0:
                continue
            n = math.ceil(need * OVERAGE)
            cands = pinned.get(cell, [])
            rng = random.Random(f"{SEED}|night_fill|{cell}")
            dash_cap = round(DASH_FILL_SHARE * n) if (pos, cls) in ((5, "0"), (6, "9")) else (
                1e9 if cls == "N" else 0)
            plan = [("dash", dash_cap), ("reading", 1e9), ("test", budget(cell, T))]
            got = pick_sources(cands, n, plan, rng)
            take(cell, got)
            queued["night_fill"][cell] += len(got)
            if len(got) < n:
                fill_short.append((cell, n, len(got)))
            q4 += [b3_row("night_fill", c) for c in got]

    def proj(cell, queues):
        return cnt[cell] + YIELD * sum(queued[q][cell] for q in queues)

    # day target per (pos, class): day >= projected flash, capped
    def day_target(pos, cls):
        fl = proj((pos, cls, "flash"), ("night_err", "night_fill"))
        return min(DAY_MAX_N if cls == "N" else DAY_MAX, math.ceil(fl))

    # ---------------- queue 3: day_err
    derrs = defaultdict(list)
    for c in pool:
        if c.kind == "err" and c.bucket == "day" and c.model != c.proposed and c.iid not in used:
            derrs[c.cell].append(c)
    pairs = {(p, l) for p, l, _ in derrs}
    front = [p for p in DAY_FRONT if p in pairs]
    order = front + sorted((p for p in pairs if p not in front),
                           key=lambda p: (cnt[(p[0], p[1], "day")], p[0], CLASSES.index(p[1])))
    q3 = []
    for pos, cls in order:
        cell = (pos, cls, "day")
        rng = random.Random(f"{SEED}|day_err|{cell}")
        plan = [("dash", DASH_ERR_CAP), ("reading", 1e9),
                ("test", budget(cell, max(day_target(pos, cls), 1))), ("other", 0)]
        got = pick_sources(derrs[cell], ERR_QUOTA_DAY, plan, rng)
        take(cell, got)
        queued["day_err"][cell] += len(got)
        q3 += [b3_row("day_err", c) for c in got]

    # ---------------- queue 5: day_fill
    dpinned = defaultdict(list)
    for c in pool:
        if c.kind in ("pinned", "dashfill") and c.bucket == "day" and c.iid not in used:
            dpinned[(c.pos, c.proposed)].append(c)
    pairs = sorted({p for p in dpinned if p[1] in VALID[p[0]]},
                   key=lambda p: (cnt[(p[0], p[1], "day")] - day_target(*p), p[0],
                                  CLASSES.index(p[1])))
    q5 = []
    for pos, cls in pairs:
        cell = (pos, cls, "day")
        D = day_target(pos, cls)
        need = D - (cnt[cell] + YIELD * queued["day_err"][cell])
        if need <= 0:
            continue
        n = math.ceil(need * OVERAGE)
        rng = random.Random(f"{SEED}|day_fill|{cell}")
        dash_cap = round(DASH_FILL_SHARE * n) if (pos, cls) in ((5, "0"), (6, "9")) else (
            1e9 if cls == "N" else 0)
        plan = [("dash", dash_cap), ("reading", 1e9), ("test", budget(cell, D))]
        got = pick_sources(dpinned[(pos, cls)], n, plan, rng)
        take(cell, got)
        queued["day_fill"][cell] += len(got)
        if len(got) < n:
            fill_short.append((cell, n, len(got)))
        q5 += [b3_row("day_fill", c) for c in got]

    # ---------------- queue 1: legacy
    q1, linfo = build_legacy(corp)

    queues = OrderedDict([("holdout", hold_rows), ("legacy", q1), ("night_err", q2),
                          ("day_err", q3), ("night_fill", q4), ("day_fill", q5)])
    paths = {name: write_queue(name, rows) for name, rows in queues.items()}
    (QDIR / "holdout_days.txt").write_text("\n".join(hold_days) + "\n", encoding="utf-8")

    # ---------------- assertions
    all_ids = Counter(r["item_id"] for rows in queues.values() for r in rows)
    dups = [k for k, v in all_ids.items() if v > 1]
    assert not dups, f"duplicate item_ids across queues: {dups[:10]}"
    for name, rows in queues.items():
        if name == "holdout":
            assert all(r["stamp"][:8] in hold_set for r in rows)
            continue
        bad = [r["item_id"] for r in rows if r["stamp"][:8] in hold_set]
        assert not bad, f"{name}: holdout-day crops {bad[:5]}"
    missing = [r["src"] for rows in queues.values() for r in rows if not os.path.isfile(r["src"])]
    assert not missing, f"missing src files: {missing[:5]}"
    vkeys = {f"{s}_dig{p}" for s, p in validation}
    leak = [k for k in all_ids if k in vkeys]
    assert not leak, f"validation_labeled crops queued: {leak[:5]}"
    vnames = {os.path.basename(v[1]) for v in validation.values()}
    leak2 = [r["src"] for rows in queues.values() for r in rows
             if os.path.basename(r["src"]) in vnames]
    assert not leak2, f"validation file names queued: {leak2[:5]}"
    hold_frames = Counter(r["group"] for r in hold_rows)
    assert all(v == 5 for v in hold_frames.values()), "incomplete holdout frame"

    # ---------------- parse with grid_review's loader
    loaded = {}
    for name, p in paths.items():
        items, warns = grid_review.load_queues([p])
        assert not warns, f"{name}: loader warnings {warns[:5]}"
        assert len(items) == len(queues[name]), f"{name}: loader dropped rows"
        loaded[name] = len(items)
    items, warns = grid_review.load_queues(list(paths.values()))
    assert not warns and len(items) == sum(loaded.values())

    # ---------------- summary
    write_summary(queues, loaded, hold_days, hstats, linfo, cnt, cnt_test, queued,
                  fill_short, skiplog, inferred, test_only, crops_all, pool, cinfo)
    return 0


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def write_summary(queues, loaded, hold_days, hstats, linfo, cnt, cnt_test, queued,
                  fill_short, skiplog, inferred, test_only, crops_all, pool, cinfo):
    L = []
    P = L.append
    secs = {"holdout": 2.0}
    P("# Batch-3 review queues\n")
    P(f"Generated {datetime.now():%Y-%m-%d %H:%M} by `tools/build_queues.py` (seed {SEED}). "
      "Review in this order; stopping after any queue leaves a usable corpus.\n")
    P("```")
    P("python tools/grid_review.py --queue work/queues/holdout.csv --frame-mode")
    P("python tools/grid_review.py --queue work/queues/legacy.csv --queue work/queues/night_err.csv "
      "--queue work/queues/day_err.csv --queue work/queues/night_fill.csv --queue work/queues/day_fill.csv")
    P("```\n")
    P("## Queues\n")
    P("| # | queue | crops | mode | est. minutes | cumulative min |")
    P("|---|---|---|---|---|---|")
    cum = 0.0
    for i, (name, rows) in enumerate(queues.items()):
        s = secs.get(name, 1.0)
        m = len(rows) * s / 60
        cum += m
        P(f"| {i} | `{name}.csv` | {len(rows)} | {'frame' if name == 'holdout' else 'grid'} "
          f"| {m:.0f} | {cum:.0f} |")
    P("\nComposition (source screen of each crop; `skip` = dig6 proposal from skip inference):\n")
    byid = {c.iid: c for c in pool}
    for name, rows in queues.items():
        if name in ("holdout", "legacy"):
            continue
        comp = Counter(byid[r["item_id"]].src_kind for r in rows)
        nskip = sum(1 for r in rows if byid[r["item_id"]].skip_note)
        ndis = sum(1 for r in rows if "model_disagrees" in r["preflag"])
        ndrift = sum(1 for r in rows if "drift" in r["preflag"])
        P(f"- `{name}`: {dict(comp)}; skip-inferred {nskip}; model_disagrees {ndis}; drift {ndrift}")
    P(f"\nTotal {sum(len(r) for r in queues.values())} crops, ~{cum:.0f} min "
      "(1 s/crop grid, 2 s/crop frame mode).\n")

    P("## Holdout\n")
    subs = [d for d in hold_days if d not in HOLDOUT_DAYS]
    P(f"Days: {', '.join(hold_days)} "
      + (f"(substituted for short days: {', '.join(subs)}). " if subs
         else "(all full 360-frame days; no substitution needed). ")
      +
      f"{len(queues['holdout']) // 5} frames.\n")
    P(f"- buckets: {dict(hstats['by_bucket'])}")
    P(f"- screens: {dict(hstats['by_screen'])}")
    P(f"- hours covered: {len(hstats['hours'])}/24")
    P("- frames with an edge-drift crop were not eligible; rows are interleaved across days so a "
      "partial review still spans every day.\n")

    P("## Legacy\n")
    P(f"- derived-only corpus crops: {linfo['derived_only']} (of which {linfo['staged']} are the staged "
      f"culls in `work/derived_only_review/`; staged-not-in-corpus: {linfo['staged_not_in_corpus']})")
    P(f"- eval_9002 disagreements with conf > {HIGH_CONF}: {linfo['eval_hi']}; "
      f"{linfo['eval_hi'] - len(linfo['eval_resolved'])} queued, "
      f"{len(linfo['eval_resolved'])} already relabelled to 9002's answer and skipped:")
    for s in linfo["eval_resolved"]:
        P(f"  - {s}")
    P("")

    P("## Skip inference (dig6)\n")
    sides = Counter((h, s) for v, h, st, s, *_ in skiplog)
    P(f"{len(skiplog)} anchor skips of +2; host side by hidden digit: "
      + ", ".join(f"{h}:{s}={n}" for (h, s), n in sorted(sides.items())))
    inf_crops = [c for c in pool if c.pos == 6 and c.skip_note]
    P(f"\n{len(inf_crops)} non-holdout dig6 crops get an inferred proposal; "
      f"{sum(c.model != c.proposed for c in inf_crops)} of them differ from the model's label "
      "(these feed the err queues).\n")

    P("## Projected corpus per cell\n")
    P(f"Existing corpus = `joes-samples/*.jpg` top level ({sum(cnt.values())} crops), light bucket by "
      f"the old hour rule. Screen typing for the test-share cap: {dict(cinfo)}.\n")
    P("Expected accepted counts, assuming 1/1.15 of queued crops survive review. "
      "Columns are cumulative: existing corpus -> +night_err -> +day_err -> +night_fill -> +day_fill. "
      "`t` = existing corpus crops from test screens (classes 0/8).\n")
    stages = ["night_err", "day_err", "night_fill", "day_fill"]
    flagged = []
    for b in BUCKETS:
        P(f"### {b}\n")
        P("| pos | class | corpus (t) | +q2 | +q3 | +q4 | +q5 | queued |")
        P("|---|---|---|---|---|---|---|---|")
        for pos in (2, 3, 4, 5, 6):
            for cls in CLASSES:
                if cls not in VALID[pos]:
                    continue
                cell = (pos, cls, b)
                vals = []
                acc = cnt[cell]
                for q in stages:
                    acc += YIELD * queued[q][cell]
                    vals.append(acc)
                nq = sum(queued[q][cell] for q in stages)
                if cnt[cell] == 0 and nq == 0 and cls == "7" and pos == 3:
                    pass
                t = f" ({cnt_test[cell]})" if cls in ("0", "8") else ""
                P(f"| dig{pos} | {cls} | {cnt[cell]}{t} | " + " | ".join(f"{v:.0f}" for v in vals)
                  + f" | {nq} |")
                if b in NIGHT and vals[-1] < 20:
                    flagged.append((cell, cnt[cell], vals[-1], nq))
        P("")
    P("## Supply-limited night cells (< 20 after all queues)\n")
    if flagged:
        for (pos, cls, b), c0, v, nq in flagged:
            P(f"- dig{pos} class {cls} {b}: corpus {c0} -> {v:.0f} (queued {nq})")
    else:
        P("- none")
    P("")
    if fill_short:
        P("## Fill cells short of their queue size (supply or cap bound)\n")
        for (pos, cls, b), n, got in fill_short:
            P(f"- dig{pos} class {cls} {b}: wanted {n}, queued {got}")
        P("")
    P("## Rules applied\n")
    P(f"- night_err quota {ERR_QUOTA_NIGHT}/cell, day_err {ERR_QUOTA_DAY}/cell; dash dig5/6 "
      f"<= {DASH_ERR_CAP}/err cell; sampled round-robin across days x contrast quartiles.")
    P(f"- night_fill targets flash {TARGET['flash']}, transition {TARGET['transition']} "
      f"(N: {TARGET_N}), counting corpus + {YIELD:.2f} x queued; queue size = shortfall x {OVERAGE}.")
    P(f"- day_fill target = min({DAY_MAX}, projected flash) per (pos, class) (N: min({DAY_MAX_N}, .)).")
    P(f"- test-screen crops <= {TEST_SHARE:.0%} of a class-0/8 cell's target (incl. corpus test crops), "
      "shared across err+fill queues. Exempt (test-only in b3): "
      + ", ".join(f"dig{p} class {c}" for p, c in sorted(test_only)) + ".")
    P("- edge-drift crops (roi_drift flag_reason contains `edge`) carry preflag `drift` and are "
      "taken only after clean crops; plain shift flags ignored.")
    P("- holdout-day crops appear only in holdout.csv; no `work/validation_labeled/` crop appears anywhere; "
      "all queues parse with `grid_review.load_queues`.")
    text = "\n".join(L) + "\n"
    (QDIR / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
