"""Assemble the training corpus from reviewed crops, decorrelating light from class.

Class balance alone is not enough. If `8` arrives almost entirely from flash-only
captures and `9` almost entirely from daylight, the network can read the glare
signature instead of the segments -- a shortcut that scores well in training and
collapses on the meter. So each class is capped not just in total but in how far
its light mix may drift from the corpus average.

    per-class flash allowance = flash_ratio * (day + transition) available

which lets classes that genuinely have daylight coverage bring more night with
them, and stops classes that only exist at night from becoming night-detectors.

Writes to the top level of `joes-samples/`, because `prepare_joe_data.py` globs
`joes-samples/*.jpg` non-recursively -- a subdirectory would be silently ignored.
Existing batch subdirs (and their holdout slices) are left alone.

Usage:
    python tools/build_corpus.py                      # report only
    python tools/build_corpus.py --apply               # write the corpus

Ledger mode (human-reviewed provenance enforced; see the "ledger mode" section):
    python tools/build_corpus.py --from-ledger                 # dry run, plan only
    python tools/build_corpus.py --from-ledger --apply         # write joes-samples/
    python tools/build_corpus.py --from-ledger --holdout-out work/holdout_b3   # + holdout
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

NAME = re.compile(
    r"^(?P<lab>[0-9N])_main_(?P<pos>dig\d)_(?P<stamp>\d{8}-\d{6})\.jpg$", re.I)

TRANSITION = {6, 19}
FLASH = {20, 21, 22, 23, 0, 1, 2, 3, 4, 5}
BUCKETS = ("day", "transition", "flash")


def bucket_for_hour(h: int) -> str:
    if h in TRANSITION:
        return "transition"
    return "flash" if h in FLASH else "day"


def load_exclusions(path: Path) -> set[tuple[str, str]]:
    """`<stamp>,<pos>` per line: crops culled by hand during review.

    Culls are made in the *built* corpus (joes-samples/), but this script rebuilds
    from work/labeled -- so without this list every rebuild silently resurrects
    them. Blank lines and `#` comments are ignored.
    """
    if not path or not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stamp, _, pos = line.partition(",")
        if pos:
            out.add((stamp.strip(), pos.strip().lower()))
    return out


def collect(sources: list[tuple[Path, bool]],
            exclude: set[tuple[str, str]] | None = None
            ) -> list[tuple[str, str, str, Path]]:
    """-> (label, pos, bucket, path). `recurse=False` keeps holdout subdirs out."""
    exclude = exclude or set()
    out, skipped = [], 0
    for root, recurse in sources:
        if not root.exists():
            continue
        it = root.rglob("*.jpg") if recurse else root.glob("*.jpg")
        for p in it:
            m = NAME.match(p.name)
            if not m:
                continue
            stamp, pos = m.group("stamp"), m.group("pos").lower()
            if (stamp, pos) in exclude:
                skipped += 1
                continue
            out.append((
                m.group("lab").upper(),
                pos,
                bucket_for_hour(int(stamp[9:11])),
                p,
            ))
    if skipped:
        print(f"excluded {skipped} hand-culled crop(s)")
    return out


def table(rows, title: str) -> None:
    tab: dict[str, Counter] = defaultdict(Counter)
    for lab, _pos, b, _p in rows:
        tab[lab][b] += 1
    print(f"\n=== {title} ({len(rows)}) ===")
    print(f"{'lbl':>4}{'day':>7}{'trans':>7}{'flash':>7}{'total':>8}{'flash%':>9}")
    for lab in sorted(tab):
        r = tab[lab]
        t = sum(r.values())
        print(f"{lab:>4}{r['day']:>7}{r['transition']:>7}{r['flash']:>7}{t:>8}"
              f"{100*r['flash']/max(t,1):>8.0f}%")
    tot = Counter()
    for r in tab.values():
        tot.update(r)
    n = sum(tot.values())
    print(f"{'ALL':>4}{tot['day']:>7}{tot['transition']:>7}{tot['flash']:>7}{n:>8}"
          f"{100*tot['flash']/max(n,1):>8.0f}%")
    shares = [100 * tab[l]['flash'] / max(sum(tab[l].values()), 1) for l in tab]
    if shares:
        print(f"     flash-share spread across classes: "
              f"{min(shares):.0f}% .. {max(shares):.0f}%  "
              f"(narrow = light carries little class information)")


def select(rows, flash_ratio: float, class_cap: int, n_cap: int, seed: int):
    rng = random.Random(seed)
    by_class: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for lab, pos, b, p in rows:
        by_class[lab][b].append((lab, pos, b, p))

    picked = []
    for lab, buckets in by_class.items():
        day = list(buckets["day"])
        trans = list(buckets["transition"])
        flash = list(buckets["flash"])
        for lst in (day, trans, flash):
            rng.shuffle(lst)

        allow_flash = int(flash_ratio * (len(day) + len(trans)))
        take = day + trans + flash[:allow_flash]

        cap = n_cap if lab == "N" else class_cap
        if len(take) > cap:
            # Trim proportionally so the light mix survives the cap.
            keep: list = []
            for lst in (day, trans, flash[:allow_flash]):
                share = max(1, round(cap * len(lst) / len(take)))
                keep.extend(lst[:share])
            take = keep[:cap]
        picked.extend(take)
    return picked


# ==========================================================================
# ledger mode (--from-ledger)
# ==========================================================================
#
# Every crop in the corpus must have human-reviewed provenance. Two sources count:
#   legacy  : work/labeled/** (recursive) or joes-samples/batch 1/ (top level only),
#             reviewed last cycle, and with no ledger rows at all.
#   ledger  : work/review_ledger.csv, read in two stages (grid_review.load_effective):
#             a crop qualifies only if its screen stage is ok / relabeled / ok_derived
#             AND (needs_verify == 0 OR its label stage is label_ok / label_fixed).
#             Label = the effective final (label-stage final beats the screen final).
#             Queue `holdout` records feed the holdout export instead.
# needs_verify comes from the union of the queue CSVs (--queues-dir); a ledger item in
# no queue counts as 0. Rejections -- screen artifact / drift / illegible, label
# label_unsure, or any crop of a frame with a drift verdict (drift is frame-wide) --
# always exclude and override legacy. A crop with ledger rows that neither qualifies
# nor is rejected (not yet screened, or needs verification) is "pending": left out.
# Crops sitting in joes-samples/ top level that are in NEITHER source (the "derived-
# only" crops) are never read; they only come back via a ledger verdict.

DEFAULT_OUT = Path("joes-samples")
OK_VERDICTS = ("ok", "relabeled", "ok_derived")
REJECT_VERDICTS = ("artifact", "drift", "illegible")
LABEL_OK_VERDICTS = ("label_ok", "label_fixed")
CULL_VERDICTS = REJECT_VERDICTS + ("label_unsure",)
POSITIONS = ("dig2", "dig3", "dig4", "dig5", "dig6")
B3_START = "20260815"            # first capture day of batch 3
TEST_SCREENS = ("test0", "test8")
SHARE_CLASSES = ("0", "8")       # classes the test-screen share cap applies to
TEST_ONLY_FRAC = 0.05            # (pos, class) with < 5% non-test crops is test-only: exempt
LEDGER_NAME = re.compile(
    r"^(?P<lab>10|[0-9N])_main_(?P<pos>dig\d)_(?P<stamp>\d{8}-\d{6})\.jpg$", re.I)
ITEM_ID = re.compile(r"^(\d{8}-\d{6})_dig(\d)$")
MANIFEST_FIELDS = ["file", "src", "label", "pos", "stamp", "bucket", "screen",
                   "provenance", "verdict", "queue", "in_legacy", "label_verdict"]


class CorpusError(Exception):
    """A hard-error condition; carries the offender lines."""

    def __init__(self, title: str, offenders: list[str]):
        super().__init__(title)
        self.title, self.offenders = title, offenders


@dataclass
class Crop:
    stamp: str
    pos: str
    label: str
    src: Path
    bucket: str = ""
    screen: str = ""
    provenance: str = "legacy"      # legacy | ledger
    verdict: str = ""
    queue: str = ""
    in_legacy: bool = False
    b3: bool = False
    label_verdict: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.stamp, self.pos)

    @property
    def day(self) -> str:
        return self.stamp[:8]

    @property
    def out_name(self) -> str:
        return f"{self.label}_main_{self.pos}_{self.stamp}.jpg"


def norm_lab(s) -> str:
    """'7'->'7'; '10'/'n'/'nan'/'-' -> 'N'; ValueError on anything else."""
    t = str(s).strip().upper()
    if t.endswith(".0") and t[:-2].isdigit():
        t = t[:-2]
    if t in ("10", "NAN", "-"):
        t = "N"
    if len(t) != 1 or t not in "0123456789N":
        raise ValueError(f"bad label {s!r}")
    return t


def norm_posname(s) -> str:
    t = str(s).strip().lower()
    return t if t.startswith("dig") else f"dig{int(t)}"


def _under(path: Path, root: Path) -> bool:
    try:
        return root.resolve() in path.resolve().parents
    except OSError:
        return False


def load_holdout_days(path: Path | None) -> tuple[set[str], bool]:
    """One YYYYMMDD (or YYYY-MM-DD, or a full stamp) per line; `#` comments."""
    if not path or not Path(path).is_file():
        return set(), False
    days = set()
    for line in Path(path).read_text(encoding="utf8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", line)
        if m:
            days.add("".join(m.groups()))
    return days, True


def load_b3_derive(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(stamp, pos) -> (bucket, screen) from work/b3_derive.csv."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    if not path or not Path(path).is_file():
        return out
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[(r["stamp"], r["pos"].lower())] = (r["bucket"], r["screen"])
    return out


def load_ledger_eff(path: Path) -> dict[tuple[str, str], dict]:
    """grid_review.load_effective -> {(stamp, 'digN'): two-stage record}.

    Record keys: the ledger columns (of the screen row, else the label row), plus
    screen_verdict / label_verdict / rejected / final (effective label) and verdict
    (= rejected reason, else screen verdict, else label verdict)."""
    tools = str(Path(__file__).resolve().parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from grid_review import load_effective
    out = {}
    for item_id, r in load_effective(path).items():
        m = ITEM_ID.match(item_id)
        if m:
            stamp, pos = m.group(1), f"dig{m.group(2)}"
        else:
            stamp, pos = (r.get("stamp") or "").strip(), norm_posname(r.get("pos"))
        out[(stamp, pos)] = dict(r, _stamp=stamp, _pos=pos)
    return out


def load_needs_verify(qdir: Path | None) -> tuple[dict[tuple[str, str], int], list[str]]:
    """Union of needs_verify over qdir/*.csv -> ({(stamp, 'digN'): 0/1}, warnings).
    An item in several queues needs verification if any says so."""
    out: dict[tuple[str, str], int] = {}
    warns: list[str] = []
    if not qdir or not Path(qdir).is_dir():
        warns.append(f"queues dir {qdir} not found: every ledger item counts as needs_verify=0")
        return out, warns
    for p in sorted(Path(qdir).glob("*.csv")):
        with p.open(newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            if "item_id" not in (rd.fieldnames or []):
                continue
            if "needs_verify" not in (rd.fieldnames or []):
                warns.append(f"{p.name} has no needs_verify column: its items count as 0")
            for r in rd:
                m = ITEM_ID.match((r.get("item_id") or "").strip())
                if not m:
                    continue
                k = (m.group(1), f"dig{m.group(2)}")
                nv = 1 if (r.get("needs_verify") or "").strip().lower() in (
                    "1", "true", "yes", "y") else 0
                out[k] = max(out.get(k, 0), nv)
    return out, warns


def ledger_status(e: dict, nv: int) -> tuple[str, str]:
    """-> ('ok' | 'rejected' | 'pending', reason) for one effective ledger record."""
    if e.get("rejected"):
        return "rejected", e["rejected"]
    if e.get("screen_verdict") not in OK_VERDICTS:
        return "pending", "not screened"
    if nv and e.get("label_verdict") not in LABEL_OK_VERDICTS:
        return "pending", "needs verification"
    return "ok", ""


def ledger_label(e: dict) -> str:
    return norm_lab((e.get("final") or "").strip() or (e.get("proposed") or "").strip())


def collect_legacy(labeled: Path, batch1: Path):
    """-> ({(stamp,pos): (label, path)}, n_unparseable, n_duplicates)"""
    out: dict[tuple[str, str], tuple[str, Path]] = {}
    bad = dup = 0
    for root, recurse in ((labeled, True), (batch1, False)):
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.jpg") if recurse else root.glob("*.jpg")):
            m = LEDGER_NAME.match(p.name)
            if not m:
                bad += 1
                continue
            k = (m.group("stamp"), m.group("pos").lower())
            if k in out:
                dup += 1
                continue
            out[k] = (norm_lab(m.group("lab")), p)
    return out, bad, dup


def scan_keys(root: Path, recurse: bool = True) -> set[tuple[str, str]]:
    out = set()
    if root and Path(root).exists():
        for p in (Path(root).rglob("*.jpg") if recurse else Path(root).glob("*.jpg")):
            m = LEDGER_NAME.match(p.name)
            if m:
                out.add((m.group("stamp"), m.group("pos").lower()))
    return out


def build_candidates(args, eff, b3, legacy, culled, nv=None):
    """Merge both provenance sources into per-crop candidates.

    Returns (crops, rejected, holdout_rows, problems, info). `rejected` are
    non-holdout ledger records with an excluding outcome (their `verdict` is the
    reason: artifact / drift / illegible / label_unsure); `holdout_rows` are ledger
    records of queue `holdout` (never trained). `nv` = needs_verify map.
    """
    nv = nv or {}
    crops: list[Crop] = []
    rejected: list[dict] = []
    holdout_rows: list[dict] = []
    problems: list[str] = []
    info = Counter()
    drift_frames = {e["_stamp"] for e in eff.values() if e.get("screen_verdict") == "drift"}

    for k in sorted(set(legacy) | set(eff)):
        stamp, pos = k
        leg = legacy.get(k)
        e = eff.get(k)
        if e is not None and (e.get("queue") or "").strip().lower() == "holdout":
            holdout_rows.append(e)
            info["ledger_holdout_rows"] += 1
            if leg:
                info["legacy_claimed_by_holdout"] += 1
            continue
        if stamp in drift_frames and not (e is not None and e.get("rejected")):
            # drift is frame-wide: a crop of a drift frame without its own drift row
            src = (e or {}).get("src") or (str(leg[1]) if leg else "")
            rejected.append(dict(e or {}, _stamp=stamp, _pos=pos, verdict="drift", src=src))
            info["frame_drift_rejected"] += 1
            if leg:
                info["legacy_rejected_by_ledger"] += 1
            continue
        if e is not None:
            status, why = ledger_status(e, nv.get(k, 0))
            queue = (e.get("queue") or "").strip()
            if status == "rejected":
                rejected.append(dict(e, verdict=why))
                info[f"ledger_{why}"] += 1
                if leg:
                    info["legacy_rejected_by_ledger"] += 1
                continue
            if status == "pending":
                info[f"ledger_pending_{why.replace(' ', '_')}"] += 1
                if leg:
                    info["legacy_pending_in_ledger"] += 1
                continue
            verdict = e["screen_verdict"]
            if verdict == "ok_derived" and args.no_ok_derived:
                info["ok_derived_dropped_by_flag"] += 1
                continue
            try:
                label = ledger_label(e)
            except ValueError:
                problems.append(f"{stamp} {pos}: ledger {verdict} record has no usable "
                                f"label ({e.get('final')!r})")
                continue
            src = Path(e["src"]) if (e.get("src") or "").strip() else None
            if (src is None or not src.is_file()) and leg:
                src = leg[1]
            if src is not None and not src.is_file() and (LEGACY_SRC / src.name).is_file():
                # joes-samples/ top level is this script's OUTPUT, so ledger rows that
                # pointed at the pre-batch-3 corpus there go stale after the first
                # --apply. work/legacy_src/ holds those 1,288 originals permanently.
                src = LEGACY_SRC / src.name
            if src is None:
                problems.append(f"{stamp} {pos}: ledger row has no src")
                continue
            if leg and leg[0] != label:
                info["ledger_overrides_legacy_label"] += 1
            if k in culled and leg:
                info["culled_but_rereviewed_ok"] += 1
            crops.append(Crop(stamp, pos, label, src, provenance="ledger",
                              verdict=verdict, queue=queue, in_legacy=leg is not None,
                              label_verdict=e.get("label_verdict", "")))
            info[f"ledger_{verdict}"] += 1
            if e.get("label_verdict"):
                info[f"ledger_{verdict}+{e['label_verdict']}"] += 1
        else:
            if k in culled:
                info["legacy_culled"] += 1
                continue
            crops.append(Crop(stamp, pos, leg[0], leg[1], provenance="legacy",
                              in_legacy=True))
            info["legacy_untouched"] += 1

    # light bucket + screen. b3 crops: work/b3_derive.csv. Everything else: the hour
    # rule, and dig_data.classify_screen over the frame's labels in the candidate set.
    frames: dict[str, dict[str, str]] = defaultdict(dict)
    for c in crops:
        frames[c.stamp][c.pos] = c.label
    fallback = 0
    tools = str(Path(__file__).resolve().parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from dig_data import classify_screen

    def legacy_screen(labels: dict[str, str]) -> str:
        """classify_screen over the frame's candidate labels. Legacy frames are often
        partial (crops were reviewed one by one), so when it cannot corroborate, use
        what a reading screen can never show: it always has dig2=5 and dig3 in
        {7,8,9}, so a dig2 of 0/8 or a dig3 of 0 is itself a test screen."""
        kind = classify_screen(labels)
        if kind != "uncertain":
            return kind
        if labels.get("dig2") in ("0", "8"):
            return "test" + labels["dig2"]
        if labels.get("dig3") == "0":
            return "test0"
        return kind

    for c in crops:
        is_b3 = "rawdigits" in str(c.src).lower() and c.day >= B3_START
        c.b3 = is_b3
        d = b3.get(c.key) if is_b3 else None
        if d:
            c.bucket, c.screen = d
        else:
            if is_b3:
                fallback += 1
            c.bucket = bucket_for_hour(int(c.stamp[9:11]))
            c.screen = legacy_screen(frames[c.stamp])
    info["b3_crops"] = sum(c.b3 for c in crops)
    info["b3_missing_in_derive_hour_fallback"] = fallback
    return crops, rejected, holdout_rows, problems, info


# ---- hard-error gates ------------------------------------------------------


def check_provenance(crops, eff, legacy, args, nv=None) -> list[str]:
    """Every crop must trace to legacy-trusted or a qualifying ledger record."""
    nv = nv or {}
    drift_frames = {e.get("_stamp") or k[0] for k, e in eff.items()
                    if e.get("screen_verdict") == "drift"}
    bad = []
    for c in crops:
        e = eff.get(c.key)
        why = None
        if c.stamp in drift_frames:
            why = "frame has a drift verdict"
        elif c.provenance == "ledger":
            status, reason = ledger_status(e, nv.get(c.key, 0)) if e is not None else ("", "")
            if e is None:
                why = "no ledger row"
            elif status != "ok":
                why = f"ledger {status}: {reason}"
            elif (e.get("queue") or "").strip().lower() == "holdout":
                why = "ledger queue is holdout"
            elif args.no_ok_derived and e["screen_verdict"] == "ok_derived":
                why = "ok_derived with --no-ok-derived"
            elif c.label != ledger_label(e):
                why = "label differs from ledger final"
        elif c.provenance == "legacy":
            leg = legacy.get(c.key)
            if leg is None:
                why = "not in work/labeled or batch 1"
            elif c.src != leg[1] or c.label != leg[0]:
                why = "path/label differs from the legacy source"
            elif e is not None:
                why = f"ledger has a row ({e['verdict']}); legacy provenance no longer applies"
        else:
            why = f"unknown provenance {c.provenance!r}"
        if why:
            bad.append(f"{c.out_name}  src={c.src}  ({why})")
    return bad


LEGACY_SRC = Path("work/legacy_src")


def run_gates(crops, eff, legacy, holdout_days, validation_keys, args, stage: str,
              nv=None):
    """-> list[CorpusError]; empty means every hard-error condition is clear."""
    errs = []
    prov = check_provenance(crops, eff, legacy, args, nv)
    if prov:
        errs.append(CorpusError(f"[{stage}] crops without qualifying provenance", prov))
    hd = [f"{c.out_name}  (day {c.day})" for c in crops if c.day in holdout_days]
    if hd:
        errs.append(CorpusError(f"[{stage}] crops on a holdout day", hd))
    val = [c.out_name for c in crops if c.key in validation_keys]
    if val:
        errs.append(CorpusError(f"[{stage}] crops in work/validation_labeled", val))
    ho = [c.out_name for c in crops
          if (eff.get(c.key, {}).get("queue") or "").strip().lower() == "holdout"]
    if ho:
        errs.append(CorpusError(f"[{stage}] crops in the holdout queue", ho))
    lab = [f"{c.out_name} (label {c.label!r})" for c in crops
           if c.label not in tuple("0123456789N") or c.out_name.startswith("10_")]
    if lab:
        errs.append(CorpusError(f"[{stage}] bad label (10 must be written N)", lab))
    miss = [f"{c.out_name}  src={c.src}" for c in crops if not Path(c.src).is_file()]
    if miss:
        errs.append(CorpusError(f"[{stage}] source file missing", miss))
    return errs


def report_errors(errs: list[CorpusError]) -> None:
    print("\n" + "!" * 70, file=sys.stderr)
    print("HARD ERROR -- nothing written", file=sys.stderr)
    for e in errs:
        print(f"\n{e.title}: {len(e.offenders)}", file=sys.stderr)
        for line in e.offenders[:200]:
            print(f"  {line}", file=sys.stderr)
        if len(e.offenders) > 200:
            print(f"  ... and {len(e.offenders) - 200} more", file=sys.stderr)
    print("!" * 70, file=sys.stderr)


# ---- selection -------------------------------------------------------------


# --stable-sampling: order by a per-crop hash instead of shuffling the whole cell, so a
# pool change (a relabel, a new reject) only moves the crops it touches. With plain
# shuffling, 9 relabels reshuffled ~100 night crops between the 9008 and 9009 builds.
STABLE = {"on": False, "seed": ""}


def _h(*parts) -> str:
    import hashlib
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()


def _round_robin(items: list[Crop], k: int, rng: random.Random) -> list[Crop]:
    if k >= len(items):
        return list(items)
    by_day: dict[str, list[Crop]] = defaultdict(list)
    for c in sorted(items, key=lambda c: c.key):
        by_day[c.day].append(c)
    days = sorted(by_day)
    if STABLE["on"]:
        days.sort(key=lambda d: _h(STABLE["seed"], d))
        for d in days:
            by_day[d].sort(key=lambda c: _h(STABLE["seed"], *c.key), reverse=True)
    else:
        rng.shuffle(days)
        for d in days:
            rng.shuffle(by_day[d])
    out: list[Crop] = []
    while len(out) < k:
        for d in days:
            if by_day[d]:
                out.append(by_day[d].pop())
                if len(out) == k:
                    break
    return out


def pick(items: list[Crop], k: int, rng: random.Random) -> list[Crop]:
    """k crops, legacy-trusted first, round-robin over capture days, seeded."""
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    legacy = [c for c in items if c.in_legacy]
    rest = [c for c in items if not c.in_legacy]
    out = _round_robin(legacy, min(k, len(legacy)), rng)
    if len(out) < k:
        out += _round_robin(rest, k - len(out), rng)
    return out


def pick_cell(items: list[Crop], cap: int, share: float, exempt: bool,
              rng: random.Random) -> list[Crop]:
    """Cap one (pos, class, bucket) cell, holding test-screen crops <= share."""
    if cap <= 0 or not items:
        return []
    tests = [c for c in items if c.screen in TEST_SCREENS]
    if exempt or share >= 1.0 or not tests:
        return pick(items, cap, rng)
    others = [c for c in items if c.screen not in TEST_SCREENS]
    t = min(len(tests), int(share * cap + 1e-9))
    n = min(len(others), cap - t)
    t = min(t, int(share * n / (1.0 - share) + 1e-9))
    return pick(others, n, rng) + pick(tests, t, rng)


def select_cells(crops: list[Crop], caps: dict[str, int], share: float,
                 flash_ratio: float, seed: int, ratio_scope: str = "class"):
    """-> (picked, notes). Caps per (pos, class, bucket); flash <= ratio*(day+trans).

    ``ratio_scope="class"`` (default) applies the flash ratio per class summed over
    positions: the network has no positional input, so the light/class confound
    (HANDOFF §3) lives at class level. Over-budget classes are trimmed by
    water-filling, so thin positions keep their night crops and only the
    plentiful ones give way. ``"cell"`` applies it per (pos, class), which strands
    night crops wherever a position's day supply is thin.

    Test-screen share (classes 0/8 only) is enforced per cell, except for a
    (pos, class) that is test-only in the pool (< 5% non-test crops, e.g. dig2 `0`
    exists almost solely on the 00000 screen): there the cap would erase the class,
    so it is exempted and listed in the report.
    """
    if ratio_scope == "class":
        return _select_cells_class_ratio(crops, caps, share, flash_ratio, seed)
    pool: dict[tuple, list[Crop]] = defaultdict(list)
    for c in crops:
        pool[(c.pos, c.label, c.bucket)].append(c)
    pairs = sorted({(p, l) for p, l, _ in pool})
    picked: list[Crop] = []
    notes = {"test_only": [], "share_emptied": [], "ratio_trimmed": []}
    for pos, lab in pairs:
        allc = [c for b in BUCKETS for c in pool.get((pos, lab, b), [])]
        n_tests = sum(c.screen in TEST_SCREENS for c in allc)
        applies = lab in SHARE_CLASSES and share < 1.0
        exempt = applies and (len(allc) - n_tests) < TEST_ONLY_FRAC * len(allc)
        if exempt and n_tests:
            notes["test_only"].append(f"{pos} class {lab} ({n_tests}/{len(allc)} test-screen)")
        eff_share = share if applies else 1.0
        kept: dict[str, list[Crop]] = {}
        for b in ("day", "transition"):
            items = pool.get((pos, lab, b), [])
            rng = random.Random(f"{seed}|{pos}|{lab}|{b}")
            kept[b] = pick_cell(items, caps[b], eff_share, exempt, rng)
        n_dt = len(kept["day"]) + len(kept["transition"])
        allowed = int(flash_ratio * n_dt + 1e-9)
        items = pool.get((pos, lab, "flash"), [])
        fcap = min(caps["flash"], allowed)
        if items and allowed < min(caps["flash"], len(items)):
            notes["ratio_trimmed"].append(
                f"{pos} class {lab}: flash {len(items)} avail -> cap {fcap} "
                f"(ratio x {n_dt} kept day+trans)")
        rng = random.Random(f"{seed}|{pos}|{lab}|flash")
        kept["flash"] = pick_cell(items, fcap, eff_share, exempt, rng)
        for b in BUCKETS:
            cap_b = fcap if b == "flash" else caps[b]
            if applies and not exempt and cap_b > 0 and pool.get((pos, lab, b)) \
                    and not kept[b]:
                notes["share_emptied"].append(f"{pos} class {lab} {b}")
            picked.extend(kept[b])
    return picked, notes


def _waterfill(want: dict, budget: int) -> dict:
    """Split ``budget`` over keys, none above its ``want``, smallest wants filled first."""
    out = {k: 0 for k in want}
    left = sorted(want, key=lambda k: (want[k], k))
    while left and budget > 0:
        share = budget // len(left)
        if share == 0:
            for k in left[:budget]:
                out[k] += 1
            break
        k = left[0]
        if want[k] - out[k] <= share:
            budget -= want[k] - out[k]
            out[k] = want[k]
            left.pop(0)
        else:
            for k in left:
                out[k] += share
            budget -= share * len(left)
    return out


def _select_cells_class_ratio(crops: list[Crop], caps: dict[str, int], share: float,
                              flash_ratio: float, seed: int):
    pool: dict[tuple, list[Crop]] = defaultdict(list)
    for c in crops:
        pool[(c.pos, c.label, c.bucket)].append(c)
    pairs = sorted({(p, l) for p, l, _ in pool})
    notes = {"test_only": [], "share_emptied": [], "ratio_trimmed": []}
    kept: dict[tuple, dict[str, list[Crop]]] = {}
    meta: dict[tuple, tuple] = {}
    for pos, lab in pairs:
        allc = [c for b in BUCKETS for c in pool.get((pos, lab, b), [])]
        n_tests = sum(c.screen in TEST_SCREENS for c in allc)
        applies = lab in SHARE_CLASSES and share < 1.0
        exempt = applies and (len(allc) - n_tests) < TEST_ONLY_FRAC * len(allc)
        if exempt and n_tests:
            notes["test_only"].append(f"{pos} class {lab} ({n_tests}/{len(allc)} test-screen)")
        eff_share = share if applies else 1.0
        meta[(pos, lab)] = (applies, exempt, eff_share)
        kept[(pos, lab)] = {}
        for b in ("day", "transition"):
            rng = random.Random(f"{seed}|{pos}|{lab}|{b}")
            kept[(pos, lab)][b] = pick_cell(pool.get((pos, lab, b), []), caps[b],
                                            eff_share, exempt, rng)

    for lab in sorted({l for _, l in pairs}):
        poss = [p for p, l in pairs if l == lab]
        n_dt = sum(len(kept[(p, lab)]["day"]) + len(kept[(p, lab)]["transition"])
                   for p in poss)
        allowed = int(flash_ratio * n_dt + 1e-9)
        want = {p: min(caps["flash"], len(pool.get((p, lab, "flash"), []))) for p in poss}
        got = _waterfill(want, allowed) if sum(want.values()) > allowed else want
        if got != want:
            trimmed = ", ".join(f"{p} {want[p]}->{got[p]}" for p in poss if got[p] < want[p])
            notes["ratio_trimmed"].append(
                f"class {lab}: flash {sum(want.values())} -> {allowed} "
                f"(ratio x {n_dt} kept day+trans): {trimmed}")
        for p in poss:
            _, exempt, eff_share = meta[(p, lab)]
            rng = random.Random(f"{seed}|{p}|{lab}|flash")
            kept[(p, lab)]["flash"] = pick_cell(pool.get((p, lab, "flash"), []), got[p],
                                                eff_share, exempt, rng)

    picked: list[Crop] = []
    for pos, lab in pairs:
        applies, exempt, _ = meta[(pos, lab)]
        for b in BUCKETS:
            if applies and not exempt and pool.get((pos, lab, b)) and not kept[(pos, lab)][b]:
                notes["share_emptied"].append(f"{pos} class {lab} {b}")
            picked.extend(kept[(pos, lab)][b])
    return picked, notes


# ---- reporting -------------------------------------------------------------


def print_cell_table(avail: list[Crop], sel: list[Crop]) -> None:
    a, s = Counter(), Counter()
    for c in avail:
        a[(c.pos, c.label, c.bucket)] += 1
    for c in sel:
        s[(c.pos, c.label, c.bucket)] += 1
    print("\n=== per (pos, class, bucket): available > selected ===")
    print(f"{'pos':<5}{'lbl':>4} | {'day':>9} | {'trans':>9} | {'flash':>9} | "
          f"{'sel':>5} {'flash%':>7}")
    for pos in sorted({k[0] for k in a}):
        for lab in sorted({k[1] for k in a if k[0] == pos}, key=lambda x: (x == "N", x)):
            cells = [f"{a[(pos, lab, b)]}>{s[(pos, lab, b)]}" for b in BUCKETS]
            tot = sum(s[(pos, lab, b)] for b in BUCKETS)
            fl = 100 * s[(pos, lab, "flash")] / tot if tot else 0
            print(f"{pos:<5}{lab:>4} | " + " | ".join(f"{x:>9}" for x in cells)
                  + f" | {tot:>5} {fl:>6.0f}%")


def print_summaries(sel: list[Crop]) -> None:
    print("\n=== per-class flash share (selected) ===")
    by: dict[str, Counter] = defaultdict(Counter)
    for c in sel:
        by[c.label][c.bucket] += 1
    shares = []
    for lab in sorted(by, key=lambda x: (x == "N", x)):
        r = by[lab]
        n = sum(r.values())
        sh = 100 * r["flash"] / max(n, 1)
        shares.append(sh)
        print(f"  {lab:>2}: n={n:<5} day={r['day']:<4} trans={r['transition']:<4} "
              f"flash={r['flash']:<4} flash%={sh:>3.0f}")
    n = len(sel)
    fl = sum(1 for c in sel if c.bucket == "flash")
    print(f"  ALL: n={n}  flash%={100 * fl / max(n, 1):.0f}"
          + (f"  spread across classes {min(shares):.0f}%..{max(shares):.0f}%"
             if shares else ""))

    print("\n=== provenance (selected) ===")
    pc = Counter(c.provenance for c in sel)
    both = sum(1 for c in sel if c.provenance == "ledger" and c.in_legacy)
    print(f"  legacy={pc['legacy']}  ledger={pc['ledger']} "
          f"(of which also legacy-trusted: {both})")
    vc = Counter(c.verdict or "(legacy, no ledger row)" for c in sel)
    print("  verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(vc.items())))
    print(f"  b3 crops: {sum(c.b3 for c in sel)}")

    print("\n=== test-screen share, classes 0 / 8 (selected) ===")
    for lab in SHARE_CLASSES:
        xs = [c for c in sel if c.label == lab]
        t = sum(c.screen in TEST_SCREENS for c in xs)
        print(f"  class {lab}: {t}/{len(xs)} = {100 * t / max(len(xs), 1):.0f}% test-screen")
        for pos in POSITIONS:
            ps = [c for c in xs if c.pos == pos]
            if ps:
                tp = sum(c.screen in TEST_SCREENS for c in ps)
                print(f"      {pos}: {tp}/{len(ps)} = {100 * tp / len(ps):.0f}%")


# ---- outputs ---------------------------------------------------------------


def write_manifest(path: Path, sel: list[Crop]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_FIELDS)
        for c in sorted(sel, key=lambda c: c.out_name):
            w.writerow([c.out_name, str(c.src), c.label, c.pos, c.stamp, c.bucket,
                        c.screen, c.provenance, c.verdict, c.queue, int(c.in_legacy),
                        c.label_verdict])


def plan_culls(rejected: list[dict], culled_path: Path, reasons_path: Path):
    """-> (new culled.txt lines, new reasons rows), skipping ones already present."""
    have = load_exclusions(culled_path)
    have_r = set()
    if reasons_path.is_file():
        with reasons_path.open(newline="", encoding="utf-8") as fh:
            have_r = {(r["stamp"], r["pos"]) for r in csv.DictReader(fh)}
    lines, rows = [], []
    for e in sorted(rejected, key=lambda e: (e["_stamp"], e["_pos"])):
        k = (e["_stamp"], e["_pos"])
        if k not in have:
            have.add(k)
            lines.append(f"{k[0]},{k[1]}")
        if k not in have_r:
            have_r.add(k)
            rows.append((k[0], k[1], e["verdict"], e.get("src", "")))
    return lines, rows


def append_culls(culled_path: Path, reasons_path: Path, lines, rows) -> None:
    if lines:
        culled_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = ""
        if culled_path.is_file() and culled_path.stat().st_size:
            with culled_path.open("rb") as fh:
                fh.seek(-1, 2)
                if fh.read(1) not in (b"\n", b"\r"):
                    prefix = "\n"
        with culled_path.open("a", encoding="utf8", newline="") as fh:
            fh.write(prefix + "".join(f"{l}\n" for l in lines))
    if rows:
        new = not reasons_path.is_file() or reasons_path.stat().st_size == 0
        reasons_path.parent.mkdir(parents=True, exist_ok=True)
        with reasons_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["stamp", "pos", "verdict", "src"])
            w.writerows(rows)


def plan_holdout(holdout_rows: list[dict], nv=None):
    """Whole, fully-qualified frames only. -> (export [(stamp,pos,label,src)],
    dropped_rejected, dropped_pending, dropped_incomplete, problems).

    A frame is dropped if any crop is rejected (artifact/drift/illegible/
    label_unsure), any crop is pending (not screened, or needs verification without
    label_ok/label_fixed), or a position is missing. Labels are the effective final.
    """
    nv = nv or {}
    frames: dict[str, dict[str, dict]] = defaultdict(dict)
    for e in holdout_rows:
        frames[e["_stamp"]][e["_pos"]] = e
    export, rej, pend, inc, problems = [], [], [], [], []
    for stamp in sorted(frames):
        rows = frames[stamp]
        st = {p: ledger_status(e, nv.get((stamp, p), 0)) for p, e in rows.items()}
        bad = sorted(p for p, (s, _) in st.items() if s == "rejected")
        if bad:
            rej.append((stamp, [f"{p}:{st[p][1]}" for p in bad]))
            continue
        waiting = sorted(p for p, (s, _) in st.items() if s == "pending")
        if waiting:
            pend.append((stamp, [f"{p}:{st[p][1]}" for p in waiting]))
            continue
        missing = [p for p in POSITIONS if p not in rows]
        if missing:
            inc.append((stamp, missing))
            continue
        for p in POSITIONS:
            e = rows[p]
            try:
                lab = ledger_label(e)
            except ValueError:
                problems.append(f"{stamp} {p}: no usable label")
                continue
            export.append((stamp, p, lab, Path(e["src"])))
    return export, rej, pend, inc, problems


def from_ledger(args) -> int:
    t0 = time.time()
    default_out = args.out.resolve() == DEFAULT_OUT.resolve()
    # never write into protected locations
    for prot in (args.batch1, args.labeled, args.validation):
        if prot and (args.out.resolve() == prot.resolve() or _under(args.out, prot)):
            print(f"refusing --out {args.out}: inside protected dir {prot}", file=sys.stderr)
            return 2
    if args.holdout_out and args.holdout_out.resolve() == args.out.resolve():
        print("--holdout-out must differ from --out", file=sys.stderr)
        return 2

    holdout_days, hd_exists = load_holdout_days(args.holdout_days)
    if not hd_exists:
        msg = f"holdout-days file not found: {args.holdout_days}"
        if args.apply:
            print(f"ERROR: {msg} (required with --apply; create it, even if empty)",
                  file=sys.stderr)
            return 2
        print(f"WARNING: {msg} -- treating as no holdout days (dry run only)")
    print(f"holdout days ({len(holdout_days)}): {sorted(holdout_days)}")

    eff = load_ledger_eff(args.ledger)
    nv, nv_warns = load_needs_verify(args.queues_dir)
    for w in nv_warns:
        print(f"WARNING: {w}")
    b3 = load_b3_derive(args.b3_derive)
    legacy, bad_names, dups = collect_legacy(args.labeled, args.batch1)
    culled = load_exclusions(args.exclude)
    validation_keys = scan_keys(args.validation)

    print(f"ledger {args.ledger}: {len(eff)} effective items; screen stage "
          f"{dict(Counter(e['screen_verdict'] or '-' for e in eff.values()))}; label stage "
          f"{dict(Counter(e['label_verdict'] or '-' for e in eff.values()))}")
    n_ver = sum(1 for k, e in eff.items()
                if nv.get(k) and e["label_verdict"] in LABEL_OK_VERDICTS)
    print(f"needs_verify from {args.queues_dir}: {len(nv)} queued items, "
          f"{sum(nv.values())} need verification ({n_ver} verified so far)")
    print(f"legacy-trusted: {len(legacy)} crops ({bad_names} unparseable names skipped, "
          f"{dups} duplicate keys skipped); hand-culled list: {len(culled)}; "
          f"b3_derive: {len(b3)} crops; validation_labeled: {len(validation_keys)} crops")

    if args.out.is_dir():
        top = scan_keys(args.out, recurse=False)
        ok_ledger = {k for k, e in eff.items() if ledger_status(e, nv.get(k, 0))[0] == "ok"
                     and (e.get("queue") or "").strip().lower() != "holdout"}
        dropped_top = [k for k in top if k not in legacy and k not in ok_ledger]
        print(f"existing top level of {args.out}: {len(top)} jpgs, of which "
              f"{len(dropped_top)} are in neither legacy source nor an ok-ledger row "
              f"(derived-only; dropped)")

    crops, rejected, holdout_rows, problems, info = build_candidates(
        args, eff, b3, legacy, culled, nv)
    if problems:
        report_errors([CorpusError("ledger rows unusable", problems)])
        return 2
    print("candidates: " + ", ".join(f"{k}={v}" for k, v in sorted(info.items())))
    if info["b3_missing_in_derive_hour_fallback"]:
        print(f"WARNING: {info['b3_missing_in_derive_hour_fallback']} b3 crop(s) missing from "
              f"b3_derive.csv -- bucket from the hour rule, screen from labels")

    errs = run_gates(crops, eff, legacy, holdout_days, validation_keys, args, "candidates",
                     nv)
    if errs:
        report_errors(errs)
        return 2

    if args.reserve:
        res = set()
        for line in Path(args.reserve).read_text(encoding="utf-8").split():
            m = re.search(r"(dig[2-6])_(\d{8}-\d{6})", line)
            if m:
                res.add((m.group(2), m.group(1)))
        before = len(crops)
        crops = [c for c in crops if c.key not in res]
        print(f"--reserve {args.reserve}: {len(res)} crop(s) held back, "
              f"{before - len(crops)} removed from the candidate pool")
    STABLE.update(on=args.stable_sampling, seed=args.seed)
    caps = {"day": args.cap_day, "transition": args.cap_transition, "flash": args.cap_flash}
    sel, notes = select_cells(crops, caps, args.test_share, args.flash_ratio, args.seed,
                              ratio_scope=args.ratio_scope)
    sel.sort(key=lambda c: c.out_name)

    errs = run_gates(sel, eff, legacy, holdout_days, validation_keys, args, "output", nv)
    if errs:
        report_errors(errs)
        return 2

    print_cell_table(crops, sel)
    print_summaries(sel)
    print(f"\ncaps: day {args.cap_day} / transition {args.cap_transition} / flash "
          f"{args.cap_flash}; test-share {args.test_share}; flash-ratio {args.flash_ratio}; "
          f"seed {args.seed}")
    print(f"pool {len(crops)} -> selected {len(sel)}")
    for key, title in (("test_only", "test-only (pos,class) exempt from the share cap"),
                       ("share_emptied", "cells emptied by the test-share cap"),
                       ("ratio_trimmed", "flash trimmed by the flash-ratio rule")):
        if notes[key]:
            print(f"\n{title} ({len(notes[key])}):")
            for line in notes[key][:40]:
                print(f"  {line}")
            if len(notes[key]) > 40:
                print(f"  ... {len(notes[key]) - 40} more")

    new_c, new_r = plan_culls(rejected, args.exclude, args.culled_reasons)
    print(f"\nrejected by ledger (non-holdout): {len(rejected)}; "
          f"new lines for {args.exclude}: {len(new_c)}")

    ho_export = []
    if args.holdout_out:
        ho_export, ho_rej, ho_pend, ho_inc, ho_problems = plan_holdout(holdout_rows, nv)
        if ho_problems:
            report_errors([CorpusError("holdout export rows unusable", ho_problems)])
            return 2
        miss = [f"{s} {p}  src={src}" for s, p, _l, src in ho_export if not src.is_file()]
        if miss:
            report_errors([CorpusError("holdout source file missing", miss)])
            return 2
        n_unver = sum(1 for _, why in ho_pend if any("verification" in w for w in why))
        print(f"\nholdout export -> {args.holdout_out}: "
              f"{len(ho_export) // len(POSITIONS)} complete frames ({len(ho_export)} crops); "
              f"dropped whole: {len(ho_rej)} frame(s) with a rejected crop, "
              f"{len(ho_pend)} frame(s) with an unverified/unscreened crop "
              f"({n_unver} awaiting verification), {len(ho_inc)} incomplete frame(s)")
        for s, why in ho_rej[:40]:
            print(f"  rejected frame {s}: {', '.join(why)}")
        for s, why in ho_pend[:40]:
            print(f"  pending frame {s}: {', '.join(why)}")
        for s, miss_p in ho_inc[:40]:
            print(f"  incomplete frame {s}: missing {', '.join(miss_p)}")

    if not args.apply:
        stale = len(list(args.out.glob("*.jpg"))) if args.out.is_dir() else 0
        print(f"\n(dry run -- pass --apply to write {len(sel)} crops to {args.out}; "
              f"would clear {stale} existing top-level jpg(s)"
              + (", backing them up first" if default_out and stale else "") + ")")
        return 0

    # ---- apply: every gate has passed. Read the sources first so an --out that
    # overlaps a source dir (a ledger src inside joes-samples/) cannot lose them.
    blobs = {c.out_name: Path(c.src).read_bytes() for c in sel}
    hblobs = {f"{lab}_main_{p}_{s}.jpg": src.read_bytes() for s, p, lab, src in ho_export}

    args.out.mkdir(parents=True, exist_ok=True)
    stale = sorted(args.out.glob("*.jpg"))
    if stale and default_out:
        backup = args.out / f"_backup_{time.strftime('%Y%m%d-%H%M%S')}"
        backup.mkdir()
        for p in stale:
            shutil.copy2(p, backup / p.name)
        if sum(1 for _ in backup.glob("*.jpg")) != len(stale):
            print("ERROR: backup incomplete, aborting before delete", file=sys.stderr)
            return 2
        print(f"\nbacked up {len(stale)} jpg(s) -> {backup}")
    for p in stale:
        p.unlink()
    for name, data in blobs.items():
        (args.out / name).write_bytes(data)
    print(f"cleared {len(stale)} existing *.jpg, wrote {len(sel)} -> {args.out}")

    write_manifest(args.manifest, sel)
    print(f"manifest -> {args.manifest}")
    append_culls(args.exclude, args.culled_reasons, new_c, new_r)
    if new_c or new_r:
        print(f"appended {len(new_c)} to {args.exclude}, {len(new_r)} to {args.culled_reasons}")

    if args.holdout_out:
        args.holdout_out.mkdir(parents=True, exist_ok=True)
        for name, data in hblobs.items():
            (args.holdout_out / name).write_bytes(data)
        extra = [p.name for p in args.holdout_out.glob("*.jpg") if p.name not in hblobs]
        print(f"holdout: wrote {len(hblobs)} -> {args.holdout_out}")
        if extra:
            print(f"WARNING: {len(extra)} stale jpg(s) already in {args.holdout_out} are NOT "
                  f"part of this export (not deleted): {extra[:5]}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


def add_ledger_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("ledger mode (--from-ledger)")
    g.add_argument("--from-ledger", action="store_true",
                   help="build from legacy-trusted crops + the review ledger "
                        "(human-reviewed provenance enforced)")
    g.add_argument("--ledger", type=Path, default=Path("work/review_ledger.csv"))
    g.add_argument("--holdout-days", type=Path, default=Path("work/queues/holdout_days.txt"),
                   help="one YYYYMMDD per line; no output crop may fall on these days")
    g.add_argument("--validation", type=Path, default=Path("work/validation_labeled"))
    g.add_argument("--b3-derive", type=Path, default=Path("work/b3_derive.csv"))
    g.add_argument("--no-ok-derived", action="store_true",
                   help="drop ok_derived crops (ablation model; a no-op when the ledger "
                        "has no ok_derived rows)")
    g.add_argument("--queues-dir", type=Path, default=Path("work/queues"),
                   help="queue CSVs whose needs_verify column gates ledger crops "
                        "(union; items in no queue count as needs_verify=0)")
    g.add_argument("--cap-day", type=int, default=60)
    g.add_argument("--cap-transition", type=int, default=40)
    g.add_argument("--cap-flash", type=int, default=80)
    g.add_argument("--test-share", type=float, default=0.3,
                   help="max test-screen share per (pos, class 0/8, bucket) cell")
    g.add_argument("--holdout-out", type=Path, default=None,
                   help="export complete holdout-queue frames here (with --apply)")
    g.add_argument("--manifest", type=Path, default=Path("work/corpus_manifest.csv"))
    g.add_argument("--culled-reasons", type=Path, default=Path("work/culled_reasons.csv"))
    g.add_argument("--reserve", type=Path, default=None,
                   help="file of crop names (or anything containing dig<P>_<stamp>) never to "
                        "select, e.g. the model-selection set")
    g.add_argument("--stable-sampling", action="store_true",
                   help="order capped cells by a per-crop hash so rebuilds change only the "
                        "crops whose verdicts changed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", type=Path, default=Path("work/labeled"))
    ap.add_argument("--batch1", type=Path, default=Path("joes-samples/batch 1"))
    # 1.0 caps every class at a 50/50 light mix where it has the coverage to do
    # so, which collapsed the cross-class flash-share spread from 24-88% to 24-50%.
    ap.add_argument("--flash-ratio", type=float, default=1.0)
    ap.add_argument("--ratio-scope", choices=("class", "cell"), default="class",
                    help="--from-ledger: apply --flash-ratio per class (default) or per (pos, class)")
    ap.add_argument("--class-cap", type=int, default=150)
    ap.add_argument("--n-cap", type=int, default=80)
    ap.add_argument("--exclude", type=Path, default=Path("work/culled.txt"),
                    help="file of <stamp>,<pos> crops culled by hand; kept out "
                         "of every rebuild (default work/culled.txt)")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    add_ledger_args(ap)
    args = ap.parse_args()
    if args.from_ledger:
        sys.exit(from_ledger(args))

    rows = collect([(args.labeled, True), (args.batch1, False)],
                   load_exclusions(args.exclude))
    table(rows, "available pool")

    picked = select(rows, args.flash_ratio, args.class_cap, args.n_cap, args.seed)
    table(picked, "selected corpus")

    print("\n=== selected: class x position ===")
    t2: dict[str, Counter] = defaultdict(Counter)
    for lab, pos, _b, _p in picked:
        t2[pos][lab] += 1
    for pos in sorted(t2):
        print(f"  {pos}: " + " ".join(f"{k}={v}" for k, v in sorted(t2[pos].items())))

    if args.apply:
        # `prepare_joe_data.py` globs joes-samples/*.jpg non-recursively, so the
        # corpus has to sit at that top level -- alongside batch subdirs and the
        # spreadsheet. Clear only the jpgs this script manages; never rmtree the
        # target, which would take the source batches and the xlsx with it.
        args.out.mkdir(parents=True, exist_ok=True)
        stale = list(args.out.glob("*.jpg"))
        for p in stale:
            p.unlink()
        kept = [c for c in args.out.iterdir() if c.name != "MANIFEST.csv"]
        for lab, pos, _b, p in picked:
            shutil.copy2(p, args.out / f"{lab}_main_{pos}_{p.name.split('_')[-1]}")
        print(f"\ncleared {len(stale)} existing *.jpg, wrote {len(picked)} -> {args.out}")
        print(f"left untouched: {len(kept)} other entries "
              f"({', '.join(sorted(c.name for c in kept)[:4])}...)")
    else:
        print("\n(report only -- pass --apply to write the corpus)")


if __name__ == "__main__":
    main()
