"""Grid review: put human eyes on every training/holdout digit crop, fast.

Shows a page of tiles that share (queue, position, proposed label, light bucket),
sorted by brightness, so the odd one out pops. Only click/mark the bad ones;
`Enter` commits the page and everything unmarked gets the page's default verdict.

Two review tasks, one at a time (`--mode`):

  verify  "Claude is not sure what this digit is -- confirm the label."
          Only queue items with needs_verify=1 that have no label_* verdict yet.
          Header `VERIFY . dig6 . is this a 1? . flash`; each tile shows the proposed
          label big and the deployed model's reading small.
          Enter = unmarked confirmed (label_ok)   0-9/n = label_fixed (that digit)
          x / ?  = label_unsure (can't tell -> excluded)    a artifact   d drift
  screen  "Flag alignment drift or artifacts only."
          Items without a screen verdict whose label needs no verification or was
          verified (label_ok / label_fixed); a fixed crop shows (and is grouped
          under) its verified label. Unverified items are skipped (count printed).
          Enter = unmarked ok   a artifact   d drift   (0-9/n relabel: escape hatch)
  consistency  "Every tile on this page should be the same digit -- spot the odd one."
          Every effectively accepted item in the LEDGER (holdout included; the queue
          files only add bucket/context), grouped by (position, current final label),
          all light buckets mixed, dark -> bright, pages of <= --page-size. Groups:
          dig6, dig5, dig4, dig3, dig2; classes 0 8 9 5 6 3 1 7 2 4 N. Header
          `CONSISTENCY · dig6 · all should be 0 · page 3/7 · group n/N · overall`;
          caption = time, date, bucket letter D/T/F, `v`/`f` if verified/fixed earlier.
          Enter = unmarked label_ok (final = current label)   0-9/n = label_fixed
          x / ?  = label_unsure    a artifact   d drift (screen-stage rows, frame-wide)
          Page ids `gc...`; resume skips items whose latest label-stage row is from a
          consistency page (a fixed item moves to its new group, already done).
          `--positions dig6,dig5` limits the run (default all).
  (none)  the original single-pass review (ok/relabeled/ok_derived/artifact/drift/
          illegible keys), kept for backward compatibility.

Drift is frame-wide: ROI misalignment comes from the per-image alignment step, so
`d` on one crop marks every crop of that capture (same stamp) on the page, and the
commit writes `drift` rows for all five positions of the frame -- including ones not
on the page or not in any loaded queue (src found next to the crop by stamp) -- with
the same page_id, so undo removes them too. Crops of a drift frame are skipped
everywhere afterwards.

Every committed page is appended to an append-only ledger (default
`work/review_ledger.csv`) and flushed, so you can stop at any time and resume
exactly where you left off. Other tools read the outcome with::

    from grid_review import load_effective    # (tools/ on sys.path)
    eff = load_effective("work/review_ledger.csv")   # item_id -> combined record

Ledger verdicts, two stages per item:
    screen stage: ok | relabeled | ok_derived | artifact | drift | illegible
    label stage:  label_ok | label_fixed | label_unsure
(+ `undone` tombstones, which cancel every row of the page_id they reference).
The latest row of each stage counts. Effective: rejected if screen in
{artifact, drift, illegible} or label = label_unsure; otherwise the final label is
the label-stage final when present (label_fixed beats a screen relabel), else the
screen-stage final. `final` is blank on rejecting rows.

Keys
    mouse hover / left-click / arrow keys   select a tile
    a   artifact (reflection streak/blob, sky cloud, dust/smear, ghosting) -> exclude
    d   drift    (glyph clipped / neighbour digit intruding) -> exclude WHOLE FRAME
    0-9, n (or -)   verify/consistency: label_fixed; screen/default: relabel
    x, ?            verify/consistency only: label_unsure (can't tell -> exclude)
    i, o            default mode only: illegible / ok-derived
    Space / c       clear back to the default    (a/d/x/i/o pressed twice also clears)
    right-click     toggle artifact           double-click / z   zoom (frame + time strip)
    v   toggle ROI view <-> upscaled 20x32 NEAREST model view
    Enter           commit page
    u / Ctrl+Z      undo last committed page of this mode (appends tombstones; page comes
                    back with its marks so you can fix the one mistake and re-commit)
    q / window X    quit (asks before discarding marks on the uncommitted page)

Usage (from repo root):
    python tools/grid_review.py --mode verify --queue work/queues/holdout.csv
    python tools/grid_review.py --mode screen --queue work/queues/legacy.csv
    python tools/grid_review.py --mode consistency --positions dig6
    python tools/grid_review.py --queue work/queues/holdout.csv --frame-mode
    python tools/grid_review.py --propagate-drift          # idempotent ledger repair
    python tools/grid_review.py --make-sample-queue work/grid_review_sample_queue.csv \
        --root ..\\AIOTED-digital-rawdigits\\20260920 --n 200
    python tools/grid_review.py --selftest

Queue CSV columns: queue,item_id,stamp,pos,src,proposed,bucket,brightness,context,
preflag,group,needs_verify
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import random
import re
import sys
import time
import uuid
from collections import Counter, OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageStat

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

LEDGER_FIELDS = ["item_id", "stamp", "pos", "src", "queue", "proposed", "final",
                 "verdict", "page_id", "reviewed_at"]
QUEUE_FIELDS = ["queue", "item_id", "stamp", "pos", "src", "proposed", "bucket",
                "brightness", "context", "preflag", "group", "needs_verify"]
SCREEN_VERDICTS = ("ok", "relabeled", "ok_derived", "artifact", "drift", "illegible")
LABEL_VERDICTS = ("label_ok", "label_fixed", "label_unsure")
VERDICTS = SCREEN_VERDICTS + LABEL_VERDICTS
EXCLUDED = ("artifact", "drift", "illegible")
SCREEN_OK = ("ok", "relabeled", "ok_derived")
LABEL_OK = ("label_ok", "label_fixed")
UNDONE = "undone"
LABEL_SET = tuple("0123456789") + ("N",)
POSITIONS = (2, 3, 4, 5, 6)
MODEL_INPUT = (20, 32)          # what prepare_joe_data.py / dig_data resize to
CELL_ORDERS = ("csv", "size", "small", "sorted")
TASKS = ("classic", "verify", "screen", "consistency", "audit")
TASK_LETTER = {"classic": "", "verify": "v", "screen": "s", "consistency": "c", "audit": "a"}
LABEL_TASKS = ("verify", "consistency", "audit")      # tasks whose default verdict is label_ok
DIV = {2: 10000, 3: 1000, 4: 100, 5: 10, 6: 1}

# which mark actions each task accepts (digits always; "clear" always)
TASK_ACTIONS = {
    "classic": {"artifact", "drift", "illegible", "ok_derived"},
    "verify": {"artifact", "drift", "unsure"},
    "screen": {"artifact", "drift"},
    "consistency": {"artifact", "drift", "unsure"},
    "audit": {"artifact", "drift", "unsure"},
}

# consistency mode: group order (positions, then classes within a position)
CONSISTENCY_POS_ORDER = (6, 5, 4, 3, 2)
CONSISTENCY_CLASS_ORDER = ("0", "8", "9", "5", "6", "3", "1", "7", "2", "4", "N")
BUCKET_LETTER = {"day": "D", "transition": "T", "flash": "F"}

# light buckets (same rule as tools/build_corpus.py / select_review_set.py)
_TRANSITION = {6, 19}
_FLASH = {20, 21, 22, 23, 0, 1, 2, 3, 4, 5}

_STAMP_RE = re.compile(r"(\d{8}-\d{6})")
_POS_RE = re.compile(r"_dig(\d)_", re.I)
_MODEL_RE = re.compile(r"(?:model saw|9002 says)\s+(10|[0-9N])", re.I)
_ITEM_RE = re.compile(r"^(\d{8}-\d{6})_dig(\d)$")


def bucket_for_hour(h: int) -> str:
    if h in _TRANSITION:
        return "transition"
    return "flash" if h in _FLASH else "day"


def norm_label(s) -> str:
    """'7' -> '7', '10'/'n'/'nan' -> 'N'. Raises on anything else."""
    t = str(s).strip().upper()
    if t in ("10", "NAN", "-"):
        t = "N"
    if t.endswith(".0") and t[:-2].isdigit():
        t = t[:-2]
        if t == "10":
            t = "N"
    if t not in LABEL_SET:
        raise ValueError(f"bad label {s!r}")
    return t


def norm_pos(s) -> int:
    t = str(s).strip().lower()
    if t.startswith("dig"):
        t = t[3:]
    p = int(t)
    if p not in POSITIONS:
        raise ValueError(f"bad position {s!r}")
    return p


def label_from_filename(name: str) -> str | None:
    tok = os.path.basename(name).split("_", 1)[0]
    try:
        return norm_label(tok)
    except ValueError:
        return None


def model_from_context(ctx: str) -> str:
    """'... model saw 9 ...' / '9002 says 8 @0.97' -> '9' / '8' ('' if absent)."""
    m = _MODEL_RE.search(ctx or "")
    if not m:
        return ""
    try:
        return norm_label(m.group(1))
    except ValueError:
        return ""


def split_item_id(item_id: str, row: dict | None = None) -> tuple[str, int | None]:
    """item_id '20260818-042958_dig4' -> ('20260818-042958', 4); falls back to the
    row's stamp/pos columns."""
    m = _ITEM_RE.match(item_id or "")
    if m:
        return m.group(1), int(m.group(2))
    row = row or {}
    try:
        p = norm_pos(row.get("pos"))
    except (ValueError, TypeError):
        p = None
    return (row.get("stamp") or "").strip(), p


def find_crop(directory, stamp: str, pos: int) -> tuple[str, str]:
    """(path, filename label) of the dig<pos> crop of <stamp> in <directory>, or ('', '')."""
    if not directory or not stamp:
        return "", ""
    d = Path(directory)
    if not d.is_dir():
        return "", ""
    hits = sorted(d.glob(f"*_dig{pos}_{stamp}.jpg")) or sorted(d.glob(f"*dig{pos}*{stamp}*.jpg"))
    if not hits:
        return "", ""
    return str(hits[0]), label_from_filename(hits[0].name) or ""


# --------------------------------------------------------------------------
# queue items
# --------------------------------------------------------------------------


@dataclass
class QItem:
    queue: str
    item_id: str
    stamp: str
    pos: int
    src: str
    proposed: str
    bucket: str = ""
    brightness: float | None = None
    context: str = ""
    preflag: str = ""
    group: str = ""
    order: int = 0
    needs_verify: bool = False
    model_label: str = ""            # deployed model's reading (from context)
    orig_proposed: str = ""          # screen mode: queue proposal before verification
    prior: str = ""                  # consistency: earlier label stage 'v' (label_ok) / 'f' (fixed)

    @property
    def cell(self) -> tuple:
        return (self.queue, self.pos, self.proposed, self.bucket)

    @property
    def frame_key(self) -> tuple:
        return (self.queue, self.group or self.stamp)

    def sort_key(self):
        b = self.brightness
        return (b is None, b if b is not None else 0.0, self.stamp, self.order)


def load_queues(paths) -> tuple[list[QItem], list[str]]:
    """Read queue CSVs in the given order. Returns (items, warnings).

    Duplicate item_ids keep the first occurrence. Rows with unparseable
    pos/label are skipped with a warning. A missing needs_verify column reads as 0.
    """
    items: list[QItem] = []
    seen: set[str] = set()
    warns: list[str] = []
    for p in paths:
        p = Path(p)
        with p.open(newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            missing = {"item_id", "stamp", "pos", "src", "proposed"} - set(rd.fieldnames or [])
            if missing:
                raise SystemExit(f"{p}: missing columns {sorted(missing)}")
            for ln, r in enumerate(rd, start=2):
                iid = (r.get("item_id") or "").strip()
                if not iid:
                    warns.append(f"{p.name}:{ln}: blank item_id, skipped")
                    continue
                if iid in seen:
                    warns.append(f"{p.name}:{ln}: duplicate item_id {iid}, skipped")
                    continue
                try:
                    pos = norm_pos(r["pos"])
                    lab = norm_label(r["proposed"])
                except (ValueError, TypeError) as e:
                    warns.append(f"{p.name}:{ln}: {e}, skipped")
                    continue
                br = (r.get("brightness") or "").strip()
                try:
                    brf = float(br) if br else None
                except ValueError:
                    brf = None
                stamp = (r.get("stamp") or "").strip()
                ctx = (r.get("context") or "").strip()
                it = QItem(
                    queue=(r.get("queue") or "").strip() or p.stem,
                    item_id=iid, stamp=stamp, pos=pos,
                    src=(r.get("src") or "").strip(), proposed=lab,
                    bucket=(r.get("bucket") or "").strip(),
                    brightness=brf,
                    context=ctx,
                    preflag=(r.get("preflag") or "").strip(),
                    group=(r.get("group") or "").strip(),
                    order=len(items),
                    needs_verify=(r.get("needs_verify") or "").strip().lower()
                    in ("1", "true", "yes", "y"),
                    model_label=model_from_context(ctx),
                )
                if it.src and not os.path.isfile(it.src):
                    warns.append(f"{p.name}:{ln}: missing image {it.src}")
                seen.add(iid)
                items.append(it)
    return items, warns


def frame_context(items) -> str:
    """Context shared by a frame's crops: the `;`-separated parts common to all
    of them (e.g. "reading 58137"), falling back to the first item's context.
    Per-crop parts ("model saw 4") are shown on hover instead."""
    ctxs = [it.context for it in items if it.context]
    if not ctxs:
        return ""
    split = [[p.strip() for p in c.split(";") if p.strip()] for c in ctxs]
    common = [p for p in split[0] if all(p in s for s in split[1:])]
    return "; ".join(common) if common else ctxs[0]


def image_brightness(path: str) -> float | None:
    """Mean luma 0..255 of the crop, or None if unreadable."""
    try:
        with Image.open(path) as im:
            im.draft("L", (im.width // 2 or 1, im.height // 2 or 1))
            return float(ImageStat.Stat(im.convert("L")).mean[0])
    except Exception:
        return None


def fill_brightness(items, fn=image_brightness, verbose=False) -> int:
    todo = [it for it in items if it.brightness is None and it.src]
    for k, it in enumerate(todo):
        it.brightness = fn(it.src)
        if verbose and k and k % 500 == 0:
            print(f"  brightness {k}/{len(todo)}", flush=True)
    return len(todo)


# --------------------------------------------------------------------------
# b3_derive index (verification aids: frame reading, time strip)
# --------------------------------------------------------------------------

DRow = namedtuple("DRow", "stamp pos path model truth status screen est bright bucket",
                  defaults=(None, ""))


def est_digit(est, pos: int) -> str:
    try:
        return str(int(est) // DIV[pos] % 10)
    except (TypeError, ValueError, KeyError):
        return "?"


class DeriveIndex:
    """work/b3_derive.csv: (stamp, pos) -> DRow, plus per-position sorted stamps of
    reading-screen frames (for the time strip)."""

    def __init__(self, path=None, rows=None):
        self.rows: dict[tuple[str, int], DRow] = {}
        self.reading: dict[int, list[str]] = defaultdict(list)
        if rows is None:
            rows = []
            if path and Path(path).is_file():
                with Path(path).open(newline="", encoding="utf-8-sig") as fh:
                    rows = list(csv.DictReader(fh))
        for r in rows:
            try:
                pos = norm_pos(r["pos"])
            except (ValueError, TypeError, KeyError):
                continue
            try:
                model = norm_label(r.get("model_label", ""))
            except ValueError:
                model = "?"
            try:
                bright = float(r.get("frame_brightness") or "")
            except ValueError:
                bright = None
            d = DRow(r["stamp"], pos, r.get("path", ""), model, (r.get("truth") or "?"),
                     r.get("status", ""), r.get("screen", ""), r.get("reading_est", ""),
                     bright, (r.get("bucket") or "").strip())
            self.rows[(d.stamp, pos)] = d
            if d.screen == "reading":
                self.reading[pos].append(d.stamp)
        for lst in self.reading.values():
            lst.sort()

    def __len__(self):
        return len(self.rows)

    def get(self, stamp: str, pos: int) -> DRow | None:
        return self.rows.get((stamp, pos))

    def neighbours(self, stamp: str, pos: int, n: int = 3) -> tuple[list[DRow], list[DRow]]:
        """Previous n and next n reading-screen frames at this position (own stamp excluded)."""
        st = self.reading.get(pos, [])
        i = bisect.bisect_left(st, stamp)
        j = bisect.bisect_right(st, stamp)
        prev = [self.rows[(s, pos)] for s in st[max(0, i - n):i]]
        nxt = [self.rows[(s, pos)] for s in st[j:j + n]]
        return prev, nxt


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


def read_ledger_rows(path) -> list[dict]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = []
        for r in csv.DictReader(fh):
            v = (r.get("verdict") or "").strip()
            if not r.get("page_id") or (v not in VERDICTS and v != UNDONE):
                continue  # torn / garbage line
            if v != UNDONE and not r.get("item_id"):
                continue
            rows.append(r)
        return rows


def _undone_pages(rows) -> set[str]:
    return {r["page_id"] for r in rows if r["verdict"] == UNDONE}


def load_ledger(path) -> dict[str, dict]:
    """item_id -> latest effective ledger row of either stage (rows of undone pages
    ignored). Raw view; use load_effective() for the two-stage outcome."""
    rows = read_ledger_rows(path)
    undone = _undone_pages(rows)
    eff: dict[str, dict] = {}
    for r in rows:
        if r["verdict"] != UNDONE and r["page_id"] not in undone:
            eff[r["item_id"]] = r
    return eff


def _norm_final(s) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    try:
        return norm_label(s)
    except ValueError:
        return s


def combine_stages(label_row: dict | None, screen_row: dict | None) -> dict:
    """Combine an item's latest label-stage and screen-stage rows.

    -> dict with the LEDGER_FIELDS of the screen row (else the label row), plus
       label_verdict / label_final / screen_verdict / screen_final,
       label_page_id / screen_page_id (page that wrote each stage's latest row),
       rejected ('' or the rejecting verdict), final (effective training label,
       '' when rejected or when no stage gives one) and verdict (rejected reason,
       else screen verdict, else label verdict).
    """
    base = screen_row or label_row or {}
    lv = (label_row or {}).get("verdict", "")
    sv = (screen_row or {}).get("verdict", "")
    lf = _norm_final((label_row or {}).get("final"))
    sf = _norm_final((screen_row or {}).get("final"))
    if sv in EXCLUDED:
        rej = sv
    elif lv == "label_unsure":
        rej = lv
    else:
        rej = ""
    if rej:
        final = ""
    elif lv in LABEL_OK and lf:
        final = lf
    elif sv in SCREEN_OK:
        final = sf or _norm_final(base.get("proposed"))
    else:
        final = ""
    out = {k: base.get(k, "") for k in LEDGER_FIELDS}
    out.update(label_verdict=lv, label_final=lf, screen_verdict=sv, screen_final=sf,
               label_page_id=(label_row or {}).get("page_id", ""),
               screen_page_id=(screen_row or {}).get("page_id", ""),
               rejected=rej, final=final, verdict=rej or sv or lv)
    return out


def effective_from_rows(rows) -> dict[str, dict]:
    undone = _undone_pages(rows)
    label: dict[str, dict] = {}
    screen: dict[str, dict] = {}
    order: dict[str, None] = {}
    for r in rows:
        if r["verdict"] == UNDONE or r["page_id"] in undone:
            continue
        (label if r["verdict"] in LABEL_VERDICTS else screen)[r["item_id"]] = r
        order[r["item_id"]] = None
    return {iid: combine_stages(label.get(iid), screen.get(iid)) for iid in order}


def load_effective(path) -> dict[str, dict]:
    """item_id -> two-stage effective record (see combine_stages)."""
    return effective_from_rows(read_ledger_rows(path))


def drift_stamps(eff: dict[str, dict]) -> set[str]:
    """Stamps of frames with an effective drift verdict on any crop."""
    out = set()
    for iid, e in eff.items():
        if e["screen_verdict"] == "drift":
            st, _ = split_item_id(iid, e)
            if st:
                out.add(st)
    return out


def effective_pages(path) -> "OrderedDict[str, list[dict]]":
    """page_id -> its rows, for committed pages that were not undone, in commit order."""
    rows = read_ledger_rows(path)
    undone = _undone_pages(rows)
    out: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        if r["verdict"] != UNDONE and r["page_id"] not in undone:
            out.setdefault(r["page_id"], []).append(r)
    return out


def page_task(page_id: str) -> str:
    """Task that wrote a page: 'g'/'f' + ('v'|'s'|'c')? + session id. 'p...' = propagation."""
    if not page_id:
        return "?"
    if page_id[0] == "p":
        return "propagate"
    if len(page_id) > 1 and page_id[1] == "v":
        return "verify"
    if len(page_id) > 1 and page_id[1] == "s":
        return "screen"
    if len(page_id) > 1 and page_id[1] == "c":
        return "consistency"
    if len(page_id) > 1 and page_id[1] == "a":
        return "audit"
    return "classic"


class LedgerWriter:
    """Append-only CSV writer; flush + fsync after every page."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file() and self.path.stat().st_size > 0:
            with self.path.open("rb") as fh:
                first = fh.readline().decode("utf-8-sig").strip()
                fh.seek(-1, os.SEEK_END)
                last = fh.read(1)
            if first.split(",") != LEDGER_FIELDS:
                raise SystemExit(f"{self.path}: unexpected header {first!r}")
            if last != b"\n":  # torn last line from a crash: terminate it
                with self.path.open("ab") as fh:
                    fh.write(b"\r\n")

    def append(self, rows: list[dict]) -> None:
        new = not self.path.is_file() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _drift_row(stamp: str, pos: int, src: str, proposed: str, queue: str,
               page_id: str, ts: str, item_id: str | None = None) -> dict:
    return {"item_id": item_id or f"{stamp}_dig{pos}", "stamp": stamp, "pos": pos,
            "src": src, "queue": queue, "proposed": proposed, "final": "",
            "verdict": "drift", "page_id": page_id, "reviewed_at": ts}


def propagate_frame_drift(ledger_path, dry_run: bool = False) -> int:
    """Idempotent: for every frame (stamp) with an effective drift verdict, append
    `drift` rows for any of its five positions whose effective screen verdict is not
    drift yet (src found next to a known crop of the frame; proposed = filename
    label; queue = that of the frame's drift row). All rows go in one page (page_id
    'p<timestamp>-<hex>'). Returns the number of rows appended (or that would be)."""
    rows = read_ledger_rows(ledger_path)
    eff = effective_from_rows(rows)
    by_stamp: dict[str, list[dict]] = defaultdict(list)
    for iid, e in eff.items():
        st, _ = split_item_id(iid, e)
        if st:
            by_stamp[st].append(e)
    pid = f"p{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    ts = now_iso()
    new = []
    for st in sorted(drift_stamps(eff)):
        known = by_stamp[st]
        ref = next(e for e in known if e["screen_verdict"] == "drift")
        dirs = [Path(e["src"]).parent for e in known if e.get("src")]
        for p in POSITIONS:
            iid = f"{st}_dig{p}"
            if eff.get(iid, {}).get("screen_verdict") == "drift":
                continue
            src, lab = (eff[iid].get("src", ""), eff[iid].get("proposed", "")) \
                if iid in eff else ("", "")
            if not src:
                for d in dirs:
                    src, lab = find_crop(d, st, p)
                    if src:
                        break
            elif src:
                lab = label_from_filename(src) or lab
            new.append(_drift_row(st, p, src, lab, ref.get("queue", ""), pid, ts, iid))
    if new and not dry_run:
        LedgerWriter(ledger_path).append(new)
    return len(new)


# --------------------------------------------------------------------------
# consistency mode: items built from the ledger itself
# --------------------------------------------------------------------------


def parse_positions(spec) -> tuple[int, ...]:
    """'dig6,dig5' / '6,5' / '' -> (6, 5) / all positions (in consistency order)."""
    toks = [t for t in str(spec or "").replace(" ", "").split(",") if t]
    if not toks:
        return CONSISTENCY_POS_ORDER
    out = []
    for t in toks:
        p = norm_pos(t)
        if p not in out:
            out.append(p)
    return tuple(out)


def _prior_marker(e: dict) -> str:
    """'v' / 'f' if the item's latest label-stage row is a label_ok / label_fixed written
    outside consistency mode, else ''."""
    if page_task(e.get("label_page_id", "")) == "consistency":
        return ""
    return {"label_ok": "v", "label_fixed": "f"}.get(e.get("label_verdict", ""), "")


def consistency_items(ledger_path, queue_paths=(), derive: DeriveIndex | None = None,
                      positions=None) -> tuple[list[QItem], set[str], Counter]:
    """Items for --mode consistency, built from the ledger (not the queue files).

    Scope: every effectively accepted item (not rejected, final label present) --
    holdout included -- plus items rejected *by a consistency page* (so the totals
    stay stable across sessions and their pages can be undone). `proposed` = the
    current effective final label (for the latter: their last final before the
    rejection). src from the ledger row, else from the queue files; bucket from the
    queue files, else b3_derive, else the hour rule; brightness from b3_derive
    frame_brightness (None -> caller computes crop luma); context / preflag / model
    reading from the queue files when present.

    Returns (items, loaded_ids, info): loaded_ids = every ledger item id at the
    selected positions (undo may cancel rows of any of them); info = counters.
    """
    positions = set(positions or POSITIONS)
    rows = read_ledger_rows(ledger_path)
    undone = _undone_pages(rows)
    last_final: dict[str, str] = {}
    for r in rows:
        if r["verdict"] != UNDONE and r["page_id"] not in undone and _norm_final(r.get("final")):
            last_final[r["item_id"]] = _norm_final(r.get("final"))
    eff = effective_from_rows(rows)
    meta: dict[str, QItem] = {}
    if queue_paths:
        qitems, _ = load_queues(queue_paths)
        meta = {it.item_id: it for it in qitems}
    items: list[QItem] = []
    loaded: set[str] = set()
    info: Counter = Counter()
    for iid, e in eff.items():
        st, pos = split_item_id(iid, e)
        if pos is None or pos not in positions:
            continue
        loaded.add(iid)
        if e["rejected"]:
            pid = e["label_page_id"] if e["rejected"] == "label_unsure" else e["screen_page_id"]
            if page_task(pid) != "consistency":
                info["rejected_before"] += 1
                continue
            lab = last_final.get(iid, "")
            info["rejected_in_consistency"] += 1
        else:
            lab = e["final"]
        try:
            lab = norm_label(lab)
        except ValueError:
            info["no_label"] += 1
            continue
        m = meta.get(iid)
        src = (e.get("src") or "").strip() or (m.src if m else "")
        if not src:
            info["no_src"] += 1
        d = derive.get(st, pos) if derive is not None else None
        bucket = (m.bucket if m and m.bucket else "") or (d.bucket if d else "")
        if not bucket and len(st) >= 11 and st[9:11].isdigit():
            bucket = bucket_for_hour(int(st[9:11]))
        model = m.model_label if m else ""
        if not model and d is not None and d.model not in ("", "?"):
            model = d.model
        items.append(QItem(
            queue=(e.get("queue") or "").strip() or (m.queue if m else "") or "ledger",
            item_id=iid, stamp=st, pos=pos, src=src, proposed=lab, bucket=bucket,
            brightness=d.bright if d is not None else None,
            context=m.context if m else "", preflag=m.preflag if m else "",
            group=m.group if m else "", order=len(items),
            needs_verify=m.needs_verify if m else False, model_label=model,
            orig_proposed=m.proposed if m else _norm_final(e.get("proposed")),
            prior=_prior_marker(e)))
    info["items"] = len(items)
    return items, loaded, info


# --------------------------------------------------------------------------
# page + marks
# --------------------------------------------------------------------------


@dataclass
class Page:
    kind: str                                  # "grid" | "frame"
    items: list[QItem]
    cell: tuple | None = None                  # grid mode
    frames: list[tuple[tuple, dict[int, QItem]]] = field(default_factory=list)
    marks: dict[str, tuple[str, str]] = field(default_factory=dict)
    restored_from: str = ""                    # page_id this was undone from
    task: str = "classic"

    @property
    def default_verdict(self) -> str:
        return "label_ok" if self.task in LABEL_TASKS else "ok"

    def _mates(self, item: QItem) -> list[QItem]:
        if not item.stamp:
            return [item]
        return [it for it in self.items if it.stamp == item.stamp] or [item]

    def mark(self, item: QItem, action: str) -> list[str]:
        """action: artifact|drift|illegible|ok_derived|unsure|clear|<label>.

        Drift is frame-wide: marking it marks every tile of the same stamp on the
        page; changing or clearing the drift mark of any of them clears the frame.
        Returns the item_ids whose mark may have changed."""
        iid = item.item_id
        cur = self.marks.get(iid)
        touched = [iid]
        if cur and cur[0] == "drift":
            # any change away from drift (incl. pressing d again) clears the frame
            for m in self._mates(item):
                if self.marks.get(m.item_id, ("",))[0] == "drift":
                    self.marks.pop(m.item_id, None)
                    touched.append(m.item_id)
            if action in ("drift", "clear"):
                return touched
            cur = None
        if action == "drift":
            for m in self._mates(item):
                self.marks[m.item_id] = ("drift", "")
                touched.append(m.item_id)
            return touched
        if action == "clear":
            self.marks.pop(iid, None)
        elif action in EXCLUDED or action in ("ok_derived", "unsure"):
            verdict = "label_unsure" if action == "unsure" else action
            if cur and cur[0] == verdict:
                self.marks.pop(iid, None)          # same key twice toggles off
            else:
                self.marks[iid] = (verdict, item.proposed if action == "ok_derived" else "")
        else:
            lab = norm_label(action)
            if lab == item.proposed:
                self.marks.pop(iid, None)
            else:
                self.marks[iid] = ("label_fixed" if self.task in LABEL_TASKS else "relabeled",
                                   lab)
        return touched

    def resolved(self, item: QItem) -> tuple[str, str]:
        return self.marks.get(item.item_id, (self.default_verdict, item.proposed))

    def commit_rows(self, page_id: str, ts: str | None = None) -> list[dict]:
        ts = ts or now_iso()
        rows = []
        for it in self.items:
            verdict, final = self.resolved(it)
            rows.append({
                "item_id": it.item_id, "stamp": it.stamp, "pos": it.pos,
                "src": it.src, "queue": it.queue, "proposed": it.proposed,
                "final": final, "verdict": verdict, "page_id": page_id,
                "reviewed_at": ts,
            })
        return rows


# --------------------------------------------------------------------------
# session (pure logic: paging, commit, undo, resume, progress)
# --------------------------------------------------------------------------


class Session:
    def __init__(self, items: list[QItem], ledger_path, *, mode: str = "grid",
                 task: str = "classic", page_size: int = 48, frames_per_page: int = 8,
                 cell_order: str = "csv", derive: DeriveIndex | None = None,
                 extra_loaded_ids=(), redo_since: str = ""):
        """items: the loaded crops. In consistency mode they come from
        consistency_items() (proposed = current effective label) and
        extra_loaded_ids its loaded_ids (undo scope)."""
        if mode not in ("grid", "frame"):
            raise ValueError(mode)
        if task == "consistency" and mode != "grid":
            raise ValueError("consistency mode is grid-only")
        if task not in TASKS:
            raise ValueError(task)
        if cell_order not in CELL_ORDERS:
            raise ValueError(cell_order)
        self.mode = mode
        self.task = task
        self.derive = derive
        # audit --redo-since: re-open crops whose last label ruling (incl. "can't tell")
        # predates this stamp, e.g. to revisit x calls with a wider time strip
        self.redo_since = redo_since
        self.page_size = max(1, page_size)
        self.frames_per_page = max(1, frames_per_page)
        self.ledger_path = Path(ledger_path)
        self.writer = LedgerWriter(self.ledger_path)
        self._reload()

        # every loaded crop (for frame neighbours / drift propagation), then the
        # task's scope
        items = [replace(it) for it in items]
        self.all_items = items
        self.by_stamp_pos = {(it.stamp, it.pos): it for it in items}
        self.loaded_ids = {it.item_id for it in items} | set(extra_loaded_ids)
        self.unverified_skipped = 0
        self.not_needing_verify = 0
        if task == "verify":
            self.not_needing_verify = sum(1 for it in items if not it.needs_verify)
            items = [it for it in items if it.needs_verify]
        elif task == "screen":
            keep = []
            for it in items:
                e = self.eff.get(it.item_id)
                lv = e["label_verdict"] if e else ""
                if it.needs_verify and not lv:
                    if not self._rejected(it):
                        self.unverified_skipped += 1
                    continue
                if lv in LABEL_OK and e["label_final"]:
                    it.orig_proposed = it.proposed
                    it.proposed = norm_label(e["label_final"])
                keep.append(it)
            items = keep
        self.items = items
        self.by_id = {it.item_id: it for it in items}
        self.preexisting = sum(1 for it in items if self.done(it))

        # queues in first-appearance (= command line) order
        self.queues: list[str] = list(OrderedDict.fromkeys(it.queue for it in items))
        self.cell_order = cell_order
        self._build_cells()
        # frames
        frame_items: OrderedDict[tuple, list[QItem]] = OrderedDict()
        for it in items:
            frame_items.setdefault(it.frame_key, []).append(it)
        self.frame_items = frame_items

        self.forced: list[str] = []
        self.forced_marks: dict[str, tuple[str, str]] = {}
        self.forced_from = ""
        self.session_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        self.page_counter = 0
        self.stats: Counter = Counter()
        self.committed = 0
        self.pages_committed = 0
        self.t0 = time.time()

    # ---- cells ----
    def cell_of(self, it: QItem) -> tuple:
        """Grid cell of an item. Consistency: ('consistency', pos, label, 'all') --
        every queue and light bucket mixed."""
        if self.task == "consistency":
            return ("consistency", it.pos, it.proposed, "all")
        return it.cell

    def _build_cells(self) -> None:
        cell_items: OrderedDict[tuple, list[QItem]] = OrderedDict()
        for it in self.items:
            cell_items.setdefault(self.cell_of(it), []).append(it)
        for lst in cell_items.values():
            lst.sort(key=QItem.sort_key)
        self.cell_items = cell_items
        if self.task == "consistency":
            def rank(c):
                pos, lab = c[1], c[2]
                return (CONSISTENCY_POS_ORDER.index(pos) if pos in CONSISTENCY_POS_ORDER
                        else 99, CONSISTENCY_CLASS_ORDER.index(lab))
            self.cells = sorted(cell_items, key=rank)
            return
        ordered = []
        for q in self.queues:
            cells = [c for c in cell_items if c[0] == q]
            if self.cell_order == "size":
                cells.sort(key=lambda c: -len(cell_items[c]))
            elif self.cell_order == "small":
                cells.sort(key=lambda c: len(cell_items[c]))
            elif self.cell_order == "sorted":
                cells.sort(key=lambda c: (c[1], LABEL_SET.index(c[2]), c[3]))
            ordered.extend(cells)
        self.cells = ordered

    def _regroup(self) -> None:
        """Consistency: follow the ledger -- an accepted item's label is its current
        effective final (a fix moves it to its new group, where it counts as done; an
        undo moves it back) and its v/f marker follows the label stage."""
        if self.task != "consistency":
            return
        for it in self.items:
            e = self.eff.get(it.item_id)
            if not e:
                continue
            if not e["rejected"] and e["final"]:
                try:
                    it.proposed = norm_label(e["final"])
                except ValueError:
                    pass
            it.prior = _prior_marker(e)
        self._build_cells()

    # ---- ledger state ----
    def _reload(self) -> None:
        rows = read_ledger_rows(self.ledger_path)
        self.eff = effective_from_rows(rows)
        self.drift = drift_stamps(self.eff)
        undone = _undone_pages(rows)
        self.page_log: OrderedDict[str, list[dict]] = OrderedDict()
        for r in rows:
            if r["verdict"] != UNDONE and r["page_id"] not in undone:
                self.page_log.setdefault(r["page_id"], []).append(r)

    @property
    def reviewed(self) -> dict[str, dict]:
        """item_id -> effective record (both stages)."""
        return self.eff

    def _rejected(self, it: QItem) -> bool:
        e = self.eff.get(it.item_id)
        return bool(it.stamp and it.stamp in self.drift) or bool(e and e["rejected"])

    # ---- progress ----
    def done(self, it: QItem) -> bool:
        if it.stamp and it.stamp in self.drift:
            return True                         # drift frames are skipped everywhere
        e = self.eff.get(it.item_id)
        if e is None:
            return False
        if self.task == "classic":
            return True
        if self.task == "audit" and self.redo_since:
            lp = e.get("label_page_id", "")
            if page_task(lp) == "audit" and lp[2:17] >= self.redo_since:
                return True
            return bool(e["rejected"]) and e["rejected"] != "label_unsure"
        if e["rejected"]:
            return True
        if self.task == "verify":
            return bool(e["label_verdict"])
        if self.task == "consistency":
            return page_task(e.get("label_page_id", "")) == "consistency"
        if self.task == "audit":
            # re-check already-labelled crops: done only once an audit page has ruled
            return page_task(e.get("label_page_id", "")) == "audit"
        return bool(e["screen_verdict"])

    def cell_progress(self, cell) -> tuple[int, int]:
        lst = self.cell_items.get(cell, [])
        return sum(self.done(it) for it in lst), len(lst)

    def queue_progress(self, queue) -> tuple[int, int]:
        lst = [it for it in self.items if it.queue == queue]
        return sum(self.done(it) for it in lst), len(lst)

    def overall_progress(self) -> tuple[int, int]:
        return sum(self.done(it) for it in self.items), len(self.items)

    def pages_left_in_cell(self, cell) -> int:
        d, t = self.cell_progress(cell)
        return math.ceil((t - d) / self.page_size)

    def group_page_numbers(self, cell) -> tuple[int, int]:
        """(this page, pages in the group) for the header, e.g. (3, 7)."""
        d, t = self.cell_progress(cell)
        total = max(1, math.ceil(t / self.page_size))
        left = self.pages_left_in_cell(cell)
        return max(1, min(total, total - left + 1)), total

    def todo_counts(self) -> Counter:
        """(pos, label) -> items not done yet."""
        return Counter(self.cell_of(it)[1:3] for it in self.items if not self.done(it))

    # ---- paging ----
    def _make_page(self, items: list[QItem]) -> Page:
        if self.mode == "grid":
            items = sorted(items, key=QItem.sort_key)
            return Page("grid", items, cell=self.cell_of(items[0]), task=self.task)
        frames: OrderedDict[tuple, dict[int, QItem]] = OrderedDict()
        for it in items:
            frames.setdefault(it.frame_key, {})[it.pos] = it
        flat = [it for _, d in frames.items() for _, it in sorted(d.items())]
        return Page("frame", flat, frames=list(frames.items()), task=self.task)

    def next_page(self) -> Page | None:
        if self.forced:
            rem = [self.by_id[i] for i in self.forced
                   if i in self.by_id and not self.done(self.by_id[i])]
            if rem:
                pg = self._make_page(rem)
                pg.marks = {k: v for k, v in self.forced_marks.items()
                            if k in {it.item_id for it in rem}}
                pg.restored_from = self.forced_from
                return pg
            self.forced, self.forced_marks, self.forced_from = [], {}, ""

        if self.mode == "grid":
            for cell in self.cells:
                rem = [it for it in self.cell_items[cell] if not self.done(it)]
                if rem:
                    n_pages = math.ceil(len(rem) / self.page_size)
                    size = math.ceil(len(rem) / n_pages)   # balanced: no 1-tile pages
                    return Page("grid", rem[:size], cell=cell, task=self.task)
            return None

        picked: list[QItem] = []
        nf = 0
        for fk, lst in self.frame_items.items():
            rem = [it for it in lst if not self.done(it)]
            if rem:
                picked.extend(rem)
                nf += 1
                if nf >= self.frames_per_page:
                    break
        return self._make_page(picked) if picked else None

    # ---- commit / undo ----
    def new_page_id(self) -> str:
        self.page_counter += 1
        return f"{self.mode[0]}{TASK_LETTER[self.task]}{self.session_id}-{self.page_counter:04d}"

    def frame_drift_rows(self, rows: list[dict], page_id: str, ts: str) -> list[dict]:
        """Extra drift rows: every position of each drift-marked frame that is not on
        the page and not already effective drift."""
        have = {r["item_id"] for r in rows}
        out = []
        seen_st: set[str] = set()
        for ref in rows:
            st = ref["stamp"]
            if ref["verdict"] != "drift" or not st or st in seen_st:
                continue
            seen_st.add(st)
            d = Path(ref["src"]).parent if ref.get("src") else None
            for p in POSITIONS:
                q = self.by_stamp_pos.get((st, p))
                iid = q.item_id if q else f"{st}_dig{p}"
                if iid in have or self.eff.get(iid, {}).get("screen_verdict") == "drift":
                    continue
                if q and q.src:
                    src, lab = q.src, q.proposed
                else:
                    src, lab = find_crop(d, st, p)
                out.append(_drift_row(st, p, src, lab, ref["queue"], page_id, ts, iid))
                have.add(iid)
        return out

    def commit(self, page: Page) -> str:
        pid = self.new_page_id()
        ts = now_iso()
        rows = page.commit_rows(pid, ts)
        rows += self.frame_drift_rows(rows, pid, ts)
        self.writer.append(rows)
        for r in rows:
            self.stats[r["verdict"]] += 1
        self.committed += len(rows)
        self.pages_committed += 1
        self.forced, self.forced_marks, self.forced_from = [], {}, ""
        self._reload()
        self._regroup()
        return pid

    def undo(self) -> tuple[bool, str]:
        """Undo the most recently committed page (this or an earlier session).

        Refuses if that page was written by another review mode, or has items not
        loaded in this session (so a stray `u` can never silently cancel work
        belonging to another queue). Frame-drift rows written for positions that
        are not loaded do not block the undo; they are removed with the page.
        """
        if not self.page_log:
            return False, "nothing to undo"
        pid, rows = next(reversed(self.page_log.items()))
        ptask = page_task(pid)
        if ptask != self.task:
            return False, (f"last committed page {pid} was written in {ptask} mode; "
                           f"not undoing from {self.task} mode")
        drift_st = {r["stamp"] for r in rows
                    if r["verdict"] == "drift" and r["item_id"] in self.loaded_ids}
        absent = [r["item_id"] for r in rows if r["item_id"] not in self.loaded_ids
                  and not (r["verdict"] == "drift" and r["stamp"] in drift_st)]
        if absent:
            return False, (f"last committed page {pid} has {len(absent)} item(s) not in "
                           f"the loaded queues (e.g. {absent[0]}); not undoing")
        ts = now_iso()
        tomb = [{**{k: r.get(k, "") for k in LEDGER_FIELDS},
                 "final": "", "verdict": UNDONE, "page_id": pid, "reviewed_at": ts}
                for r in rows]
        self.writer.append(tomb)
        if self.session_id in pid:
            for r in rows:
                self.stats[r["verdict"]] -= 1
            self.committed -= len(rows)
            self.pages_committed -= 1
        self._reload()
        self._regroup()
        # the page's own tiles (not the frame-drift rows it dragged in from other
        # cells): rows are written page tiles first, so the first loaded row
        # defines the grid cell
        prim = [r for r in rows if r["item_id"] in self.by_id
                and self.by_id[r["item_id"]].queue == r["queue"]]
        if pid[:1] == "g" and prim:
            cell0 = self.cell_of(self.by_id[prim[0]["item_id"]])
            prim = [r for r in prim if self.cell_of(self.by_id[r["item_id"]]) == cell0]
        default = "label_ok" if self.task in LABEL_TASKS else "ok"
        self.forced = [r["item_id"] for r in prim]
        self.forced_marks = {r["item_id"]: (r["verdict"], r["final"]) for r in prim
                             if r["verdict"] != default}
        self.forced_from = pid
        return True, (f"undid page {pid} ({len(rows)} rows) -- marks restored, fix and Enter")

    # ---- neighbours for zoom ----
    def frame_neighbours(self, it: QItem) -> dict[int, tuple[str, str]]:
        """pos -> (path, label) for the other digit positions of the same frame."""
        out: dict[int, tuple[str, str]] = {}
        d = Path(it.src).parent if it.src else None
        for p in POSITIONS:
            if p == it.pos:
                out[p] = (it.src, it.proposed)
                continue
            q = self.by_stamp_pos.get((it.stamp, p))
            if q and q.src:
                out[p] = (q.src, q.proposed)
                continue
            src, lab = find_crop(d, it.stamp, p)
            if src:
                out[p] = (src, lab or "?")
        return out

    def frame_reading(self, it: QItem) -> str:
        """One-line derived reading of the item's frame (b3_derive, else context)."""
        dr = self.derive.get(it.stamp, it.pos) if self.derive else None
        if dr is not None:
            per = []
            for p in POSITIONS:
                x = self.derive.get(it.stamp, p)
                per.append(f"{x.truth if x and x.truth not in ('', '?') else '?'}")
            return (f"screen {dr.screen}; reading_est {dr.est or '?'}; truth "
                    f"{''.join(per)}; status {dr.status}")
        m = re.search(r"reading[_ a-z]*\s+(\d{5})", it.context or "")
        return f"reading {m.group(1)} (from context)" if m else "no derived reading"

    def time_strip(self, it: QItem, n: int = 3) -> list[dict]:
        """Same position in the previous n / next n reading-screen frames (b3_derive),
        with the item itself in the middle. Each entry: stamp, path, lines (caption),
        current. Non-b3 items: the nearest crops of that position in the same
        directory (filename labels only)."""
        def hhmm(st: str) -> str:
            same_day = st[:8] == it.stamp[:8]
            t = f"{st[9:11]}:{st[11:13]}" if len(st) >= 13 else st
            return t if same_day else f"{st[4:6]}-{st[6:8]} {t}"

        def drow_entry(d: DRow, cur: bool) -> dict:
            truth = d.truth if d.truth not in ("", "?") else "?"
            # h = the human label currently in effect for this neighbour (blank if never
            # reviewed, x if rejected) -- the derivation (t/e) can be wrong for hours
            e = self.eff.get(f"{d.stamp}_dig{d.pos}")
            h = "" if e is None else ("x" if e["rejected"] else (e["final"] or ""))
            return {"stamp": d.stamp, "path": d.path, "current": cur,
                    "lines": [hhmm(d.stamp) + (f"  {d.est}" if d.est else ""),
                              f"t {truth}  e {est_digit(d.est, d.pos)}  m {d.model}",
                              f"h {h or '-'}"]}

        if self.derive is not None and (self.derive.get(it.stamp, it.pos) is not None):
            prev, nxt = self.derive.neighbours(it.stamp, it.pos, n)
            me = self.derive.get(it.stamp, it.pos)
            cur = drow_entry(me, True)
            cur["path"] = it.src or me.path
            return [drow_entry(d, False) for d in prev] + [cur] + \
                [drow_entry(d, False) for d in nxt]
        # fallback: neighbouring crops of the same position in the same directory
        cur = {"stamp": it.stamp, "path": it.src, "current": True,
               "lines": [hhmm(it.stamp), f"prop {it.proposed}"
                         + (f"  m {it.model_label}" if it.model_label else "")]}
        if not it.src or not Path(it.src).parent.is_dir():
            return [cur]
        cands = []
        for f in Path(it.src).parent.glob(f"*_dig{it.pos}_*.jpg"):
            m = _STAMP_RE.search(f.name)
            if m and m.group(1) != it.stamp:
                cands.append((m.group(1), f))
        cands.sort()
        stamps = [c[0] for c in cands]
        i = bisect.bisect_left(stamps, it.stamp)

        def ent(c):
            return {"stamp": c[0], "path": str(c[1]), "current": False,
                    "lines": [hhmm(c[0]), f"file {label_from_filename(c[1].name) or '?'}"]}
        return [ent(c) for c in cands[max(0, i - n):i]] + [cur] + \
            [ent(c) for c in cands[i:i + n]]

    # ---- summary ----
    def summary(self) -> str:
        mins = max((time.time() - self.t0) / 60.0, 1e-9)
        d, t = self.overall_progress()
        counts = "  ".join(f"{v}={self.stats[v]}" for v in VERDICTS if self.stats[v])
        rate = self.committed / mins
        return (f"grid_review {self.task} session {self.session_id}: {self.committed} rows in "
                f"{self.pages_committed} pages, {mins:.1f} min, {rate:.1f} crops/min"
                f" ({60.0 / rate if rate else 0:.2f} s/crop) | {counts or 'no verdicts'}"
                f" | overall {d}/{t} done | ledger {self.ledger_path}")


# --------------------------------------------------------------------------
# sample queue builder (for exercising the tool before real queues exist)
# --------------------------------------------------------------------------


def make_sample_queue(out, root, n: int = 200, seed: int = 0,
                      queue_name: str = "sample", preflag_frac: float = 0.0) -> int:
    root = Path(root)
    frames: OrderedDict[str, list[tuple[int, str, Path]]] = OrderedDict()
    for f in sorted(root.rglob("*.jpg")):
        lab = label_from_filename(f.name)
        m_s, m_p = _STAMP_RE.search(f.name), _POS_RE.search(f.name)
        if lab is None or not m_s or not m_p:
            continue
        pos = int(m_p.group(1))
        if pos not in POSITIONS:
            continue
        frames.setdefault(m_s.group(1), []).append((pos, lab, f))
    stamps = list(frames)
    rng = random.Random(seed)
    rng.shuffle(stamps)
    rows = []
    for s in stamps:
        for pos, lab, f in sorted(frames[s]):
            rows.append((s, pos, lab, f))
        if len(rows) >= n:
            break
    rows = rows[:n]
    rows.sort(key=lambda r: (r[0], r[1]))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=QUEUE_FIELDS)
        w.writeheader()
        for s, pos, lab, f in rows:
            reading = "".join(lab2 for _, lab2, _ in sorted(frames[s]))
            w.writerow({
                "queue": queue_name, "item_id": f"{s}_dig{pos}", "stamp": s,
                "pos": pos, "src": str(f.resolve()), "proposed": lab,
                "bucket": bucket_for_hour(int(s[9:11])),
                "brightness": f"{image_brightness(str(f)) or 0:.1f}",
                "context": f"filename reading {reading}; file {f.parent.name}/{f.name}",
                "preflag": "drift" if rng.random() < preflag_frac else "",
                "group": "", "needs_verify": "0",
            })
    return len(rows)


# --------------------------------------------------------------------------
# Tk view
# --------------------------------------------------------------------------

BG = "#1e1e1e"
FG = "#eeeeee"
SEL = "#ffffff"
MARK_COLORS = {
    "ok": "#3a3a3a",
    "label_ok": "#3a3a3a",
    "relabeled": "#ffd24a",
    "label_fixed": "#ffd24a",
    "ok_derived": "#3fc3ff",
    "artifact": "#ff3b3b",
    "drift": "#ff9a1f",
    "illegible": "#c070ff",
    "label_unsure": "#c070ff",
}
MARK_TEXT = {"ok_derived": "OK-D", "artifact": "ART", "drift": "DRIFT", "illegible": "ILL",
             "label_unsure": "??"}

# NumLock off: Windows sends navigation keysyms from the keypad.
KP_NUMLOCK_OFF = {
    "KP_Insert": "0", "KP_End": "1", "KP_Down": "2", "KP_Next": "3",
    "KP_Left": "4", "KP_Begin": "5", "KP_Right": "6", "KP_Home": "7",
    "KP_Up": "8", "KP_Prior": "9",
}
ACTION_KEYS = {"a": "artifact", "d": "drift", "i": "illegible", "o": "ok_derived",
               "x": "unsure", "question": "unsure", "c": "clear", "space": "clear"}

FOOTERS = {
    "classic": ("hover/click/arrows select | a artifact  d drift (whole frame)  i illegible  "
                "o ok-derived  0-9/n relabel  Space/c clear  right-click artifact | "
                "z/dbl-click zoom  v ROI<->20x32 | Enter commit page  u/Ctrl+Z undo  q quit"),
    "audit": ("AUDIT: the model confidently disagrees with these labels | Enter = every "
              "unmarked tile IS the label shown | 0-9/n = it is that digit  x/? = can't tell | "
              "a artifact  d drift | z zoom (frame + time strip)  v ROI<->20x32  u undo  q quit"),
    "verify": ("VERIFY: Enter = every unmarked tile IS the label shown | 0-9/n = it is that "
               "digit  x/? = can't tell | a artifact  d drift (whole frame)  Space/c clear | "
               "z/dbl-click zoom (frame + time strip)  v ROI<->20x32  u undo  q quit"),
    "screen": ("SCREEN: only flag a artifact / d drift (whole frame); Enter = unmarked ok | "
               "Space/c clear  right-click artifact | z/dbl-click zoom  v ROI<->20x32 | "
               "u/Ctrl+Z undo  q quit"),
    "consistency": ("CONSISTENCY: Enter = every unmarked tile IS the label in the header | "
                    "0-9/n = it is that digit  x/? = can't tell | a artifact  d drift (whole "
                    "frame)  Space/c clear | z zoom (frame + time strip)  v ROI<->20x32  "
                    "u undo  q quit      caption: D/T/F light bucket, v verified / f fixed "
                    "earlier"),
}
BUCKET_COLORS = {"D": "#ffe08a", "T": "#ff9a1f", "F": "#6fb7ff"}
PRIOR_COLORS = {"v": "#7fdc7f", "f": "#ffd24a"}


def key_action(ev) -> str | None:
    sym = ev.keysym
    if sym in KP_NUMLOCK_OFF:
        return KP_NUMLOCK_OFF[sym]
    if sym.startswith("KP_") and sym[3:].isdigit():
        return sym[3:]
    if len(sym) == 1 and sym.isdigit():
        return sym
    if sym in ("n", "N", "minus", "KP_Subtract"):
        return "N"
    low = sym.lower() if len(sym) == 1 else sym
    return ACTION_KEYS.get(low)


def action_allowed(task: str, action: str) -> bool:
    if action == "clear" or action in LABEL_SET:
        return True
    return action in TASK_ACTIONS[task]


def badge_text(preflag: str, task: str) -> str:
    """Tile badge from the preflag tokens. Verify mode hides model_disagrees (the
    caption shows the model's reading); screen mode shows only the drift flags."""
    if task == "classic":
        return f"!{preflag[:5]}" if preflag else ""
    toks = [t for t in (preflag or "").split("|") if t]
    if task in ("verify", "screen", "consistency", "audit"):
        toks = [t for t in toks if t != "model_disagrees"]
    if task in ("screen", "consistency"):
        toks = [t for t in toks if t.startswith("drift")]
    if not toks:
        return ""
    return "!" + toks[0][:6]


def _dpi_aware() -> None:
    """1:1 physical pixels on scaled Windows displays (no bitmap blur)."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


@dataclass
class Tile:
    item: QItem
    x0: int
    y0: int
    gr: int
    gc: int
    border: int = 0
    tag_bg: int = 0
    tag_txt: int = 0


class GridApp:
    BORDER = 4
    GAP = 6
    LABEL_W = 190

    def __init__(self, session: Session, args):
        import tkinter as tk
        from tkinter import font as tkfont
        self.tk = tk
        self.s = session
        self.args = args
        self.task = session.task
        self.CAP = {"verify": 24, "audit": 24, "consistency": 30}.get(self.task, 18)
        self.th = args.tile_h
        # tile box follows the crop aspect (~94x202); the 20x32 model view is
        # drawn into the same box, i.e. un-distorted back to crop geometry
        aspect = 94 / 202
        for it in session.items[:20]:
            try:
                with Image.open(it.src) as im:
                    aspect = im.width / im.height
                break
            except Exception:
                continue
        self.box_w = max(16, round(self.th * aspect))
        self.cw = self.box_w + 2 * self.BORDER + self.GAP
        self.ch = self.th + 2 * self.BORDER + self.CAP + self.GAP
        self.model_view = bool(getattr(args, "model_view", False))
        self.page: Page | None = None
        self.tiles: list[Tile] = []
        self.sel = 0
        self.photos: list = []
        self.zphotos: list = []
        self.cache: OrderedDict[str, Image.Image] = OrderedDict()
        self.zoom_win = None
        self.msg = ""

        self.root = tk.Tk()
        self.root.title(f"grid_review {'' if self.task == 'classic' else self.task}")
        self.root.configure(bg=BG)
        self.f_small = tkfont.Font(family="Consolas", size=8)
        self.f_tag = tkfont.Font(family="Consolas", size=9, weight="bold")
        self.f_big = tkfont.Font(family="Consolas", size=13, weight="bold")
        self.header = tk.Label(self.root, font=("Consolas", 12), fg=FG, bg=BG,
                               justify="left", anchor="w")
        self.header.pack(fill="x", padx=10, pady=(6, 2))
        if session.mode == "grid":
            w = args.cols * self.cw + 8
            h = math.ceil(session.page_size / args.cols) * self.ch + 4
        else:
            fc = args.frame_cols
            w = fc * (self.LABEL_W + 5 * self.cw + 14)
            h = math.ceil(session.frames_per_page / fc) * self.ch + 4
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(padx=8, anchor="w")
        self.status = tk.Label(self.root, font=("Consolas", 10), fg="#9cf", bg=BG,
                               anchor="w", justify="left")
        self.status.pack(fill="x", padx=10)
        self.footer = tk.Label(self.root, font=("Consolas", 10), fg="#888", bg=BG,
                               anchor="w", justify="left", text=FOOTERS[self.task])
        self.footer.pack(fill="x", padx=10, pady=(0, 6))

        self.root.bind("<Key>", self.on_key)
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", lambda e: self.show_status())
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Double-Button-1>", self.on_double)
        self.canvas.bind("<Button-3>", self.on_right)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.geometry("+0+0")
        self.root.focus_force()
        self.load_page()

    # ---------- images ----------
    def pil(self, path: str) -> Image.Image | None:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception:
            img = None
        self.cache[path] = img
        if len(self.cache) > 600:
            self.cache.popitem(last=False)
        return img

    def render_img(self, img: Image.Image, h: int, model: bool, roi_resample=None,
                   store=None):
        from PIL import ImageTk
        w = max(1, round(img.width * h / img.height))
        if model:
            small = img.resize(MODEL_INPUT, Image.Resampling.NEAREST)
            out = small.resize((w, h), Image.Resampling.NEAREST)
        else:
            rs = roi_resample or (Image.Resampling.NEAREST if h >= img.height
                                  and h % img.height == 0 else Image.Resampling.LANCZOS)
            out = img.resize((w, h), rs)
        ph = ImageTk.PhotoImage(out)
        (self.photos if store is None else store).append(ph)
        return ph

    # ---------- page ----------
    def load_page(self) -> None:
        self.page = self.s.next_page()
        self.sel = 0
        if self.page and self.page.restored_from:
            # select the first restored mark so the fix is one keypress away
            for k, it in enumerate(self.page.items):
                if it.item_id in self.page.marks:
                    self.sel = k
                    break
        self.draw()

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        self.photos.clear()
        self.tiles = []
        pg = self.page
        if pg is None:
            d, t = self.s.overall_progress()
            extra = ""
            if self.task == "screen" and self.s.unverified_skipped:
                extra = (f"\n{self.s.unverified_skipped} crop(s) still need verification "
                         f"(--mode verify) before they can be screened.")
            c.create_text(20, 20, anchor="nw", fill=FG, font=("Consolas", 16),
                          text=f"All {self.task if self.task != 'classic' else ''} "
                               f"crops done ({d}/{t}).  u undo, q quit.{extra}")
            self.update_header()
            self.show_status()
            return
        B = self.BORDER
        if pg.kind == "grid":
            for k, it in enumerate(pg.items):
                r, col = divmod(k, self.args.cols)
                self.tiles.append(Tile(it, 4 + col * self.cw, 2 + r * self.ch, r, col))
        else:
            fc = self.args.frame_cols
            block_w = self.LABEL_W + 5 * self.cw + 14
            for fi, (fk, slots) in enumerate(pg.frames):
                r, bc = divmod(fi, fc)
                bx, by = bc * block_w, 2 + r * self.ch
                self.draw_frame_label(bx, by, slots)
                for p, it in sorted(slots.items()):
                    pc = POSITIONS.index(p)
                    self.tiles.append(Tile(it, bx + self.LABEL_W + pc * self.cw, by,
                                           r, bc * 5 + pc))
        for t in self.tiles:
            it = t.item
            x0, y0 = t.x0, t.y0
            x1, y1 = x0 + self.box_w + 2 * B, y0 + self.th + 2 * B
            t.border = c.create_rectangle(x0 + B / 2, y0 + B / 2, x1 - B / 2, y1 - B / 2,
                                          outline=MARK_COLORS["ok"], width=B)
            img = self.pil(it.src) if it.src else None
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if img is None:
                c.create_rectangle(x0 + B, y0 + B, x1 - B, y1 - B, fill="#444", outline="")
                c.create_text(cx, cy, text="missing", fill="#f88", font=self.f_small)
            else:
                ph = self.render_img(img, self.th, self.model_view)
                c.create_image(cx, cy, image=ph)
            hhmm = f"{it.stamp[9:11]}:{it.stamp[11:13]} {it.stamp[4:6]}-{it.stamp[6:8]}" \
                if len(it.stamp) >= 13 else it.stamp
            if self.task in ("verify", "audit"):
                # the label being verified, big; the model's reading + time small
                c.create_text(x0 + 1, y1, anchor="nw", text=it.proposed, fill="#ffd24a",
                              font=self.f_big)
                small = (f"m{it.model_label or '?'} " if it.model_label else "m? ") + \
                    (hhmm[:5] if pg.kind == "grid" else f"dig{it.pos}")
                c.create_text(x0 + 18, y1 + 4, anchor="nw", text=small, fill="#aaa",
                              font=self.f_small)
            elif self.task == "consistency":
                # time + date, then light bucket letter and the earlier-review marker
                c.create_text(x0 + 1, y1 + 1, anchor="nw", text=hhmm, fill="#aaa",
                              font=self.f_small)
                bl = BUCKET_LETTER.get(it.bucket, "?")
                c.create_text(x0 + 1, y1 + 13, anchor="nw", text=bl,
                              fill=BUCKET_COLORS.get(bl, "#888"), font=self.f_tag)
                if it.prior:
                    c.create_text(x0 + 16, y1 + 13, anchor="nw", text=it.prior,
                                  fill=PRIOR_COLORS[it.prior], font=self.f_tag)
            elif pg.kind == "frame":   # time/date are in the frame label
                c.create_text(x0 + 1, y1 + 1, anchor="nw",
                              text=f"dig{it.pos}  {it.proposed}", fill="#8f8",
                              font=self.f_tag)
            else:
                cap = hhmm
                if it.orig_proposed and it.orig_proposed != it.proposed:
                    cap = f"{hhmm[:5]} fixed"
                c.create_text(x0 + 1, y1 + 1, anchor="nw", text=cap, fill="#aaa",
                              font=self.f_small)
            badge = badge_text(it.preflag, self.task)
            if badge:   # orange badge, top-right inside the image
                pt = c.create_text(x1 - B - 2, y0 + B + 1, anchor="ne",
                                   text=badge, fill="#000", font=self.f_small)
                bb = c.bbox(pt)
                pb = c.create_rectangle(bb[0] - 1, bb[1], bb[2] + 1, bb[3], fill="#ff9a1f",
                                        outline="")
                c.tag_raise(pt, pb)
            t.tag_bg = c.create_rectangle(x0 + B, y0 + B, x0 + B, y0 + B, fill="", outline="")
            t.tag_txt = c.create_text(x0 + B + 2, y0 + B + 1, anchor="nw", text="",
                                      font=self.f_tag)
            self.refresh_tile(t)
        self.sel_rect = c.create_rectangle(0, 0, 0, 0, outline=SEL, width=2, dash=(4, 2))
        self.move_sel(self.sel)
        self.update_header()
        self.show_status()

    def draw_frame_label(self, bx, by, slots: dict[int, QItem]) -> None:
        its = [slots[p] for p in sorted(slots)]
        st = its[0].stamp
        when = f"{st[0:4]}-{st[4:6]}-{st[6:8]}\n{st[9:11]}:{st[11:13]}:{st[13:15]}" \
            if len(st) >= 15 else st
        ctx = frame_context(its)
        if len(ctx) > 150:
            ctx = ctx[:147] + "..."
        prop = " ".join(slots[p].proposed if p in slots else "." for p in POSITIONS)
        flags = ", ".join(sorted({badge_text(i.preflag, self.task)[1:] for i in its
                                  if badge_text(i.preflag, self.task)}))
        txt = f"{when}\nproposed  {prop}\n{ctx}" + (f"\n! {flags}" if flags else "")
        self.canvas.create_text(bx + 4, by + 4, anchor="nw", text=txt, fill=FG,
                                width=self.LABEL_W - 10, font=("Consolas", 9))

    def refresh_tile(self, t: Tile) -> None:
        c = self.canvas
        verdict, final = self.page.resolved(t.item)
        color = MARK_COLORS[verdict]
        c.itemconfigure(t.border, outline=color)
        if verdict in ("ok", "label_ok"):
            c.itemconfigure(t.tag_txt, text="")
            c.coords(t.tag_bg, 0, 0, 0, 0)
            c.itemconfigure(t.tag_bg, fill="")
            return
        txt = f"->{final}" if verdict in ("relabeled", "label_fixed") else MARK_TEXT[verdict]
        c.itemconfigure(t.tag_txt, text=txt, fill="#000")
        bb = c.bbox(t.tag_txt)
        if bb:
            c.coords(t.tag_bg, bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1)
        c.itemconfigure(t.tag_bg, fill=color)
        c.tag_raise(t.tag_bg)
        c.tag_raise(t.tag_txt)

    def update_header(self) -> None:
        pg, s = self.page, self.s
        od, ot = s.overall_progress()
        view = "20x32 model view" if self.model_view else "ROI view"
        if pg is None:
            self.header.config(text=f"{self.task.upper()} done   overall {od}/{ot}")
            return
        q = pg.items[0].queue
        qd, qt = s.queue_progress(q)
        nm = len(pg.marks)
        undo = f"   [restored from {pg.restored_from}]" if pg.restored_from else ""
        skipped = (f"   ({s.unverified_skipped} awaiting verification, skipped)"
                   if self.task == "screen" and s.unverified_skipped else "")
        if self.task == "consistency":
            _, pos, lab, _ = pg.cell
            k, kt = s.group_page_numbers(pg.cell)
            gi = s.cells.index(pg.cell) + 1 if pg.cell in s.cells else 0
            cd, ct = s.cell_progress(pg.cell)
            self.header.config(text=(
                f"CONSISTENCY · dig{pos} · all should be {lab} · page {k}/{kt} · "
                f"group {gi}/{len(s.cells)} · overall {od}/{ot}      group {cd}/{ct}\n"
                f"page: {len(pg.items)} crops, all light buckets, dark->bright, "
                f"{nm} marked   {view}{undo}"))
            return
        if pg.kind == "grid":
            _, pos, lab, bucket = pg.cell
            cd, ct = s.cell_progress(pg.cell)
            left = s.pages_left_in_cell(pg.cell)
            if self.task in ("verify", "audit"):
                title = f"VERIFY · dig{pos} · is this a {lab}? · {bucket or '-'}"
            elif self.task == "screen":
                title = (f"SCREEN (artifacts/drift only) · dig{pos} · {lab} "
                         f"· {bucket or '-'}")
            else:
                title = f"{q}   dig{pos} . proposed {lab} . {bucket or '-'}"
            self.header.config(text=(
                f"{title}      [{q}] cell {cd}/{ct} ({left} page{'s' if left != 1 else ''} "
                f"left)   queue {qd}/{qt}   overall {od}/{ot}\n"
                f"page: {len(pg.items)} crops, dark->bright, {nm} marked   {view}{undo}"
                f"{skipped}"))
        else:
            title = {"verify": "VERIFY", "audit": "AUDIT (model disagrees)", "screen": "SCREEN (artifacts/drift only)"}.get(
                self.task, q)
            self.header.config(text=(
                f"{title}   [{q}] frame mode: {len(pg.frames)} frames / {len(pg.items)} crops"
                f"   queue {qd}/{qt}   overall {od}/{ot}\n"
                f"{nm} marked   {view}{undo}{skipped}"))

    def show_status(self, t: Tile | None = None) -> None:
        if t is None:
            self.status.config(text=self.msg or " ")
            return
        it = t.item
        b = f"{it.brightness:.0f}" if it.brightness is not None else "?"
        flag = f"  [{it.preflag}]" if it.preflag else ""
        fixed = (f" (queue proposed {it.orig_proposed})"
                 if it.orig_proposed and it.orig_proposed != it.proposed else "")
        if it.prior:
            fixed += {"v": " [label verified earlier]", "f": " [label fixed earlier]"}[it.prior]
        if self.task == "consistency":
            fixed += f"  {it.bucket or '?'}  [{it.queue}]"
        self.status.config(text=(f"{it.item_id}  dig{it.pos} proposed {it.proposed}{fixed}  "
                                 f"bright {b}{flag}  {it.context}   "
                                 f"({os.path.basename(it.src)})"))

    # ---------- selection ----------
    def tile_at(self, x, y) -> int | None:
        B2 = 2 * self.BORDER
        for k, t in enumerate(self.tiles):
            if t.x0 <= x <= t.x0 + self.box_w + B2 and t.y0 <= y <= t.y0 + self.th + B2 + self.CAP:
                return k
        return None

    def move_sel(self, k: int) -> None:
        if not self.tiles:
            return
        self.sel = max(0, min(k, len(self.tiles) - 1))
        t = self.tiles[self.sel]
        B2 = 2 * self.BORDER
        self.canvas.coords(self.sel_rect, t.x0 - 2, t.y0 - 2,
                           t.x0 + self.box_w + B2 + 2, t.y0 + self.th + B2 + 2)
        self.canvas.tag_raise(self.sel_rect)

    def arrow(self, sym: str) -> None:
        if not self.tiles:
            return
        cur = self.tiles[self.sel]
        dr, dc = {"Left": (0, -1), "Right": (0, 1), "Up": (-1, 0), "Down": (1, 0)}[sym]
        pos = {(t.gr, t.gc): k for k, t in enumerate(self.tiles)}
        maxr = max(t.gr for t in self.tiles)
        maxc = max(t.gc for t in self.tiles)
        r, c = cur.gr, cur.gc
        for _ in range(max(maxr, maxc) + 1):
            r, c = r + dr, c + dc
            if r < 0 or c < 0 or r > maxr or c > maxc:
                break
            if (r, c) in pos:
                self.move_sel(pos[(r, c)])
                self.show_status(self.tiles[self.sel])
                return
            if dr:  # vertical: snap to nearest tile in that row
                row = [(abs(t.gc - c), k) for k, t in enumerate(self.tiles) if t.gr == r]
                if row:
                    self.move_sel(min(row)[1])
                    self.show_status(self.tiles[self.sel])
                    return

    def on_motion(self, ev) -> None:
        k = self.tile_at(ev.x, ev.y)
        if k is None:
            return
        if k != self.sel:
            self.move_sel(k)
        self.show_status(self.tiles[k])

    def on_click(self, ev) -> None:
        k = self.tile_at(ev.x, ev.y)
        if k is not None:
            self.move_sel(k)
            self.show_status(self.tiles[k])

    def on_double(self, ev) -> None:
        k = self.tile_at(ev.x, ev.y)
        if k is not None:
            self.move_sel(k)
            self.zoom()

    def on_right(self, ev) -> None:
        k = self.tile_at(ev.x, ev.y)
        if k is not None:
            self.move_sel(k)
            self.apply("artifact")

    # ---------- actions ----------
    def apply(self, action: str, k: int | None = None) -> None:
        if not self.tiles or self.page is None:
            return
        if not action_allowed(self.task, action):
            self.msg = f"key for '{action}' is not used in {self.task} mode"
            self.show_status()
            return
        t = self.tiles[self.sel if k is None else k]
        touched = set(self.page.mark(t.item, action))
        for tt in self.tiles:
            if tt.item.item_id in touched:
                self.refresh_tile(tt)
        self.update_header()

    def commit(self) -> None:
        if self.page is None:
            return
        pg = self.page
        pid = self.s.commit(pg)
        c = Counter(pg.resolved(it)[0] for it in pg.items)
        self.msg = (f"committed {pid}: " +
                    "  ".join(f"{v} {c[v]}" for v in VERDICTS if c[v]))
        self.load_page()

    def undo(self) -> None:
        ok, msg = self.s.undo()
        self.msg = msg
        if ok:
            self.load_page()
        else:
            self.show_status()

    def toggle_view(self) -> None:
        self.model_view = not self.model_view
        sel = self.sel
        self.draw()
        self.move_sel(sel)

    def quit(self) -> None:
        if self.page is not None and self.page.marks:
            from tkinter import messagebox
            if not messagebox.askyesno(
                    "grid_review",
                    f"Discard {len(self.page.marks)} mark(s) on this uncommitted page and quit?\n"
                    f"(Committed pages are already saved.)", parent=self.root):
                return
        self.root.destroy()

    def on_key(self, ev) -> None:
        if ev.state & 0x4:  # Ctrl-combos other than Ctrl+Z
            return
        sym = ev.keysym
        low = sym.lower() if len(sym) == 1 else sym
        if sym in ("Return", "KP_Enter"):
            self.commit()
        elif low == "q":
            self.quit()
        elif low == "u":
            self.undo()
        elif low == "v":
            self.toggle_view()
        elif low == "z":
            self.zoom()
        elif sym in ("Left", "Right", "Up", "Down"):
            self.arrow(sym)
        else:
            act = key_action(ev)
            if act:
                self.apply(act)

    # ---------- zoom ----------
    def zoom(self) -> None:
        if not self.tiles:
            return
        tk = self.tk
        if self.zoom_win is not None:
            self.zoom_win.destroy()
        self.zphotos = []
        k = self.sel
        it = self.tiles[k].item
        nb = self.s.frame_neighbours(it)
        top = tk.Toplevel(self.root, bg=BG)
        top.title(f"zoom {it.item_id}")
        top.transient(self.root)            # stays above the grid window
        if self.root.attributes("-topmost"):
            top.attributes("-topmost", True)
        self.zoom_win = top
        verify = self.task in LABEL_TASKS
        z = max(1, self.args.zoom - 1) if verify else self.args.zoom
        mv_h = 120 if verify else max(64, self.th)
        der = self.s.derive
        head = (f"{it.item_id}   dig{it.pos}  "
                f"{'label' if self.task == 'consistency' else 'proposed'} {it.proposed}"
                + (f"   model {it.model_label}" if it.model_label else "")
                + f"\nframe: {self.s.frame_reading(it)}\n{it.context}\n"
                  f"mark keys apply to the yellow crop and close;  Esc / z / Enter close")
        tk.Label(top, bg=BG, fg=FG, font=("Consolas", 11), justify="left",
                 text=head).grid(row=0, column=0, columnspan=7, sticky="w", padx=8, pady=4)
        fr_row = tk.Frame(top, bg=BG)
        fr_row.grid(row=1, column=0, columnspan=7, sticky="w")
        for col, p in enumerate(POSITIONS):
            fr = tk.Frame(fr_row, bg="#ffd24a" if p == it.pos else BG, padx=3, pady=3)
            fr.grid(row=0, column=col, padx=4, pady=4, sticky="n")
            if p not in nb:
                tk.Label(fr, text=f"dig{p}\n(not found)", bg=BG, fg="#888",
                         font=("Consolas", 10)).pack()
                continue
            path, lab = nb[p]
            img = self.pil(path)
            if img is None:
                tk.Label(fr, text=f"dig{p}\n(unreadable)", bg=BG, fg="#f88").pack()
                continue
            ph = self.render_img(img, img.height * z, False, Image.Resampling.NEAREST,
                                 store=self.zphotos)
            tk.Label(fr, image=ph, bd=0).pack()
            mv = self.render_img(img, mv_h, True, store=self.zphotos)
            tk.Label(fr, image=mv, bd=0, bg=BG).pack(pady=(4, 0))
            cap = f"dig{p}  {lab}"
            d = der.get(it.stamp, p) if der else None
            if d is not None:
                cap += f"\nt {d.truth}  m {d.model}"
            tk.Label(fr, text=cap, bg=fr["bg"], fg="#000" if p == it.pos else "#bbb",
                     font=("Consolas", 11, "bold")).pack()
        if verify:
            strip = self.s.time_strip(it, self.args.strip)
            tk.Label(top, bg=BG, fg="#9cf", font=("Consolas", 10), anchor="w",
                     text=(f"dig{it.pos} in the nearest reading-screen frames "
                           f"(t = derived truth, e = reading_est digit, m = model, h = your label):")
                     ).grid(row=2, column=0, columnspan=7, sticky="w", padx=8)
            sr = tk.Frame(top, bg=BG)
            sr.grid(row=3, column=0, columnspan=7, sticky="w", padx=4, pady=(0, 6))
            for col, ent in enumerate(strip):
                fr = tk.Frame(sr, bg="#ffd24a" if ent["current"] else BG, padx=2, pady=2)
                fr.grid(row=0, column=col, padx=3, sticky="n")
                img = self.pil(ent["path"]) if ent["path"] else None
                if img is None:
                    tk.Label(fr, text="(missing)", bg=BG, fg="#888").pack()
                else:
                    ph = self.render_img(img, img.height, False, Image.Resampling.NEAREST,
                                         store=self.zphotos)
                    tk.Label(fr, image=ph, bd=0).pack()
                tk.Label(fr, text="\n".join(ent["lines"]), bg=fr["bg"],
                         fg="#000" if ent["current"] else "#bbb", justify="left",
                         font=("Consolas", 9)).pack()

        def close(_e=None):
            if self.zoom_win is not None:
                self.zoom_win.destroy()
                self.zoom_win = None
            self.root.focus_force()

        def on_zkey(ev):
            if ev.keysym in ("Escape", "z", "Z", "Return", "KP_Enter", "q"):
                close()
                return "break"
            act = key_action(ev)
            if act:
                close()
                self.apply(act, k)
            return "break"

        top.bind("<Key>", on_zkey)
        top.protocol("WM_DELETE_WINDOW", close)
        top.geometry("+20+20")
        top.focus_force()

    # ---------- debug helpers ----------
    def demo_marks(self) -> None:
        acts = {"classic": ["artifact", "drift", "illegible", "ok_derived", "7"],
                "verify": ["artifact", "unsure", "7", "drift"],
                "audit": ["artifact", "unsure", "7", "drift"],
                "screen": ["artifact", "drift", "7"],
                "consistency": ["artifact", "unsure", "7", "drift"]}[self.task]
        for k, act in enumerate(acts):
            if k + 1 < len(self.tiles):
                self.apply(act, k + 1)
        self.msg = "demo marks (not committed)"
        self.show_status()

    def screenshot(self, path: str) -> None:
        from PIL import ImageGrab
        self.root.update()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        print(f"screenshot {w}x{h} -> {path}")
        if self.zoom_win is not None:
            zw = self.zoom_win
            zw.lift()
            zw.update()
            x, y = zw.winfo_rootx(), zw.winfo_rooty()
            w, h = zw.winfo_width(), zw.winfo_height()
            zp = Path(path).with_name(Path(path).stem + "_zoom.png")
            ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(zp)
            print(f"zoom screenshot {w}x{h} -> {zp}")

    def run(self) -> None:
        a = self.args
        if a.demo_marks:
            self.demo_marks()
        if a.autoclose_ms:
            def _auto():
                if a.screenshot:
                    self.screenshot(a.screenshot)
                self.root.destroy()
            if a.zoom_demo:
                self.root.after(max(200, a.autoclose_ms // 3), self.zoom)
            self.root.attributes("-topmost", True)
            self.root.after(a.autoclose_ms, _auto)
        self.root.mainloop()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", action="append", default=[], type=Path,
                    help="queue CSV (repeatable; presented in the given order)")
    ap.add_argument("--mode", choices=("verify", "screen", "consistency", "audit"), default=None,
                    help="verify: confirm labels of needs_verify items; screen: flag "
                         "artifacts/drift only; consistency: every accepted item in the "
                         "ledger, one page per (position, current label), all light buckets "
                         "mixed. Omit for the original single-pass review.")
    ap.add_argument("--positions", default="",
                    help="consistency mode: positions to review, e.g. dig6,dig5 (default all, "
                         "in the order dig6, dig5, dig4, dig3, dig2)")
    ap.add_argument("--ledger", type=Path, default=Path("work/review_ledger.csv"))
    ap.add_argument("--b3-derive", type=Path, default=Path("work/b3_derive.csv"),
                    help="verify/consistency mode: frame readings + time strip in the zoom "
                         "window (consistency: also brightness + light bucket)")
    ap.add_argument("--frame-mode", action="store_true",
                    help="rows = frames (dig2..dig6), for the holdout")
    ap.add_argument("--page-size", type=int, default=48)
    ap.add_argument("--cols", type=int, default=12, help="grid mode columns")
    ap.add_argument("--tile-h", type=int, default=180,
                    help="tile image height px (crop is ~202 tall)")
    ap.add_argument("--frames-per-page", type=int, default=8)
    ap.add_argument("--frame-cols", type=int, default=2,
                    help="frame mode: frames side by side per row")
    ap.add_argument("--cell-order", choices=CELL_ORDERS, default="csv",
                    help="cell order within a queue: csv (first appearance), size "
                         "(largest first), small (smallest first), sorted (pos,label,bucket)")
    ap.add_argument("--zoom", type=int, default=3, help="zoom window scale")
    ap.add_argument("--strip", type=int, default=3,
                    help="verify/audit zoom: reading frames shown on each side of the crop")
    ap.add_argument("--redo-since", default="", metavar="YYYYmmdd-HHMMSS",
                    help="audit: re-open items whose latest label ruling (including "
                         "can't-tell) was made before this stamp")
    ap.add_argument("--model-view", action="store_true",
                    help="start in the upscaled 20x32 model view (v toggles)")
    ap.add_argument("--no-brightness", action="store_true",
                    help="don't compute missing brightness from the images")
    ap.add_argument("--propagate-drift", action="store_true",
                    help="append drift rows for missing positions of drift frames in "
                         "--ledger (idempotent), print the count, exit")
    # sample queue builder
    ap.add_argument("--make-sample-queue", type=Path, metavar="OUT")
    ap.add_argument("--root", type=Path, help="raw crop dir for --make-sample-queue")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-name", default="sample")
    ap.add_argument("--preflag-frac", type=float, default=0.0,
                    help="sample builder: fraction of rows given preflag=drift (demo)")
    # testing / debug
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--autoclose-ms", type=int, default=0,
                    help="debug: render first page, close after N ms (never commits)")
    ap.add_argument("--screenshot", default="",
                    help="debug with --autoclose-ms: save window grab here")
    ap.add_argument("--demo-marks", action="store_true",
                    help="debug: pre-mark a few tiles (uncommitted) to show colours")
    ap.add_argument("--zoom-demo", action="store_true",
                    help="debug: open the zoom window on the first tile")
    args = ap.parse_args(argv)

    if args.selftest:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_grid_review
        return test_grid_review.main()

    if args.propagate_drift:
        n = propagate_frame_drift(args.ledger)
        print(f"propagate_frame_drift: appended {n} drift row(s) to {args.ledger}")
        return 0

    if args.make_sample_queue:
        if not args.root:
            ap.error("--make-sample-queue needs --root")
        n = make_sample_queue(args.make_sample_queue, args.root, args.n, args.seed,
                              args.sample_name, args.preflag_frac)
        print(f"wrote {n} rows -> {args.make_sample_queue}")
        return 0

    if args.mode == "consistency":
        return run_consistency(args, ap)

    if not args.queue:
        ap.error("give at least one --queue (or --make-sample-queue / --selftest)")
    items, warns = load_queues(args.queue)
    for w in warns[:20]:
        print("warning:", w)
    if len(warns) > 20:
        print(f"warning: ... {len(warns) - 20} more")
    if not items:
        raise SystemExit("no items in the queue(s)")
    if not args.no_brightness:
        n = fill_brightness(items, verbose=True)
        if n:
            print(f"computed brightness for {n} items")
    task = args.mode or "classic"
    derive = None
    if task in ("verify", "audit"):
        derive = DeriveIndex(args.b3_derive)
        print(f"b3_derive: {len(derive)} crops indexed ({args.b3_derive})")

    sess = Session(items, args.ledger,
                   mode="frame" if args.frame_mode else "grid", task=task,
                   page_size=args.page_size, frames_per_page=args.frames_per_page,
                   cell_order=args.cell_order, derive=derive,
                   redo_since=args.redo_since)
    d, t = sess.overall_progress()
    print(f"{task} mode: {t} items in {len(sess.queues)} queue(s), {len(sess.cells)} cells; "
          f"{d} already done in {args.ledger}; {len(sess.drift)} drift frame(s) skipped")
    if task in ("verify", "audit"):
        print(f"  ({sess.not_needing_verify} loaded item(s) need no verification)")
    if task == "screen" and sess.unverified_skipped:
        print(f"  {sess.unverified_skipped} item(s) need verification first "
              f"(run --mode verify on the same queues) -- skipped")
    if t - d == 0:
        print("nothing to do in this mode.")
    return run_app(sess, args)


def run_app(sess: Session, args) -> int:
    _dpi_aware()
    app = GridApp(sess, args)
    app.run()
    summ = sess.summary()
    print(summ)
    if sess.committed or sess.pages_committed:
        log = args.ledger.with_name(args.ledger.stem + "_sessions.log")
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"{now_iso()} {summ}\n")
    return 0


def consistency_report(sess: Session, s_per_crop: float = 0.6) -> str:
    """Per-position (and per-class) counts still to do, with a time estimate."""
    todo = sess.todo_counts()
    lines = []
    tot = 0
    for p in CONSISTENCY_POS_ORDER:
        per = [(lab, todo[(p, lab)]) for lab in CONSISTENCY_CLASS_ORDER if todo[(p, lab)]]
        n = sum(v for _, v in per)
        if not n and not any(c[1] == p for c in sess.cells):
            continue
        tot += n
        lines.append(f"  dig{p}: {n:5d} to do (~{n * s_per_crop / 60:.0f} min)   "
                     + "  ".join(f"{lab}:{v}" for lab, v in per))
    lines.append(f"  total {tot} crops to do, ~{tot * s_per_crop / 60:.0f} min at "
                 f"{s_per_crop} s/crop")
    return "\n".join(lines)


def run_consistency(args, ap) -> int:
    if args.frame_mode:
        ap.error("--mode consistency is grid-only (drop --frame-mode)")
    try:
        positions = parse_positions(args.positions)
    except ValueError as e:
        ap.error(f"--positions: {e}")
    derive = DeriveIndex(args.b3_derive)
    print(f"b3_derive: {len(derive)} crops indexed ({args.b3_derive})")
    qpaths = list(args.queue)
    if not qpaths:
        qdir = args.ledger.parent / "queues"
        qpaths = sorted(qdir.glob("*.csv")) if qdir.is_dir() else []
    print(f"queue metadata (bucket/context/src fallback) from {len(qpaths)} queue file(s)")
    items, loaded, info = consistency_items(args.ledger, qpaths, derive, positions)
    if not items:
        raise SystemExit(f"no accepted items in {args.ledger} at positions {positions}")
    if not args.no_brightness:
        n = fill_brightness(items, verbose=True)
        if n:
            print(f"computed crop brightness for {n} items not in b3_derive")
    if info["no_src"]:
        print(f"warning: {info['no_src']} item(s) without an image path")
    sess = Session(items, args.ledger, task="consistency", page_size=args.page_size,
                   derive=derive, extra_loaded_ids=loaded)
    d, t = sess.overall_progress()
    print(f"consistency mode: {t} accepted items at "
          f"{', '.join(f'dig{p}' for p in positions)} in {len(sess.cells)} groups; "
          f"{d} already done in consistency pages of {args.ledger}"
          + (f" ({info['rejected_in_consistency']} rejected there)"
             if info["rejected_in_consistency"] else ""))
    print(consistency_report(sess))
    if t - d == 0:
        print("nothing to do in this mode.")
    return run_app(sess, args)


if __name__ == "__main__":
    sys.exit(main())
