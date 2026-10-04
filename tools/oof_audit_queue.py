"""Build a label-audit queue from cross-validation out-of-fold predictions.

A crop the model gets confidently wrong *out of fold* -- it never trained on that
crop, or on anything from the same day -- is the best-aimed place to look for a bad
label (HANDOFF §5f: a high-confidence disagreement on a legible crop is usually a
label error). This writes those crops as a grid_review queue for
``--mode audit``, which re-shows crops even when they already carry a label verdict.

    python tools/oof_audit_queue.py --min-conf 0.7
    python tools/grid_review.py --mode audit --queue work/queues/oof_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter
from pathlib import Path

from grid_review import QUEUE_FIELDS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oof-glob", default="work/cv_oof_fold*.csv")
    ap.add_argument("--manifest", default="work/corpus_manifest.csv")
    ap.add_argument("--derive", default="work/b3_derive.csv")
    ap.add_argument("--min-conf", type=float, default=0.7,
                    help="only disagreements where the model's top class has >= this prob")
    ap.add_argument("--out", default="work/queues/oof_audit.csv")
    args = ap.parse_args()

    manifest = {}
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            manifest[r["file"]] = r
    bright = {}
    if os.path.exists(args.derive):
        with open(args.derive, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                bright[(r["stamp"], str(r["pos"]).replace("dig", ""))] = r["frame_brightness"]

    files = sorted(glob.glob(args.oof_glob))
    rows, seen, n_all, n_err = [], set(), 0, 0
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                n_all += 1
                if r["label"] == r["pred"]:
                    continue
                n_err += 1
                conf = float(r["conf"])
                if conf < args.min_conf:
                    continue
                name = os.path.basename(r["path"])
                m = manifest.get(name)
                if m is None:
                    print(f"warning: {name} not in manifest -- skipped")
                    continue
                pos = m["pos"].replace("dig", "")
                item_id = f"{m['stamp']}_dig{pos}"
                if item_id in seen:
                    continue
                seen.add(item_id)
                src = m["src"]
                if not os.path.isabs(src):
                    src = str(Path(src).resolve())
                rows.append({
                    "queue": "oof_audit", "item_id": item_id, "stamp": m["stamp"],
                    "pos": pos, "src": src, "proposed": m["label"], "bucket": m["bucket"],
                    "brightness": bright.get((m["stamp"], pos), ""),
                    "context": f"OOF model says {r['pred']} @{conf:.2f} ({m['screen']}, "
                               f"{m['queue'] or m['provenance']})",
                    "preflag": "model_disagrees", "group": "", "needs_verify": "1"})

    rows.sort(key=lambda r: (-int(r["pos"]), r["proposed"], r["bucket"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(files)} OOF file(s): {n_all} crops, {n_err} OOF errors, "
          f"{len(rows)} at conf >= {args.min_conf} -> {args.out}")
    by = Counter((f"dig{r['pos']}", r["proposed"] + "->" + r["context"].split()[3], r["bucket"])
                 for r in rows)
    for k, n in by.most_common(25):
        print(f"  {k[0]} {k[1]:<6} {k[2]:<10} x{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
