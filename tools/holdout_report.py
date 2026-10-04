"""Stratified holdout report for one or more eval CSVs (``eval_dig_model.py --csv``).

Joins each crop to ``work/b3_derive.csv`` for its measured light bucket and screen,
then prints the numbers the batch-3 gates are judged on: reading-screen accuracy per
(position, bucket) with dig6/dig5 at night first, per-frame accuracy, confusion
axes, and -- with two or more models -- fixed/broken lists against the first.

    python tools/holdout_report.py work/eval_9002_holdoutb3.csv work/eval_9008_holdoutb3.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

BUCKETS = ("flash", "transition", "day")


def load_derive(path):
    idx = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx[(r["stamp"], str(r["pos"]).replace("dig", ""))] = (r["bucket"], r["screen"])
    return idx


def load_eval(path, derive):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pos = str(r["position"]).replace("dig", "")
            bucket, screen = derive.get((r["frame"], pos), ("?", r.get("screen", "?")))
            key = os.path.basename(r["path"]).split("_", 1)[1]  # drop the label prefix
            rows[key] = dict(label=r["label"], pred=r["pred"], pos=pos, frame=r["frame"],
                             bucket=bucket, screen=screen, conf=float(r["conf"]))
    return rows


def pct(ok, n):
    return f"{ok:>4}/{n:<4} {100 * ok / n:6.2f}%" if n else f"{'-':>15}"


def report(name, rows):
    print(f"\n=== {name} ===")
    n = len(rows)
    ok = sum(r["label"] == r["pred"] for r in rows.values())
    print(f"all crops            {pct(ok, n)}")
    frames = defaultdict(list)
    for r in rows.values():
        frames[r["frame"]].append(r["label"] == r["pred"])
    print(f"whole frames         {pct(sum(all(v) for v in frames.values()), len(frames))}")
    for b in BUCKETS:
        sel = [r for r in rows.values() if r["bucket"] == b]
        print(f"  {b:<18} {pct(sum(r['label'] == r['pred'] for r in sel), len(sel))}")

    print("reading screens, per position x bucket:")
    print(f"  {'pos':<5}" + "".join(f"{b:>18}" for b in BUCKETS))
    for pos in "65432":
        cells = []
        for b in BUCKETS:
            sel = [r for r in rows.values()
                   if r["pos"] == pos and r["bucket"] == b and r["screen"] == "reading"]
            cells.append(pct(sum(r["label"] == r["pred"] for r in sel), len(sel)))
        print(f"  dig{pos:<2}" + "".join(f"{c:>18}" for c in cells))
    for scr in ("test0", "test8", "dash"):
        sel = [r for r in rows.values() if r["screen"] == scr]
        print(f"  screen {scr:<12} {pct(sum(r['label'] == r['pred'] for r in sel), len(sel))}")

    axes = Counter((f"dig{r['pos']}", f"{r['label']}->{r['pred']}", r["bucket"])
                   for r in rows.values() if r["label"] != r["pred"])
    print("error axes: " + ", ".join(f"{p} {a} {b} x{c}" for (p, a, b), c in axes.most_common(12)))
    ones = sum(1 for r in rows.values() if r["label"] == "1" and r["pred"] == "7")
    print(f"1->7 count: {ones}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--derive", default="work/b3_derive.csv")
    args = ap.parse_args()
    derive = load_derive(args.derive)
    evals = [(os.path.basename(p), load_eval(p, derive)) for p in args.csvs]
    for name, rows in evals:
        report(name, rows)
    base_name, base = evals[0]
    for name, rows in evals[1:]:
        common = base.keys() & rows.keys()
        fixed = [k for k in common if base[k]["pred"] != base[k]["label"]
                 and rows[k]["pred"] == rows[k]["label"]]
        broken = [k for k in common if base[k]["pred"] == base[k]["label"]
                  and rows[k]["pred"] != rows[k]["label"]]
        print(f"\n{name} vs {base_name}: fixed {len(fixed)}, broken {len(broken)}")
        for k in sorted(broken):
            r = rows[k]
            print(f"  BROKEN {k}: {r['label']}->{r['pred']} ({r['bucket']}, {r['screen']}, "
                  f"conf {r['conf']:.2f})")


if __name__ == "__main__":
    main()
