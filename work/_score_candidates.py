"""One-off: score candidate models on holdout_b3, the night selection set, and the probe verdicts."""
import csv, os, subprocess, sys, collections
sys.path.insert(0, "tools")
from grid_review import read_ledger_rows

# `--resize stb` (first argument) scores with the firmware's resize; default nearest.
# Non-default resizes get their own cached eval CSVs (eval_<m>_<tag>_<resize>.csv).
args = sys.argv[1:]
RESIZE = "nearest"
if args[:1] == ["--resize"]:
    RESIZE, args = args[1], args[2:]
models = args
der = {}
for r in csv.DictReader(open("work/b3_derive.csv")):
    der[(r["stamp"], str(r["pos"]).replace("dig", ""))] = r
rows = read_ledger_rows("work/review_ledger.csv")
und = {r["page_id"] for r in rows if r["verdict"] == "undone"}
probe = {r["stamp"]: r["final"] for r in rows if r["queue"] == "probe_sel"
         and r["page_id"] not in und and r["verdict"] in ("label_ok", "label_fixed")}


def ev(m, data, tag):
    out = f"work/eval_{m}_{tag}.csv" if RESIZE == "nearest" else f"work/eval_{m}_{tag}_{RESIZE}.csv"
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(f"models/dig-class11_{m}_s2.tflite") \
            or os.path.getmtime(out) < max(os.path.getmtime(os.path.join(data, f)) for f in os.listdir(data)):
        subprocess.run([sys.executable, "tools/eval_dig_model.py", "--model", f"models/dig-class11_{m}_s2.tflite",
                        "--data", data, "--resize", RESIZE, "--csv", out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return list(csv.DictReader(open(out)))


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "  -  "


print(f"{'model':<6} | {'hold all':>8} {'frames':>7} {'nt rdg frm':>10} {'nt dig6':>8} {'nt dig5':>8} | "
      f"{'sel all':>7} {'sel d6':>7} | {'probe':>6}")
for m in models:
    H = ev(m, "work/holdout_b3", "holdoutb3")
    S = ev(m, "work/selection_night", "sel")
    ok = lambda r: r["label"] == r["pred"]
    fr = collections.defaultdict(list); nrf = collections.defaultdict(list); d6 = [0, 0]; d5 = [0, 0]
    for r in H:
        p = str(r["position"]).replace("dig", ""); x = der[(r["frame"], p)]
        fr[r["frame"]].append(ok(r))
        if x["bucket"] == "flash" and x["screen"] == "reading":
            nrf[r["frame"]].append(ok(r))
            if p == "6": d6[0] += ok(r); d6[1] += 1
            if p == "5": d5[0] += ok(r); d5[1] += 1
    s6 = [r for r in S if str(r["position"]).endswith("6")]
    pr = [r for r in s6 if os.path.basename(r["path"])[:-4].split("_")[3] in probe]
    pok = sum(r["pred"] == probe[os.path.basename(r["path"])[:-4].split("_")[3]] for r in pr)
    print(f"{m:<6} | {pct(sum(map(ok, H)), len(H)):>8} {pct(sum(all(v) for v in fr.values()), len(fr)):>7} "
          f"{pct(sum(all(v) for v in nrf.values()), len(nrf)):>10} {pct(*d6):>8} {pct(*d5):>8} | "
          f"{pct(sum(map(ok, S)), len(S)):>7} {pct(sum(map(ok, s6)), len(s6)):>7} | {pok:>3}/{len(pr)}")
