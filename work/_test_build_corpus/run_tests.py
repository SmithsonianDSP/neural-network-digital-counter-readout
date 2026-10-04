"""Synthetic end-to-end test for `build_corpus.py --from-ledger`.

Run from the repo root:  python work/_test_build_corpus/run_tests.py
Everything is written under work/_test_build_corpus/ (fixtures + outputs). The real
joes-samples/, work/culled.txt, work/corpus_manifest.csv are never touched: every run
passes scratch paths for --out / --exclude / --manifest / --culled-reasons, and the
default-out (backup) scenario runs in a scratch cwd that has its own joes-samples/.
"""
import csv
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
T = Path(__file__).resolve().parent
SCRIPT = REPO / "tools" / "build_corpus.py"
sys.path.insert(0, str(REPO / "tools"))
import build_corpus as B  # noqa: E402

FIELDS = ["item_id", "stamp", "pos", "src", "queue", "proposed", "final", "verdict",
          "page_id", "reviewed_at"]
POS = ["dig2", "dig3", "dig4", "dig5", "dig6"]
fails = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


def run(*args, cwd=REPO):
    r = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=cwd,
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def row(stamp, pos, src, queue, proposed, final, verdict, page):
    return {"item_id": f"{stamp}_{pos}", "stamp": stamp, "pos": pos[3:], "src": str(src),
            "queue": queue, "proposed": proposed, "final": final, "verdict": verdict,
            "page_id": page, "reviewed_at": "2026-10-03T12:00:00-05:00"}


def write_ledger(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- fixtures
shutil.rmtree(T / "work", ignore_errors=True)
W = T / "work"
W.mkdir()
rng = random.Random(7)

derive = list(csv.DictReader(open(REPO / "work/b3_derive.csv", encoding="utf-8-sig")))
frames = defaultdict(dict)
for r in derive:
    if r["day"] >= "20260815" and Path(r["path"]).is_file():
        frames[r["stamp"]][r["pos"]] = r
frames = {s: d for s, d in frames.items() if len(d) == 5}
HOLD_DAY = "20260925"
train_stamps = sorted(s for s in frames if s[:8] != HOLD_DAY)
hold_stamps = sorted(s for s in frames if s[:8] == HOLD_DAY)
assert len(hold_stamps) > 12, len(hold_stamps)
days = sorted({s[:8] for s in train_stamps})
# 40 frames (200 crops) spread across days, covering every bucket / screen
by_day = defaultdict(list)
for s in train_stamps:
    by_day[s[:8]].append(s)
chosen = []
while len(chosen) < 40:
    for d in days:
        if by_day[d] and len(chosen) < 40:
            chosen.append(by_day[d].pop(rng.randrange(len(by_day[d]))))
print(f"fixture: {len(chosen)} train frames over {len(days)} days, "
      f"{len(hold_stamps)} frames on holdout day {HOLD_DAY}")

rows, expect = [], {}      # expect[(stamp,pos)] = (verdict, label)
verdict_cycle = (["ok"] * 14 + ["relabeled"] * 2 + ["ok_derived"] * 2 +
                 ["artifact", "drift", "illegible"])
page = 0
for fi, s in enumerate(chosen):
    page += 1
    for pi, p in enumerate(POS):
        d = frames[s][p]
        prop = d["model_label"]
        v = verdict_cycle[(fi * 5 + pi) % len(verdict_cycle)]
        final = prop
        if v == "relabeled":
            final = "7" if prop != "7" else "1"
        if v in B.REJECT_VERDICTS:
            final = ""
        rows.append(row(s, p, d["path"], "night_err", prop, final, v, f"g{page}"))
        expect[(s, p)] = (v, final)

# drift is frame-wide: every crop of a frame with a drift verdict is rejected, even
# though the fixture ledger has `ok` rows for the frame's other positions (an old
# ledger written before frame-wide propagation)
drift_st = {s for (s, p), (v, _) in expect.items() if v == "drift"}
n_frame_drift = sum(1 for k, (v, _) in expect.items() if k[0] in drift_st and v != "drift"
                    and v not in B.REJECT_VERDICTS)
for k in list(expect):
    if k[0] in drift_st and expect[k][0] not in B.REJECT_VERDICTS:
        expect[k] = ("drift", "")

# needs_verify (two-stage) cases among plain `ok` day/transition crops
okday = sorted(k for k, (v, _) in expect.items()
               if v == "ok" and frames[k[0]][k[1]]["bucket"] != "flash")
relday = sorted(k for k, (v, _) in expect.items()
                if v == "relabeled" and frames[k[0]][k[1]]["bucket"] != "flash")
NV_PEND, NV_OK, NV_OK_EARLY, NV_UNS, NOQ_FIX = okday[1], okday[4], okday[7], okday[10], okday[13]
NV_FIX = relday[0]
print(f"nv cases: pend={NV_PEND} ok={NV_OK} ok_early={NV_OK_EARLY} unsure={NV_UNS} "
      f"fix={NV_FIX} noqueue_fix={NOQ_FIX}")


def other(lab, *avoid):
    return next(x for x in "3456789" if x != lab and x not in avoid)


fix_to = other(expect[NV_FIX][1], frames[NV_FIX[0]][NV_FIX[1]]["model_label"])
noq_to = other(expect[NOQ_FIX][1])
for k, v, fin in ((NV_OK, "label_ok", expect[NV_OK][1]),
                  (NV_FIX, "label_fixed", fix_to),
                  (NV_UNS, "label_unsure", ""),
                  (NOQ_FIX, "label_fixed", noq_to)):
    page += 1
    d = frames[k[0]][k[1]]
    rows.append(row(k[0], k[1], d["path"], "night_err", d["model_label"], fin, v, f"gv{page}"))
page += 1
d = frames[NV_OK_EARLY[0]][NV_OK_EARLY[1]]
rows.insert(0, row(NV_OK_EARLY[0], NV_OK_EARLY[1], d["path"], "night_err", d["model_label"],
                   expect[NV_OK_EARLY][1], "label_ok", f"gv{page}"))   # label before screen
expect[NV_PEND] = ("pending", "")
expect[NV_UNS] = ("label_unsure", "")
expect[NV_FIX] = ("relabeled", fix_to)
expect[NOQ_FIX] = ("ok", noq_to)
nv_keys = {NV_PEND, NV_OK, NV_OK_EARLY, NV_UNS, NV_FIX}

# holdout queue on the holdout day: frames 0..4 clean, 5 has an artifact, 6 incomplete,
# 7 has a relabel + ok_derived (still exported). Two-stage cases: frame 0 dig6 needs
# verification and was label_fixed (exported with the fixed label); frame 1 dig5 needs
# verification and has none (dropped, pending); frame 2 dig2 label_unsure (dropped,
# rejected, never culled); frame 3 dig4 needs verification, label_ok (exported).
hold_expect = {}
for i, s in enumerate(hold_stamps[:8]):
    page += 1
    for pi, p in enumerate(POS):
        d = frames[s][p]
        prop = d["model_label"]
        v, final = "ok", prop
        if i == 5 and p == "dig4":
            v, final = "artifact", ""
        if i == 6 and p in ("dig5", "dig6"):
            continue
        if i == 7 and p == "dig3":
            v, final = "relabeled", ("2" if prop != "2" else "3")
        if i == 7 and p == "dig6":
            v = "ok_derived"
        rows.append(row(s, p, d["path"], "holdout", prop, final, v, f"g{page}"))
        hold_expect[(s, p)] = (v, final)
H0, H1, H2, H3 = ((hold_stamps[0], "dig6"), (hold_stamps[1], "dig5"),
                  (hold_stamps[2], "dig2"), (hold_stamps[3], "dig4"))
h0_to = other(hold_expect[H0][1])
for k, v, fin in ((H0, "label_fixed", h0_to), (H2, "label_unsure", ""),
                  (H3, "label_ok", hold_expect[H3][1])):
    page += 1
    rows.append(row(k[0], k[1], frames[k[0]][k[1]]["path"], "holdout",
                    frames[k[0]][k[1]]["model_label"], fin, v, f"fv{page}"))

# legacy crops: pick four from work/labeled
leg, _, _ = B.collect_legacy(REPO / "work/labeled", REPO / "joes-samples/batch 1")
culled = B.load_exclusions(Path(__file__).parent / "culled_seed.txt")
legkeys = sorted(k for k in leg if k not in culled and k[0][:8] != HOLD_DAY)
L_REJ, L_RELAB, L_OKD, L_HOLD = (legkeys[10], legkeys[200], legkeys[400], legkeys[600])
relab_to = "9" if leg[L_RELAB][0] != "9" else "4"
for k, v, fin, q in ((L_REJ, "artifact", "", "night_err"),
                     (L_RELAB, "relabeled", relab_to, "night_err"),
                     (L_OKD, "ok_derived", leg[L_OKD][0], "night_err"),
                     (L_HOLD, "ok", leg[L_HOLD][0], "holdout")):
    page += 1
    rows.append(row(k[0], k[1], leg[k][1], q, leg[k][0], fin, v, f"g{page}"))
# an undone page must be ignored: this would reject a legacy crop, then is undone
k_undone = legkeys[800]
rows.append(row(k_undone[0], k_undone[1], leg[k_undone][1], "night_err", leg[k_undone][0],
                "", "drift", "gUNDO"))
rows.append({"item_id": "", "stamp": "", "pos": "", "src": "", "queue": "", "proposed": "",
             "final": "", "verdict": "undone", "page_id": "gUNDO2", "reviewed_at": ""})
# (tombstone references gUNDO via its page_id column)
rows[-1]["page_id"] = "gUNDO"

LEDGER = W / "ledger.csv"
write_ledger(LEDGER, rows)
QD = W / "queues"
QD.mkdir()
with (QD / "night_err.csv").open("w", newline="", encoding="utf-8") as fh:
    qw = csv.writer(fh)
    qw.writerow(["queue", "item_id", "stamp", "pos", "src", "proposed", "needs_verify"])
    for (s_, p_), (v_, f_) in sorted(expect.items()):
        if (s_, p_) == NOQ_FIX:
            continue                                    # in no queue -> needs_verify 0
        qw.writerow(["night_err", f"{s_}_{p_}", s_, p_[3:], frames[s_][p_]["path"],
                     frames[s_][p_]["model_label"], int((s_, p_) in nv_keys)])
with (QD / "holdout.csv").open("w", newline="", encoding="utf-8") as fh:
    qw = csv.writer(fh)
    qw.writerow(["queue", "item_id", "stamp", "pos", "src", "proposed", "needs_verify"])
    for (s_, p_) in sorted(hold_expect):
        qw.writerow(["holdout", f"{s_}_{p_}", s_, p_[3:], frames[s_][p_]["path"],
                     frames[s_][p_]["model_label"], int((s_, p_) in (H0, H1, H2, H3))])
HDAYS = W / "holdout_days.txt"
HDAYS.write_text(f"# synthetic\n{HOLD_DAY}\n", encoding="utf8")
shutil.copy2(Path(__file__).parent / "culled_seed.txt", W / "culled.txt")  # frozen pre-batch-3 list
n_culled0 = len((W / "culled.txt").read_text().strip().splitlines())
(W / "empty_validation").mkdir()
print(f"fixture ledger: {len(rows)} rows; legacy picks rej={L_REJ} relab={L_RELAB} "
      f"okd={L_OKD} hold={L_HOLD} undone={k_undone}")


def common(out, *extra, ledger=LEDGER, hdays=HDAYS, culled_p=W / "culled.txt",
           reasons=W / "culled_reasons.csv", manifest=W / "manifest.csv",
           validation=W / "empty_validation"):
    return ["--from-ledger", "--ledger", ledger, "--holdout-days", hdays,
            "--out", out, "--exclude", culled_p, "--culled-reasons", reasons,
            "--manifest", manifest, "--validation", validation, "--queues-dir", QD, *extra]


def manifest_rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


# caps that cannot bind: used wherever a test asserts presence / exact counts
BIG = ["--cap-day", 100000, "--cap-transition", 100000, "--cap-flash", 100000,
       "--flash-ratio", 1000, "--test-share", 1]

# ---------------------------------------------------------------- 1. dry run
print("\n1. dry run writes nothing")
OUT = W / "out"
code, o, e = run(*common(OUT))
check("exit 0", code == 0, e[-300:])
check("no out dir / manifest / culled change in dry run",
      not OUT.exists() and not (W / "manifest.csv").exists()
      and len((W / "culled.txt").read_text().strip().splitlines()) == n_culled0)
check("dry run mentions it would append culls", "new lines for" in o)

# ---------------------------------------------------------------- 2. apply
print("\n2. --apply, caps that cannot bind")
HOUT = W / "holdout_b3"
code, o, e = run(*common(OUT, "--apply", "--holdout-out", HOUT, *BIG))
check("exit 0", code == 0, e[-400:])
files = {p.name for p in OUT.glob("*.jpg")}
man = manifest_rows(W / "manifest.csv")
check("manifest rows == output files", {r["file"] for r in man} == files and len(man) == len(files),
      f"({len(files)})")
check("manifest has required columns", set(man[0]) >= {
    "file", "src", "label", "pos", "stamp", "bucket", "screen", "provenance", "verdict", "queue"})
check("no '10_' names", not any(f.startswith("10_") for f in files))
check("every crop is legacy-trusted or ledger ok/relabeled/ok_derived",
      all((r["provenance"] == "legacy" and r["verdict"] == "") or
          (r["provenance"] == "ledger" and r["verdict"] in B.OK_VERDICTS) for r in man))
mkey = {(r["stamp"], r["pos"]): r for r in man}
rej = [k for k, (v, _) in expect.items() if v in B.CULL_VERDICTS]
check("ledger rejections absent", not any(k in mkey for k in rej), f"({len(rej)})")
check("legacy crop rejected by ledger is absent", L_REJ not in mkey)
check("legacy relabel overrides label",
      L_RELAB in mkey and mkey[L_RELAB]["label"] == relab_to
      and mkey[L_RELAB]["file"].startswith(relab_to + "_main_"),
      f"(legacy {leg[L_RELAB][0]} -> {mkey.get(L_RELAB, {}).get('label')})")
check("legacy ok_derived kept (ledger provenance)",
      L_OKD in mkey and mkey[L_OKD]["verdict"] == "ok_derived")
check("legacy crop in holdout queue not trained", L_HOLD not in mkey)
check("undone rejection ignored (crop still legacy)",
      k_undone in mkey and mkey[k_undone]["provenance"] == "legacy")
rl = [k for k, (v, f) in expect.items() if v == "relabeled" and k in mkey]
check("relabeled crops use final label", rl and all(mkey[k]["label"] == expect[k][1] for k in rl),
      f"({len(rl)})")
check("no output crop on holdout day", not any(r["stamp"][:8] == HOLD_DAY for r in man))
check("frame-wide drift: every crop of a drift frame is absent",
      drift_st and not any(r["stamp"] in drift_st for r in man),
      f"({len(drift_st)} frames, {n_frame_drift} ok rows overridden)")
check("needs_verify, screened ok, unverified -> pending, absent", NV_PEND not in mkey)
check("needs_verify + label_ok (after screen) -> present, label = proposal",
      NV_OK in mkey and mkey[NV_OK]["label"] == expect[NV_OK][1]
      and mkey[NV_OK]["label_verdict"] == "label_ok")
check("needs_verify + label_ok written before the screen row -> present",
      NV_OK_EARLY in mkey and mkey[NV_OK_EARLY]["verdict"] == "ok")
check("needs_verify + label_fixed beats the screen relabel",
      NV_FIX in mkey and mkey[NV_FIX]["label"] == fix_to and mkey[NV_FIX]["verdict"] == "relabeled"
      and mkey[NV_FIX]["label_verdict"] == "label_fixed", f"(-> {fix_to})")
check("item in no queue (needs_verify 0) + label_fixed -> fixed label",
      NOQ_FIX in mkey and mkey[NOQ_FIX]["label"] == noq_to)
check("label_unsure -> absent", NV_UNS not in mkey)
check("report counts the pending crop", "ledger_pending_needs_verification=1" in o, "")
check("no holdout-queue crop in output",
      not any((r["stamp"], r["pos"]) in hold_expect for r in man))
check("b3 buckets come from b3_derive",
      all(r["bucket"] == next(d["bucket"] for d in [frames[r["stamp"]][r["pos"]]])
          for r in man if r["stamp"] in frames))
check("b3 screens come from b3_derive",
      all(r["screen"] == frames[r["stamp"]][r["pos"]]["screen"]
          for r in man if r["stamp"] in frames))
n_ledger_b3 = sum(1 for k, (v, _) in expect.items() if v in B.OK_VERDICTS)
pair_dt = Counter((r["pos"], r["label"]) for r in man if r["bucket"] != "flash")
missing = [k for k, (v, f) in expect.items() if v in B.OK_VERDICTS and k not in mkey]
only_night = [k for k in missing
              if frames[k[0]][k[1]]["bucket"] == "flash"
              and pair_dt[(k[1], expect[k][1])] == 0]
check("all ok-ish synthetic b3 crops present, except night-only (pos,class) pairs "
      "(flash-ratio rule: flash <= ratio x (day+transition) = 0)",
      len(only_night) == len(missing),
      f"({n_ledger_b3} ok-ish, {len(missing)} night-only dropped)")

# holdout export
hf = sorted(p.name for p in HOUT.glob("*.jpg"))
exp_frames = [s for s in hold_stamps[:8] if s not in (hold_stamps[1], hold_stamps[2],
                                                      hold_stamps[5], hold_stamps[6])]
check("holdout export = complete, un-rejected frames only", len(hf) == 5 * len(exp_frames),
      f"({len(hf)} files, {len(exp_frames)} frames)")
check("holdout export dropped the rejected frame and the incomplete frame",
      not any(hold_stamps[5] in f or hold_stamps[6] in f for f in hf)
      and hold_stamps[5] in o and hold_stamps[6] in o)
check("holdout: frame with an unverified needs_verify crop dropped (pending)",
      not any(hold_stamps[1] in f for f in hf)
      and f"pending frame {hold_stamps[1]}: dig5:needs verification" in o)
check("holdout: frame with a label_unsure crop dropped (rejected)",
      not any(hold_stamps[2] in f for f in hf)
      and f"rejected frame {hold_stamps[2]}: dig2:label_unsure" in o)
check("holdout: label_fixed crop exported with the fixed label",
      f"{h0_to}_main_dig6_{hold_stamps[0]}.jpg" in hf, f"(-> {h0_to})")
check("holdout: label_ok crop's frame exported",
      any(f.endswith(f"dig4_{hold_stamps[3]}.jpg") for f in hf))
check("holdout summary line counts pending frames",
      "1 frame(s) with an unverified/unscreened crop (1 awaiting verification)" in o)
s7 = hold_stamps[7]
check("holdout export applies relabel", any(
    f == f"{hold_expect[(s7, 'dig3')][1]}_main_dig3_{s7}.jpg" for f in hf))
check("holdout export includes ok_derived crop", any(f.endswith(f"dig6_{s7}.jpg") for f in hf))

# culled
cl = (W / "culled.txt").read_text().strip().splitlines()
new = cl[n_culled0:]
want = sorted({f"{k[0]},{k[1]}" for k in rej} | {f"{L_REJ[0]},{L_REJ[1]}"})
check("culled.txt: rejected crops appended in `stamp,pos` format",
      sorted(new) == want, f"({len(new)} new; {len(want)} expected)")
rr = list(csv.DictReader(open(W / "culled_reasons.csv", encoding="utf-8")))
check("culled_reasons.csv parallel rows (stamp,pos,verdict,src)",
      len(rr) == len(want) and set(rr[0]) == {"stamp", "pos", "verdict", "src"}
      and {r["verdict"] for r in rr} <= set(B.CULL_VERDICTS))
check("label_unsure culled with that verdict",
      any((r["stamp"], r["pos"], r["verdict"]) == (NV_UNS[0], NV_UNS[1], "label_unsure")
          for r in rr))
check("frame-drift crops culled as drift (crops with their own rejection keep it)",
      all(any((r["stamp"], r["pos"], r["verdict"]) == (k[0], k[1], expect[k][0]) for r in rr)
          for k in expect if k[0] in drift_st))
check("pending (unverified) crop NOT culled", f"{NV_PEND[0]},{NV_PEND[1]}" not in cl)
check("holdout-queue artifact is NOT culled",
      f"{hold_stamps[5]},dig4" not in cl)
check("holdout-queue label_unsure is NOT culled", f"{hold_stamps[2]},dig2" not in cl)

code, od, ed = run(*common(W / "out_default_caps", "--apply",
                           "--manifest", W / "manifest_dc.csv"))
check("default caps (60/40/80, share 0.3, ratio 1.0) run exits 0", code == 0, ed[-300:])
mdc = manifest_rows(W / "manifest_dc.csv")
check("default caps trim the pool (legacy + ledger > caps)", len(mdc) < len(man),
      f"({len(mdc)} < {len(man)})")

print("\n3. re-run is idempotent (determinism + no duplicate culls)")
snap = {p.name: p.read_bytes() for p in OUT.glob("*.jpg")}
code, o2, e2 = run(*common(OUT, "--apply", "--holdout-out", HOUT, *BIG))
check("exit 0", code == 0, e2[-300:])
check("same output files / bytes", snap == {p.name: p.read_bytes() for p in OUT.glob("*.jpg")})
check("culled.txt / reasons unchanged",
      (W / "culled.txt").read_text().strip().splitlines() == cl
      and len(list(csv.DictReader(open(W / "culled_reasons.csv", encoding="utf-8")))) == len(rr))

print("\n4. --no-ok-derived")
OUT2 = W / "out_nookd"
code, o, e = run(*common(OUT2, "--apply", "--no-ok-derived", *BIG, "--manifest", W / "manifest2.csv"))
man2 = manifest_rows(W / "manifest2.csv")
okd = [k for k, (v, _) in expect.items() if v == "ok_derived"] + [L_OKD]
check("exit 0", code == 0, e[-300:])
check("ok_derived crops dropped", not any((r["stamp"], r["pos"]) in set(okd) for r in man2)
      and not any(r["verdict"] == "ok_derived" for r in man2), f"({len(okd)} dropped)")
check("everything else unchanged", len(man2) == len(man) - len(okd))

print("\n5. caps (small caps, tight share) are respected")
OUT3 = W / "out_caps"
code, o, e = run(*common(OUT3, "--apply", "--cap-day", 2, "--cap-transition", 1,
                         "--cap-flash", 3, "--test-share", 0.3,
                         "--manifest", W / "manifest3.csv"))
check("exit 0", code == 0, e[-300:])
m3 = manifest_rows(W / "manifest3.csv")
cells = Counter((r["pos"], r["label"], r["bucket"]) for r in m3)
cap = {"day": 2, "transition": 1, "flash": 3}
check("every (pos,class,bucket) cell within cap", all(n <= cap[b] for (_, _, b), n in cells.items()))
bad = []
for lab in {l for _, l, _ in cells}:
    fl = sum(n for (_, l, b), n in cells.items() if l == lab and b == "flash")
    dt = sum(n for (_, l, b), n in cells.items() if l == lab and b != "flash")
    if fl > 1.0 * dt:
        bad.append((lab, fl, dt))
check("flash <= flash_ratio x (day+transition) per class (default --ratio-scope class)", not bad, str(bad))
tb = []
for (pos, lab, b), n in cells.items():
    if lab in ("0", "8"):
        t = sum(1 for r in m3 if (r["pos"], r["label"], r["bucket"]) == (pos, lab, b)
                and r["screen"] in B.TEST_SCREENS)
        if t > 0.3 * n + 1e-9 and "exempt" not in o:
            tb.append((pos, lab, b, t, n))
exempt = set()
for line in o.splitlines():
    if line.strip().startswith("dig") and "test-screen)" in line:
        exempt.add(tuple(line.split()[:3]))
tb = [x for x in tb if (x[0], "class", x[1]) not in exempt]
check("test-screen share <= 0.3 in each class 0/8 cell (non-exempt pairs)", not tb, str(tb))
# preference check straight from the data
pool = defaultdict(lambda: [0, 0])
for r in man:   # default-caps run = full pool
    pool[(r["pos"], r["label"], r["bucket"])][0] += 1
viol = 0
leg_in = {(r["stamp"], r["pos"]) for r in man if r["in_legacy"] == "1"}
for cell in {(r["pos"], r["label"], r["bucket"]) for r in man}:
    full = [r for r in man if (r["pos"], r["label"], r["bucket"]) == cell
            and r["screen"] not in B.TEST_SCREENS]
    kept = [r for r in m3 if (r["pos"], r["label"], r["bucket"]) == cell
            and r["screen"] not in B.TEST_SCREENS]
    full_leg = sum(1 for r in full if r["in_legacy"] == "1")
    kept_leg = sum(1 for r in kept if r["in_legacy"] == "1")
    if kept_leg < min(full_leg, len(kept)):
        viol += 1
check("legacy-trusted kept first when trimming", viol == 0, f"({viol} cells)")
# diversity: in a trimmed cell the kept b3 crops should span distinct days
cells_days = []
for cell in {(r["pos"], r["label"], r["bucket"]) for r in m3}:
    full = [r for r in man if (r["pos"], r["label"], r["bucket"]) == cell and r["in_legacy"] != "1"]
    kept = [r for r in m3 if (r["pos"], r["label"], r["bucket"]) == cell and r["in_legacy"] != "1"]
    if len(kept) >= 2 and len(full) > len(kept):
        nd_full = len({r["stamp"][:8] for r in full})
        cells_days.append(len({r["stamp"][:8] for r in kept}) >= min(len(kept), nd_full))
check("trimmed cells are round-robined over capture days", all(cells_days), f"({len(cells_days)} trimmed cells)")
code, o3, _ = run(*common(W / "out_caps_b", "--apply", "--cap-day", 2, "--cap-transition", 1,
                          "--cap-flash", 3, "--manifest", W / "manifest3b.csv"))
check("seeded: identical selection on rerun", [r["file"] for r in m3] ==
      [r["file"] for r in manifest_rows(W / "manifest3b.csv")])

print("\n6. hard errors (nothing written, non-zero exit)")
# 6a: ledger row on a holdout day outside the holdout queue
bad_rows = rows + [row(hold_stamps[9], "dig3", frames[hold_stamps[9]]["dig3"]["path"],
                       "night_err", "7", "7", "ok", "gBAD")]
write_ledger(W / "ledger_hday.csv", bad_rows)
OUT4 = W / "out_hday"
code, o, e = run(*common(OUT4, "--apply", ledger=W / "ledger_hday.csv",
                         manifest=W / "m4.csv"))
check("holdout-day crop -> exit 2", code == 2)
check("offender listed", hold_stamps[9] in e and "holdout day" in e)
check("nothing written", not OUT4.exists() and not (W / "m4.csv").exists())

# 6b: output crop in validation_labeled
VAL = W / "validation_fake"
VAL.mkdir()
shutil.copy2(REPO / "work/validation_labeled/0_main_dig2_20260802-224010.jpg",
             VAL / f"{leg[L_RELAB][0]}_main_{L_RELAB[1]}_{L_RELAB[0]}.jpg")
code, o, e = run(*common(W / "out_val", "--apply", validation=VAL, manifest=W / "m5.csv"))
check("validation_labeled overlap -> exit 2", code == 2 and "validation_labeled" in e
      and L_RELAB[0] in e)
check("nothing written", not (W / "out_val").exists())

# 6c: holdout-days file missing with --apply
code, o, e = run(*common(W / "out_nohd", "--apply", hdays=W / "nope.txt", manifest=W / "m6.csv"))
check("missing holdout_days + --apply -> exit 2", code == 2 and not (W / "out_nohd").exists())
code, o, e = run(*common(W / "out_nohd", hdays=W / "nope.txt", manifest=W / "m6.csv"))
check("missing holdout_days dry run -> warns, exit 0", code == 0 and "WARNING" in o)

# 6d: label 10 in ledger is normalised to N, never emitted as 10
rows10 = [dict(r) for r in rows]
tgt = next(r for r in rows10 if r["verdict"] == "ok" and r["queue"] == "night_err"
           and (r["stamp"], "dig" + r["pos"]) not in nv_keys and r["stamp"] not in drift_st)
tgt["final"] = tgt["proposed"] = "10"
write_ledger(W / "ledger10.csv", rows10)
code, o, e = run(*common(W / "out10", "--apply", ledger=W / "ledger10.csv",
                         manifest=W / "m7.csv"))
f10 = [p.name for p in (W / "out10").glob("*.jpg")]
check("ledger label '10' written as N", code == 0 and not any(f.startswith("10_") for f in f10)
      and any(f.startswith(f"N_main_{tgt['pos'] and 'dig' + tgt['pos']}_{tgt['stamp']}") for f in f10))
# ...and the gate itself rejects a '10' label that slipped through
c10 = B.Crop("20260901-120000", "dig3", "10", Path(__file__), "day", "reading",
             "ledger", "ok", "x", False)
errs = B.run_gates([c10], {}, {}, set(), set(), type("A", (), {"no_ok_derived": False}), "t")
check("gate flags label '10'", any("bad label" in x.title for x in errs))

# 6e: provenance gate -- inject an unreviewed crop at the candidate stage
real_bc = B.build_candidates


def injecting(*a, **k):
    crops, rej_, ho, prob, info = real_bc(*a, **k)
    fake = rows[0]
    crops.append(B.Crop("20260901-120000", "dig3", "7",
                        Path(frames[chosen[0]]["dig3"]["path"]), "day", "reading",
                        provenance="legacy", in_legacy=True))   # claims legacy, is not
    return crops, rej_, ho, prob, info


B.build_candidates = injecting
ns = type("NS", (), {})()
import argparse  # noqa: E402
ap = argparse.ArgumentParser()
B.add_ledger_args(ap)
ap.add_argument("--out", type=Path, default=B.DEFAULT_OUT)
ap.add_argument("--labeled", type=Path, default=REPO / "work/labeled")
ap.add_argument("--batch1", type=Path, default=REPO / "joes-samples/batch 1")
ap.add_argument("--exclude", type=Path, default=W / "culled.txt")
ap.add_argument("--flash-ratio", type=float, default=1.0)
ap.add_argument("--ratio-scope", default="class")
ap.add_argument("--seed", type=int, default=20260807)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args(["--from-ledger", "--ledger", str(LEDGER), "--holdout-days", str(HDAYS),
                   "--queues-dir", str(QD),
                   "--validation", str(W / "empty_validation"), "--out", str(W / "out_inject"),
                   "--manifest", str(W / "m8.csv"), "--culled-reasons", str(W / "cr8.csv"),
                   "--apply"])
a.labeled, a.batch1 = REPO / "work/labeled", REPO / "joes-samples/batch 1"
import contextlib  # noqa: E402
import io  # noqa: E402
buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    rc = B.from_ledger(a)
B.build_candidates = real_bc
check("injected unreviewed crop -> from_ledger returns 2", rc == 2)
check("offender listed with reason", "without qualifying provenance" in buf.getvalue()
      and "20260901-120000" in buf.getvalue(), buf.getvalue().strip().splitlines()[2:4].__repr__())
check("nothing written", not (W / "out_inject").exists())
# a ledger-provenance crop whose ledger row is a rejection also trips the gate
ledx = B.load_ledger_eff(LEDGER)
rej_key = rej[0]
cbad = B.Crop(rej_key[0], rej_key[1], "5", Path(__file__), "day", "reading", "ledger", "ok", "")
check("gate flags ledger-provenance crop whose verdict is a rejection",
      bool(B.check_provenance([cbad], ledx, {}, type("A", (), {"no_ok_derived": False}))))
nvx, _ = B.load_needs_verify(QD)
cpend = B.Crop(NV_PEND[0], NV_PEND[1], expect.get(NV_PEND, ("", "5"))[1] or "5", Path(__file__),
               "day", "reading", "ledger", "ok", "night_err")
bad_p = B.check_provenance([cpend], ledx, {}, type("A", (), {"no_ok_derived": False}), nvx)
check("gate flags an unverified needs_verify crop (ledger pending)",
      bool(bad_p) and "needs verification" in bad_p[0], str(bad_p[:1]))
check("...but not without the needs_verify map (item counts as 0)",
      not B.check_provenance([B.Crop(NV_PEND[0], NV_PEND[1],
                                     B.ledger_label(ledx[NV_PEND]), Path(__file__), "day",
                                     "reading", "ledger", "ok", "night_err")],
                             ledx, {}, type("A", (), {"no_ok_derived": False})))

print("\n7. out-dir safety / backup only for the default out dir")
code, o, e = run(*common(W / "out", "--apply", "--manifest", W / "m9.csv"))
check("custom --out: no backup dir created",
      code == 0 and not list((W / "out").glob("_backup_*")) and "backed up" not in o)
code, o, e = run(*common(REPO / "joes-samples/batch 1", "--apply"))
check("--out inside batch 1 refused", code == 2 and "protected" in e)
code, o, e = run(*common(REPO / "work/labeled/a", "--apply"))
check("--out inside work/labeled refused", code == 2)

# default-out scenario in a scratch cwd that owns a joes-samples/
CWD = W / "cwd_default"
(CWD / "joes-samples" / "batch 1").mkdir(parents=True)
b1 = CWD / "joes-samples" / "batch 1"
shutil.copy2(leg[legkeys[3]][1], b1 / leg[legkeys[3]][1].name)
stale = ["3_main_dig4_20260101-000000.jpg", "N_main_dig2_20260101-000100.jpg"]
for n in stale:
    shutil.copy2(leg[legkeys[3]][1], CWD / "joes-samples" / n)
(CWD / "joes-samples" / "notes.xlsx").write_bytes(b"x")
b1_before = {p.name: p.read_bytes() for p in b1.iterdir()}
args = common(Path("joes-samples"), "--apply", "--labeled", W / "no_labeled",
              "--batch1", b1, manifest=CWD / "man.csv")
args = [str(x) for x in args]
r = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=CWD, capture_output=True, text=True)
bk = list((CWD / "joes-samples").glob("_backup_*"))
check("default out: exit 0", r.returncode == 0, r.stderr[-300:])
check("default out: backup subdir created with the stale top-level jpgs",
      len(bk) == 1 and sorted(p.name for p in bk[0].glob("*.jpg")) == sorted(stale))
check("default out: stale jpgs cleared from top level", not any(
    (CWD / "joes-samples" / n).exists() for n in stale))
check("default out: batch 1 and xlsx untouched",
      {p.name: p.read_bytes() for p in b1.iterdir()} == b1_before
      and (CWD / "joes-samples" / "notes.xlsx").exists())
check("default out: ledger ok crops written to top level",
      len(list((CWD / "joes-samples").glob("*.jpg"))) > 0)
check("default out: backup is a subdir (prepare_joe_data's top-level glob won't see it)",
      not any(p.parent != CWD / "joes-samples" for p in (CWD / "joes-samples").glob("*.jpg")))

print()
if fails:
    print(f"FAILED ({len(fails)}): {fails}")
    sys.exit(1)
print("ALL TESTS PASSED")
