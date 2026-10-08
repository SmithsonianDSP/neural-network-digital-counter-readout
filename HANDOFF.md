# Handoff: electric meter LCD — batch 3 and the 9015 retrain

Written 2026-10-03. Supersedes `HANDOFF-9002.md` (the 2026-08-08 handoff, archived
unchanged — read it for the 9001/9002 history, §5c's class-3 lesson and §5d's
derived-label technique). Original problem statement: `HANDOFF-ELECTRIC-LCD.md`.

---

## 1. State

- **DEPLOYED: `models/dig-class11_9015_s2.tflite`** (float, 349 KB), loaded 2026-10-03
  over the network (no SD pull needed for model swaps). Rollback: `dig-class11_9002_s2`
  is still on the card.
- **Not yet verified on device.** The ROIs were re-set at the last remount (early Oct, to
  pull the SD card), so the current framing is in no test set. Verdict = 2–3 nights of new
  history CSVs vs 9002's Aug 15–Sep 30 baseline (§7.1).
- **Batch 3:** 82,795 crops / 16,559 frames, 2026-08-15 → 09-30 (+2 frames on 08-07), in
  `AIOTED-digital-rawdigits/<YYYYmmdd>/<HH>/`, labels = 9002's predictions. Daily history
  CSVs `data_YYYY-MM-DD.csv` (08-07 → 09-30, no header, 13 cols: ts, name, raw, value, pre,
  rate, abs, error, dig2..dig6). Image dirs 08-08 → 08-14 do not exist.
- **Review ledger:** `work/review_ledger.csv` (append-only; backups
  `work/review_ledger.backup-*.csv`). Every training and holdout crop has a human verdict.
- **Candidate held, not deployed: `models/dig-class11_9018_s2.tflite`** (2026-10-08).
  Best on every test set (§5a), but 3× more fragile under int8 and on the same 7/3 knife
  edge as the bad seeds. Decision deferred to the mid-Nov rollover batch (§7.1).
- **Corpus:** `joes-samples/` top level = 6,785 crops (`work/corpus_manifest.csv`) — the
  9018–9020 corpus, built by `build_corpus.py --from-ledger --stable-sampling --reserve
  work/selection_night_manifest.txt`. 9015's corpus is the newest `joes-samples/_backup_*/`;
  older: 140814 = the old 1,288; 182230 = 9008's; 183546 = 9009's; 190016 = 9010–12's.
- **Holdout:** `work/holdout_b3/` — 790 crops / 158 complete frames from 7 whole held-out
  days (`work/queues/holdout_days.txt`: 08-18, 08-25, 09-01, 09-09, 09-16, 09-23, 09-29).
  No crop from those days is in any corpus. Includes dash/N. The old 240-crop
  `work/validation_labeled/` is kept for information only.
- **Selection set:** `work/selection_night/` — 662 reviewed night crops that no candidate
  trained on (excludes every corpus snapshot). Used to *pick* among candidates so the
  holdout stays an honest confirmation.

## 2. What batch 3 changed (read before trusting anything older)

- **The meter crossed 58000 on 08-15 and 59000 on 09-11** (57687 → 59424). dig3 is 7/8/9 on
  real readings; every dig2≡5/dig3≡7 assumption is dead. `select_review_set.py` now uses
  reading = 50000 + 1000·d3 + 100·d4 + 10·d5 + d6 and types the screen from dig2. 60000 is
  ~40 days out at current usage (~12–14 kWh/day); dig2 will change then — check the
  toolchain before that batch.
- **Light buckets are brightness-based now** (`dig_data.light_bucket`: flash ≤ 81, day ≥ 96
  frame-median luma). The hour rule was wrong by late Sept (hour 18 at night level).
- **The derivation inherits consistent misreads.** After the 09-04 framing step, 9002 read
  daytime dig6 9→8 and 5→6 consistently; a monotone fit sees "198 lasting twice as long"
  and agrees. The skip guard in `select_review_set.py` catches some; human consistency
  passes caught the rest (16 relabels in day/night fill crops that "model and derivation
  agreed" on). **"Model + derivation agree" is not proof for dig6 5/6, 8/9.**
- **Framing:** every SD pull = unmount → remount → re-set reference/markers/ROIs, so a
  few px of framing change between deployments is normal. `tools/roi_drift.py` measures
  it (found ~4–8 px full-res at 09-04); Joseph checked by eye and judged it not
  meaningful. **Drift as a review verdict means per-frame alignment failure**
  (clipped glyph / neighbour intruding) — it is frame-wide, and mostly occurred *before*
  09-04 (glare mornings). Automatic pre-flagging was tried and is not good enough
  (best: 11/17 caught at 61% precision) — drift stays a human call.
- **Joseph's HA filtered sensor** (already in place) is rate-aware, knows sun/AC state, and
  is strict at night (never > 4 kWh/h). It rejects downward readings and tens-digit jumps.
  **So the only night errors that reach his data are small *upward* dig6 errors.** Rank
  models on that (§5), not on raw dig5 accuracy.

## 3. Rules (carried forward, revised, and new)

Carried forward unchanged: label naming `<label>_main_dig<pos>_<stamp>.jpg`, never `10_`;
never run notebooks 01–03; architecture fixed (`dig_model.py`); ship float; light/class
decorrelation; `--user-weight 1`; keep upstream but never gate on it; fresh holdout before
selection; per-crop selection for training, complete frames in the holdout.

New or revised this cycle (all confirmed with Joseph):

- **Nothing trains or enters the holdout without a human verdict.** Enforced: `build_corpus.py
  --from-ledger` hard-fails on any crop without qualifying provenance. Derived labels are
  *proposals* shown to the reviewer.
- **Artifact rule:** localized non-display artifacts → exclude (reflection streaks/blobs,
  dust/smear, mid-transition ghosting, per-frame misalignment). Uniform haze/glare wash/flash
  over-exposure → keep (§5c of the old handoff still holds).
- **No "illegible" verdict.** Human legibility is not the criterion. Joseph cannot beat
  chance on night dig6 0/8/9 by eye; those labels rest on the **timeline**. For such crops
  the reviewer judges artifacts + whether the timeline settles the digit, and `x`s it if not.
- **Use the wide time strip (`--strip 8`) for night label calls.** The 3-reading strip was
  error-prone: revisiting 42 `x` calls with 8 readings each side changed 24 of them
  (incl. all 9 dig6 "4"s → really 1s, the old silent `1→4`) and recovered 37.
- **Flash-ratio is per class** (`--ratio-scope class`, default), not per (pos, class) — the
  network has no positional input, so the confound lives at class level; per-cell stranded
  night data wherever a position's day supply was thin.
- **CV folds group whole days** (`--fold-by day`, default). Round-robin by frame leaked
  lighting between adjacent frames.
- **Never pick a model from one training run.** At night, seed alone swings night dig6
  by 15–25 points on the same corpus (§5). Train ≥3 seeds, pick on the selection set,
  confirm on the holdout.

## 4. Toolchain (new/changed — all in `tools/`, run from repo root)

| script | purpose |
|---|---|
| `select_review_set.py --derive` | full-reading derivation (rollover-aware longest monotone chain, CSV anchors, skip guard) → `work/b3_derive.csv`; `--stats-cache work/b3_frame_stats.csv` |
| `dig_data.py` | + `light_bucket`, `frame_brightness`, `classify_screen` |
| `roi_drift.py` | per-crop / per-frame shift vs templates → `work/roi_drift.csv`, plots |
| `build_queues.py` | holdout + priority review queues → `work/queues/*.csv` (`needs_verify` column) |
| `grid_review.py` | the reviewer. Modes: `--mode verify` (confirm uncertain labels), `screen` (artifacts/drift only), `consistency` (pages of one label, all light mixed), `audit` (re-check crops already labelled; `--redo-since STAMP` re-opens `x` calls). `--strip N` sets the zoom time strip. `d` is frame-wide. 22 tests in `test_grid_review.py` |
| `build_corpus.py --from-ledger` | ledger-gated corpus + `--holdout-out`; per-cell caps, test-share cap, per-class flash ratio; `--stable-sampling` (always use), `--reserve FILE`; writes manifest, culls with reasons; backs up `joes-samples/` before `--apply`. Tests: `work/_test_build_corpus/run_tests.py` |
| `oof_audit_queue.py` | CV out-of-fold disagreements → audit queue |
| `grid_review.py` zoom | the time strip's `h` line shows the human label of each neighbouring frame |
| `holdout_report.py` | stratified holdout report + fixed/broken lists |
| `train_dig_class11.py` | + `--fold-by`, rep-dataset capped at `--rep-size` |
| `drift_preflag.py` | the (failed) automatic drift pre-flag analysis |

`work/_score_candidates.py` is a one-off scorer (holdout / selection / probe) — promote it
if reused. `work/legacy_src/` holds the original 1,288 corpus files permanently (ledger rows
for pre-batch-3 crops point at `joes-samples/`, which is an *output* dir).

## 5. Results

Day-grouped 5-fold CV on the 9008 corpus: best epochs 30/100/69/52/79 → **E\* = 75**;
pooled OOF 97.77%. All finals: `--final --epochs 75 --user-weight 1`, ~9–10 s/epoch with
three runs in parallel (fine on this box; four+ slows everything).

| model | corpus | seed | hold all | night rdg frames | hold nt dig6 | hold nt dig5 | sel nt dig6 | probe | up-errors* |
|---|---|---|---|---|---|---|---|---|---|
| 9002 | (Aug) | — | 90.5% | 41.7% | 58.3% | 79.2% | 27.9% | 14/71 | 70 |
| 9008 | pre-audit | 42 | 98.2% | 85.4% | 91.7% | 93.8% | 92.9% | 62/71 | 7 |
| 9016 | pre-audit | 7 | 95.2% | 66.7% | 72.9% | 91.7% | 64.3% | 27/71 | |
| 9017 | pre-audit | 123 | 96.8% | 70.8% | 72.9% | 95.8% | 81.4% | 58/71 | |
| 9013 | final | 42 | 97.3% | 85.4% | 85.4% | 100% | 71.4% | 34/71 | |
| 9014 | final | 7 | 97.2% | 75.0% | 83.3% | 100% | 78.6% | 42/71 | |
| **9015** | final | 123 | 97.6% | 77.1% | 87.5% | 97.9% | **93.6%** | **63/71** | **2** |

\*upward night dig6 errors on reading screens, holdout + selection (n = 187).
"probe" = agreement with Joseph's wide-strip verdicts on 71 disputed selection crops.

9015 gates: op set identical ✅; `1→7` = 0 ✅; ROI-shift 0.984/0.957 vs 9002's 0.934/0.882
on the same corpus ✅; old Aug holdout 98.33% vs 97.92% ✅; day/transition 100% ✅;
11 broken vs 9002 — 6 are dig3 8→0 on the single night of 08-18 (−8000, always rejected),
the rest low-confidence downward night misreads. Upstream 89.3% (info only; every batch-3
model drifts there, 9008 92.8%) — don't use 9015 on a different meter.

`_q` (int8) of 9015 is viable now (rep set fixed): ~6 pts worse on night dig6/frames,
0 upward night dig6 errors, 0 false `N`. Float is shipped; `_q` is the fallback if memory
is ever needed.

**Lessons with evidence:**
- **The audit hit rate is high where the model is confident.** OOF disagreements at
  conf ≥ 0.7: 40/92 were label errors (43%); below 0.7: 5/56 (9%). Cheapest label cleanup
  there is.
- **Removing boundary examples moves the boundary** (old §5c, again): the first audit `x`'d
  ~11 of ~35 night dig6 0s, and two of three seeds then showed 0→8. The wide-strip revisit
  recovered most of them.
- **9008 was a lucky draw**, not a better corpus: its corpus with seeds 7/123 averaged well
  below it. Without the seed sweep we'd have shipped on luck and blamed the audit.
- **A blind human-vs-model test was skipped** (Joseph: he can't beat chance on night dig6
  0/8/9). The model reads signal the eye can't; that only works because the timeline
  labels are good — keep them good.

## 5a. Follow-up (2026-10-08): right epoch count, the 7/3 cluster, 9018

**E\* was wrong for the cleaned corpus.** Fresh day-grouped CV on the final corpus: best
epochs 103/91/150/146/96 → **E\* = 125** (not 75); pooled OOF **98.47%** (was 97.77% on
the 9008 corpus — the label cleanup measurably helped). 9013–9015 were trained at 75.

| model | seed | epochs | hold all | night rdg frames | hold nt dig6 / dig5 | sel nt dig6 | probe | up +1/+2 | up ≥+3 |
|---|---|---|---|---|---|---|---|---|---|
| 9015 (deployed) | 123 | 75 | 97.6% | 77.1% | 87.5 / 97.9% | 93.6% | 63/71 | 0 | 2 |
| **9018** | 42 | 125 | **98.4%** | **87.5%** | **91.7 / 100%** | **95.0%** | **64/70** | 0 | 5 |
| 9019 | 7 | 125 | 95.7% | 68.8% | 70.8 / 95.8% | 55.4% | 21/70 | 7 | 6 |
| 9020 | 123 | 125 | 96.1% | 70.8% | 75.0 / 95.8% | 59.0% | 21/70 | 0 | 9 |

9018 gates: op set ✅, `1→7` 0 ✅, ROI-shift 0.979/0.950 ✅, old Aug holdout 97.92% ✅,
only 3 broken vs 9002 ✅, upstream 92.4% (info). int8: 9018 `_q` loses 16 pts on selection
night dig6 (95.0 → 79.1%) and 5 daytime crops; 9015 `_q` loses 3 pts. **If a quantized
build is ever needed, use 9015 `_q`.**

**The seed variance is one cluster flipping.** 52 night dig6 **7s** in the selection set
(mostly the Sept stretches where 9002 read them as 0) decide everything: 9019 reads 46 of
them as 3, 9020 42, 9018 `_q` 12, 9015 5, 9018 float 0 — and every model's 7→3 errors are
a subset of the bad seeds'. **All 52 were verified as 7s with the wide strip** (46 in the
probe, the last 6 on 2026-10-08). So the training data supports two near-equal rules for
this washout; at 75 or 125 epochs, about one seed in three lands on the right one. Longer
training changed *which* seed wins, not the odds. 9018 is confident on these 7s (mean
p(3) 0.05) yet flips under int8 — the fragility is in the weights, not the output
confidence, so on-device confidence cannot reveal it.

**Rules from this:** ≥3 seeds per candidate is mandatory, not optional; pick on the
selection set (never the holdout); check the int8 build as a robustness probe even when
shipping float; never reuse an E\* across a changed corpus — re-run CV.

## 6. Known gaps

> **⚠ Scrutinize every step of +2 kWh or more between consecutive accepted readings —
> *especially* when no error was registered.** (Joseph, 2026-10-07.) Larger steps (+3,
> +4, …) are not automatically "catch-ups after rejected reads"; a big step soon after
> the previous accepted reading is the misread itself. Judge each step against the time
> since the previous accepted reading.
>
> At this meter's night usage, consecutive accepted readings normally differ by 0 or +1.
> A +2 step means a value was skipped. A skipped value is the footprint of the one
> failure that gets past every filter: an upward dig6 misread small enough to look like
> real usage. Live example, 9015's first night (2026-10-04, early morning): dig6 went
> 2 → **7** → (resync) → 4 with no accepted 3. The 7 was a misread, most likely 3→7 (the
> known night axis, §5), possibly 2→7. The HA filter caught that one because +5 was too
> big. A +1 or +2 misread (2→3 read early, 3→5, 7→9) would not be caught, and leaves
> exactly this signature: a +2 step and **no error anywhere**.
>
> - **Monitoring:** a +2 step in the accepted series is a suspected silent misread, not
>   "usage". Check the raw reads (CSV col 3, all frames incl. rejected) around it: did the
>   skipped value ever appear? Did the larger value appear early or flicker?
> - **Label derivation:** the same signature corrupts training labels. A consistent
>   misread makes a value look like it "lasted twice as long, then skipped one"
>   (§2, 9→8 / 5→6 after 09-04). `select_review_set.py`'s skip guard unpins runs around
>   skips. Keep it, and treat crops next to any +2 step as Mode-1 (verify with the wide
>   strip), never as "model + derivation agree".
> - **Model evaluation:** "zero registered errors" is not the same as "zero wrong
>   readings". Count +2 steps when judging a deployed model.

**Label-consistency checks (2026-10-08).** Two cheap nets for labels that "read clean":
- *Model refuses its own training label:* run the shipped model over `joes-samples/`;
  of 9015's 80 disagreements, 8 were real label errors (10%) — after every other pass.
- *Count-up rule on human labels:* within a frame's human dig5+dig6 labels, the
  two-digit value must never decrease over time. **Group by the human labels, never by
  the derivation's `reading_est`** — the first attempt grouped by the derived tens digit,
  which is wrong in exactly the weak stretches, and produced 30/30 false alarms. Done
  correctly on 563 frames: 0 violations.
- Worked example of a derivation wrong for an hour: 2026-09-03 01:23–02:31, dig6 was 4
  while 9002 read 9 every frame; the fit held "58683" throughout. The wide strip now
  shows the human label of each neighbour (`h`), which made this visible.

- **The night dig6 7/3 boundary is under-supported** (§5a) — the root of the seed
  lottery. Top model-side gap.
- **Cap-sampling churn — fixed as a flag.** `--stable-sampling` orders capped cells by a
  per-crop hash, so a rebuild moves only the crops whose verdicts changed. It is *off by
  default* (old builds stay reproducible); always pass it from now on. Switching it on
  cost a one-time churn of ~580 crops vs 9015's corpus. `--reserve FILE` keeps listed
  crops (the selection set) out of any corpus.
- **~1,900 reviewed, accepted crops are capped out of training** (pool 8,712 → 6,793). Free
  boundary data; using it costs the selection set unless a new one is carved.
- **~550 "soft" night dig5/dig6 labels** (derivation ambiguous, verified with the 3-reading
  strip) remain in corpus + holdout: 330 dig6 + 188 dig5 in the corpus, 27 in the holdout.
  The 71-crop probe confirmed 70/71 of a disputed sample, so the error rate is probably
  low, but the `x` revisit shows narrow-strip calls can be wrong.
- **Night `00000`-screen 0s at dig3/5/6 are nearly absent from training** (test-share cap
  ~4–5% of those cells) — the source of the 0→8 bias some seeds show. Their labels are
  certain (screen-pinned).
- **Aug-era washed dig6 8s** (`work/dig6night_test`, old mount, extreme washout): every
  batch-3 model reads them as 0 (9006's 11/15 came from an "8" bias). Direction is downward
  (rejected), so low priority unless that regime recurs.
- **Night dig5 3→7** is the most common remaining dig5 axis; harmless with the HA filter
  (tens jump), costs readings.
- **Winter:** night share of captures grows. The current night numbers are the ones that
  matter more each month.

## 7. Next actions, highest value first

1. **On-device verification** after 2–3 nights: drop new CSVs (and crops, if handy) into
   `AIOTED-digital-rawdigits`; compare night accepted-reading rate and non-monotone count
   vs 9002's Aug 15–Sep 30 baseline. Rollback = switch the model file back to 9002.
   **List every step of +2 kWh or more in the accepted series** (§6 note) and read the raw dig6
   sequence around each; for 2026-10-04's 2→7, see whether the 7 sat in the 2-run or the
   3-run.
   **Also decide 9015 vs 9018 here:** score both (float and `_q`) on the rollover batch —
   the first data with post-remount framing and a new dig2 digit. Swap earlier only if the
   HA dashboard shows 9015 doing worse on device than its test numbers.
2. **Support the night dig6 7/3 boundary** (§5a). First carve a *new* selection set from
   held-back crops in the same Sept stretches (the current one can't judge a cluster once
   it's trained on). Then add wide-strip-reviewed night dig6 7s with this washout plus the
   washed 3s they're confused with (~15–20 min of review) and the ~1,900 reviewed crops
   currently capped out. Success test: **all 3 seeds** get the cluster right.
3. **Variance reduction in the recipe:** stochastic weight averaging over the last ~15
   epochs of a final run (still one exported model). Re-run the 3-seed protocol and
   compare spread.
4. **Night `00000` 0s:** queue ~60 dig3/5/6 night `00000` crops for a screen pass
   (labels certain) and relax the test-share cap for class 0 at night.
5. **Dimmer-flash experiment** (AIOTE LED intensity): one night's capture. The night
   failure is over-exposure washing single segments — a physical check beats more analysis.
6. **Wide-strip re-verify of the ~550 soft night labels** — only if §7.1 shows night dig6
   errors still reaching HA.
7. **Joseph's idea: train hopeless night dig6 crops as `N`** so the model abstains instead
   of guessing. Candidate set exists (his `x` calls). Risks: N bleeding to other positions
   (no positional input), contradictory pairs. Measure as a variant on the same holdout:
   abstentions gained vs wrong readings removed. Only worth it if §7.1 shows upward dig6
   errors still getting through the HA filter.
8. Before the meter reaches 60000 (~mid-Nov): dig2 changes 5→6 — re-check screen typing
   (dig2 is the screen discriminator) and derivation.

## 8. Environment

Python 3.12.2, TF 2.21.0 (CPU), Keras 3.15.1, Pillow 12.2.0, ai_edge_litert. Background
shells die at ~60 min — one fold / one final per command; three in parallel is fine.
Never point tests at `work/review_ledger.csv`. Subagent tiering: see Joseph's preference
(Haiku/Sonnet for mechanical work, Opus for design; the main session reviews everything).
