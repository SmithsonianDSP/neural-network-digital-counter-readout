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
- **Corpus:** `joes-samples/` top level = 6,793 crops (`work/corpus_manifest.csv`), built by
  `build_corpus.py --from-ledger`. Earlier corpora are kept in `joes-samples/_backup_*/`
  (140814 = the old 1,288; 182230 = 9008's; 183546 = 9009's; 190016 = 9010–12's).
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
| `build_corpus.py --from-ledger` | ledger-gated corpus + `--holdout-out`; per-cell caps, test-share cap, per-class flash ratio; writes manifest, culls with reasons; backs up `joes-samples/` before `--apply`. Tests: `work/_test_build_corpus/run_tests.py` |
| `oof_audit_queue.py` | CV out-of-fold disagreements → audit queue |
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

## 6. Known gaps

- **Cap-sampling churn.** Per-cell caps sample with `random.Random(seed|cell)` over the
  pool, so any pool change (even 9 relabels) reshuffles the whole cell — ~100 night crops
  swapped per rebuild. Makes corpus comparisons noisy. Fix first (§7.2).
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
2. **Stable cap sampling** in `build_corpus.py`: rank each candidate by a hash of its
   item_id and take the top N per cell, so pool changes only move the crops that changed.
3. **Variance reduction in the recipe:** stochastic weight averaging over the last ~15
   epochs of a final run (still one exported model), and/or raise caps to use the ~1,900
   reviewed-but-unused crops (carve a fresh selection set first). Then re-run the 3-seed
   protocol and compare spread.
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
