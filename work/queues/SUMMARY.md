# Batch-3 review queues

Generated 2026-10-03 12:11 by `tools/build_queues.py` (seed 20261003). Review in this order; stopping after any queue leaves a usable corpus.

```
python tools/grid_review.py --queue work/queues/holdout.csv --frame-mode
python tools/grid_review.py --queue work/queues/legacy.csv --queue work/queues/night_err.csv --queue work/queues/day_err.csv --queue work/queues/night_fill.csv --queue work/queues/day_fill.csv
```

## Queues

| # | queue | crops | mode | est. minutes | cumulative min |
|---|---|---|---|---|---|
| 0 | `holdout.csv` | 875 | frame | 29 | 29 |
| 1 | `legacy.csv` | 160 | grid | 3 | 32 |
| 2 | `night_err.csv` | 1281 | grid | 21 | 53 |
| 3 | `day_err.csv` | 147 | grid | 2 | 56 |
| 4 | `night_fill.csv` | 3105 | grid | 52 | 107 |
| 5 | `day_fill.csv` | 2134 | grid | 36 | 143 |

Composition (source screen of each crop; `skip` = dig6 proposal from skip inference):

- `night_err`: {'reading': 1112, 'test': 131, 'dash': 38}; skip-inferred 155; model_disagrees 1281; drift 5
- `day_err`: {'dash': 20, 'reading': 109, 'test': 18}; skip-inferred 71; model_disagrees 147; drift 0
- `night_fill`: {'reading': 2633, 'test': 251, 'dash': 221}; skip-inferred 0; model_disagrees 0; drift 0
- `day_fill`: {'reading': 1832, 'test': 187, 'dash': 115}; skip-inferred 0; model_disagrees 0; drift 0

Total 7702 crops, ~143 min (1 s/crop grid, 2 s/crop frame mode).

## Holdout

Days: 20260818, 20260825, 20260901, 20260909, 20260916, 20260923, 20260929 (all full 360-frame days; no substitution needed). 175 frames.

- buckets: {'flash': 84, 'day': 63, 'transition': 28}
- screens: {'reading': 105, 'test0': 24, 'test8': 24, 'dash': 22}
- hours covered: 24/24
- frames with an edge-drift crop were not eligible; rows are interleaved across days so a partial review still spans every day.

## Legacy

- derived-only corpus crops: 149 (of which 90 are the staged culls in `work/derived_only_review/`; staged-not-in-corpus: 0)
- eval_9002 disagreements with conf > 0.85: 14; 11 queued, 3 already relabelled to 9002's answer and skipped:
  - 20260804-035525 dig4: csv label 5, file now work\labeled\c\8_main_dig4_20260804-035525.jpg (= 9002's 8)
  - 20260804-035525 dig2: csv label 5, file now work\labeled\c\8_main_dig2_20260804-035525.jpg (= 9002's 8)
  - 20260804-035525 dig3: csv label 7, file now work\labeled\c\8_main_dig3_20260804-035525.jpg (= 9002's 8)

## Skip inference (dig6)

108 anchor skips of +2; host side by hidden digit: 1:lower=1, 3:upper=2, 5:contradicts-prior=1, 5:unresolved=3, 5:upper=48, 7:lower=3, 7:upper=2, 8:upper=13, 9:contradicts-prior=3, 9:lower=31, 9:unresolved=1

605 non-holdout dig6 crops get an inferred proposal; 578 of them differ from the model's label (these feed the err queues).

## Projected corpus per cell

Existing corpus = `joes-samples/*.jpg` top level (1288 crops), light bucket by the old hour rule. Screen typing for the test-share cap: {'from_eval_9002_corpus': 1103, 'assumed_test': 111, 'from_frame_labels': 18, 'unknown_nontest': 56}.

Expected accepted counts, assuming 1/1.15 of queued crops survive review. Columns are cumulative: existing corpus -> +night_err -> +day_err -> +night_fill -> +day_fill. `t` = existing corpus crops from test screens (classes 0/8).

### flash

| pos | class | corpus (t) | +q2 | +q3 | +q4 | +q5 | queued |
|---|---|---|---|---|---|---|---|
| dig2 | 0 | 23 (23) | 27 | 27 | 80 | 80 | 66 |
| dig2 | 5 | 44 | 44 | 44 | 81 | 81 | 42 |
| dig2 | 8 | 20 (20) | 20 | 20 | 80 | 80 | 69 |
| dig2 | N | 16 | 20 | 20 | 40 | 40 | 28 |
| dig3 | 0 | 0 (0) | 52 | 52 | 80 | 80 | 92 |
| dig3 | 7 | 64 | 64 | 64 | 64 | 64 | 0 |
| dig3 | 8 | 5 (5) | 5 | 5 | 81 | 81 | 87 |
| dig3 | 9 | 0 | 52 | 52 | 80 | 80 | 92 |
| dig3 | N | 10 | 10 | 10 | 40 | 40 | 35 |
| dig4 | 0 | 3 (3) | 5 | 5 | 80 | 80 | 89 |
| dig4 | 1 | 0 | 0 | 0 | 80 | 80 | 92 |
| dig4 | 2 | 0 | 0 | 0 | 80 | 80 | 92 |
| dig4 | 3 | 0 | 0 | 0 | 80 | 80 | 92 |
| dig4 | 4 | 1 | 1 | 1 | 80 | 80 | 91 |
| dig4 | 5 | 27 | 27 | 27 | 80 | 80 | 61 |
| dig4 | 6 | 50 | 50 | 50 | 80 | 80 | 35 |
| dig4 | 7 | 0 | 6 | 6 | 80 | 80 | 92 |
| dig4 | 8 | 18 (18) | 19 | 19 | 81 | 81 | 72 |
| dig4 | 9 | 0 | 0 | 0 | 80 | 80 | 92 |
| dig4 | N | 14 | 14 | 14 | 40 | 40 | 30 |
| dig5 | 0 | 49 (36) | 53 | 53 | 80 | 80 | 36 |
| dig5 | 1 | 38 | 41 | 41 | 81 | 81 | 49 |
| dig5 | 2 | 9 | 16 | 16 | 80 | 80 | 82 |
| dig5 | 3 | 29 | 81 | 81 | 81 | 81 | 60 |
| dig5 | 4 | 45 | 55 | 55 | 81 | 81 | 41 |
| dig5 | 5 | 15 | 16 | 16 | 80 | 80 | 75 |
| dig5 | 6 | 1 | 18 | 18 | 80 | 80 | 91 |
| dig5 | 7 | 19 | 71 | 71 | 81 | 81 | 71 |
| dig5 | 8 | 37 (35) | 89 | 89 | 89 | 89 | 60 |
| dig5 | 9 | 0 | 52 | 52 | 80 | 80 | 92 |
| dig6 | 0 | 40 (40) | 92 | 92 | 92 | 92 | 60 |
| dig6 | 1 | 6 | 58 | 58 | 81 | 81 | 86 |
| dig6 | 2 | 1 | 31 | 31 | 80 | 80 | 91 |
| dig6 | 3 | 0 | 52 | 52 | 80 | 80 | 92 |
| dig6 | 4 | 23 | 75 | 75 | 80 | 80 | 66 |
| dig6 | 5 | 0 | 52 | 52 | 80 | 80 | 92 |
| dig6 | 6 | 1 | 53 | 53 | 80 | 80 | 91 |
| dig6 | 7 | 10 | 62 | 62 | 80 | 80 | 81 |
| dig6 | 8 | 30 (30) | 82 | 82 | 82 | 82 | 60 |
| dig6 | 9 | 39 | 91 | 91 | 91 | 91 | 60 |

### transition

| pos | class | corpus (t) | +q2 | +q3 | +q4 | +q5 | queued |
|---|---|---|---|---|---|---|---|
| dig2 | 0 | 2 (2) | 5 | 5 | 40 | 40 | 44 |
| dig2 | 5 | 22 | 22 | 22 | 40 | 40 | 21 |
| dig2 | 8 | 2 (2) | 2 | 2 | 40 | 40 | 44 |
| dig2 | N | 7 | 9 | 9 | 40 | 40 | 38 |
| dig3 | 0 | 2 (2) | 37 | 37 | 40 | 40 | 44 |
| dig3 | 7 | 25 | 25 | 25 | 25 | 25 | 0 |
| dig3 | 8 | 1 (1) | 1 | 1 | 40 | 40 | 45 |
| dig3 | 9 | 0 | 52 | 52 | 52 | 52 | 60 |
| dig3 | N | 6 | 6 | 6 | 41 | 41 | 40 |
| dig4 | 0 | 5 (5) | 7 | 7 | 41 | 41 | 41 |
| dig4 | 1 | 0 | 2 | 2 | 40 | 40 | 46 |
| dig4 | 2 | 0 | 0 | 0 | 40 | 40 | 46 |
| dig4 | 3 | 0 | 0 | 0 | 40 | 40 | 46 |
| dig4 | 4 | 5 | 5 | 5 | 41 | 41 | 41 |
| dig4 | 5 | 6 | 6 | 6 | 41 | 41 | 40 |
| dig4 | 6 | 22 | 22 | 22 | 40 | 40 | 21 |
| dig4 | 7 | 0 | 0 | 0 | 40 | 40 | 46 |
| dig4 | 8 | 3 (3) | 5 | 5 | 40 | 40 | 43 |
| dig4 | 9 | 0 | 0 | 0 | 39 | 39 | 45 |
| dig4 | N | 4 | 4 | 4 | 41 | 41 | 42 |
| dig5 | 0 | 6 (3) | 6 | 6 | 41 | 41 | 40 |
| dig5 | 1 | 0 | 0 | 0 | 40 | 40 | 46 |
| dig5 | 2 | 8 | 8 | 8 | 40 | 40 | 37 |
| dig5 | 3 | 6 | 23 | 23 | 41 | 41 | 40 |
| dig5 | 4 | 7 | 7 | 7 | 40 | 40 | 38 |
| dig5 | 5 | 4 | 4 | 4 | 41 | 41 | 42 |
| dig5 | 6 | 5 | 5 | 5 | 41 | 41 | 41 |
| dig5 | 7 | 6 | 6 | 6 | 41 | 41 | 40 |
| dig5 | 8 | 6 (2) | 42 | 42 | 42 | 42 | 41 |
| dig5 | 9 | 0 | 16 | 16 | 40 | 40 | 46 |
| dig6 | 0 | 4 (4) | 7 | 7 | 41 | 41 | 42 |
| dig6 | 1 | 2 | 2 | 2 | 40 | 40 | 44 |
| dig6 | 2 | 5 | 5 | 5 | 41 | 41 | 41 |
| dig6 | 3 | 5 | 5 | 5 | 41 | 41 | 41 |
| dig6 | 4 | 6 | 11 | 11 | 41 | 41 | 40 |
| dig6 | 5 | 5 | 42 | 42 | 42 | 42 | 43 |
| dig6 | 6 | 7 | 9 | 9 | 40 | 40 | 38 |
| dig6 | 7 | 11 | 11 | 11 | 41 | 41 | 34 |
| dig6 | 8 | 6 (4) | 11 | 11 | 41 | 41 | 40 |
| dig6 | 9 | 11 | 34 | 34 | 41 | 41 | 34 |

### day

| pos | class | corpus (t) | +q2 | +q3 | +q4 | +q5 | queued |
|---|---|---|---|---|---|---|---|
| dig2 | 0 | 5 (5) | 5 | 7 | 7 | 61 | 64 |
| dig2 | 5 | 18 | 18 | 18 | 18 | 61 | 49 |
| dig2 | 8 | 6 (6) | 6 | 6 | 6 | 61 | 63 |
| dig2 | N | 8 | 8 | 11 | 11 | 40 | 37 |
| dig3 | 0 | 4 (4) | 4 | 7 | 7 | 61 | 65 |
| dig3 | 7 | 23 | 23 | 23 | 23 | 53 | 35 |
| dig3 | 8 | 6 (6) | 6 | 7 | 7 | 61 | 63 |
| dig3 | 9 | 0 | 0 | 2 | 2 | 60 | 69 |
| dig3 | N | 8 | 8 | 8 | 8 | 40 | 37 |
| dig4 | 0 | 4 (4) | 4 | 4 | 4 | 61 | 65 |
| dig4 | 1 | 0 | 0 | 0 | 0 | 60 | 69 |
| dig4 | 2 | 0 | 0 | 0 | 0 | 60 | 69 |
| dig4 | 3 | 0 | 0 | 0 | 0 | 60 | 69 |
| dig4 | 4 | 20 | 20 | 20 | 20 | 60 | 46 |
| dig4 | 5 | 5 | 5 | 5 | 5 | 61 | 64 |
| dig4 | 6 | 5 | 5 | 5 | 5 | 61 | 64 |
| dig4 | 7 | 0 | 0 | 0 | 0 | 60 | 69 |
| dig4 | 8 | 9 (9) | 9 | 9 | 9 | 60 | 59 |
| dig4 | 9 | 0 | 0 | 1 | 1 | 60 | 69 |
| dig4 | N | 7 | 7 | 7 | 7 | 40 | 38 |
| dig5 | 0 | 28 (22) | 28 | 29 | 29 | 60 | 37 |
| dig5 | 1 | 7 | 7 | 7 | 7 | 60 | 61 |
| dig5 | 2 | 9 | 9 | 9 | 9 | 60 | 59 |
| dig5 | 3 | 8 | 8 | 22 | 22 | 60 | 60 |
| dig5 | 4 | 24 | 24 | 24 | 24 | 61 | 42 |
| dig5 | 5 | 4 | 4 | 4 | 4 | 61 | 65 |
| dig5 | 6 | 7 | 7 | 7 | 7 | 60 | 61 |
| dig5 | 7 | 4 | 4 | 4 | 4 | 61 | 65 |
| dig5 | 8 | 8 (8) | 8 | 16 | 16 | 60 | 60 |
| dig5 | 9 | 5 | 5 | 5 | 5 | 61 | 64 |
| dig6 | 0 | 12 (4) | 12 | 12 | 12 | 61 | 56 |
| dig6 | 1 | 12 | 12 | 15 | 15 | 61 | 56 |
| dig6 | 2 | 10 | 10 | 10 | 10 | 60 | 58 |
| dig6 | 3 | 15 | 15 | 15 | 15 | 60 | 52 |
| dig6 | 4 | 14 | 14 | 15 | 15 | 60 | 53 |
| dig6 | 5 | 11 | 11 | 46 | 46 | 61 | 57 |
| dig6 | 6 | 7 | 7 | 10 | 10 | 60 | 61 |
| dig6 | 7 | 6 | 6 | 15 | 15 | 61 | 63 |
| dig6 | 8 | 19 (18) | 19 | 29 | 29 | 61 | 48 |
| dig6 | 9 | 41 | 41 | 76 | 76 | 76 | 40 |

## Supply-limited night cells (< 20 after all queues)

- none

## Fill cells short of their queue size (supply or cap bound)

- dig4 class 9 transition: wanted 46, queued 45
- dig3 class 7 flash: wanted 19, queued 0
- dig3 class 7 transition: wanted 18, queued 0
- dig3 class 7 day: wanted 43, queued 35

## Rules applied

- night_err quota 60/cell, day_err 40/cell; dash dig5/6 <= 15/err cell; sampled round-robin across days x contrast quartiles.
- night_fill targets flash 80, transition 40 (N: 40), counting corpus + 0.87 x queued; queue size = shortfall x 1.15.
- day_fill target = min(60, projected flash) per (pos, class) (N: min(40, .)).
- test-screen crops <= 30% of a class-0/8 cell's target (incl. corpus test crops), shared across err+fill queues. Exempt (test-only in b3): dig2 class 0, dig2 class 8, dig3 class 0.
- edge-drift crops (roi_drift flag_reason contains `edge`) carry preflag `drift` and are taken only after clean crops; plain shift flags ignored.
- holdout-day crops appear only in holdout.csv; no `work/validation_labeled/` crop appears anywhere; all queues parse with `grid_review.load_queues`.
