# Handoff: electric meter LCD — batch 2 corpus and the 9002 retrain

Written 2026-08-07, **updated 2026-08-08** (§5e retracted, §5f added, §7 reordered).
Supersedes the 2026-07-31 handoff (which covered the 9001 cycle);
prior context is `HANDOFF-ELECTRIC-LCD.md` (original problem statement — physical
remediation is exhausted, glare/UV-haze are permanent input properties). Gas-meter
precedent: `C:\Users\josep\source\repos\AIOTED-analog\HANDOFF.md`.

---

## 1. State as of this writing

- **Deployed:** `models/dig-class11_9001_s2.tflite` (float, 349 KB), flashed
  2026-07-31, trained on 228 daylight user crops (×4) + 1290 upstream. Near-perfect
  dawn-to-dusk; weakest in flash-only conditions, which is what this cycle addresses.
- **New capture:** 9,125 raw crops / 1,825 frames, 2026-08-02 → 08-07, in repo
  `AIOTED-digital-rawdigits` (`<YYYYmmdd>/<HH>/`). ~38 kWh/day; reading spanned
  57500 → 57692.
- **Reviewed:** 2,788 crops queued → **2,185 human-verified** in `work/labeled/`
  (subfolders `a`–`e` mirror the review tiers). The rest were dropped as illegible
  at 20×32 or for ROI drift (glyph clipped / neighbour digit intruding).
- **Corpus:** `joes-samples/` top level = **1,288 crops / 755 frames** (mirrored into
  `04_joe_lcd_20x32/`). Started as 1,143 from `tools/build_corpus.py`; Joseph culled and
  restored by hand, then 178 derived-label night crops were added (§5d).
  **1,139 are human-reviewed; 149 carry derived labels only** — the split is
  reproducible by diffing against `work/labeled` + `joes-samples/batch 1`.
- **Holdout:** `work/validation_labeled/` — 240 crops / 50 complete frames, human
  verified, never trained. Drawn *before* any training selection.
- **DEPLOYED:** `models/dig-class11_9002_s2.tflite` (float, 349 KB), loaded 2026-08-07.
  Holdout 97.92% vs 9001's 91.67%. Gate results in §5a.
- **Also built, not deployed:** 9003–9007 (§5c, §5d). 9006 is the strongest on
  nighttime dig6, and as of 2026-08-08 that number is trustworthy — see §5e.
- **⚠ Open:** Joseph culled 71 of the 149 derived crops, but the culls are staged in
  `work/derived_only_review/` only — **`joes-samples/` is untouched**. Re-examine those
  culls in light of §5e's retraction; several were probably culled for the reflection.
- **Mineable pool identified (2026-08-08):** 1,271 human-reviewed crops that never
  entered the corpus, carrying 52 known 9002 failures — no capture needed. §5f.

Stale `work/cv_oof.csv` and `work/cv_oof_fold5.csv` from the 9001 cycle were moved to
`work/archive_9001/` — the pooled report globs `cv_oof_fold*.csv` and would have mixed
the old corpus's fold 5 into 9002's numbers.

## 2. What batch 2 taught us (read before changing anything)

**The display cycles four screens, not three.** Only ~40% of frames are readings:

| screen | what it is | frames (raw) |
|---|---|---|
| `5 7 d d d` | the meter reading | 721 |
| `00000` | test screen | 374 |
| `88888` | test screen; also lights the `.` and `°` annunciators | 280 |
| `- - - 0 9` | dash screen — dig2/3/4 blank, **dig5/dig6 are not meter digits** | 218 |

The dash screen's trailing digits are **constant `0 9`** (confirmed: 48 reviewed crops
came back 25 zeros / 23 nines, nothing else). `prepare_joe_data.py` types it `blank09`.
AIOTE's own rate limits filter this screen out, so it never reaches Home Assistant.

**Ground truth is derivable from monotonicity.** `reading = 57000 + 100·dig4 +
10·dig5 + dig6`, with dig2≡5 and dig3≡7. `tools/select_review_set.py` fits a monotone
step function through the model's own readings and uses it to corroborate screen type
and pin each digit — non-circular for dig2/dig3, because they are confirmed by
dig4/5/6 landing where a fit built from *other* frames predicts.

**But the derivation is a good triage and a bad labeller.** Measured against human
review:

| tier | n | model correct | derivation correct |
|---|---|---|---|
| A (model vs fit conflict) | 63 | **47** | 13 |
| B (fit can't pin dig6) | 60 | **36** | 4 |
| E (both agreed) | 376 | 357 | **357** |

Bulk agreement is ~95%; its *conflict* calls were mostly false alarms, because tiers
A/B sit exactly where the fit is weakest (reading screens are only 40% of frames, so
anchors are 10–20 min apart and dig6 moves every ~9 captures). **Use it to decide what
to look at. Do not adopt its labels on conflicts unreviewed.**

## 3. Non-negotiable rules

Carried forward from the 9001 cycle:

- **Labeling:** `<label>_main_dig<pos>_<YYYYmmdd-HHMMSS>.jpg`, label ∈ `0`–`9` or `N`.
  **Never `10_`** — upstream writes it for the NaN class (1,018 files in batch 2) but
  the notebook reads only the first character and `tools/dig_data.py` hard-errors on it.
  All repo tooling normalises `10` → `N` on load.
- **Never run notebooks 01/02/03.** nb01's first cell deletes the training dir; nb02's
  dedupe collapses the near-duplicates that are the point; nb03 exports a stale
  SavedModel. Everything lives in `tools/`.
- **ROIs are frozen** (94×202). Retuning invalidates the corpus and means full recapture.
- **Always keep a fresh holdout**, drawn before selection, never trained.
- **Background shells die at ~60 min.** One fold / one diagnostic per command.
  `--only-fold K` exists for this; per-fold results persist immediately.
- **Ship the float build.**

New this cycle:

- **Decorrelate light from class.** Class balance alone is not enough. Raw batch 2 had
  `8` at 88% flash and `2` at 24% — feed that in and the network learns the glare
  signature instead of the segments. `build_corpus.py --flash-ratio` caps each class's
  flash share; 1.0 collapsed the cross-class spread from **24–88% to 24–50%** and the
  corpus from 73% to 48% flash. Any future batch gets the same treatment.
- **Select per-crop, not per-frame, for training.** Five crops of one frame share
  exposure and glare, so whole frames give far less diversity per image — and each one
  force-feeds a `5`@dig2 and a `7`@dig3 into cells you meant to cap. **Holdout is the
  exception**: complete frames there, so per-frame accuracy is measurable.
- **`USER_WEIGHT` is now a flag.** `×4` was calibrated for 228 images against 1290
  upstream. At 1,143 verified and genuinely diverse images, use **`--user-weight 1`**
  (near parity); replicating real diversity only reintroduces class skew. The module
  default is still 4 — **pass the flag explicitly**.
- **Derived truth ≠ legible crop.** The fit tells you what the *meter* read, never
  whether the *crop* carries signal. A pitch-black night crop gets a "correct" label
  that is untrainable noise. `label_frames.py` shows the 20×32 model view precisely so
  this call is made against what the network sees.

## 4. The toolchain (all in `tools/`, run from repo root)

| script | purpose |
|---|---|
| `dig_data.py` | shared loader / label parser / screen typing / contrast |
| `select_review_set.py` | **new** — `--derive` fits the monotone reading series and reports per-crop status; `--emit` writes the holdout + tiered review queue |
| `label_frames.py` | **new** — tkinter keyboard labeller; frame mode walks dig2→dig6, one keystroke per position; shows ROI *and* 20×32 model view; resumable journal |
| `build_corpus.py` | **new** — assembles `joes-samples/` top level from `work/labeled` + batch 1, applying class caps and the flash-ratio decorrelation |
| `prepare_joe_data.py` | `--audit` (per-frame table + contact sheets), `--build [--force]` (`joes-samples/*.jpg` → `04_joe_lcd_20x32/`, 20×32 NEAREST). **Globs top level only** — subdirectories are silently ignored |
| `eval_dig_model.py` | the one evaluator: `--model --data --compare-to --roi-shift --csv --resize --exclude` |
| `augment_lcd.py` | photometric augmenter (`AugmentConfig`, glare lobe, veiling haze) |
| `augment_preview.py` | `--stats`, `--sheet`, `--probe` |
| `dig_model.py` | `build_dig_class11()` — 88,023 params; do not change |
| `train_dig_class11.py` | `--cv K [--only-fold K]`, `--diagnostic`, `--final --epochs E`, `--export-only`, `--user-weight` |
| `check_tflite_compat.py` | pre-flash op-set / AllocateTensors gate |

## 5. Where the 9002 cycle stands

Done:

```bash
python tools/build_corpus.py --flash-ratio 1.0 --apply     # -> joes-samples/ (1143)
python tools/prepare_joe_data.py --audit                   # 647 frames, sheets in work/
python tools/prepare_joe_data.py --build --force           # -> 04_joe_lcd_20x32/
python tools/augment_preview.py --stats                    # all three gates pass
```

Augmenter needed **no retune**: real p5 29.8 vs augmented 24.3, ratio 0.78–0.88 across
quantiles, 10.7% of augmented below real p5. It is now only ~15–20% harsher than
reality, because reality got harder when real night data entered the corpus. Only dial
`veil_k` down if `1→7` reappears in the upstream regression (§2.1 of the old handoff).

Remaining:

```bash
python tools/train_dig_class11.py --cv 5 --only-fold 1 --user-weight 1 --oof-csv work/cv_oof_fold1.csv
# ... folds 2..5 likewise; E* = median of best epochs, rounded UP to nearest 25
python tools/train_dig_class11.py --diagnostic lopo-dig3 --user-weight 1
python tools/train_dig_class11.py --final --epochs <E*> --user-weight 1 --version 9002
```

Then the gates — all must pass before flashing:

```bash
python tools/check_tflite_compat.py --model models/dig-class11_9002_s2.tflite --reference dig-class11_2000_s2.tflite
python tools/eval_dig_model.py --model models/dig-class11_9002_s2.tflite --data work/validation_labeled --compare-to models/dig-class11_9001_s2.tflite --csv work/eval_9002_holdout.csv
python tools/eval_dig_model.py --model models/dig-class11_9002_s2.tflite --data 03_data_resize_all-use_for_training --csv work/eval_9002_upstream.csv
python tools/eval_dig_model.py --model models/dig-class11_9002_s2.tflite --data joes-samples --compare-to models/dig-class11_9001_s2.tflite --roi-shift --csv work/eval_9002.csv
```

Criteria: holdout ≥ current deployed; `1→7` stays 0 on user data; ROI-shift agreement
not worse than the deployed model *measured on the same corpus*; **zero regressions in
the compare-to list** — a previously-correct image going wrong is a red flag even if
aggregates rise.

**The 2-point upstream-regression budget is retired** (Joseph's call, 2026-08-07, and
the evidence supports it). Report upstream for information; do not gate on it. 9006
traded 0.15 pts of upstream for +6 pts of nighttime dig6, which was plainly correct.
See §5d for the ablation showing upstream is no longer load-bearing.

Careful with ROI-shift comparisons: 9001's "1.00 on dig3/class-7" was measured on the
old 228-image daylight corpus. Re-run the *old* model on the *current* corpus before
calling anything a regression — done that way, 9002 beat 9001 on every stratum.

**Check the light confound explicitly.** Evaluate per class *stratified by bucket*. If
9002 learned glare-as-feature, night accuracy will hold on the night-heavy classes
(`3 5 6 7 8`) and collapse on the day-heavy ones (`2 9`). That asymmetry is the
signature; the aggregate will hide it.

## 5a. 9002 results (2026-08-07)

CV best epochs 103 / 243 / 277 / 146 / 70 → median 146 → **E\* = 150**. Pooled
out-of-fold **1133/1143 = 99.13%** (leakage-inflated; an upper bound, not a verdict).

| gate | 9001 | 9002 | verdict |
|---|---|---|---|
| tflite op set / AllocateTensors | — | identical to reference | **PASS** |
| holdout (240 crops, leak-free) | 91.67% | **97.92%** | **PASS** |
| upstream regression | 98.37% | 97.75% (−0.62, budget 2) | **PASS** |
| ROI-shift ALL s=0.03/0.06 | 0.915 / 0.857 | **0.993 / 0.958** | **PASS** |
| ROI-shift dig3 | 0.927 / 0.881 | **0.994 / 0.989** | **PASS** |
| zero broken vs 9001 | — | 8 broken / 157 fixed | see below |

Holdout strata: low-contrast tercile **81.25% → 97.50%**, dig5 82% → 100%,
dig6 80% → 95%. **`1→7` and `7→1` are zero** in both the OOF pool and the holdout —
the failure mode that motivated the whole project is gone.

Light-confound check clean: day-heavy classes read at night as well as night-heavy
ones (class `2` flash 10/10, class `9` flash 41/41). Class `2`'s n is small, so this
is absence of evidence rather than a cleared confound.

**The 8 broken files are `0↔8`, and only 2 are operationally relevant.** Reconstructing
full frames: 3 are zeros-screen crops (`dig3=0`, other positions dropped), 2 are
`88888`, 1 is ambiguous, and **2 are real readings** — `7→0` at dig3 (conf 0.35) and
`7→3` at dig5 (conf **0.997**, also wrong out-of-fold, so consistently wrong).

## 5b. Known root cause of the residual `0↔8` confusion

**Classes `0` and `8` in the corpus are built almost entirely from test screens.** From
the OOF CSV's screen column:

```
class 8:  test8=126   kwh=10     <- 93% of the model's 8 exposure is the 88888 screen
class 0:  zeros=106  blank09=18  kwh=12
class 9:  kwh=78     blank09=20  <- healthy, and class 9 has no such problem
```

The model's `8` is largely a *test-screen* 8 detector, which is exactly the risk of the
`88888` screen also lighting the `.` and `°` annunciators. `build_corpus.py` caps by
class and by flash-ratio but **has no screen-type cap**, so test-screen crops flooded
both classes — the "~45 test screens, hard cap" from the original plan was specified
and never implemented.

Note this is a *composition* problem, not a noise problem: pruning obscured nighttime
0/8/9 samples would shrink an already-tiny real-reading population (n=10 and n=12)
without touching the cause. The fix is a per-screen cap in `build_corpus.py` so real
reading 0s/8s are not drowned; `work/labeled` holds substantially more of them than the
corpus selected.

**Caveat added later:** 9001 had the *same* test-screen bias (~2 kWh-8s of 42) and
never developed `0→8`, so composition alone does not explain 9002's error. The washed-out
middle-segment explanation fits better — see §5c.

## 5c. Model bake-off (2026-08-07)

Four models, identical recipe (150 epochs, `--user-weight 1`), differing only in corpus.
All evaluated on the same 240-crop leak-free holdout.

| | 9001 | 9002 | 9003 | 9004 |
|---|---|---|---|---|
| corpus | 228 | 1,143 | 1,090 | 1,110 |
| class 0 / 3 / 8 | — | 136/58/136 | 122/43/117 | 122/**63**/117 |
| **holdout** | 91.67% | **97.92%** | 96.25% | 97.50% |
| upstream | 98.37% | 97.75% | 98.60% | **98.68%** |
| low-contrast tercile | 81.25% | **97.50%** | 91.25% | 93.75% |
| ROI-shift ALL s=0.06 | 0.857 | 0.958 | — | **0.978** |
| repeating error axes (≥2) | 6 | 1 (`0→8`) | 2 (`3→7`, `0→4`) | **none** |

**The load-bearing lesson: do not cull hard-but-correctly-labeled samples.** 9003 removed
15 class-3 crops whose washed-out bottom segments "looked too much like 7s". Those were
the samples defining the 3/7 boundary under haze — removing them produced `3→7` ×3 at
**high confidence** (0.81–0.91), all on the darkest holdout frames (contrast 27.5–51.8).
Restoring them in 9004 eliminated the axis completely (class 3 back to 7/7). The
observation was correct; the training decision inverted it.

9003 also showed the mirror effect: a cleaner corpus scored *better* on clean upstream
data (98.60%) and *worse* under haze (low-contrast 91.25% vs 9002's 97.50%).

9002 vs 9004 on the holdout is 2 fixed / 3 broken — one image, statistically
indistinguishable (McNemar p≈0.17 for the 9002/9003 comparison; 9002/9004 is weaker
still). The tiebreakers with real resolving power are upstream (+0.93 pts = 12 images to
9004) and error structure (9004 has no repeating axis).

Five class-3 crops recovered from a corrupted SD card carry synthetic names —
`3_main_dig6_19700101-1200NN.jpg`. **The 1970 date is a deliberate sentinel** marking
provenance-lost data; dig6 and daylight were supplied by Joseph, not derived. Content
hashing confirmed no duplicates and no holdout leak. ~2.2k more recovered crops remain
unlabelled; the classes worth mining are **`2`** (42 crops, 10 at night) and `9`.

## 5d. Derived labels make illegible crops trainable (9005 / 9006 / 9007)

**The technique.** A crop's label does not require anyone to read it. On a test screen
the other four digits pin it (`00000`→dig6 is 0, `88888`→8, `- - - 0 9`→dig5 is 0 and
dig6 is 9); on a reading screen the monotone fit pins it. So crops dropped as illegible
during review can still enter training with correct labels. This is what broke the
nighttime dig6 wall.

The gap it closed: `build_corpus.py` reads only `work/labeled`, so crops never sent to
human review never entered the pool. 268 nighttime dig6 crops had certain labels and
**exactly 2 were in the corpus**, while the dig6-flash cells read `0:0  8:0  9:39`.
The model wasn't confusing shapes — with no mass in those cells it fell back on the
cell prior and answered `9`.

| model | corpus | holdout 240 | dig6-night (73) | washed dig6 (45) | upstream |
|---|---|---|---|---|---|
| 9002 | 1,143 | **235** | 68% | 13/45 (29%) | 97.75% |
| 9005 | 1,180 (+70 dig6 night) | 234 | 71% | 23/45 (51%) | 98.29% |
| 9006 | 1,288 (+108 dig5 night) | 232 | **77%** | **30/45 (67%)** | 97.60% |
| 9007 | 1,288, **no upstream** | **236** | 71% | 27/45 (60%) | 39.07% |

**Transfer is real and cross-position.** 9006 received *no* new dig6 data, only dig5 —
yet nighttime dig6 class 8 went **0/15 → 11/15** and dig6-night rose 71%→77%. The
network learned segment topology, not a per-position brightness operating point.

**A claim recorded here earlier was wrong.** After 9005 left class 8 at 0/15 I argued
nighttime dig6 `8` was *physically irrecoverable* — that a washed-out 8 and 0 are the
same image because the distinguishing middle segment is destroyed. 9006 refuted it. The
model needed washed-out 8 examples, and dig5 had them. Do not re-derive that argument.

**Upstream ablation (9007).** Dropping all 1,290 upstream images: holdout *improved*
(236/240, best of any model) and ROI-shift *improved* (0.984/0.959 vs 9006's
0.971/0.943), while dig6-night fell 6 pts. Every gap is inside noise and they point in
opposite directions across metrics. Conclusion: upstream is not load-bearing at 1,288
user images. Keep it (free, and it is the only hedge against conditions outside the
five-day capture window, which no holdout here can test) but never gate on it.

**Method warning.** The dig5 test set was built badly — drawn from "unused with a
certain label" rather than from the dropped-as-illegible pool. 9002 scored 28/30 on it,
so those crops were mostly legible and 9006's 100% there means little. The dig6 result
is the clean one because 9006 got zero dig6 data.

## 5e. ~~The dig6 night results are provisional~~ — RETRACTED 2026-08-08

**This section previously said the dig6 vertical smear was a physical obscurity on the
glass, present across the entire Aug 2–07 window, and that it invalidated all dig6-night
measurements. That was wrong.** Joseph investigated further on 2026-08-08: the smear is
a **daytime reflection**, and he has minimised it optically.

What follows from the correction:

- **Night dig6 measurements are clean.** A daytime reflection cannot contaminate them.
  9005/9006's dig6-night gains are real segment learning, not "learned to read around a
  mark". The `0`/`8` washout at night is genuine over-exposure of the middle segment.
- **The contradictory-pairs ceiling does not apply at night.** The earlier claim that a
  middle-obscured `8` and `0` are the same image with different labels — and therefore
  cap achievable accuracy regardless of method — was a consequence of the artifact
  theory and dies with it. Nighttime 0/8 is an attackable discrimination problem.
- **dig6 work is unblocked.** The previous instruction "do not resume corpus surgery on
  dig6 until there is a post-cleaning night capture" no longer holds. Proceed.
- **The residue moved to daylight.** Daytime dig6 crops in `joes-samples/` and in the
  240-crop holdout carry a feature the meter no longer produces, so they are mildly
  off-distribution. Benign direction (the input got cleaner, and daytime dig6 already
  reads 100%), but it does mean the holdout slightly understates present-day daytime
  performance and a fresh capture is still wanted to re-baseline.
- Several of the 71 staged culls in `work/derived_only_review/` were probably culled for
  the reflection. Re-examine before applying.

Still true regardless: the `1→7` fix, the class-3 boundary finding (§5c), the
light-decorrelation work, and everything at dig2–dig5 in daylight.

**Process note worth keeping.** The artifact theory was constructed from a real
observation (the crops genuinely do look alike) plus a wrong cause, and it produced a
confident, load-bearing "stop work here" instruction that survived a full handoff. It
was falsified by looking at the meter, not by any amount of analysis of the corpus. This
is the second time this project has recorded a confident claim that later inverted (the
first is in §5d). Prefer a physical check over a data argument when one is available.

## 5f. The unused-pool sweep (2026-08-08) — where the remaining errors actually are

Question asked: rather than capturing more data, how much is there to gain from
supplementing the corpus with crops the model *gets wrong*? Answer: a measurable amount,
concentrated in two places, and it needs a `build_corpus.py` change first.

Deployed 9002 was run over all 2,185 human-reviewed crops
(`work/eval_9002_labeledpool.csv`, `--resize nearest`):

| set | n | errors | acc |
|---|---|---|---|
| in corpus (trained) | 914 | 2 | 99.78% |
| **never trained** | **1,271** | **52** | **95.91%** |
| overlap with holdout | 0 | — | — |

No holdout contamination. The in-corpus 99.78% is memorisation and carries no signal.
**All 52 errors are night or transition crops; zero daytime errors in 1,271 images.**

**≥3 of the 52 are label errors, not model errors.** Frame `20260804-035525` reads
`8/8/8/8` at dig2/3/4/5 with confidence 1.000/0.986/1.000/0.978 but is labelled
`5/7/5/8` and typed `kwh` — an `88888` test screen stamped with the constant dig2≡5 /
dig3≡7 on autopilot. It supplies 3 of the 4 most confident errors in the whole pool.
**Sort disagreements by confidence to get a targeted label-audit list** (~12 above 0.85);
a high-confidence disagreement on a legible crop is usually a bad label. This is cheaper
and better-aimed than another full review pass.

Error rate against how well the corpus covers each (position, class, bucket) cell:

| corpus samples in cell | unused n | wrong | err rate |
|---|---|---|---|
| empty (0) | 9 | 5 | 55.6% |
| 1–9 | 147 | 12 | 8.2% |
| 10–29 | 456 | 7 | 1.5% |
| 30+ | 659 | 28 | 4.2% |

The remaining ~49 real errors split into two different problems:

- **Coverage (~17 errors).** The §5d pattern repeating. Standout, and **new — not
  previously recorded anywhere**: **dig6 class `1` at night — 6 corpus samples, 9 wrong
  of 16 available (56%), as `1→4` ×5 and `1→2` ×4, all on real reading screens.** The
  documented dig6 story has been entirely the 0/8/9 three-way; this is a separate axis,
  and a `1→4` in the ones place is a +3 kWh error that the rate limits cannot catch (§6).
  Highest operational value item currently known. Also empty: dig3 class `0` night (0
  corpus, 3/3 wrong), dig6 classes `5`/`3`/`6` night.
- **Difficulty (~24 errors).** Class `8` at night, dig5 (18) and dig6 (6), as `8→0` and
  `8→9` — the entire 30+ bin. Those cells hold 37 and 30 corpus samples and still fail
  ~13%, so this is not emptiness. With §5e retracted it is a genuine washout
  discrimination problem. Expect partial gains only: 9006 showed 0/8/9 mass
  redistributes rather than resolving.

**⚠ The blocker is the flash-ratio rule, not data supply.** Every one of the 52 is a
night crop. Class `8` is already 73% night in the corpus and the unused night supply is
460 crops — adding it takes class 8 to **92% night**, straight back past the 88% that
`--flash-ratio` exists to prevent (§3). Under the current per-(class, bucket) cap,
error mining and light-decorrelation are in direct conflict. **The per-(class × position
× bucket) cap in §8 is therefore the enabling change, not a backlog nicety** — it is what
lets an empty dig6-class-1-night cell be filled without re-correlating glare with class 8
globally.

Expected value, calibrated: dig6 class 1 is a near-certain win (§5d precedent for filling
a near-empty cell: 0/15 → 11/15). Class 8 at night is partial at best. Aggregate holdout
gain is capped around +1–2 pts and is **unmeasurable at n=240 — gate this work on
dig6-night, not on the holdout.**

Mining does not replace capture: it cannot produce evidence about a season, sun angle or
haze regime never sampled, and the holdout still carries the removed reflection (§5e).

## 6. Known gaps

- **dig6 after dark is the only failure mode the rate limits cannot catch.** An `8→9`
  or `1→4` in the ones place is a ±1–3 kWh error, small enough to pass `MaxRateValue`.
  Every other position's errors are caught. Judge models on dig6-night, not aggregate —
  9006's dig6-night is 77% while its aggregate holdout is 96.7%, and the aggregate hides
  it because daytime dig6 is 100%.
- **Confidence is not an exposed variable** in the firmware (accepts ≳0.5), so a
  confident wrong answer costs exactly as much as an unsure one. Do not weigh
  confidence in ship decisions.
- **Class `2` is thin** — 42 crops, 10 at night. Class `9` at dig6 became the weak one
  in 9006 (6/15, down from 12/15) as 0 and 8 improved; the three-way 0/8/9 confusion
  redistributes rather than resolving.
- **dig6 class `1` at night fails 56% of the time** and is not caught by anything —
  `1→4` / `1→2` on real readings, 6 corpus samples. Discovered 2026-08-08, §5f.
- **The holdout contains no `N`** — validation frames were drawn from corroborated
  reading + test screens, so dash screens (the `N` source) were excluded by
  construction. `N` regressions will not be caught by it.
- **Every crop in the project is from 2026-08-02→07.** One season, one sun angle, one
  haze regime. No holdout here can detect loss of generalisation outside that window.
- Light buckets are by capture hour: **06/19 transition, 20–05 flash-only, 07–18 day**.
- `work/culled.txt` records 40 hand-culled crops; `build_corpus.py --exclude` honours it.
  Culls made only in `joes-samples/` are lost on rebuild — record them there.

## 7. Next actions, highest value first

Reordered 2026-08-08 after §5e's retraction unblocked dig6 and §5f located the
remaining errors. The old #1 ("capture one clean night, everything is blocked on it")
is no longer blocking anything — it has moved to #5.

1. **Audit the ~12 high-confidence disagreements in `work/eval_9002_labeledpool.csv`**
   (§5f). Start with frame `20260804-035525`. Cheap, and bad labels at a decision
   boundary are the one input that reliably backfires (§5c).
2. **Implement the per-(class × position × bucket) cap in `build_corpus.py`** (§8). This
   is the gate on everything below it — without it, adding the night crops that fix
   dig6 re-correlates glare with class 8 and undoes the §3 decorrelation work.
3. **Fill the thin night cells wholesale** — 152 crops available in cells with <10
   corpus samples. **dig6 class `1` first.** Add whole cells, correct crops included;
   do not add only the misclassified ones (that shifts the prior and, per §2, the cheap
   oracles for "the model got this wrong" are unreliable here).
4. **Class `8` at night (dig5/dig6) as a separate experiment**, gated on dig6-night
   rather than the holdout. Partial gains expected; watch class `9` for the 9006-style
   redistribution.
5. Capture one clean night post-reflection-fix, re-baseline 9002/9006, and rebuild the
   dig6 holdout. Still worth doing — the current holdout carries the removed reflection
   in its daytime dig6 crops — but no longer a prerequisite.
6. Re-examine the 71 staged culls in `work/derived_only_review/`; several were likely
   culled for the reflection rather than for genuine illegibility.
7. Extend the derived-label technique (§5d) to dig2–dig4 if night gaps remain — it has
   not been tried there, and ~218 unused night dig5 zeros plus ~90 dig6 zeros remain.
8. Exposure experiment: dimmer flash + longer exposure. The night failure is
   *over*-exposure blowing out middle segments, not under-exposure. Now better
   motivated than it was — §5f confirms the residual hard axis is class `8` washout at
   night, and §5e no longer offers the artifact as an alternative explanation.
9. Joseph has ~2.2k unlabelled crops recovered from a corrupted SD card (filenames
   lost; he can infer dig5 vs dig6 from the brightness gradient, since dig6 sits
   farthest from the flash). Mine them for the cells §5f flags as empty — not the
   `2`/`9` guessed earlier.

### 7a. Overnight behaviour — weighed, deliberately deferred

Current overnight failure is **tolerable in practice**: readings go bursty (a gap of a
couple of hours then a +2 kWh jump), or a dig6 misread inflates `PreValue` and real
usage takes a few hours to catch up. Cost is hourly-level resolution, not daily totals,
and most consumption is daytime A/C. **Open risk: winter**, when daylight hours shrink
and the night share of captures grows. Re-assess then.

Options already considered — do not re-propose without new information:

- **Tightening `MaxRateValue` does not work.** The display is whole-kWh, so a genuine
  increment and a false `8→9` are both exactly +1 — the identical observation, separable
  by no threshold. Worse, the rate is computed per-minute since the last valid
  `PreValue`, so a real +1 over 4 min reads as 0.25 kWh/min and gets rejected, then
  0.125 at 8 min, and so on until the window grows enough to pass.
- **Relabel near-indiscernible dig6 `0/8/9` as `N`** so failures are abstentions rather
  than plausible wrong digits. Attractive because errors otherwise *ratchet* — downward
  misreads get filtered by `AllowNegativeRates`, upward ones survive and accumulate.
  Costs ~50% of night readings and risks `N` bleeding to other positions (the model has
  no positional input).
- **HA-side timing guard** — at ~38 kWh/day a real increment arrives roughly every 38
  min, so an increment landing far sooner is suspect. Stateful check AIOTE's single
  threshold cannot express.
- **Skip overnight captures** — crudest, most reliable; on a cumulative counter it costs
  resolution only.

## 8. Deferred backlog

- Synthetic 7-segment compositing (`HANDOFF-ELECTRIC-LCD.md` §4) — only if real
  coverage still proves short.
- Day-trained → night-tested transfer diagnostic. Largely superseded by §5d, which
  showed transfer works across positions.
- ~~Per-(class × position × bucket) cells in `build_corpus.py`.~~ **Promoted to next
  action #2 on 2026-08-08.** The current cap is per-(class, bucket), which is how dig6
  ended up with empty night cells for 0/3/5/8 while class 8 had 68 flash samples
  elsewhere. Structural fix for §5d's root cause, and per §5f it now gates the whole
  error-mining path.
- Cross-ROI wrong-screen rejection — all-five-digits-identical is the signal; per-digit
  confidence cannot do it (test-screen 8s read at 0.955). Firmware or HA-side work.

## 9. Environment

Python 3.12.2, TF 2.21.0 (CPU-only — no GPU on native Windows for TF ≥ 2.11),
Keras 3.15.1, Pillow 12.2.0, ai_edge_litert. ~2 s/epoch at batch 4 on the ~2,200-image
pool; a fold runs ~15 min, a 5-fold CV ~1–1.5 h chunked. Training runs locally — do not
move it to a sandbox (gas-meter handoff §9 for why).
