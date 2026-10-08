#!/usr/bin/env python3
"""Train / calibrate / export the 11-class digit readout model.

Three modes, all driven off the same data pool and the same fit recipe so the
number that comes out of calibration actually describes the model that ships:

``--cv 5``
    Grouped 5-fold cross-validation **over the 49 user-captured frames only**.
    The group key is the capture timestamp, so every digit position of one
    frame lands in the same fold -- otherwise dig2 of a frame in training would
    leak the glare and exposure of dig5 of the same frame in validation.
    Folds are laid out by sorting frames by screen type and dealing them
    round-robin, which keeps the 26 kwh / 10 test8 / 7 zeros / 6 blank09 frames
    roughly proportional across folds.

    The upstream corpus is *always* fully in training; only user frames are
    held out. User frames are replicated x4 **after** the split, so no replica
    of a held-out frame can reach the training set.

    Out-of-fold predictions over all 228 user images are pooled into one honest
    report -- this is the number to trust, not the final model's score on
    joes-samples (which it was trained on).

``--diagnostic lopo-dig3`` / ``--diagnostic loso-zeros``
    One fold-shaped run holding out an entire digit position / screen type.
    These probe glare-position overfit and screen generalisation. They are
    expected to look worse than the CV number and are diagnostics, not gates.

``--final --epochs E``
    Train on 100% of the pool (upstream x1 + user x4), no validation, no early
    stopping, for a fixed epoch budget, then export tflite.

Usage::

    python tools/train_dig_class11.py --cv 5
    python tools/train_dig_class11.py --diagnostic lopo-dig3
    python tools/train_dig_class11.py --final --epochs 275
    python tools/train_dig_class11.py --export-only

Training recipe (all opt-in; the defaults are the 9001-9020 recipe, bit for bit):

``--aug-profile {legacy,night_aware}``
    Photometric chain (``augment_lcd.AUG_PROFILES``). ``night_aware`` gates on the
    crop's mean luma: dark crops (night dig5/dig6) get no veil / glare / white
    balance and a narrow exposure/contrast jitter; bright crops get the legacy
    chain with roughly halved probabilities.
``--geom {legacy,gentle}``
    IDG geometry (``augment_lcd.GEOM_PROFILES``). ``gentle``: shift {-1,0,1} px,
    zoom 0.9-1.1, rotation 2 deg, no brightness_range.
``--lr-decay-frac F`` / ``--lr-end-frac R``
    0 (default) = Adadelta's constant lr. F > 0: hold the compiled lr for the
    first (1-F) of the run's epochs, then decay linearly to R x lr (default 0.05)
    at the last epoch. The run length is ``--max-epochs`` or ``--epochs``; in CV
    with early stopping, use a fixed budget (e.g. --epochs 125 --patience 999) or
    the decay never arrives.

Recipe "R1" = ``--aug-profile night_aware --geom gentle --lr-decay-frac 0.2`` on a
``prepare_joe_data.py --resize stb`` corpus.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

# See dig_model.py for why Keras is reached by attribute access on `tf`.
try:
    keras = tf.keras
    EarlyStopping = keras.callbacks.EarlyStopping
    LearningRateScheduler = keras.callbacks.LearningRateScheduler
    ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
except AttributeError:  # pragma: no cover - stand-alone Keras 3 install
    import keras  # type: ignore
    EarlyStopping = keras.callbacks.EarlyStopping
    LearningRateScheduler = keras.callbacks.LearningRateScheduler
    from keras.src.legacy.preprocessing.image import ImageDataGenerator  # type: ignore

from augment_lcd import AUG_PROFILES, GEOM_PROFILES  # noqa: E402
from augment_lcd import make_idg_preprocessing_fn  # noqa: E402
from dig_data import CLASS_NAMES, load_dirs  # noqa: E402
from dig_model import build_dig_class11  # noqa: E402
from eval_dig_model import report, write_csv  # noqa: E402

NCLS = len(CLASS_NAMES)

UPSTREAM_DIR = "03_data_resize_all-use_for_training"
USER_DIR = "04_joe_lcd_20x32"
# Replication factor for the user corpus. 4 was calibrated when the user set was
# 228 images against 1290 upstream and needed the weight to register at all. Pass
# --user-weight to override; at the ~1143-image corpus it is 1 (near parity), and
# replicating real diversity would only reintroduce class skew.
USER_WEIGHT = 4

SCREEN_ORDER = ["kwh", "test8", "zeros", "blank09"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_pools(upstream_dir, user_dir, resize="nearest"):
    """Load both corpora *unreplicated*. Replication happens after splitting."""
    x_up, y_up, m_up = load_dirs([upstream_dir], resize=resize)
    x_us, y_us, m_us = load_dirs([user_dir], resize=resize)
    print(f"upstream {upstream_dir}: {len(y_up)} images")
    print(f"user     {user_dir}: {len(y_us)} images "
          f"({len({m['frame'] for m in m_us})} frames)")
    return (x_up, y_up, m_up), (x_us, y_us, m_us)


def onehot(y):
    return np.eye(NCLS, dtype=np.float32)[np.asarray(y, dtype=np.int64)]


def assign_folds(user_meta, k=5, by="day"):
    """Assign frames to folds, never splitting a group across train/val.

    ``by="frame"`` (the 9001-9007 behaviour): sort frames by screen type and
    deal them round-robin. Grouping by frame keeps the five crops of one
    capture together, but temporally adjacent frames -- same hour, same glare,
    same haze -- still land in different folds, which leaks lighting.

    ``by="day"`` (default from batch 3): whole capture days go to one fold,
    greedily balanced by frame count. Removes that leak; needs >= k days.
    """
    frame_screen = {}
    for m in user_meta:
        frame_screen[m["frame"]] = m["screen"]
    frames = sorted(frame_screen,
                    key=lambda f: (SCREEN_ORDER.index(frame_screen[f])
                                   if frame_screen[f] in SCREEN_ORDER else 99, f))
    if by == "day":
        days = {}
        for f in frames:
            days.setdefault(f[:8], []).append(f)
        if len(days) < k:
            raise SystemExit(f"--fold-by day needs >= {k} capture days, have {len(days)}")
        load = [0] * k
        fold_of_frame = {}
        for day in sorted(days, key=lambda d: (-len(days[d]), d)):
            fold = min(range(k), key=lambda i: (load[i], i))
            load[fold] += len(days[day])
            for f in days[day]:
                fold_of_frame[f] = fold
    else:
        fold_of_frame = {f: i % k for i, f in enumerate(frames)}

    print(f"\nfold layout ({len(frames)} user frames, k={k}):")
    print(f"  {'fold':<6} " + "".join(f"{s:>9}" for s in SCREEN_ORDER) + f"{'frames':>9}")
    for fold in range(k):
        ff = [f for f in frames if fold_of_frame[f] == fold]
        counts = [sum(1 for f in ff if frame_screen[f] == s) for s in SCREEN_ORDER]
        print(f"  {fold:<6} " + "".join(f"{c:>9}" for c in counts) + f"{len(ff):>9}")
    return fold_of_frame


def make_datagen(seed, aug_profile="legacy", geom="legacy"):
    """The geometric IDG plus the LCD photometric chain.

    IDG runs its geometry first and then calls ``preprocessing_function``, which
    is the physically right order: haze and glare sit in front of the display,
    so they are applied to the already-registered glyph. The defaults are the
    notebook geometry + ``augment_lcd.DEFAULT``.
    """
    return ImageDataGenerator(
        **GEOM_PROFILES[geom],
        preprocessing_function=make_idg_preprocessing_fn(AUG_PROFILES[aug_profile], seed),
    )


def lr_schedule(base_lr, epochs, decay_frac, end_frac=0.05):
    """Hold ``base_lr`` for the first (1-F) of ``epochs``, then decay linearly so the
    LAST epoch runs at ``end_frac * base_lr``. Returns None when decay is off."""
    if not decay_frac or decay_frac <= 0:
        return None
    if not 0 < decay_frac <= 1:
        raise SystemExit("--lr-decay-frac must be in (0, 1]")
    start = min(epochs - 1, max(0, int(round(epochs * (1.0 - decay_frac)))))
    span = epochs - start  # decayed epochs; the last one lands on end_frac

    def fn(epoch, lr=None):
        if epoch < start:
            return float(base_lr)
        t = (epoch - start + 1) / span
        return float(base_lr * (1.0 - (1.0 - end_frac) * t))

    fn.start, fn.span = start, span  # type: ignore[attr-defined]
    return fn


def describe_recipe(args) -> str:
    """The training-log header: every knob that defines the recipe."""
    aug = AUG_PROFILES[args.aug_profile]
    lines = [
        "=" * 62,
        "RECIPE",
        f"  user dir     : {args.user}   upstream: {args.upstream}   "
        f"user-weight x{args.user_weight}",
        f"  aug profile  : {args.aug_profile}",
    ]
    if hasattr(aug, "dark"):
        lines += [f"    gate       : mean luma <= {aug.dark_luma_max:g} -> dark",
                  f"    dark       : {aug.dark}",
                  f"    bright     : {aug.bright}"]
    else:
        lines += [f"    config     : {aug}"]
    lines += [f"  geom         : {args.geom}  {GEOM_PROFILES[args.geom]}"]
    if args.lr_decay_frac and args.lr_decay_frac > 0:
        lines += [f"  lr           : compiled lr, linear decay over the last "
                  f"{args.lr_decay_frac:g} of the run's epochs to x{args.lr_end_frac:g}"]
    else:
        lines += ["  lr           : constant (compiled optimizer, no schedule)"]
    lines += [f"  seed {args.seed}  batch {args.batch}  epochs {args.epochs}"
              + (f"  max-epochs {args.max_epochs}" if args.max_epochs else ""),
              "=" * 62]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# one train run
# --------------------------------------------------------------------------


def fit_once(x_train, y_train, x_val, y_val, *, epochs, batch, patience, seed,
             tag="", aug_profile="legacy", geom="legacy", lr_decay_frac=0.0,
             lr_end_frac=0.05):
    """Fit a fresh model. ``x_val`` may be None for the final (no-holdout) run."""
    keras.utils.set_random_seed(seed)
    model = build_dig_class11()

    datagen = make_datagen(seed, aug_profile, geom)
    flow = datagen.flow(x_train.astype(np.float32), onehot(y_train),
                        batch_size=batch, seed=seed)

    callbacks = []
    sched = None
    if lr_decay_frac and lr_decay_frac > 0:
        base_lr = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
        sched = lr_schedule(base_lr, epochs, lr_decay_frac, lr_end_frac)
        callbacks.append(LearningRateScheduler(sched, verbose=0))
        print(f"[{tag}] lr {base_lr:g} for epochs 1-{sched.start}, then linear to "
              f"{sched(epochs - 1):g} at epoch {epochs}")
    validation = None
    if x_val is not None and len(x_val):
        validation = (x_val.astype(np.float32), onehot(y_val))
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True, verbose=1))

    t0 = time.time()
    hist = model.fit(flow, validation_data=validation, epochs=epochs,
                     verbose=2, callbacks=callbacks)
    dt = time.time() - t0

    ran = len(hist.history["loss"])
    best = None
    if validation is not None:
        best = int(np.argmin(hist.history["val_loss"])) + 1
        print(f"[{tag}] ran {ran} epochs in {dt:.0f}s ({dt / max(ran, 1):.2f} s/ep), "
              f"best epoch {best} (val_loss {min(hist.history['val_loss']):.5f}, "
              f"val_acc {hist.history['val_accuracy'][best - 1]:.4f})")
    else:
        print(f"[{tag}] ran {ran} epochs in {dt:.0f}s ({dt / max(ran, 1):.2f} s/ep), "
              f"final loss {hist.history['loss'][-1]:.5f}, "
              f"acc {hist.history['accuracy'][-1]:.4f}")
    return model, best, hist


def recipe_kwargs(args) -> dict:
    return dict(aug_profile=args.aug_profile, geom=args.geom,
                lr_decay_frac=args.lr_decay_frac, lr_end_frac=args.lr_end_frac)


def build_train_arrays(x_up, y_up, x_us, y_us, keep_mask, weight=None):
    """Upstream x1 + the kept user rows replicated x `weight`."""
    w = USER_WEIGHT if weight is None else weight
    xs = [x_up] + [x_us[keep_mask]] * w
    ys = [y_up] + [y_us[keep_mask]] * w
    return np.concatenate(xs), np.concatenate(ys)


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------


def run_cv(args, up, us):
    x_up, y_up, _ = up
    x_us, y_us, m_us = us

    fold_of_frame = assign_folds(m_us, k=args.cv, by=args.fold_by)
    fold_of_row = np.array([fold_of_frame[m["frame"]] for m in m_us])

    oof = np.zeros((len(y_us), NCLS), dtype=np.float32)
    best_epochs = []

    # Which folds this invocation runs. The assignment above is deterministic
    # (it depends only on the user corpus and k), so `--only-fold K` reproduces
    # exactly the split fold K would have got in a full run.
    folds = range(args.cv) if args.only_fold is None else [args.only_fold - 1]
    if args.only_fold is not None:
        if not 1 <= args.only_fold <= args.cv:
            raise SystemExit(f"--only-fold must be in 1..{args.cv}")
        print(f"\nrunning ONLY fold {args.only_fold} of {args.cv}")

    for fold in folds:
        val_mask = fold_of_row == fold
        train_mask = ~val_mask
        x_train, y_train = build_train_arrays(x_up, y_up, x_us, y_us, train_mask, args.user_weight)
        x_val, y_val = x_us[val_mask], y_us[val_mask]
        print(f"\n=== fold {fold + 1}/{args.cv}: train {len(y_train)} "
              f"({len(y_up)} upstream + {int(train_mask.sum())} user x{args.user_weight}), "
              f"val {len(y_val)} user images ===")

        model, best, _ = fit_once(x_train, y_train, x_val, y_val,
                                  epochs=args.max_epochs or args.epochs,
                                  batch=args.batch, patience=args.patience,
                                  seed=args.seed + fold, tag=f"fold{fold + 1}",
                                  **recipe_kwargs(args))
        best_epochs.append(best)
        probs = model.predict(x_val.astype(np.float32), verbose=0)
        oof[val_mask] = probs

        # Persist immediately. A five-fold run is over an hour of compute and
        # the pooled CSV is only written at the very end; one killed shell used
        # to throw away every completed fold with it.
        fold_meta = [m for m, keep in zip(m_us, val_mask) if keep]
        fold_csv = f"{os.path.splitext(args.oof_csv)[0]}_fold{fold + 1}.csv"
        write_csv(fold_csv, fold_meta, y_us[val_mask], probs.argmax(axis=1),
                  probs.max(axis=1), probs)
        print(f"[fold{fold + 1}] best epoch {best} persisted to {fold_csv}")
        keras.backend.clear_session()

    if args.only_fold is not None:
        print(f"\nfold {args.only_fold} best epoch: {best_epochs[0]}")
        print("(single-fold run: no pooled out-of-fold report)")
        return None

    pred = oof.argmax(axis=1)
    conf = oof.max(axis=1)

    report("OUT-OF-FOLD (grouped 5-fold CV over user frames)",
           x_us, y_us, m_us, pred, conf, oof)
    write_csv(args.oof_csv, m_us, y_us, pred, conf, oof)

    print("\n" + "=" * 62)
    print("per-fold best epoch: " + ", ".join(str(b) for b in best_epochs))
    median = float(np.median(best_epochs))
    e_star = int(np.ceil(median / 25.0) * 25)
    print(f"median best epoch = {median:g}  ->  E* (round up to 25) = {e_star}")
    print(f"\n  python tools/train_dig_class11.py --final --epochs {e_star}")
    print("=" * 62)
    return e_star


def run_diagnostic(args, up, us):
    x_up, y_up, _ = up
    x_us, y_us, m_us = us

    if args.diagnostic == "lopo-dig3":
        val_mask = np.array([m["position"] == 3 for m in m_us])
        what = "all user dig3 images (leave-one-position-out)"
    elif args.diagnostic == "loso-zeros":
        val_mask = np.array([m["screen"] == "zeros" for m in m_us])
        what = "all user zeros-screen images (leave-one-screen-out)"
    else:  # pragma: no cover - argparse restricts this
        raise SystemExit(f"unknown diagnostic {args.diagnostic}")

    train_mask = ~val_mask
    x_train, y_train = build_train_arrays(x_up, y_up, x_us, y_us, train_mask, args.user_weight)
    print(f"\n=== diagnostic {args.diagnostic}: hold out {what} ===")
    print(f"train {len(y_train)} ({len(y_up)} upstream + {int(train_mask.sum())} "
          f"user x{args.user_weight}), held out {int(val_mask.sum())} user images")

    model, best, _ = fit_once(x_train, y_train, x_us[val_mask], y_us[val_mask],
                              epochs=args.max_epochs or args.epochs,
                              batch=args.batch, patience=args.patience,
                              seed=args.seed, tag=args.diagnostic,
                              **recipe_kwargs(args))

    probs = model.predict(x_us[val_mask].astype(np.float32), verbose=0)
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    meta = [m for m, keep in zip(m_us, val_mask) if keep]
    report(f"DIAGNOSTIC {args.diagnostic} (held-out slice only, best epoch {best})",
           x_us[val_mask], y_us[val_mask], meta, pred, conf, probs)
    if args.csv:
        write_csv(args.csv, meta, y_us[val_mask], pred, conf, probs)


def run_final(args, up, us):
    x_up, y_up, _ = up
    x_us, y_us, _ = us
    keep = np.ones(len(y_us), dtype=bool)
    x_train, y_train = build_train_arrays(x_up, y_up, x_us, y_us, keep, args.user_weight)
    epochs = args.max_epochs or args.epochs
    print(f"\n=== FINAL: train on 100% of the pool, {len(y_train)} images "
          f"({len(y_up)} upstream + {len(y_us)} user x{args.user_weight}), "
          f"{epochs} epochs, no validation, no early stopping ===")

    model, _, _ = fit_once(x_train, y_train, None, None, epochs=epochs,
                           batch=args.batch, patience=args.patience,
                           seed=args.seed, tag="final", **recipe_kwargs(args))

    args.out.mkdir(parents=True, exist_ok=True)
    keras_path = args.out / f"{args.stem}.keras"
    model.save(str(keras_path))
    print(f"saved {keras_path}")
    export_tflite(model, args, x_up, x_us)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def representative_dataset(x_up, x_us, n=256, seed=0):
    """256 DISTINCT images, weighted toward the user capture domain.

    All 228 user images plus enough random upstream ones to reach ``n``, sampled
    without replacement. Feeding the same image repeatedly (or feeding random
    noise) gives the converter a calibration range that does not match what the
    device actually sees, and the int8 model degrades in ways the float one does
    not.
    """
    rng = np.random.default_rng(seed)
    if len(x_us) > n:
        # The user corpus outgrew the calibration budget (1,143+ vs 256): sample
        # it rather than feeding every image to the quantiser.
        x_us = x_us[rng.choice(len(x_us), size=n, replace=False)]
    take_up = max(0, n - len(x_us))
    idx = rng.choice(len(x_up), size=min(take_up, len(x_up)), replace=False)
    pool = np.concatenate([x_us, x_up[idx]]).astype(np.float32)
    print(f"  representative dataset: {len(pool)} distinct images "
          f"({len(x_us)} user + {len(idx)} upstream)")

    def gen():
        for i in range(len(pool)):
            yield [pool[i : i + 1]]

    return gen


def export_tflite(model, args, x_up, x_us) -> None:
    """Convert to tflite in a form AI-on-the-Edge-Device can actually load.

    ``TFLiteConverter.from_keras_model`` on Keras 3 leaves the batch dimension
    dynamic, so Flatten is emitted as a *computed* reshape - SHAPE +
    STRIDED_SLICE + PACK feeding RESHAPE. Those three ops are not registered in
    the firmware's TFLite-for-Microcontrollers op resolver, and the symptom is
    an unhelpful ``AllocateTensors() failed`` that looks like a corrupt file.

    Pinning the batch dimension to 1 makes every shape static, so Flatten
    collapses to a plain RESHAPE and the op set matches the shipped models:
    ADD, CONV_2D, FULLY_CONNECTED, MAX_POOL_2D, MUL, RESHAPE, SOFTMAX.
    """
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.stem

    # Pin the batch dimension by re-running the *same layer objects* on a
    # fixed-shape input. Two approaches that look equivalent and are not:
    #   - converting from a tf.function concrete function leaves the weights as
    #     VAR_HANDLE / READ_VARIABLE graph variables, which TFLM cannot handle;
    #   - clone_model() + set_weights() silently mis-assigns the BatchNorm
    #     moving statistics, producing a model that converts cleanly and
    #     predicts garbage.
    # Reusing the layer objects avoids both: there is no weight copy to get wrong.
    static_input = keras.Input(batch_shape=(1, 32, 20, 3))
    y = static_input
    for layer in model.layers[1:]:
        y = layer(y)
    static = keras.Model(static_input, y)

    # Fail loudly rather than shipping a silently broken model.
    probe = np.random.default_rng(0).uniform(0, 255, (1, 32, 20, 3)).astype(np.float32)
    delta = float(np.abs(model.predict(probe, verbose=0)
                         - static.predict(probe, verbose=0)).max())
    if delta > 1e-4:
        raise SystemExit(f"Static-shape rebuild changed predictions by {delta:.2e}")
    print(f"Static-shape rebuild verified (max delta {delta:.2e})")

    float_path = args.out / f"{stem}.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(static)
    float_path.write_bytes(converter.convert())

    rep = representative_dataset(x_up, x_us, n=args.rep_size, seed=args.seed)

    # Per-channel quantisation bumps CONV_2D to v5 and FULLY_CONNECTED to v12.
    # The firmware's op resolver accepts the v3 / v4 that the shipped models
    # use, so ask for per-tensor instead. `_q` disables it for the dense layers
    # only; `_qpt` disables it everywhere.
    q_path = args.out / f"{stem}_q.tflite"
    quant = tf.lite.TFLiteConverter.from_keras_model(static)
    quant.optimizations = [tf.lite.Optimize.DEFAULT]
    quant.representative_dataset = rep
    quant._experimental_disable_per_channel_quantization_for_dense_layers = True
    q_path.write_bytes(quant.convert())

    qpt_path = args.out / f"{stem}_qpt.tflite"
    qpt = tf.lite.TFLiteConverter.from_keras_model(static)
    qpt.optimizations = [tf.lite.Optimize.DEFAULT]
    qpt.representative_dataset = rep
    qpt._experimental_disable_per_channel_quantization_for_dense_layers = True
    qpt._experimental_disable_per_channel = True
    qpt_path.write_bytes(qpt.convert())

    for p in (float_path, q_path, qpt_path):
        print(f"exported {p}  ({p.stat().st_size / 1024:.0f} KB)")


def run_export_only(args, up, us):
    keras_path = args.out / f"{args.stem}.keras"
    if not keras_path.exists():
        raise SystemExit(f"--export-only needs {keras_path}; refusing to export "
                         f"an untrained model")
    model = keras.saving.load_model(str(keras_path))
    print(f"loaded {keras_path}")
    export_tflite(model, args, up[0], us[0])


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cv", type=int, metavar="K",
                      help="grouped K-fold CV over the user frames")
    mode.add_argument("--diagnostic", choices=["lopo-dig3", "loso-zeros"])
    mode.add_argument("--final", action="store_true",
                      help="train on 100%% of the pool and export")
    mode.add_argument("--export-only", action="store_true",
                      help="re-export tflite from the saved .keras")

    ap.add_argument("--upstream", default=UPSTREAM_DIR)
    ap.add_argument("--user", default=USER_DIR)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--version", default="9001")
    ap.add_argument("--size", default="s2")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="hard cap on epochs, overriding --epochs (smoke tests)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--user-weight", type=int, default=USER_WEIGHT,
                    help="replication factor for the user corpus "
                         f"(default {USER_WEIGHT}; use 1 for the ~1143-image corpus)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rep-size", type=int, default=256,
                    help="images in the int8 representative dataset")
    ap.add_argument("--only-fold", type=int, default=None, metavar="K",
                    help="run just fold K (1-based) of the same deterministic "
                         "assignment, so a long CV can be chunked across shells")
    ap.add_argument("--fold-by", choices=("day", "frame"), default="day",
                    help="CV grouping: whole capture days (default) or "
                         "round-robin frames (the 9001-9007 behaviour)")
    ap.add_argument("--aug-profile", choices=sorted(AUG_PROFILES), default="legacy",
                    help="photometric augmentation profile "
                         "(default legacy = augment_lcd.DEFAULT)")
    ap.add_argument("--geom", choices=sorted(GEOM_PROFILES), default="legacy",
                    help="IDG geometry profile (default legacy = the notebook's)")
    ap.add_argument("--lr-decay-frac", type=float, default=0.0, metavar="F",
                    help="0 = constant lr (default); F > 0 = linear decay over the "
                         "last F of the run's epochs")
    ap.add_argument("--lr-end-frac", type=float, default=0.05, metavar="R",
                    help="with --lr-decay-frac: final lr as a fraction of the start lr")
    ap.add_argument("--oof-csv", default="work/cv_oof.csv")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    args.stem = f"dig-class11_{args.version}_{args.size}"
    print(describe_recipe(args))

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    keras.utils.set_random_seed(args.seed)

    model = build_dig_class11()
    print(f"dig-class11 graph: {model.count_params()} params")
    del model

    up, us = load_pools(args.upstream, args.user)
    if len(us[1]) == 0:
        raise SystemExit(f"no user images in {args.user}")

    if args.cv:
        run_cv(args, up, us)
    elif args.diagnostic:
        run_diagnostic(args, up, us)
    elif args.final:
        run_final(args, up, us)
    else:
        run_export_only(args, up, us)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
