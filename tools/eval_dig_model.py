"""Evaluate an 11-class digit readout model (.tflite or .keras) on digit crops.

Usage:
    python tools/eval_dig_model.py --model <path .tflite|.keras> --data DIR [DIR...]
        [--resize nearest|bilinear|area|lanczos] [--compare-to MODEL2]
        [--roi-shift] [--csv OUT.csv] [--exclude FILE]

Contract: raw 0-255 float32 input of shape (N, 32, 20, 3); no normalisation.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dig_data import (  # noqa: E402
    CLASS_NAMES,
    RESAMPLE,
    TARGET_H,
    TARGET_W,
    contrast,
    load_dirs,
    parse_label,
    parse_position,
)

NCLS = len(CLASS_NAMES)


# --------------------------------------------------------------------------
# model wrappers
# --------------------------------------------------------------------------


class TFLiteModel:
    def __init__(self, path: str):
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.backend = "ai_edge_litert"
        except ImportError:  # pragma: no cover - fallback path
            from tensorflow.lite import Interpreter  # type: ignore
            self.backend = "tf.lite"
        self.path = path
        self.itp = Interpreter(model_path=path)
        self.itp.allocate_tensors()
        self.inp = self.itp.get_input_details()[0]
        self.out = self.itp.get_output_details()[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        probs = np.empty((len(x), NCLS), dtype=np.float32)
        idx_in, idx_out = self.inp["index"], self.out["index"]
        for i in range(len(x)):
            self.itp.set_tensor(idx_in, x[i : i + 1])
            self.itp.invoke()
            probs[i] = self.itp.get_tensor(idx_out)[0]
        return probs


class KerasModel:
    def __init__(self, path: str):
        import keras
        self.backend = "keras"
        self.path = path
        self.model = keras.saving.load_model(path)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return np.asarray(self.model.predict(x, verbose=0), dtype=np.float32)


def load_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tflite":
        return TFLiteModel(path)
    if ext in (".keras", ".h5"):
        return KerasModel(path)
    raise ValueError(f"unsupported model extension: {path}")


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def hr(title: str) -> None:
    print()
    print(title)
    print("-" * max(len(title), 60))


def pct(n, d) -> str:
    return f"{100.0 * n / d:6.2f}%" if d else "    n/a"


def stratum_table(rows, header, name_w=12):
    """rows: list of (name, n_correct, n_total). Prints name/acc/n."""
    print(f"{header:<{name_w}} {'acc':>8} {'n':>6}")
    for name, c, t in rows:
        print(f"{str(name):<{name_w}} {pct(c, t):>8} {t:>6}")
    if not rows:
        print("(no data)")


# --------------------------------------------------------------------------
# core evaluation
# --------------------------------------------------------------------------


def evaluate(model, x, y, meta):
    probs = model.predict(x.astype(np.float32))
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return probs, pred, conf


def terciles(values):
    """Return (edges, assign_fn) splitting values into 3 near-equal groups."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return (0.0, 0.0), (lambda z: 0)
    lo, hi = np.percentile(v, [100.0 / 3.0, 200.0 / 3.0])

    def assign(z):
        if z <= lo:
            return 0
        if z <= hi:
            return 1
        return 2

    return (lo, hi), assign


def report(model_path, x, y, meta, pred, conf, probs):
    n = len(y)
    correct = pred == y
    hr(f"MODEL: {model_path}")
    print(f"Overall accuracy: {correct.sum()}/{n} = {pct(correct.sum(), n).strip()}")

    # --- confusion matrix ---
    hr("Confusion matrix (rows = true, cols = pred, class 10 = N)")
    cm = np.zeros((NCLS, NCLS), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print("true\\pred " + "".join(f"{c:>5}" for c in CLASS_NAMES) + f"{'tot':>7}")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"{cname:>8}  " + "".join(f"{cm[i, j]:>5}" for j in range(NCLS)) + f"{cm[i].sum():>7}")
    print(f"{'tot':>8}  " + "".join(f"{cm[:, j].sum():>5}" for j in range(NCLS)) + f"{cm.sum():>7}")

    hr("Off-diagonal cells with count >= 2 (ranked)")
    offs = [
        (cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
        for i in range(NCLS)
        for j in range(NCLS)
        if i != j and cm[i, j] >= 2
    ]
    offs.sort(reverse=True)
    if offs:
        for c, t, p in offs:
            print(f"  {t}->{p} x{c}")
    else:
        print("  (none)")

    # --- per position ---
    hr("Per digit position (user-style '_dig<N>_' files only)")
    rows = []
    positions = sorted({m["position"] for m in meta if m["position"] is not None})
    for p in positions:
        idx = [i for i, m in enumerate(meta) if m["position"] == p]
        rows.append((f"dig{p}", int(correct[idx].sum()), len(idx)))
    stratum_table(rows, "position")

    # --- per screen ---
    hr("Per screen type")
    rows = []
    for s in ["kwh", "test8", "zeros", "blank09"]:
        idx = [i for i, m in enumerate(meta) if m["screen"] == s]
        if idx:
            rows.append((s, int(correct[idx].sum()), len(idx)))
    unk = [i for i, m in enumerate(meta) if m["screen"] is None]
    if unk:
        rows.append(("(unknown)", int(correct[unk].sum()), len(unk)))
    stratum_table(rows, "screen")

    # --- per contrast tercile ---
    hr("Per contrast tercile (p95-p5 luma, terciles over the evaluated set)")
    cvals = np.array([m["contrast"] for m in meta], dtype=np.float64)
    (lo, hi), assign = terciles(cvals)
    groups = np.array([assign(c) for c in cvals])
    labels = [f"low <={lo:.1f}", f"mid <={hi:.1f}", f"high >{hi:.1f}"]
    rows = []
    for g in range(3):
        idx = np.where(groups == g)[0]
        rows.append((labels[g], int(correct[idx].sum()), len(idx)))
    stratum_table(rows, "tercile", name_w=16)

    # --- confidence ---
    hr("Confidence (max softmax)")
    for name, mask in (("correct", correct), ("wrong", ~correct)):
        c = conf[mask]
        if len(c):
            print(
                f"  {name:<8} n={len(c):<5} mean={c.mean():.4f} "
                f"median={np.median(c):.4f} p10={np.percentile(c, 10):.4f}"
            )
        else:
            print(f"  {name:<8} n=0")
    print(f"  mean P(N) over all images = {probs[:, 10].mean():.4f}")

    # --- worst errors ---
    hr("Worst 20 errors (most confident mistakes first)")
    err = np.where(~correct)[0]
    err = err[np.argsort(-conf[err])][:20]
    if len(err):
        print(f"{'file':<48} {'true':>5} {'pred':>5} {'conf':>7}")
        for i in err:
            print(
                f"{os.path.basename(meta[i]['path']):<48} "
                f"{CLASS_NAMES[y[i]]:>5} {CLASS_NAMES[pred[i]]:>5} {conf[i]:>7.4f}"
            )
    else:
        print("  (no errors)")


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def compare(meta, y, predA, confA, predB, confB, nameA, nameB):
    okA, okB = predA == y, predB == y
    hr(f"PAIRED COMPARISON  A={nameA}  B={nameB}")

    def block(title, keyfn, order=None):
        print()
        print(f"  {title}")
        keys = {keyfn(i) for i in range(len(y))}
        keys = [k for k in (order or sorted(keys, key=str)) if k in keys]
        print(f"    {'stratum':<16} {'A acc':>9} {'B acc':>9} {'n':>6}")
        for k in keys:
            idx = [i for i in range(len(y)) if keyfn(i) == k]
            print(
                f"    {str(k):<16} {pct(okA[idx].sum(), len(idx)):>9} "
                f"{pct(okB[idx].sum(), len(idx)):>9} {len(idx):>6}"
            )

    print(f"    {'overall':<16} {pct(okA.sum(), len(y)):>9} {pct(okB.sum(), len(y)):>9} {len(y):>6}")
    block("by true class", lambda i: CLASS_NAMES[y[i]], order=CLASS_NAMES)
    block("by position", lambda i: f"dig{meta[i]['position']}" if meta[i]["position"] else "n/a")
    block("by screen", lambda i: meta[i]["screen"] or "n/a")

    fixed = [i for i in range(len(y)) if okA[i] and not okB[i]]
    broken = [i for i in range(len(y)) if okB[i] and not okA[i]]
    for title, idxs, pr, cf in (
        (f"fixed by A ({nameA}) vs B", fixed, predB, confB),
        (f"broken by A ({nameA}) vs B", broken, predA, confA),
    ):
        print()
        print(f"  {title}: {len(idxs)} file(s)" + (" (showing 20)" if len(idxs) > 20 else ""))
        for i in idxs[:20]:
            print(
                f"    {os.path.basename(meta[i]['path']):<48} true={CLASS_NAMES[y[i]]} "
                f"other_pred={CLASS_NAMES[pr[i]]} conf={cf[i]:.4f}"
            )


# --------------------------------------------------------------------------
# ROI shift stability
# --------------------------------------------------------------------------


def roi_shift(model, meta, resize="nearest", shifts=(0.03, 0.06)):
    """Crop-jitter stability on native-resolution (94x202) source images."""
    native = [m for m in meta if Image.open(m["path"]).size != (TARGET_W, TARGET_H)]
    hr("ROI-SHIFT STABILITY")
    if not native:
        print("  (no native-resolution images in the evaluated set -- skipped)")
        return

    filt = RESAMPLE[resize]
    records = []  # (meta, s, agree_frac, n_distinct, mean_conf)

    for m in native:
        img = Image.open(m["path"]).convert("RGB")
        W, H = img.size
        for s in shifts:
            cw, ch = int(round((1 - 2 * s) * W)), int(round((1 - 2 * s) * H))
            crops = []
            for dy in (-s, 0.0, s):
                for dx in (-s, 0.0, s):
                    left = int(round((s + dx) * W))
                    top = int(round((s + dy) * H))
                    left = max(0, min(left, W - cw))
                    top = max(0, min(top, H - ch))
                    crop = img.crop((left, top, left + cw, top + ch))
                    crops.append(np.array(crop.resize((TARGET_W, TARGET_H), filt), dtype=np.uint8))
            probs = model.predict(np.stack(crops).astype(np.float32))
            preds = probs.argmax(axis=1)
            centre = preds[4]  # dy=0, dx=0
            records.append(
                {
                    "position": m["position"],
                    "label": m["label"],
                    "s": s,
                    "agree": float((preds == centre).mean()),
                    "distinct": int(len(set(preds.tolist()))),
                    "conf": float(probs.max(axis=1).mean()),
                    "centre": int(centre),
                }
            )

    def summarise(title, keyfn, keys=None):
        print()
        print(f"  {title}")
        print(
            f"    {'stratum':<10} {'s':>6} {'agree':>8} {'distinct':>9} "
            f"{'maxsoft':>8} {'n':>5}"
        )
        allk = keys if keys is not None else sorted({keyfn(r) for r in records}, key=str)
        for k in allk:
            for s in shifts:
                sub = [r for r in records if keyfn(r) == k and r["s"] == s]
                if not sub:
                    continue
                print(
                    f"    {str(k):<10} {s:>6.2f} "
                    f"{np.mean([r['agree'] for r in sub]):>8.3f} "
                    f"{np.mean([r['distinct'] for r in sub]):>9.3f} "
                    f"{np.mean([r['conf'] for r in sub]):>8.4f} {len(sub):>5}"
                )

    print()
    print(f"  {len(native)} native-resolution images x 9 windows x {len(shifts)} shift scales")
    print(f"    {'overall':<10} {'s':>6} {'agree':>8} {'distinct':>9} {'maxsoft':>8} {'n':>5}")
    for s in shifts:
        sub = [r for r in records if r["s"] == s]
        print(
            f"    {'ALL':<10} {s:>6.2f} {np.mean([r['agree'] for r in sub]):>8.3f} "
            f"{np.mean([r['distinct'] for r in sub]):>9.3f} "
            f"{np.mean([r['conf'] for r in sub]):>8.4f} {len(sub):>5}"
        )
    summarise("by position", lambda r: f"dig{r['position']}")
    summarise(
        "by true class",
        lambda r: CLASS_NAMES[r["label"]],
        keys=[c for c in CLASS_NAMES],
    )


# --------------------------------------------------------------------------
# csv
# --------------------------------------------------------------------------


def write_csv(path, meta, y, pred, conf, probs):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cols = (
        ["path", "label", "pred", "conf"]
        + [f"p{c}" for c in CLASS_NAMES[:10]]
        + ["pN", "position", "frame", "screen", "contrast"]
    )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, m in enumerate(meta):
            w.writerow(
                [m["path"], CLASS_NAMES[y[i]], CLASS_NAMES[pred[i]], f"{conf[i]:.6f}"]
                + [f"{p:.6f}" for p in probs[i]]
                + [m["position"], m["frame"], m["screen"], f"{m['contrast']:.3f}"]
            )
    print(f"\nwrote {path} ({len(meta)} rows)")


# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--resize", default="nearest", choices=sorted(RESAMPLE))
    ap.add_argument("--compare-to", default=None)
    ap.add_argument("--roi-shift", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--exclude", default=None)
    args = ap.parse_args(argv)

    x, y, meta = load_dirs(args.data, exclude=args.exclude, resize=args.resize)
    print(f"loaded {len(y)} images from {', '.join(args.data)} (resize={args.resize})")
    if len(y) == 0:
        print("nothing to evaluate")
        return 1

    model = load_model(args.model)
    print(f"model backend: {model.backend}")
    probs, pred, conf = evaluate(model, x, y, meta)
    report(args.model, x, y, meta, pred, conf, probs)

    if args.csv:
        write_csv(args.csv, meta, y, pred, conf, probs)

    if args.compare_to:
        model_b = load_model(args.compare_to)
        probs_b, pred_b, conf_b = evaluate(model_b, x, y, meta)
        report(args.compare_to, x, y, meta, pred_b, conf_b, probs_b)
        compare(meta, y, pred, conf, pred_b, conf_b, args.model, args.compare_to)

    if args.roi_shift:
        roi_shift(model, meta, resize=args.resize)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
