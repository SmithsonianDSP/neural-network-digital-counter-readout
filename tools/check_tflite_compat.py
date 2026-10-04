#!/usr/bin/env python3
"""
Check a .tflite model before flashing it to AI-on-the-Edge-Device.

Two failure modes cost a full round-trip each, and both are cheap to catch here:

**Unsupported operators.** The firmware runs TFLite *for Microcontrollers*,
whose op resolver registers a fixed set of operators at fixed version ranges.
A model using anything outside that set fails at load with

    <ERR> [TFLITE] AllocateTensors() failed

which reads like file corruption and is nothing of the sort. Keras 3's default
converter is a reliable way to trigger it: it leaves the batch dimension
dynamic, so Flatten becomes SHAPE + STRIDED_SLICE + PACK feeding RESHAPE.

**Untrained weights.** A randomly initialised model converts perfectly, loads
perfectly, and predicts nonsense. Scoring against a known capture catches it.

Usage::

    python tools/check_tflite_compat.py --model models/ana-cont/ana-cont_2010_s0.tflite
    python tools/check_tflite_compat.py --model <new> --reference models/ana-cont/ana-cont_1901_s0.tflite
    python tools/check_tflite_compat.py --model <new> --captures joes-samples/ana2 --truth 5.1
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Operators registered by the firmware's micro op resolver. Conservative: this
# is the set upstream ana-cont models actually use.
KNOWN_GOOD = {
    "ADD", "CONV_2D", "FULLY_CONNECTED", "MAX_POOL_2D", "MUL", "RESHAPE",
    "QUANTIZE", "DEQUANTIZE", "AVERAGE_POOL_2D", "SOFTMAX", "RELU",
    "DEPTHWISE_CONV_2D", "LOGISTIC", "PAD", "MEAN",
}


def read_ops(path: Path) -> tuple[set[str], int]:
    try:
        import tflite
    except ImportError:
        raise SystemExit("pip install tflite")

    names = {v: k for k, v in vars(tflite.BuiltinOperator).items()
             if isinstance(v, int)}
    model = tflite.Model.GetRootAsModel(path.read_bytes(), 0)

    ops = set()
    for i in range(model.OperatorCodesLength()):
        code = model.OperatorCodes(i)
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        ops.add(f"{names.get(builtin, '?%d' % builtin)} v{code.Version()}")
    return ops, model.Version()


def load_interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def read_value(interpreter, rgb) -> tuple[float, float]:
    import cv2
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]

    x = cv2.resize(rgb, (in_d["shape"][2], in_d["shape"][1]),
                   interpolation=cv2.INTER_CUBIC)
    if in_d["dtype"] == np.uint8:
        scale, zero = in_d["quantization"]
        x = np.clip(x.astype(np.float32) / scale + zero if scale else x,
                    0, 255).astype(np.uint8)
    else:
        x = x.astype(np.float32)

    interpreter.set_tensor(in_d["index"], x[None])
    interpreter.invoke()
    p = interpreter.get_tensor(out_d["index"])[0].astype(np.float32)
    if out_d["dtype"] == np.uint8:
        scale, zero = out_d["quantization"]
        if scale:
            p = (p - zero) * scale

    value = math.atan2(p[0], p[1]) / (2 * math.pi) * 10 % 10
    return value, float(np.hypot(p[0], p[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None,
                        help="A model known to load on the device; its op set "
                             "becomes the allow-list")
    parser.add_argument("--captures", type=Path, default=None,
                        help="Directory of real captures at a known reading")
    parser.add_argument("--truth", type=float, default=None,
                        help="Geometric dial value for those captures")
    args = parser.parse_args()

    problems = []

    ops, schema = read_ops(args.model)
    size_kb = args.model.stat().st_size / 1024
    print(f"{args.model.name}  {size_kb:.0f} KB  schema v{schema}")
    print(f"  ops: {sorted(ops)}")

    allowed = KNOWN_GOOD
    if args.reference:
        ref_ops, _ = read_ops(args.reference)
        print(f"  reference {args.reference.name}: {sorted(ref_ops)}")
        extra = ops - ref_ops
        if extra:
            print(f"  ! not present in the reference model: {sorted(extra)}")
            bare = {o.split(" v")[0] for o in extra}
            if bare - {o.split(" v")[0] for o in ref_ops}:
                problems.append(f"operators absent from reference: {sorted(bare)}")
            else:
                print("    (same operators, higher versions - usually fine, "
                      "but the likeliest remaining cause if it fails to load)")
        else:
            print("  OK: op set is a subset of the reference model")
    else:
        unknown = {o.split(" v")[0] for o in ops} - allowed
        if unknown:
            problems.append(f"operators outside the known-good set: {sorted(unknown)}")
        else:
            print("  OK: all operators are in the known-good set")

    try:
        interpreter = load_interpreter(args.model)
        print("  OK: AllocateTensors() succeeded on the desktop runtime")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"desktop AllocateTensors failed: {exc}")
        interpreter = None

    if interpreter is not None and args.captures:
        # read_value() decodes a sin/cos unit vector, which is specific to
        # ana-cont's 2-output regression head. On a different output size
        # (e.g. dig-class11's 11-way softmax) it would silently read the
        # first two class scores as sin/cos and report a bogus "untrained"
        # verdict, so the ana-cont-specific readout is guarded behind an
        # output-size check here. The op-set listing, AllocateTensors probe,
        # and reference comparison above are output-shape agnostic and are
        # unaffected by this guard - they already run for any model.
        out_size = interpreter.get_output_details()[0]["shape"][-1]
        if out_size != 2:
            print(f"  (skipping value/confidence probe: this readout is "
                  f"ana-cont-specific (2 outputs), model has {out_size})")
        else:
            import cv2
            paths = sorted(glob.glob(str(args.captures / "*.jpg")))
            if not paths:
                problems.append(f"no .jpg captures in {args.captures}")
            else:
                values, confs = [], []
                for path in paths:
                    bgr = cv2.imread(path)
                    if bgr is None:
                        continue
                    v, c = read_value(interpreter, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    values.append(v)
                    confs.append(c)

                values = np.array(values)
                confidence = float(np.mean(confs))
                print(f"  {len(values)} captures: mean confidence {confidence:.2f}")

                # ana-cont emits sin/cos; a well-trained model returns a unit vector.
                # An untrained one does not, which is the cheapest tell there is.
                if not 0.7 <= confidence <= 1.3:
                    problems.append(
                        f"output vector magnitude {confidence:.2f} is far from 1.0 - "
                        f"the model is very likely untrained or mis-converted")

                if args.truth is not None:
                    d = np.abs(values - args.truth)
                    err = np.minimum(d, 10 - d)
                    print(f"  vs truth {args.truth}: MAE {err.mean():.3f}, "
                          f"worst {err.max():.2f}")
                    if err.mean() > 0.5:
                        problems.append(f"MAE {err.mean():.3f} against stated truth")

    print()
    if problems:
        print("NOT SAFE TO FLASH:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("PASS - no problems detected")


if __name__ == "__main__":
    main()
