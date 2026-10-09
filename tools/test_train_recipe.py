"""The training recipe flags are opt-in: the defaults must be the 9001-9020 recipe.

    python tools/test_train_recipe.py

Checks that a default ``train_dig_class11.py`` invocation builds exactly the old
geometry, the old photometric config and no LR schedule -- and that the batches
it produces are bit-identical to the pre-profile code path -- plus sanity checks
on the R1 pieces (luma gate, LR schedule).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import augment_lcd as A  # noqa: E402
import train_dig_class11 as T  # noqa: E402

# The literals as they stood in train_dig_class11.make_datagen / augment_lcd.DEFAULT
# before the profiles existed (commit a8ae7a69).
OLD_GEOM = dict(width_shift_range=[-1, 1], height_shift_range=[-1, 1],
                brightness_range=[0.8, 1.2], zoom_range=[0.7, 1.3], rotation_range=5)
# gentle as shipped in R1 / 9021 (must not drift either)
OLD_GENTLE = dict(width_shift_range=[-1, 0, 1], height_shift_range=[-1, 0, 1],
                  zoom_range=[0.9, 1.1], rotation_range=2)
OLD_AUG = dict(blur_p=0.45, blur_sigma=(0.3, 0.9), motion_p=0.15, motion_len=(2, 4),
               glare_p=0.55, glare_sigma_frac=(0.15, 0.9), glare_gain=(25.0, 140.0),
               veil_p=0.7, veil_k=(0.05, 0.42), veil_grid=(4, 3), veil_field_gain=(0.6, 1.0),
               veil_field_bias=(0.0, 0.4), veil_level=(150.0, 255.0),
               exposure=(0.60, 1.40), contrast=(0.55, 1.22), wb=(0.88, 1.12),
               noise_sigma=(0.5, 6.0), jpeg_p=0.5, jpeg_quality=(45, 95),
               scale_blur_to_size=True)


def _old_preprocessing_fn(cfg, seed):
    """The pre-profile closure: augment_lcd called directly."""
    root = np.random.SeedSequence(seed)
    state = {"rng": None, "pid": None}

    def _pre(img):
        pid = os.getpid()
        if state["rng"] is None or state["pid"] != pid:
            child = np.random.SeedSequence(entropy=root.entropy, spawn_key=(pid,))
            state["rng"] = np.random.default_rng(child)
            state["pid"] = pid
        return A.augment_lcd(img, state["rng"], cfg)

    return _pre


def _batches(idg, x, y, seed, n=6):
    flow = idg.flow(x, y, batch_size=4, seed=seed)
    return [flow[i][0].copy() for i in range(n)]


def test_default_configs_are_legacy():
    assert A.AUG_PROFILES["legacy"] is A.DEFAULT
    assert A.DEFAULT == A.AugmentConfig(**OLD_AUG)
    assert A.GEOM_PROFILES["legacy"] == OLD_GEOM
    assert A.GEOM_PROFILES["gentle"] == OLD_GENTLE


def test_default_cli_is_legacy():
    import argparse

    seen = {}

    def fake_parse(self, argv=None, namespace=None):
        ns = real_parse(self, argv, namespace)
        seen["ns"] = ns
        raise SystemExit(0)

    real_parse = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = fake_parse
    try:
        try:
            T.main(["--final"])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    ns = seen["ns"]
    assert ns.aug_profile == "legacy" and ns.geom == "legacy", ns
    assert ns.lr_decay_frac == 0.0
    assert T.lr_schedule(1.0, 125, ns.lr_decay_frac) is None
    assert T.recipe_kwargs(ns) == dict(aug_profile="legacy", geom="legacy",
                                       lr_decay_frac=0.0, lr_end_frac=0.05)


def test_default_batches_bit_identical():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (24, 32, 20, 3)).astype(np.float32)
    x[:8] *= 0.3  # some dark crops too
    y = T.onehot(rng.integers(0, 11, 24))
    seed = 123

    new = T.make_datagen(seed)
    old = T.ImageDataGenerator(**OLD_GEOM, preprocessing_function=_old_preprocessing_fn(A.DEFAULT, seed))
    for k in ("width_shift_range", "height_shift_range", "brightness_range", "zoom_range",
              "rotation_range", "fill_mode", "cval", "interpolation_order"):
        assert np.array_equal(np.asarray(getattr(new, k)), np.asarray(getattr(old, k))), k

    b_new = _batches(new, x, y, seed)
    b_old = _batches(old, x, y, seed)
    assert all(np.array_equal(a, b) for a, b in zip(b_new, b_old)), "default batches differ"

    # and the R1 recipe really is different
    r1 = T.make_datagen(seed, "night_aware", "gentle")
    b_r1 = _batches(r1, x, y, seed)
    assert not all(np.array_equal(a, b) for a, b in zip(b_r1, b_old))


def test_gentle_batches_bit_identical():
    """R1's geometry path (plain IDG kwargs) is untouched by the split profiles."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 255, (24, 32, 20, 3)).astype(np.float32)
    x[:8] *= 0.3
    y = T.onehot(rng.integers(0, 11, 24))
    seed = 9
    new = T.make_datagen(seed, "night_aware", "gentle")
    old = T.ImageDataGenerator(**OLD_GENTLE,
                               preprocessing_function=A.make_idg_preprocessing_fn(A.NIGHT_AWARE, seed))
    assert all(np.array_equal(a, b) for a, b in zip(_batches(new, x, y, seed), _batches(old, x, y, seed)))


def _counting(idg, box, key):
    real = idg.get_random_transform

    def wrapped(shape, seed=None):
        box[key] += 1
        return real(shape, seed)

    idg.get_random_transform = wrapped


def test_split_geometry_routing():
    dark = np.full((32, 20, 3), 50, np.float32)
    dark[8:24, 6:14] = 90  # a glyph, so a transform is visible
    bright = np.full((32, 20, 3), 140, np.float32)
    bright[8:24, 6:14] = 30
    assert A.mean_luma(dark) <= A.DARK_LUMA_MAX < A.mean_luma(bright)

    sg = A.GEOM_PROFILES["split_gentle"]
    assert sg.dark == OLD_GENTLE and sg.dark_luma_max == A.DARK_LUMA_MAX
    assert "brightness_range" not in sg.bright
    for name in ("split_gentle", "split_none"):
        fn = T.make_geometry_fn(A.GEOM_PROFILES[name])
        box = {"dark": 0, "bright": 0}
        if fn.dark_idg is not None:
            _counting(fn.dark_idg, box, "dark")
        _counting(fn.bright_idg, box, "bright")
        np.random.seed(0)
        outs_d = [fn(dark.copy()) for _ in range(20)]
        outs_b = [fn(bright.copy()) for _ in range(20)]
        assert box["bright"] == 20, (name, box)
        assert fn.route(dark) == "dark" and fn.route(bright) == "bright"
        assert any(not np.array_equal(o, bright) for o in outs_b), name
        if name == "split_none":
            assert box["dark"] == 0
            assert all(np.array_equal(o, dark) for o in outs_d), "split_none moved a dark crop"
        else:
            assert box["dark"] == 20
            assert any(not np.array_equal(o, dark) for o in outs_d)

    # threshold is inclusive (<= 78 is dark), same as the photometric gate
    fn = T.make_geometry_fn(A.GEOM_PROFILES["split_none"])
    edge = np.full((32, 20, 3), 77.9, np.float32)
    assert np.array_equal(fn(edge), edge)


def test_split_datagen_runs():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 255, (16, 32, 20, 3)).astype(np.float32)
    x[:8] *= 0.3
    y = T.onehot(rng.integers(0, 11, 16))
    for name in ("split_gentle", "split_none"):
        b = _batches(T.make_datagen(5, "night_aware", name), x, y, 5, n=4)
        assert all(a.shape == (4, 32, 20, 3) and np.isfinite(a).all() for a in b)


def test_lr_schedule():
    f = T.lr_schedule(1.0, 125, 0.2, 0.05)
    lrs = [f(e) for e in range(125)]
    assert f.start == 100 and f.span == 25
    assert all(v == 1.0 for v in lrs[:100])
    assert abs(lrs[-1] - 0.05) < 1e-9
    assert all(a > b for a, b in zip(lrs[99:], lrs[100:]))  # strictly decreasing tail
    g = T.lr_schedule(1.0, 2, 0.2, 0.05)  # smoke-test length: epoch 2 decays
    assert g(0) == 1.0 and abs(g(1) - 0.05) < 1e-9


def test_luma_gate():
    gate = A.NIGHT_AWARE
    dark = np.full((32, 20, 3), 50, np.float32)
    bright = np.full((32, 20, 3), 120, np.float32)
    assert gate.pick(dark) is A.NIGHT_DARK
    assert gate.pick(bright) is A.NIGHT_BRIGHT
    assert A.NIGHT_DARK.veil_p == 0 and A.NIGHT_DARK.glare_p == 0 and A.NIGHT_DARK.wb == (1.0, 1.0)
    out = A.augment(dark, np.random.default_rng(0), gate)
    assert out.shape == dark.shape and out.dtype == np.float32


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
