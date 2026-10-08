"""Photometric augmentation for 7-segment LCD digit crops seen through glare.

Derived from ``AIOTED-analog/src/utils/dial_synth.py::augment()``, with the
geometry stage removed and a new veiling-haze term added.

Why no geometry
---------------
This augmenter is wired into the notebook's legacy Keras ``ImageDataGenerator``
as ``preprocessing_function``. IDG applies its geometric transforms (shift,
zoom, rotation, brightness) *first* and then calls ``preprocessing_function``,
so IDG owns geometry and this module owns the photometric chain. That ordering
is also the physically correct one: the haze and glare live in front of the
display, so they are applied to the already-registered glyph.

Why veiling haze
----------------
The target install views a 7-segment LCD through permanent glare and a
UV-hazed acrylic window. Diffuse scatter through the haze adds a low-frequency
bright field to the scene: it lifts the blacks *and* compresses contrast, which
is exactly the failure mode behind the dominant shipped-model error (a washed
out top segment turning ``7`` into ``1``). A plain brightness/contrast jitter
cannot produce that, because it is spatially uniform; the veil is not.

Dependency-light on purpose: numpy + cv2 only, no Keras/TF.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "AugmentConfig",
    "DEFAULT",
    "LumaGatedConfig",
    "NIGHT_AWARE",
    "AUG_PROFILES",
    "GEOM_PROFILES",
    "DARK_LUMA_MAX",
    "augment",
    "augment_lcd",
    "make_idg_preprocessing_fn",
    "mean_luma",
]

# The model input geometry. Blur sigmas below are expressed in pixels at this
# resolution and are scaled up when the caller augments a larger crop (e.g. the
# preview sheet, which works at the native 94x202 capture resolution).
REF_W = 20
REF_H = 32


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentConfig:
    """Every probability and range used by :func:`augment_lcd`.

    Frozen so a config can be shared safely; use ``dataclasses.replace`` to
    derive a tuned variant without editing this module.
    """

    # --- focus blur ---
    blur_p: float = 0.45
    blur_sigma: tuple = (0.3, 0.9)

    # --- motion blur (analog values, unscaled) ---
    motion_p: float = 0.15
    motion_len: tuple = (2, 4)  # integers, half-open [lo, hi)

    # --- specular glare lobe; sigma is a fraction of max(h, w) ---
    glare_p: float = 0.55
    glare_sigma_frac: tuple = (0.15, 0.9)
    glare_gain: tuple = (25.0, 140.0)

    # --- veiling haze: low-frequency bright field, diffuse scatter ---
    veil_p: float = 0.7
    veil_k: tuple = (0.05, 0.42)  # opacity of the blend
    veil_grid: tuple = (4, 3)  # (rows, cols) of the low-res field
    veil_field_gain: tuple = (0.6, 1.0)
    veil_field_bias: tuple = (0.0, 0.4)
    veil_level: tuple = (150.0, 255.0)  # brightness of the veil plane

    # --- camera response ---
    # Tuned down from the drafted (0.50, 1.40) / (0.45, 1.20): those stacked a
    # second, spatially-uniform darkening on top of the veil and pushed 30% of
    # augmented samples below the real p5 contrast. Trimming the two global
    # multipliers keeps the physically-motivated veil term near full strength.
    exposure: tuple = (0.60, 1.40)
    contrast: tuple = (0.55, 1.22)  # about the image mean
    wb: tuple = (0.88, 1.12)  # per-channel gain

    # --- sensor / codec ---
    noise_sigma: tuple = (0.5, 6.0)
    jpeg_p: float = 0.5
    jpeg_quality: tuple = (45, 95)  # integers, half-open [lo, hi)

    # --- behaviour ---
    scale_blur_to_size: bool = True

    def replace(self, **kw) -> "AugmentConfig":
        """Convenience wrapper around ``dataclasses.replace``."""
        from dataclasses import replace as _replace

        return _replace(self, **kw)


DEFAULT = AugmentConfig()


# --------------------------------------------------------------------------
# night-aware profile (recipe "R1", 2026-10-08)
# --------------------------------------------------------------------------
#
# DEFAULT was tuned in the 9001 cycle on 228 *daytime* crops and is applied to
# every crop. Flash-only night crops are already washed out; on night dig6 only
# ~56% of DEFAULT-augmented views still read correctly with 9018 (day ~88%):
# veil / glare / white-balance turn faint 3s into 7-looking or blank images that
# are still labelled 3. NIGHT_AWARE gates on the image's own mean luma.
#
# Threshold, measured on 20x32 crops (04_joe_lcd_20x32 and an stb build agree to
# 0.2) against work/corpus_manifest.csv `bucket`:
#   flash      n=3015  bimodal BY POSITION: dig6 p5/p50/p95 44/49/62, dig5 67/70/74,
#                      dig4 84/88/93, dig3 87/92/106, dig2 89/96/106 (the flash
#                      lights the middle of the display; dig5/dig6 sit in its fall-off)
#   transition n=1582  p1 64  p5 73  p50 94
#   day        n=2188  p0 80  p1 97  p5 101  p50 130
# The flash histogram is empty between 75 and 82 (48.8% <= 75, 49.9% <= 82).
# 78 sits in that gap and below every day crop: it selects flash dig5/dig6
# (+ ~10% of transition crops, the dimmest), 0% of day. Flash dig2-dig4 overlap
# the day tail by luma alone and stay on the (gentler) bright chain.
DARK_LUMA_MAX = 78.0

# Dark (night dig5/dig6): no veil, no glare lobe, no white-balance shift; narrow
# exposure/contrast; mild blur / sensor noise / JPEG only.
NIGHT_DARK = AugmentConfig(
    blur_p=0.30, blur_sigma=(0.3, 0.7),
    motion_p=0.08, motion_len=(2, 3),
    glare_p=0.0,
    veil_p=0.0,
    exposure=(0.85, 1.15),
    contrast=(0.85, 1.10),
    wb=(1.0, 1.0),
    noise_sigma=(0.5, 3.0),
    jpeg_p=0.30, jpeg_quality=(60, 95),
)

# Bright (day, transition, flash dig2-4): today's chain, probabilities ~halved.
NIGHT_BRIGHT = DEFAULT.replace(
    blur_p=0.22, motion_p=0.08, glare_p=0.28, veil_p=0.35, jpeg_p=0.25,
)


def mean_luma(img) -> float:
    """Rec.601 mean luma of an (H, W, 3) / (H, W[, 1]) array in 0-255 units."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return float((0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean())
    return float(a.mean())


@dataclass(frozen=True)
class LumaGatedConfig:
    """Pick ``dark`` when the image's own mean luma <= ``dark_luma_max``, else ``bright``.

    The gate reads the image as :func:`augment` receives it -- inside the IDG that
    is *after* geometry (and after IDG's ``brightness_range``, if the geometry
    profile has one; the ``gentle`` geometry does not).
    """

    dark: AugmentConfig
    bright: AugmentConfig
    dark_luma_max: float = DARK_LUMA_MAX

    def pick(self, img) -> AugmentConfig:
        return self.dark if mean_luma(img) <= self.dark_luma_max else self.bright


NIGHT_AWARE = LumaGatedConfig(dark=NIGHT_DARK, bright=NIGHT_BRIGHT)

# Named photometric profiles. "legacy" is DEFAULT itself, so a legacy run makes
# exactly the same augment_lcd() calls (and RNG draws) as before the profiles.
AUG_PROFILES = {
    "legacy": DEFAULT,
    "night_aware": NIGHT_AWARE,
}

# Named ImageDataGenerator geometry profiles (kwargs, minus preprocessing_function).
# NB on Keras' list semantics: a list shift range is np.random.choice(list) times a
# random sign, so legacy's [-1, 1] shifts EVERY image by exactly 1 px on both axes
# (never 0); gentle's [-1, 0, 1] gives 0 px a third of the time. zoom draws zx, zy
# independently. brightness_range (legacy only) duplicates the photometric exposure.
GEOM_PROFILES = {
    "legacy": dict(  # jomjol's notebook, used for 9001-9020
        width_shift_range=[-1, 1],
        height_shift_range=[-1, 1],
        brightness_range=[0.8, 1.2],
        zoom_range=[0.7, 1.3],
        rotation_range=5,
    ),
    "gentle": dict(  # fixed, aligned ROIs: +-1 px, +-10% zoom, 2 deg, no brightness
        width_shift_range=[-1, 0, 1],
        height_shift_range=[-1, 0, 1],
        zoom_range=[0.9, 1.1],
        rotation_range=2,
    ),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _u(rng: np.random.Generator, rng_pair) -> float:
    lo, hi = rng_pair
    return float(rng.uniform(float(lo), float(hi)))


def _motion_kernel(rng: np.random.Generator, length: int) -> np.ndarray:
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    theta = float(rng.uniform(0.0, 180.0))
    rot = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), theta, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    return kernel


def _veil_field(rng: np.random.Generator, h: int, w: int, cfg: AugmentConfig) -> np.ndarray:
    """A smooth 0-1 field: a tiny random grid bicubically blown up to (h, w)."""
    rows, cols = int(cfg.veil_grid[0]), int(cfg.veil_grid[1])
    small = rng.random((rows, cols)).astype(np.float32)
    fld = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(fld.min()), float(fld.max())
    fld = (fld - lo) / (hi - lo) if hi - lo > 1e-6 else np.full_like(fld, 0.5)
    fld = fld * _u(rng, cfg.veil_field_gain) + _u(rng, cfg.veil_field_bias)
    return np.clip(fld, 0.0, 1.0)


def _jpeg_roundtrip(x: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode through JPEG at the given quality; returns float32 RGB."""
    u8 = np.clip(x, 0.0, 255.0).astype(np.uint8)
    if u8.shape[2] == 3:
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return x
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if dec is None:
            return x
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)

    ok, buf = cv2.imencode(".jpg", u8[..., 0], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return x
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if dec is None:
        return x
    return dec.astype(np.float32)[..., None]


# --------------------------------------------------------------------------
# the augmenter
# --------------------------------------------------------------------------


def augment_lcd(img: np.ndarray, rng: np.random.Generator,
                cfg: AugmentConfig = DEFAULT) -> np.ndarray:
    """Apply capture-realistic photometric degradation to a digit crop.

    Parameters
    ----------
    img : ndarray
        ``(H, W, 3)`` (or ``(H, W)`` / ``(H, W, 1)``) in RAW 0-255 units. The
        canonical shape is ``(32, 20, 3)`` but any size works, so the preview
        tooling can run at the native capture resolution.
    rng : numpy.random.Generator
        All randomness is drawn from here, so a given seed reproduces exactly.
    cfg : AugmentConfig
        Probabilities and ranges. Defaults are tuned against the real
        ``joes-samples`` contrast distribution (see ``augment_preview.py
        --stats``).

    Returns
    -------
    ndarray
        float32, same shape as ``img``, finite, clipped to 0-255. The input is
        never modified.

    Notes
    -----
    Stage order is physical: optics (blur) -> scene-side additive light
    (glare lobe, veiling haze) -> camera response (exposure, contrast, white
    balance) -> sensor noise -> codec. There is deliberately no geometry stage;
    see the module docstring.
    """
    src = np.asarray(img)
    squeeze_back = src.ndim == 2
    x = np.array(src, dtype=np.float32, copy=True)
    if squeeze_back:
        x = x[..., None]

    h, w = x.shape[:2]
    chans = x.shape[2]

    # Blur sigmas are calibrated for the 20x32 model input; on a larger crop
    # the same optical blur covers proportionally more pixels.
    size_scale = 1.0
    if cfg.scale_blur_to_size:
        size_scale = max(1.0, max(h / float(REF_H), w / float(REF_W)))

    # --- focus blur ------------------------------------------------------
    if rng.random() < cfg.blur_p:
        sigma = _u(rng, cfg.blur_sigma) * size_scale
        if sigma > 1e-3:
            x = cv2.GaussianBlur(x, (0, 0), sigma)
            if x.ndim == 2:
                x = x[..., None]

    # --- motion blur -----------------------------------------------------
    if rng.random() < cfg.motion_p:
        length = int(rng.integers(int(cfg.motion_len[0]), int(cfg.motion_len[1])))
        if length >= 2:
            kernel = _motion_kernel(rng, length)
            ksum = float(kernel.sum())
            if ksum > 1e-6:
                x = cv2.filter2D(x, -1, kernel / ksum)
                if x.ndim == 2:
                    x = x[..., None]

    # --- glare: a bright off-centre lobe, as from a specular reflection --
    if rng.random() < cfg.glare_p:
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        yy, xx = np.mgrid[0:h, 0:w]
        sigma = _u(rng, cfg.glare_sigma_frac) * max(h, w)
        lobe = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        x = x + (lobe * _u(rng, cfg.glare_gain)).astype(np.float32)[..., None]

    # --- veiling haze: diffuse scatter through the UV-hazed acrylic ------
    # Blending *toward* a bright plane lifts the blacks and compresses the
    # dynamic range in one step, which a multiplicative brightness term
    # cannot do. This is the term that produces washed-out top segments.
    if rng.random() < cfg.veil_p:
        k = _u(rng, cfg.veil_k)
        fld = _veil_field(rng, h, w, cfg)
        veil = fld[..., None] * _u(rng, cfg.veil_level)
        x = x * (1.0 - k) + veil * k

    # --- exposure, contrast, white balance -------------------------------
    x = x * _u(rng, cfg.exposure)
    mean = float(x.mean())
    x = (x - mean) * _u(rng, cfg.contrast) + mean
    x = x * rng.uniform(cfg.wb[0], cfg.wb[1], size=chans).astype(np.float32)

    # --- sensor noise and JPEG artefacts ---------------------------------
    x = x + rng.normal(0.0, _u(rng, cfg.noise_sigma), x.shape).astype(np.float32)
    x = np.clip(x, 0.0, 255.0)

    if rng.random() < cfg.jpeg_p:
        quality = int(rng.integers(int(cfg.jpeg_quality[0]), int(cfg.jpeg_quality[1])))
        x = _jpeg_roundtrip(x, quality)

    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0)
    x = x.astype(np.float32, copy=False)
    return x[..., 0] if squeeze_back else x


def augment(img: np.ndarray, rng: np.random.Generator, cfg=DEFAULT) -> np.ndarray:
    """:func:`augment_lcd` for either a plain AugmentConfig or a LumaGatedConfig."""
    if isinstance(cfg, LumaGatedConfig):
        cfg = cfg.pick(img)
    return augment_lcd(img, rng, cfg)


# --------------------------------------------------------------------------
# Keras ImageDataGenerator glue
# --------------------------------------------------------------------------


def make_idg_preprocessing_fn(cfg=DEFAULT, seed: int | None = None):
    """Return a single-argument closure for ``IDG(preprocessing_function=...)``.

    IDG calls this once per image with a float32 ``(32, 20, 3)`` array of RAW
    0-255 values, already geometrically transformed.

    RNG handling: one long-lived Generator is advanced across calls, so
    consecutive images get different draws (naively building
    ``default_rng(seed)`` inside the closure would hand every image the same
    degradation). The generator is re-derived per process id, so if Keras ever
    forks worker processes they do not all replay the same stream.
    """
    root = np.random.SeedSequence(seed)
    state: dict = {"rng": None, "pid": None}

    def _preprocess(img):
        pid = os.getpid()
        if state["rng"] is None or state["pid"] != pid:
            child = np.random.SeedSequence(entropy=root.entropy, spawn_key=(pid,))
            state["rng"] = np.random.default_rng(child)
            state["pid"] = pid
        return augment(img, state["rng"], cfg)

    _preprocess.cfg = cfg  # type: ignore[attr-defined]
    return _preprocess
