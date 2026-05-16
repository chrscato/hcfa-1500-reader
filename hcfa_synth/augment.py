"""Image augmentation tiers — the difficulty spectrum.

Each tier transforms a clean rendered form into a sample that resembles
real-world scan/fax/photo conditions. All tiers are deterministic given
a seeded random.Random.

Tier ladder:
  pristine     — passthrough
  clean_scan   — modern office scanner
  worn_scan    — old MFP / multi-generation photocopy
  fax          — faxed claim (binary, streaks)
  phone_photo  — phone-snapped printout (perspective, lighting)
  worst        — stress test (crinkles, staples, stains, redactions)
"""

from __future__ import annotations

import io
import math
import random
from typing import Callable, Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


TIER_NAMES: List[str] = [
    "pristine",
    "clean_scan",
    "worn_scan",
    "fax",
    "phone_photo",
    "worst",
]


# --------------------------------------------------------------------------
# Primitive transforms
# --------------------------------------------------------------------------

def _to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _rotate(img: Image.Image, angle_deg: float, fill: int = 255) -> Image.Image:
    arr = _to_cv(img)
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        arr, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(fill, fill, fill),
        flags=cv2.INTER_LINEAR,
    )
    return _to_pil(rotated)


def _gaussian_noise(img: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    arr = np.asarray(img).astype(np.int16)
    noise = np_rng.normal(0, sigma, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _salt_pepper(img: Image.Image, rate: float, rng: random.Random) -> Image.Image:
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    n = int(h * w * rate)
    ys = np_rng.integers(0, h, n)
    xs = np_rng.integers(0, w, n)
    vals = np_rng.choice([0, 255], n)
    for y, x, v in zip(ys, xs, vals):
        arr[y, x] = v
    return Image.fromarray(arr)


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _fade(img: Image.Image, black_point: int) -> Image.Image:
    """Push darks toward gray — black ink → black_point value."""
    arr = np.asarray(img).astype(np.float32)
    arr = black_point + (arr / 255.0) * (255 - black_point)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _binarize(img: Image.Image, threshold: int) -> Image.Image:
    gray = np.asarray(img.convert("L"))
    binary = (gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary).convert("RGB")


def _vertical_streaks(img: Image.Image, n_streaks: int, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    for _ in range(n_streaks):
        x = rng.randint(0, w - 1)
        width = rng.randint(1, 3)
        darkness = rng.randint(0, 80)
        x1 = max(0, x - width // 2)
        x2 = min(w, x + width // 2 + 1)
        arr[:, x1:x2] = np.minimum(arr[:, x1:x2], darkness)
    return Image.fromarray(arr)


def _downsample_upsample(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((int(w * factor), int(h * factor)), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.NEAREST)


def _perspective(img: Image.Image, max_shift_frac: float, rng: random.Random) -> Image.Image:
    arr = _to_cv(img)
    h, w = arr.shape[:2]
    shift = lambda: rng.uniform(-max_shift_frac, max_shift_frac)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * shift(), h * shift()],
        [w * (1 + shift()), h * shift()],
        [w * (1 + shift()), h * (1 + shift())],
        [w * shift(), h * (1 + shift())],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        arr, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
        flags=cv2.INTER_LINEAR,
    )
    return _to_pil(warped)


def _lighting_gradient(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Multiplicative diagonal gradient — one corner brighter than another."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    angle = rng.uniform(0, 2 * math.pi)
    cx, cy = math.cos(angle), math.sin(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (cx * (xx / w) + cy * (yy / h))
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
    grad = 1.0 - strength + grad * strength * 2  # range [1-s, 1+s]
    arr = arr * grad[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _shadow(img: Image.Image, side: str, depth: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    if side in ("left", "right"):
        x = np.linspace(0, 1, w)
        if side == "right":
            x = 1 - x
        falloff = 1 - depth * (1 - x)
        falloff = falloff[None, :, None]
    else:
        y = np.linspace(0, 1, h)
        if side == "bottom":
            y = 1 - y
        falloff = 1 - depth * (1 - y)
        falloff = falloff[:, None, None]
    arr = arr * falloff
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _staple_holes(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    # Two holes near the top corners, common for stapled stacks
    radius = max(6, w // 200)
    margin = max(20, w // 50)
    spots = [
        (margin + rng.randint(-5, 5), margin + rng.randint(-5, 5)),
        (w - margin + rng.randint(-5, 5), margin + rng.randint(-5, 5)),
    ]
    for cx, cy in spots:
        cv2.circle(arr, (cx, cy), radius, (20, 20, 20), -1)
    return Image.fromarray(arr)


def _coffee_stain(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))
    radius = rng.randint(int(min(h, w) * 0.05), int(min(h, w) * 0.15))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1 - dist / radius, 0, 1) ** 2  # soft round mask
    # Brown tint: multiply blue/green channels down, leave red
    stain = mask * 0.35
    arr[..., 0] = arr[..., 0] * (1 - stain * 0.4)  # red dimmed slightly
    arr[..., 1] = arr[..., 1] * (1 - stain * 0.7)  # green more
    arr[..., 2] = arr[..., 2] * (1 - stain * 0.8)  # blue most
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _redaction_bars(img: Image.Image, n: int, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    for _ in range(n):
        bar_w = rng.randint(int(w * 0.05), int(w * 0.20))
        bar_h = rng.randint(8, 22)
        x = rng.randint(0, w - bar_w)
        y = rng.randint(int(h * 0.1), int(h * 0.9))
        arr[y:y + bar_h, x:x + bar_w] = 0
    return Image.fromarray(arr)


def _crinkle(img: Image.Image, amplitude: float, rng: random.Random) -> Image.Image:
    """Cheap wavy displacement — gives a 'rumpled paper' feel."""
    arr = _to_cv(img)
    h, w = arr.shape[:2]
    freq_x = rng.uniform(0.5, 2.0) / w * 6 * math.pi
    freq_y = rng.uniform(0.5, 2.0) / h * 6 * math.pi
    phase = rng.uniform(0, 2 * math.pi)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + amplitude * np.sin(ys * freq_y + phase)
    map_y = ys + amplitude * np.cos(xs * freq_x + phase)
    warped = cv2.remap(
        arr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return _to_pil(warped)


# --------------------------------------------------------------------------
# Tier compositions
# --------------------------------------------------------------------------

def _t_pristine(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.copy()


def _t_clean_scan(img: Image.Image, rng: random.Random) -> Image.Image:
    img = _rotate(img, rng.uniform(-1.0, 1.0))
    img = _gaussian_noise(img, sigma=2, rng=rng)
    img = _fade(img, black_point=10)
    return img


def _t_worn_scan(img: Image.Image, rng: random.Random) -> Image.Image:
    img = _rotate(img, rng.uniform(-3.0, 3.0))
    img = _gaussian_noise(img, sigma=5, rng=rng)
    img = _fade(img, black_point=40)
    img = _salt_pepper(img, rate=0.0008, rng=rng)
    img = _jpeg_compress(img, quality=60)
    return img


def _t_fax(img: Image.Image, rng: random.Random) -> Image.Image:
    img = _rotate(img, rng.uniform(-2.0, 2.0))
    img = _binarize(img, threshold=160)
    img = _vertical_streaks(img, n_streaks=rng.randint(6, 15), rng=rng)
    img = _downsample_upsample(img, factor=0.5)
    return img


def _t_phone_photo(img: Image.Image, rng: random.Random) -> Image.Image:
    img = _perspective(img, max_shift_frac=0.03, rng=rng)
    img = _rotate(img, rng.uniform(-1.5, 1.5))
    img = _lighting_gradient(img, strength=0.20, rng=rng)
    img = _shadow(img, side=rng.choice(["left", "right", "top", "bottom"]), depth=0.25)
    img = _blur(img, radius=0.8)
    img = _jpeg_compress(img, quality=80)
    return img


def _t_worst(img: Image.Image, rng: random.Random) -> Image.Image:
    img = _crinkle(img, amplitude=4.0, rng=rng)
    img = _rotate(img, rng.uniform(-5.0, 5.0))
    img = _gaussian_noise(img, sigma=6, rng=rng)
    img = _fade(img, black_point=50)
    img = _shadow(img, side=rng.choice(["left", "right"]), depth=0.30)
    img = _coffee_stain(img, rng=rng)
    img = _staple_holes(img, rng=rng)
    img = _redaction_bars(img, n=rng.randint(1, 3), rng=rng)
    img = _jpeg_compress(img, quality=50)
    return img


_TIERS: Dict[str, Callable[[Image.Image, random.Random], Image.Image]] = {
    "pristine": _t_pristine,
    "clean_scan": _t_clean_scan,
    "worn_scan": _t_worn_scan,
    "fax": _t_fax,
    "phone_photo": _t_phone_photo,
    "worst": _t_worst,
}


def apply_tier(img: Image.Image, tier: str, seed: int) -> Image.Image:
    """Apply a named tier transformation, deterministic with seed."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}; valid: {list(_TIERS)}")
    rng = random.Random(seed)
    return _TIERS[tier](img, rng)
