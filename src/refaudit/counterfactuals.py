"""Capture-condition counterfactuals.

Each factor changes ONE aspect of how a photo was taken, leaving the skin condition itself in
place. Strengths are graded 1..5 and kept within what ordinary phone cameras produce, so the true
label should not change. These are "capture-condition changes", not "skin-tone changes": in photos,
tone and lighting are entangled (see arXiv 2602.19055), so we never claim to change tone itself.

All functions take and return RGB PIL images and are deterministic.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

from refaudit.groups import skin_mask

SEVERITIES = (1, 2, 3, 4, 5)


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")


def underexpose(img: Image.Image, severity: int) -> Image.Image:
    """Dim lighting: scale exposure down (in linear light) by up to ~2.3 stops."""
    stops = [0.25, 0.5, 1.0, 1.6, 2.3][severity - 1]
    arr = _to_array(img)
    lin = arr**2.2 * (2.0**-stops)
    return _to_image(lin ** (1 / 2.2))


def overexpose(img: Image.Image, severity: int) -> Image.Image:
    """Harsh light / flash: scale exposure up, with highlight clipping."""
    stops = [0.25, 0.5, 0.8, 1.2, 1.6][severity - 1]
    arr = _to_array(img)
    lin = arr**2.2 * (2.0**stops)
    return _to_image(lin ** (1 / 2.2))


def white_balance(img: Image.Image, severity: int, warm: bool = True) -> Image.Image:
    """Colour cast from indoor (warm, tungsten) or shade (cool) light.

    Applied as per-channel gains in linear light, approximating a white-balance error of
    roughly 500 K to 3000 K. Overall brightness is preserved.
    """
    shift = [0.04, 0.08, 0.13, 0.19, 0.26][severity - 1]
    r, b = (1 + shift, 1 - shift) if warm else (1 - shift, 1 + shift)
    gains = np.array([r, 1.0, b])
    gains = gains / gains.mean()
    lin = _to_array(img) ** 2.2 * gains
    return _to_image(lin ** (1 / 2.2))


def cool_white_balance(img: Image.Image, severity: int) -> Image.Image:
    return white_balance(img, severity, warm=False)


def shadow(img: Image.Image, severity: int) -> Image.Image:
    """Uneven lighting: a smooth left-to-right shadow (e.g. a phone or hand casting shade)."""
    depth = [0.15, 0.3, 0.45, 0.6, 0.75][severity - 1]
    arr = _to_array(img)
    w = arr.shape[1]
    ramp = np.linspace(1.0 - depth, 1.0, w, dtype=np.float32)[None, :, None]
    lin = arr**2.2 * ramp
    return _to_image(lin ** (1 / 2.2))


def blur(img: Image.Image, severity: int) -> Image.Image:
    """Out of focus / hand shake: Gaussian blur, radius scaled to image size."""
    frac = [0.002, 0.004, 0.007, 0.011, 0.016][severity - 1]
    radius = max(0.5, frac * max(img.size))
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius))


def jpeg(img: Image.Image, severity: int) -> Image.Image:
    """Messaging-app compression: re-encode as JPEG at decreasing quality."""
    quality = [70, 50, 30, 18, 10][severity - 1]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def low_resolution(img: Image.Image, severity: int) -> Image.Image:
    """Old camera / photo taken from far away: downscale then upscale back."""
    factor = [0.7, 0.5, 0.35, 0.25, 0.17][severity - 1]
    w, h = img.size
    small = img.convert("RGB").resize((max(8, int(w * factor)), max(8, int(h * factor))),
                                      Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def background_swap(
    img: Image.Image, severity: int, mask: np.ndarray, rng_seed: int = 0
) -> Image.Image:
    """Replace the non-skin background, keeping masked pixels (lesion + skin) untouched.

    `mask` is a boolean HxW array, True where the pixel must be kept. Severity controls how
    busy the replacement is: 1 = plain grey, 5 = high-contrast random clutter.
    """
    arr = _to_array(img)
    if mask.shape != arr.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {arr.shape[:2]}")
    rng = np.random.default_rng(rng_seed)
    h, w = mask.shape
    base = np.full((h, w, 3), 0.55, dtype=np.float32)
    if severity > 1:
        blocks = 2 ** (severity + 1)
        noise = rng.random((blocks, blocks, 3), dtype=np.float32)
        noise_img = Image.fromarray((noise * 255).astype(np.uint8), "RGB").resize((w, h),
                                                                                   Image.NEAREST)
        amount = [0, 0.3, 0.55, 0.8, 1.0][severity - 1]
        base = (1 - amount) * base + amount * (np.asarray(noise_img, dtype=np.float32) / 255.0)
    out = np.where(mask[..., None], arr, base)
    return _to_image(out)


def keep_mask_from_skin(img: Image.Image) -> np.ndarray:
    """Pixels to keep for a background swap: detected skin, smoothed, holes (e.g. the lesion)
    filled and slightly dilated so the lesion border and nearby skin are never touched."""
    m = skin_mask(np.asarray(img.convert("RGB")))
    m = binary_closing(m, iterations=3)
    m = binary_fill_holes(m)
    return binary_dilation(m, iterations=4)


def background_auto(img: Image.Image, severity: int) -> Image.Image:
    """Background swap using an automatic skin mask (skin + lesion kept, everything else
    replaced). If almost everything is skin (a tight close-up) the image is returned unchanged."""
    mask = keep_mask_from_skin(img)
    if mask.mean() > 0.97:
        return img.convert("RGB")
    return background_swap(img, severity, mask)


FACTORS: dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "underexpose": underexpose,
    "overexpose": overexpose,
    "warm_wb": white_balance,
    "cool_wb": cool_white_balance,
    "shadow": shadow,
    "blur": blur,
    "jpeg": jpeg,
    "low_res": low_resolution,
    "background": background_auto,
}
"""All capture factors. `background` uses an automatic skin mask; for datasets without people
(e.g. Waterbirds) pass an explicit mask to `background_swap` instead or leave it out."""


def apply(img: Image.Image, factor: str, severity: int) -> Image.Image:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    if factor not in FACTORS:
        raise KeyError(f"unknown factor {factor!r}; choose from {sorted(FACTORS)}")
    return FACTORS[factor](img, severity)


def random_augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Baseline: generic random test-time augmentation (flip + small crop + small colour jitter).

    This is the 'random TTA' baseline our attribute-specific probe must beat.
    """
    out = img.convert("RGB")
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    w, h = out.size
    s = rng.uniform(0.85, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    out = out.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
    arr = _to_array(out) * rng.uniform(0.9, 1.1)
    return _to_image(arr)


def rotate(img: Image.Image, degrees: float) -> Image.Image:
    """Baseline: ASRS-style rotation (arXiv 2510.01683 uses +-15 and +-30 degrees)."""
    return img.convert("RGB").rotate(degrees, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
