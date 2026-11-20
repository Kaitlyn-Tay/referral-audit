"""Skin-tone grouping helpers.

Three sources of tone information are supported:
- Fitzpatrick skin type (FST, 1-6), as labelled in Fitzpatrick17k and SCIN.
- Monk Skin Tone (MST, 1-10), as labelled in SCIN.
- Individual Typology Angle (ITA, degrees), measured from pixels. Used where no label exists
  (e.g. wound images). Measure it on the ORIGINAL image only: lighting changes shift ITA.
"""

from __future__ import annotations

import numpy as np

FST_BINS = {1: "light", 2: "light", 3: "medium", 4: "medium", 5: "dark", 6: "dark"}
MST_BINS = {
    1: "light", 2: "light", 3: "light",
    4: "medium", 5: "medium", 6: "medium",
    7: "dark", 8: "dark", 9: "dark", 10: "dark",
}
GROUP_ORDER = ["light", "medium", "dark"]


def fst_group(fst: int | float | None) -> str | None:
    """Map a Fitzpatrick type to light / medium / dark. Invalid or missing values give None."""
    if fst is None or (isinstance(fst, float) and np.isnan(fst)):
        return None
    return FST_BINS.get(int(fst))


def mst_group(mst: int | float | None) -> str | None:
    """Map a Monk Skin Tone value to light / medium / dark."""
    if mst is None or (isinstance(mst, float) and np.isnan(mst)):
        return None
    return MST_BINS.get(int(mst))


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB array (..., 3) in [0, 255] to CIELAB (D65)."""
    c = _srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255.0)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = c @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lightness = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def ita_degrees(rgb_pixels: np.ndarray) -> float:
    """Individual Typology Angle from skin pixels: arctan((L* - 50) / b*) in degrees.

    Higher ITA = lighter skin. Common cut-offs (Chardon et al. 1991):
    > 55 very light, 41-55 light, 28-41 intermediate, 10-28 tan, -30-10 brown, < -30 dark.
    Uses the median L* and b* over the given pixels for robustness.
    """
    px = np.asarray(rgb_pixels).reshape(-1, 3)
    if px.size == 0:
        raise ValueError("no pixels given")
    lab = rgb_to_lab(px)
    lightness = float(np.median(lab[:, 0]))
    b = float(np.median(lab[:, 2]))
    return float(np.degrees(np.arctan2(lightness - 50.0, b)))


def ita_group(ita: float) -> str:
    """Three-way ITA grouping aligned roughly with the FST light / medium / dark bins."""
    if ita > 41:
        return "light"
    if ita > 10:
        return "medium"
    return "dark"


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Rough skin-pixel mask from the classic YCrCb rule (Cr 133-173, Cb 77-127), excluding
    near-black and near-white pixels. Crude, and itself possibly tone-biased, which is exactly
    why ITA-based groups are checked against the true labels before being trusted."""
    rgb = np.asarray(rgb, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y > 25) & (y < 245)


def ita_from_image(rgb: np.ndarray, min_skin_frac: float = 0.02) -> float:
    """ITA over detected skin pixels, NaN if too little skin is found. Because the lesion is often
    inside the mask, this measures 'skin as captured', not constitutive skin tone."""
    rgb = np.asarray(rgb)
    m = skin_mask(rgb)
    if m.mean() < min_skin_frac:
        return float("nan")
    return ita_degrees(rgb[m])
