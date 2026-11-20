"""Polite, resumable parallel image downloader that also records photo quality."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from refaudit.quality import image_quality

HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset download; refaudit)"}


def fetch_image(
    url: str, dest: Path, timeout: float = 20.0, max_side: int = 768
) -> dict[str, float] | None:
    """Download one image, measure its quality on the raw file, downscale so the longest side is
    at most max_side, and save as JPEG (quality 95). Returns the quality dict, or None on failure.
    If dest already exists, quality is measured on the saved file instead."""
    if dest.exists():
        try:
            return image_quality(dest)
        except Exception:
            return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return None
        raw = Image.open(io.BytesIO(r.content))
        quality = image_quality(raw)
        img = raw.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        img.save(tmp, format="JPEG", quality=95)
        tmp.replace(dest)
        return quality
    except Exception:
        return None


def fetch_many(
    jobs: list[tuple[str, Path]], workers: int = 16, desc: str = "download"
) -> dict[Path, dict[str, float]]:
    """Download (url, dest) pairs in parallel. Returns {dest: quality} for successful ones."""
    ok: dict[Path, dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_image, url, dest): dest for url, dest in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            q = f.result()
            if q is not None:
                ok[futs[f]] = q
    return ok


def save_pil(img: Image.Image, dest: Path, max_side: int = 768) -> dict[str, float]:
    """Measure quality on an in-memory image, then save it like fetch_image does."""
    quality = image_quality(img)
    out = img.convert("RGB")
    out.thumbnail((max_side, max_side), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest, format="JPEG", quality=95)
    return quality
