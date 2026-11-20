"""Polite, resumable parallel image downloader."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset download; refaudit)"}


def fetch_image(url: str, dest: Path, timeout: float = 20.0, max_side: int = 768) -> bool:
    """Download one image, verify it decodes, downscale so the longest side <= max_side, save as
    JPEG (quality 95). Returns True on success. Skips if dest already exists."""
    if dest.exists():
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return False
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        img.save(tmp, format="JPEG", quality=95)
        tmp.replace(dest)
        return True
    except Exception:
        return False


def fetch_many(
    jobs: list[tuple[str, Path]], workers: int = 16, desc: str = "download"
) -> set[Path]:
    """Download (url, dest) pairs in parallel. Returns the set of dests that now exist."""
    ok: set[Path] = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_image, url, dest): dest for url, dest in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            if f.result():
                ok.add(futs[f])
    return ok
