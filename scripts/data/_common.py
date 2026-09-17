"""Helpers shared by the data preparation scripts in this folder."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Iterable, List

# Every Hugging Face source is pinned to the revision used to build the paper's manifests.
HF_REVISIONS = {
    "OpenSound/AudioCaps": "b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded",
    "CLAPv2/Clotho": "b491ad6569dba180ca60a0e2d17a1d6a0d5d9f4a",
    "cvssp/WavCaps": "0930ec11ded28fa0eaa910fde2f6fc3538acbeac",
    "Loie/Auto-ACD": "3a3a73497c14b4ed6cd0bf44c9d75cfe3c153763",
    "BAAI/Infinity-Instruct": "bddc39a8feadbd679c30623197f4e736b7e75b48",
}

DEFAULT_ROOT = Path("data")


def write_manifest(rows: Iterable[dict], path: Path) -> int:
    """Write a JSON-list manifest; ``audio_path`` values become relative to the manifest folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent.resolve()
    out: List[dict] = []
    for row in rows:
        row = dict(row)
        if "audio_path" in row:
            rel = os.path.relpath(Path(row["audio_path"]).resolve(), base)
            row["audio_path"] = Path(rel).as_posix()
        out.append(row)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {len(out):,} rows -> {path}")
    return len(out)


def read_manifest(path: Path) -> List[dict]:
    """Read a manifest and return rows with ``audio_path`` resolved to an absolute path."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    base = path.parent.resolve()
    for row in rows:
        if "audio_path" in row and not Path(row["audio_path"]).is_absolute():
            row["audio_path"] = str(base / row["audio_path"])
    return rows


def download_url(url: str, dest: Path, expected_size: int | None = None) -> Path:
    """Download ``url`` to ``dest`` unless a complete copy is already there."""
    dest = Path(dest)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  have {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  downloading {dest.name}")
    urllib.request.urlretrieve(url, tmp)
    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise RuntimeError(f"{dest.name}: got {tmp.stat().st_size} bytes, expected {expected_size}")
    tmp.replace(dest)
    return dest
