"""Download WavCaps and build its Phase 1 manifest.

Source: Hugging Face ``cvssp/WavCaps`` (pinned revision): caption JSONs plus multi-volume zip archives.
CARD uses three of the four partitions (FreeSound is not used):

    AudioSet_SL         108,317 clips  (~36 GB)
    SoundBible            1,232 clips  (~0.6 GB)
    BBC_Sound_Effects    31,201 clips  (~130 GB)
                        -------
                        140,750 rows in wavcaps/train.json

Extracting the multi-volume archives needs the ``7z`` command line tool (p7zip).

Output under ``<root>/wavcaps/``:

    raw/                 files downloaded from the Hub
    audio/<partition>/   extracted audio
    train.json           {"audio_path", "caption"}, partitions in the order above, JSON order within each

Usage:
    python scripts/data/prepare_wavcaps.py --root data
    python scripts/data/prepare_wavcaps.py --root data --skip-download --skip-extract   # rebuild the manifest
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from _common import DEFAULT_ROOT, HF_REVISIONS, write_manifest

REPO_ID = "cvssp/WavCaps"
PARTITION_JSON = {
    "AudioSet_SL": "json_files/AudioSet_SL/as_final.json",
    "SoundBible": "json_files/SoundBible/sb_final.json",
    "BBC_Sound_Effects": "json_files/BBC_Sound_Effects/bbc_final.json",
    "FreeSound": "json_files/FreeSound/fsd_final.json",
}
DEFAULT_PARTITIONS = ["AudioSet_SL", "SoundBible", "BBC_Sound_Effects"]
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg"}
ID_KEYS = ["id", "audio_id", "file_name", "filename", "audiocap_id", "audiocaps_id", "youtube_id"]


def download(raw_dir: Path, partitions: list) -> None:
    from huggingface_hub import snapshot_download

    patterns = [f"json_files/{p}/*" for p in partitions] + [f"Zip_files/{p}/*" for p in partitions]
    snapshot_download(REPO_ID, repo_type="dataset", revision=HF_REVISIONS[REPO_ID],
                      allow_patterns=patterns, local_dir=str(raw_dir))


def extract(raw_dir: Path, audio_dir: Path, partition: str) -> None:
    """Extract every master ``.zip`` of a partition (7z resolves the ``.z01``, ``.z02`` parts)."""
    target = audio_dir / partition
    target.mkdir(parents=True, exist_ok=True)
    for master in sorted((raw_dir / "Zip_files" / partition).glob("*.zip")):
        marker = target / f".{master.stem}.done"
        if marker.exists():
            print(f"  {master.name}: already extracted")
            continue
        multi_volume = any(master.parent.glob(f"{master.stem}.z[0-9]*"))
        seven_zip = shutil.which("7z") or shutil.which("7zz")
        print(f"  extracting {master.name} -> {target}")
        if seven_zip:
            subprocess.run([seven_zip, "x", str(master), f"-o{target}", "-y"], check=True)
        elif multi_volume:
            raise RuntimeError(f"{master.name} is a multi-volume archive; install 7z (p7zip) to extract it")
        else:
            with zipfile.ZipFile(master) as zf:
                zf.extractall(target)
        marker.touch()


def index_audio(root: Path) -> dict:
    """Map both file stem and file name to the path (first file found wins)."""
    index = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            index.setdefault(p.stem, p)
            index.setdefault(p.name, p)
    return index


def load_records(json_path: Path) -> list:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["data"] if isinstance(data, dict) and "data" in data else data


def match_records(records: list, index: dict) -> tuple[list, int]:
    rows, missing = [], 0
    for r in records:
        caption = (r.get("caption") or r.get("Caption") or r.get("description") or "").strip()
        if not caption:
            continue
        audio_path = None
        for key in ID_KEYS:
            if not r.get(key):
                continue
            raw = str(r[key])
            for candidate in (raw, raw.split(".")[0], Path(raw).stem, Path(raw).name):
                if candidate in index:
                    audio_path = index[candidate]
                    break
            if audio_path:
                break
        if audio_path is None:
            missing += 1
            continue
        rows.append({"audio_path": str(audio_path), "caption": caption})
    return rows, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="data root (default: data)")
    ap.add_argument("--partitions", nargs="+", default=DEFAULT_PARTITIONS, choices=list(PARTITION_JSON))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = args.root / "wavcaps"
    raw_dir, audio_dir = out_dir / "raw", out_dir / "audio"
    partitions = [p for p in PARTITION_JSON if p in args.partitions]  # fixed order

    if not args.skip_download:
        print(f"Downloading {REPO_ID}@{HF_REVISIONS[REPO_ID][:7]}: {', '.join(partitions)}")
        download(raw_dir, partitions)
    if not args.skip_extract:
        for p in partitions:
            extract(raw_dir, audio_dir, p)

    rows = []
    for p in partitions:
        records = load_records(raw_dir / PARTITION_JSON[p])
        matched, missing = match_records(records, index_audio(audio_dir / p))
        print(f"  {p:<18} records={len(records):,} matched={len(matched):,} missing={missing:,}")
        rows.extend(matched)
    write_manifest(rows, out_dir / "train.json")


if __name__ == "__main__":
    main()
