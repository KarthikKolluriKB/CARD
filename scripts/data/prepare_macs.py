"""Download MACS and build its Phase 1 manifest.

MACS (Zenodo record 5114771) ships captions only, for 3,930 clips of the TAU Urban Acoustic
Scenes 2019 development set. The TAU audio archives (Zenodo record 2589280, about 36 GB) are
downloaded and only the MACS clips are extracted.

Output under ``<root>/macs/``:

    MACS.yaml, LICENSE.txt    annotations and licence
    tau2019/                  TAU archives (delete with --delete-archives once extracted)
    audio/                    the 3,930 MACS clips
    train.json                {"audio_path", "caption"}, first annotator sentence per clip, YAML order

Usage:
    python scripts/data/prepare_macs.py --root data
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from _common import DEFAULT_ROOT, download_url, write_manifest

MACS_RECORD = "5114771"
TAU_RECORD = "2589280"


def zenodo_files(record: str) -> list:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as r:
        files = json.load(r)["files"]
    return [{"name": f["key"], "size": f["size"], "url": f["links"]["self"]} for f in files]


def parse_macs(yaml_path: Path) -> list:
    import yaml

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    records = data.get("files", data) if isinstance(data, dict) else data
    out = []
    for r in records:
        filename = r.get("filename") or r.get("file")
        if not filename:
            continue
        sentences = []
        for ann in r.get("annotations") or []:
            s = ann.get("sentence") if isinstance(ann, dict) else str(ann)
            if s:
                sentences.append(s.strip())
        if sentences:
            out.append({"filename": filename, "captions": sentences})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="data root (default: data)")
    ap.add_argument("--delete-archives", action="store_true", help="remove the TAU zips after extraction")
    args = ap.parse_args()

    out_dir = args.root / "macs"
    for f in zenodo_files(MACS_RECORD):
        if f["name"] in ("MACS.yaml", "LICENSE.txt"):
            download_url(f["url"], out_dir / f["name"], f["size"])
    records = parse_macs(out_dir / "MACS.yaml")
    wanted = {r["filename"] for r in records}
    print(f"  MACS: {len(records):,} captioned clips")

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tau_dir = out_dir / "tau2019"
    archives = [f for f in zenodo_files(TAU_RECORD) if f["name"].endswith(".zip") and "audio" in f["name"].lower()]
    print(f"  TAU2019: {len(archives)} audio archives, {sum(f['size'] for f in archives) / 1e9:.1f} GB")
    for f in archives:
        zip_path = download_url(f["url"], tau_dir / f["name"], f["size"])
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name in wanted and not (audio_dir / name).exists():
                    with zf.open(member) as src, open(audio_dir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        if args.delete_archives:
            zip_path.unlink()

    rows, missing = [], 0
    for r in records:
        path = audio_dir / r["filename"]
        if not path.exists():
            missing += 1
            continue
        rows.append({"audio_path": str(path), "caption": r["captions"][0]})
    print(f"  matched {len(rows):,}, missing {missing:,}")
    write_manifest(rows, out_dir / "train.json")


if __name__ == "__main__":
    main()
