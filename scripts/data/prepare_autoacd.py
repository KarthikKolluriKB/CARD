"""Build the Auto-ACD Phase 1 manifest on top of the WavCaps AudioSet_SL audio.

Auto-ACD provides GPT-generated captions for AudioSet clips. No new audio is downloaded: its captions
are attached to the AudioSet_SL clips already extracted by ``prepare_wavcaps.py``, one row per
(clip, caption). Captions: ``train.csv`` of Hugging Face ``Loie/Auto-ACD`` (pinned revision).

A clip is matched through the 11-character YouTube id in its WavCaps file name (``Y<id>.flac``),
looked up as ``<id>``, ``Y<id>``, then ``<id>`` without leading dashes. With the full AudioSet_SL
partition this gives 82,268 rows.

Output: ``<root>/autoacd/train.json`` with {"audio_path", "caption"}.

Usage:
    python scripts/data/prepare_autoacd.py --root data
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from _common import DEFAULT_ROOT, HF_REVISIONS, write_manifest

REPO_ID = "Loie/Auto-ACD"
FILENAME = "train.csv"


def load_captions(csv_path: Path) -> dict:
    captions = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("youtube_id") or "").strip()
            caption = (row.get("caption") or "").strip()
            if key and caption:
                captions.setdefault(key, []).append(caption)
    return captions


def youtube_id(audio_path: str):
    """``Y--cB2ZVjpnA.flac`` / ``--cB2ZVjpnA_30.000.flac`` -> ``--cB2ZVjpnA``; None if not parsable."""
    name = Path(audio_path).stem
    if name.startswith("Y"):
        name = name[1:]
    name = re.sub(r"_[\d.]+$", "", name)
    m = re.match(r"^([\w\-]{11})", name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="data root (default: data)")
    args = ap.parse_args()

    import json

    from huggingface_hub import hf_hub_download

    wavcaps_manifest = args.root / "wavcaps" / "train.json"
    if not wavcaps_manifest.exists():
        raise SystemExit(f"{wavcaps_manifest} not found; run prepare_wavcaps.py first")

    out_dir = args.root / "autoacd"
    csv_path = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset",
                               revision=HF_REVISIONS[REPO_ID], local_dir=str(out_dir / "raw"))
    captions = load_captions(Path(csv_path))
    print(f"  Auto-ACD: {len(captions):,} ids, {sum(map(len, captions.values())):,} captions")

    with open(wavcaps_manifest, encoding="utf-8") as f:
        wavcaps = json.load(f)
    base = wavcaps_manifest.parent.resolve()
    # Paths are stored relative to wavcaps/, e.g. "audio/AudioSet_SL/...".
    audioset = [r for r in wavcaps if "audioset" in r["audio_path"].lower()]

    rows, matched = [], 0
    for r in audioset:
        yid = youtube_id(r["audio_path"])
        if yid is None:
            continue
        caps = captions.get(yid) or captions.get(f"Y{yid}") or captions.get(yid.lstrip("-"))
        if not caps:
            continue
        matched += 1
        audio_path = str(base / r["audio_path"])
        rows.extend({"audio_path": audio_path, "caption": c} for c in caps)
    print(f"  matched {matched:,}/{len(audioset):,} AudioSet_SL clips")
    write_manifest(rows, out_dir / "train.json")


if __name__ == "__main__":
    main()
