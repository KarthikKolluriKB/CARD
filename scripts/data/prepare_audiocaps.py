"""Download AudioCaps and build its manifests.

Source: Hugging Face ``OpenSound/AudioCaps`` (about 44 GB with audio), pinned revision.
Every row of the dataset is one (clip, caption) pair and is written to its own WAV file.

Output under ``<root>/audiocaps/``:

    train.json             45,178 rows  {"audio_path", "caption"}             Phase 1
    train_expanded.json   225,890 rows  {"audio_path", "prompt", "response"}  Phase 2
    validation.json                     {"audio_path", "caption"}             validation during training
    test.json               4,411 rows  {"audio_path", "caption"}             one row per reference
    test_multiref.json        884 rows  {"audio_path", "captions": [...]}     evaluation

The evaluation manifest groups test rows whose audio files are byte-identical, in order of first
appearance, and keeps the last row's file for each group. This is the protocol used for the paper.
On the Hub one of the five rows of clip 473wBEwC35M carries a different waveform, so that clip
appears twice (4 + 1 references); clip PWjEfOkb6ro has a single row.

Phase 2 applies five prompt templates to every training caption and shuffles with seed 42.

Usage:
    python scripts/data/prepare_audiocaps.py --root data
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import OrderedDict
from pathlib import Path

from _common import DEFAULT_ROOT, HF_REVISIONS, write_manifest

REPO_ID = "OpenSound/AudioCaps"

# Phase 2 prompt templates for AudioCaps (the response is always the original caption).
PROMPT_TEMPLATES = [
    "Describe this audio.",
    "What sounds can you hear in this audio?",
    "List the sound events in this audio clip.",
    "Describe what is happening in this audio clip.",
    "Summarize the audio.",
]
EXPAND_SEED = 42


def export_split(split_name: str, split_data, out_dir: Path) -> list:
    """Write one WAV per row and return {"audio_path", "caption"} rows."""
    import soundfile as sf

    audio_dir = out_dir / split_name / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, item in enumerate(split_data):
        audio = item.get("audio")
        if audio is None:
            continue
        audio_path = audio_dir / f"{split_name}_{i:06d}.wav"
        if not audio_path.exists():
            sf.write(str(audio_path), audio["array"], audio["sampling_rate"])
        caption = item.get("caption", item.get("text", ""))
        if not caption:
            continue
        rows.append({"audio_path": str(audio_path), "caption": caption})
        if (i + 1) % 5000 == 0:
            print(f"    {split_name}: {i + 1:,}/{len(split_data):,}")
    return rows


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def group_references(rows: list, audio_key=file_md5) -> list:
    """Group rows with identical audio (first appearance order); the last row's file is kept."""
    groups: "OrderedDict[str, dict]" = OrderedDict()
    for row in rows:
        g = groups.setdefault(audio_key(row["audio_path"]), {"audio_path": None, "captions": []})
        g["audio_path"] = row["audio_path"]
        g["captions"].append(row["caption"])
    return list(groups.values())


def expand_prompts(rows: list, seed: int = EXPAND_SEED) -> list:
    expanded = []
    for row in rows:
        caption = row.get("caption")
        if not caption or not isinstance(caption, str):
            continue
        for prompt in PROMPT_TEMPLATES:
            expanded.append({"audio_path": row["audio_path"], "prompt": prompt,
                             "response": caption.strip()})
    random.Random(seed).shuffle(expanded)
    return expanded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="data root (default: data)")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = args.root / "audiocaps"
    print(f"Loading {REPO_ID}@{HF_REVISIONS[REPO_ID][:7]}")
    ds = load_dataset(REPO_ID, revision=HF_REVISIONS[REPO_ID])

    rows_by_split = {}
    for split_name, split_data in ds.items():
        print(f"  split {split_name}: {len(split_data):,} rows")
        rows = export_split(split_name, split_data, out_dir)
        rows_by_split[split_name] = rows
        write_manifest(rows, out_dir / f"{split_name}.json")
        if split_name == "test":
            write_manifest(group_references(rows), out_dir / "test_multiref.json")

    write_manifest(expand_prompts(rows_by_split["train"]), out_dir / "train_expanded.json")


if __name__ == "__main__":
    main()
