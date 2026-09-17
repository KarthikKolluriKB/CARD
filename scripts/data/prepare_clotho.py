"""Download Clotho v2.1 and build its manifests.

Source: Hugging Face ``CLAPv2/Clotho`` (pinned revision). That copy stores the five captions of a
clip joined into one ``text`` field.

Training uses the ``train`` split (the official Clotho development split). Its five captions are
restored from the official caption CSV (Zenodo record 4783391) by exact match on the joined text;
every row must match.

Evaluation follows the protocol used for the paper: the Hugging Face ``test`` split, which is the
official Clotho *validation* split (1,045 clips), with references obtained by splitting the joined
text on periods. For most clips this gives the five captions; where a caption has no final period,
two captions stay joined, so 442 clips have fewer than five references (4,680 in total).

Output under ``<root>/clotho/``:

    train.json             3,839 rows  {"audio_path", "captions": [5]}        Phase 1 (first caption)
    train_expanded.json   95,975 rows  {"audio_path", "prompt", "response"}   Phase 2 (5 captions x 5 prompts)
    test_multiref.json     1,045 rows  {"audio_path", "captions": [...]}      evaluation (paper protocol)

Usage:
    python scripts/data/prepare_clotho.py --root data
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _common import DEFAULT_ROOT, HF_REVISIONS, download_url, write_manifest

REPO_ID = "CLAPv2/Clotho"
DEVELOPMENT_CAPTIONS_URL = "https://zenodo.org/records/4783391/files/clotho_captions_development.csv?download=1"

# Phase 2 prompt templates for Clotho (the response is always one of the clip's captions).
PROMPT_TEMPLATES = [
    "Describe this audio.",
    "What sounds can you hear?",
    "List the sound events in this audio clip.",
    "Describe what is happening in this audio clip.",
    "Summarize the audio.",
]


def load_official_captions(csv_path: Path) -> dict:
    """Map the space-joined five captions to the list of captions, in CSV column order."""
    by_text = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            caps = [row[f"caption_{i}"] for i in range(1, 6)]
            key = " ".join(caps)
            if key in by_text:
                raise ValueError(f"{csv_path.name}: duplicate caption set for {row['file_name']}")
            by_text[key] = [c.strip() for c in caps if c.strip()]
    return by_text


def split_references(text: str) -> list:
    """Evaluation references: the joined text split on periods (whole text if there is one piece)."""
    parts = [s.strip() for s in text.split(".") if s.strip()]
    return parts if len(parts) > 1 else [text.strip()]


def export_audio(item, split: str, index: int, out_dir: Path) -> str:
    import soundfile as sf

    audio_dir = out_dir / split / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{split}_{index:05d}.wav"
    if not audio_path.exists():
        audio = item["audio"]
        sf.write(str(audio_path), audio["array"], audio["sampling_rate"])
    return str(audio_path)


def expand_prompts(rows: list) -> list:
    return [
        {"audio_path": row["audio_path"], "prompt": prompt, "response": caption}
        for row in rows
        for caption in row["captions"]
        for prompt in PROMPT_TEMPLATES
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="data root (default: data)")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = args.root / "clotho"
    print(f"Loading {REPO_ID}@{HF_REVISIONS[REPO_ID][:7]}")
    ds = load_dataset(REPO_ID, revision=HF_REVISIONS[REPO_ID])

    csv_path = download_url(DEVELOPMENT_CAPTIONS_URL, out_dir / "clotho_captions_development.csv")
    by_text = load_official_captions(csv_path)
    train = []
    for i, item in enumerate(ds["train"]):
        captions = by_text.get(item["text"])
        if captions is None:
            raise ValueError(f"train row {i} has no exact caption match in {csv_path.name}")
        train.append({"audio_path": export_audio(item, "train", i, out_dir), "captions": captions})
    if len(train) != len(by_text):
        raise ValueError(f"matched {len(train)} training clips, the CSV lists {len(by_text)}")
    write_manifest(train, out_dir / "train.json")
    write_manifest(expand_prompts(train), out_dir / "train_expanded.json")

    test = [{"audio_path": export_audio(item, "test", i, out_dir), "captions": split_references(item["text"])}
            for i, item in enumerate(ds["test"])]
    write_manifest(test, out_dir / "test_multiref.json")


if __name__ == "__main__":
    main()
