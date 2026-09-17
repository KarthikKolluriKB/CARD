"""Build the text-only instruction manifest interleaved during Phase 1.

Source: the ``Gen`` subset of Hugging Face ``BAAI/Infinity-Instruct`` (pinned revision).
90,000 conversations are sampled with seed 42; each becomes one (instruction, response) pair from
its first human and first assistant turn. Pairs with fewer than 3 words on either side are dropped
and responses are cut to 512 words. Phase 1 reads the first 50,000 rows (``num_text_clips``).

Output: ``<root>/text_instructions/train.json`` with {"instruction", "response"}.

Usage:
    python scripts/data/prepare_text_instructions.py --root data
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from _common import DEFAULT_ROOT, HF_REVISIONS, write_manifest

REPO_ID = "BAAI/Infinity-Instruct"
SUBSET = "Gen"


def first_turns(conversations: list) -> tuple[str, str]:
    instruction = response = ""
    for turn in conversations:
        role, value = turn.get("from", ""), turn.get("value", "")
        if role == "human" and not instruction:
            instruction = value
        elif role == "gpt" and not response:
            response = value
        if instruction and response:
            break
    return instruction, response


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="data root (default: data)")
    ap.add_argument("--num-samples", type=int, default=90_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"Loading {REPO_ID}@{HF_REVISIONS[REPO_ID][:7]} ({SUBSET})")
    ds = load_dataset(REPO_ID, SUBSET, split="train", revision=HF_REVISIONS[REPO_ID])
    n = min(args.num_samples, len(ds))
    indices = sorted(random.Random(args.seed).sample(range(len(ds)), n))
    print(f"  sampling {n:,} of {len(ds):,} (seed {args.seed})")

    rows, skipped = [], 0
    for idx in indices:
        instruction, response = first_turns(ds[idx].get("conversations", []))
        if len(instruction.split()) < 3 or len(response.split()) < 3:
            skipped += 1
            continue
        if len(response.split()) > 512:
            response = " ".join(response.split()[:512])
        rows.append({"instruction": instruction.strip(), "response": response.strip()})
    print(f"  kept {len(rows):,}, skipped {skipped:,} short pairs")
    write_manifest(rows, args.root / "text_instructions" / "train.json")


if __name__ == "__main__":
    main()
