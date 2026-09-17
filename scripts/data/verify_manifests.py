"""Check built manifests against the dataset sizes reported in the paper (Table I).

Counts rows as the training code reads them and, with --check-audio, confirms that every referenced
audio file exists. Exits non-zero on any mismatch.

Usage:
    python scripts/data/verify_manifests.py --root data
    python scripts/data/verify_manifests.py --root data --check-audio
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from _common import DEFAULT_ROOT, read_manifest

# (manifest, expected rows, description). None = report only.
EXPECTED = [
    ("wavcaps/train.json", 140_750, "Phase 1  WavCaps"),
    ("autoacd/train.json", 82_268, "Phase 1  Auto-ACD"),
    ("audiocaps/train.json", 45_178, "Phase 1  AudioCaps"),
    ("clotho/train.json", 3_839, "Phase 1  Clotho"),
    ("macs/train.json", 3_930, "Phase 1  MACS"),
    ("audiocaps/train_expanded.json", 225_890, "Phase 2  AudioCaps"),
    ("clotho/train_expanded.json", 95_975, "Phase 2  Clotho"),
    ("audiocaps/test_multiref.json", 884, "Eval     AudioCaps test"),
    ("clotho/test_multiref.json", 1_045, "Eval     Clotho (official validation split)"),
]
# Evaluation references: (manifest, total references, sha256 of json.dumps([row["captions"], ...])).
EVAL_REFERENCES = [
    ("audiocaps/test_multiref.json", 4_411, None),
    ("clotho/test_multiref.json", 4_680, "b017b63fe53bf08444b914e2064304e97660a8c1cce5b0e7ae0b5db92fd6da30"),
]
TEXT_MANIFEST = "text_instructions/train.json"
TEXT_ROWS_USED = 50_000


def has_target(row: dict) -> bool:
    if "prompt" in row and "response" in row:
        return bool(row["response"])
    if "captions" in row:
        caps = row["captions"] if isinstance(row["captions"], list) else [row["captions"]]
        return bool(caps and caps[0])
    return bool(row.get("caption"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="data root (default: data)")
    ap.add_argument("--check-audio", action="store_true", help="also check that audio files exist")
    args = ap.parse_args()

    failures = 0
    for rel, expected, label in EXPECTED:
        path = args.root / rel
        if not path.exists():
            print(f"MISSING  {label:<52} {rel}")
            failures += 1
            continue
        rows = [r for r in read_manifest(path) if r.get("audio_path") and has_target(r)]
        status = "ok" if expected is None or len(rows) == expected else "MISMATCH"
        want = "" if expected is None else f" (paper: {expected:,})"
        print(f"{status:<8} {label:<52} {len(rows):>8,}{want}")
        failures += status == "MISMATCH"
        if args.check_audio:
            absent = [r["audio_path"] for r in rows if not Path(r["audio_path"]).exists()]
            if absent:
                print(f"         {len(absent):,} audio files missing, e.g. {absent[0]}")
                failures += 1

    for rel, n_refs, sha in EVAL_REFERENCES:
        path = args.root / rel
        if not path.exists():
            continue
        refs = [r["captions"] for r in read_manifest(path)]
        got_sha = hashlib.sha256(json.dumps(refs).encode()).hexdigest()
        ok = sum(map(len, refs)) == n_refs and (sha is None or got_sha == sha)
        print(f"{'ok' if ok else 'MISMATCH':<8} {'         references in ' + rel:<52} {sum(map(len, refs)):>8,} "
              f"(paper: {n_refs:,}{'' if sha is None else ', fingerprint ' + ('matches' if got_sha == sha else 'differs')})")
        failures += not ok

    path = args.root / TEXT_MANIFEST
    if not path.exists():
        print(f"MISSING  {'Phase 1  text-instruction mix':<52} {TEXT_MANIFEST}")
        failures += 1
    else:
        rows = read_manifest(path)[:TEXT_ROWS_USED]
        usable = sum(1 for r in rows if r.get("instruction") and r.get("response"))
        status = "ok" if usable == TEXT_ROWS_USED else "MISMATCH"
        print(f"{status:<8} {'Phase 1  text-instruction mix (first 50,000 rows)':<52} "
              f"{usable:>8,} (paper: {TEXT_ROWS_USED:,})")
        failures += status == "MISMATCH"

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
