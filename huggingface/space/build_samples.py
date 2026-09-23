#!/usr/bin/env python
"""Pick sample clips for the demo Space and write huggingface/space/samples/ (audio + samples.json).

Run on the machine that holds the test audio. Two steps:

1. List good candidates from an evaluation run (ranked by word overlap between the model's
   caption and the closest human reference):

    python huggingface/space/build_samples.py \\
        --manifest /mnt/storage/datasets/kk/audiocaps/test_multiref.json \\
        --eval-results outputs/aora_split_stage_2_2_r_16_a_16/merged/eval_results.json \\
        --suggest 20

2. Copy the chosen clips and write samples.json:

    python huggingface/space/build_samples.py \\
        --manifest /mnt/storage/datasets/kk/audiocaps/test_multiref.json \\
        --eval-results outputs/aora_split_stage_2_2_r_16_a_16/merged/eval_results.json \\
        --pick <clip1> <clip2> <clip3> --names "Dog and traffic" "Rain" "Train horn" \\
        --dataset "AudioCaps test set" --source "YouTube (AudioSet)" --license "see the original video"

``--pick`` takes file names with or without extension. ``samples.json`` stores every reference
caption and the caption the evaluation run produced (``expected_caption``), which
``python app.py --selftest`` compares against.

For Clotho clips, ``--metadata-csv clotho_metadata_evaluation.csv`` fills each clip's Freesound
link and license from the official metadata file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def overlap_f1(candidate: str, reference: str) -> float:
    c, r = tokens(candidate), tokens(reference)
    if not c or not r:
        return 0.0
    common = 0
    pool = list(r)
    for t in c:
        if t in pool:
            pool.remove(t)
            common += 1
    if common == 0:
        return 0.0
    p, rc = common / len(c), common / len(r)
    return 2 * p * rc / (p + rc)


def load_manifest(path: Path) -> "OrderedDict[str, list[str]]":
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    refs: "OrderedDict[str, list[str]]" = OrderedDict()
    for row in rows:
        caps = row.get("captions", row.get("caption", []))
        caps = caps if isinstance(caps, list) else [caps]
        bucket = refs.setdefault(row["audio_path"], [])
        for c in caps:
            if c and c not in bucket:
                bucket.append(c)
    return refs


def load_generated(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)["results"]
    out: dict[str, str] = {}
    for r in results:
        out.setdefault(r["audio"], r["generated"])
    return out


def load_metadata(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return {row["file_name"]: row for row in csv.DictReader(f)}


def find_clip(refs, key: str) -> str:
    matches = [a for a in refs if Path(a).name == key or Path(a).stem == key]
    if len(matches) != 1:
        raise SystemExit(f"--pick {key!r}: expected one clip in the manifest, found {len(matches)}")
    return matches[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path, help="test manifest with reference captions")
    p.add_argument("--eval-results", type=Path, help="eval_results.json of the model shown in the demo")
    p.add_argument("--suggest", type=int, default=0, help="list the N best-matching clips and exit")
    p.add_argument("--pick", nargs="+", default=[], help="clip file names to include, in display order")
    p.add_argument("--names", nargs="+", default=[], help="display names, one per --pick")
    p.add_argument("--dataset", default="", help='shown in the app, e.g. "AudioCaps test set"')
    p.add_argument("--source", default="", help="source note shown for every clip")
    p.add_argument("--license", default="", help="license note shown for every clip")
    p.add_argument("--metadata-csv", type=Path, help="Clotho metadata csv (per-clip Freesound link and license)")
    p.add_argument("--model-id", default="KarthikKB1998/CARD-Qwen3-4B-AudioCaps")
    p.add_argument("--out", type=Path, default=HERE / "samples")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    refs = load_manifest(args.manifest)
    generated = load_generated(args.eval_results)
    print(f"[samples] {len(refs)} clips in manifest, {len(generated)} captions in eval results")

    if args.suggest:
        if not generated:
            raise SystemExit("--suggest needs --eval-results")
        ranked = []
        for audio, rs in refs.items():
            cap = generated.get(audio)
            if cap and len(tokens(cap)) >= 5:
                ranked.append((max(overlap_f1(cap, r) for r in rs), audio, cap, rs))
        ranked.sort(key=lambda x: -x[0])
        for i, (score, audio, cap, rs) in enumerate(ranked[: args.suggest], 1):
            print(f"\n{i:2d}. {Path(audio).name}  (overlap {score:.2f})")
            print(f"    model:     {cap}")
            for r in rs:
                print(f"    reference: {r}")
        return 0

    if not args.pick:
        raise SystemExit("give --suggest N to list candidates or --pick to build the samples")
    if args.names and len(args.names) != len(args.pick):
        raise SystemExit("--names needs exactly one name per --pick")
    if args.out.exists() and any(args.out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.out} is not empty (use --overwrite)")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_csv)
    samples = []
    for i, key in enumerate(args.pick):
        audio = find_clip(refs, key)
        src = Path(audio)
        dst_name = f"{i + 1:02d}_{src.name}"
        shutil.copy2(src, args.out / dst_name)
        size_mb = src.stat().st_size / 1e6
        meta = metadata.get(src.name, {})
        sample = {
            "name": args.names[i] if args.names else src.stem,
            "file": dst_name,
            "dataset": args.dataset,
            "source": meta.get("sound_link") or args.source,
            "license": meta.get("license") or args.license,
            "references": refs[audio],
        }
        if audio in generated:
            sample["expected_caption"] = generated[audio]
        samples.append(sample)
        print(f"[samples] {sample['name']}: {src.name} ({size_mb:.1f} MB, {len(refs[audio])} references)"
              + ("  WARNING: large file" if size_mb > 10 else ""))

    with open(args.out / "samples.json", "w", encoding="utf-8") as f:
        json.dump({"model_id": args.model_id, "samples": samples}, f, indent=2, ensure_ascii=True)
        f.write("\n")
    print(f"[samples] wrote {args.out / 'samples.json'} with {len(samples)} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
