#!/usr/bin/env python
"""Check that a CARD model reproduces the captions of an earlier evaluation, clip by clip.

Beam search is deterministic, so a model whose weights equal those behind an evaluation run
must produce exactly the same caption for every clip. This loads the model with
``card.load_card`` (an export folder or a Hub repo) and compares its captions with the
``results`` list of an ``eval_results.json`` written by that evaluation.

    CUDA_VISIBLE_DEVICES=1 python huggingface/verify_captions.py \\
        --model ~/Karthik/hub/CARD-Qwen3-4B-Clotho \\
        --eval-results ../outputs/aora_split_r16_clotho_ft/merged/eval_results.json

Exit code 0 only if every compared clip matches.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from card.hub import load_card  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="export folder or Hub repo id")
    p.add_argument("--eval-results", required=True, type=Path, help="eval_results.json of the reference run")
    p.add_argument("--mode", default="merged", choices=("merged", "adapters"))
    p.add_argument("--limit", type=int, default=0, help="compare only the first N clips (0 = all)")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    with open(args.eval_results, "r", encoding="utf-8") as f:
        results = json.load(f)["results"]

    # one reference caption per clip; flat manifests repeat each clip once per reference
    reference = {}
    inconsistent = 0
    for r in results:
        prev = reference.setdefault(r["audio"], r["generated"])
        inconsistent += prev != r["generated"]
    clips = list(reference)
    if args.limit:
        clips = clips[: args.limit]
    print(f"[verify] {len(results)} rows, {len(reference)} unique clips, comparing {len(clips)}")
    if inconsistent:
        print(f"[verify] note: {inconsistent} repeated rows in the reference run disagree with the first "
              f"caption of their clip")

    model = load_card(args.model, mode=args.mode, device=args.device)
    print(f"[verify] {model}")
    warnings.filterwarnings("ignore", message="Audio is .* s; CARD was trained")

    same, mismatches = 0, []
    t0 = time.time()
    for i, audio in enumerate(clips, 1):
        cap = model.caption(audio)
        if cap == reference[audio]:
            same += 1
        else:
            mismatches.append((audio, reference[audio], cap))
        if i % 50 == 0 or i == len(clips):
            rate = (time.time() - t0) / i
            print(f"[verify] {i}/{len(clips)}  identical so far: {same}  "
                  f"({rate:.2f} s/clip, about {rate * (len(clips) - i) / 60:.0f} min left)", flush=True)

    for audio, want, got in mismatches[:10]:
        print(f"[verify] DIFF {Path(audio).name}\n    reference: {want}\n    model:     {got}")
    pct = 100.0 * same / max(1, len(clips))
    ok = same == len(clips)
    print(f"[verify] identical captions: {same}/{len(clips)} ({pct:.1f}%)  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
