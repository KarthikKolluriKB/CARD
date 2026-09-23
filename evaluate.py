"""Evaluate a CARD model on an audio captioning test set.

Captions every clip of an evaluation manifest with beam search (4 beams, 40 new tokens), then
scores them with aac-metrics. Writes ``predictions.jsonl`` (one row per clip) and ``metrics.json``.

    python evaluate.py --model KarthikKB1998/CARD-Qwen3-4B-AudioCaps \\
        --manifest data/audiocaps/test_multiref.json --output-dir results/card_star_audiocaps
    python evaluate.py --model outputs/card_star_clotho/merged \\
        --manifest data/clotho/test_multiref.json --output-dir results/card_star_clotho

    # score an existing predictions file again
    python evaluate.py --predictions results/card_star_audiocaps/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from card.evaluation import (
    PAPER_METRICS,
    caption_metrics,
    load_references,
    read_predictions,
    require_java,
    write_predictions,
)

LABELS = {"cider_d": "CIDEr-D", "spider": "SPIDEr", "spice": "SPICE", "meteor": "METEOR"}


def generate(args) -> tuple[list[dict], Path]:
    from card import load_card

    rows = load_references(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    model = load_card(args.model, mode=args.mode, device=args.device)
    print(f"[eval] {model}")
    print(f"[eval] {len(rows)} clips from {args.manifest}")
    warnings.filterwarnings("ignore", message="Audio is .* s; CARD was trained")

    predictions, t0 = [], time.time()
    for i, (audio_path, resolved, refs) in enumerate(rows, 1):
        caption = model.caption(resolved, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
        predictions.append({"audio_path": audio_path, "prediction": caption, "references": refs})
        if i % 50 == 0 or i == len(rows):
            rate = (time.time() - t0) / i
            print(f"[eval] {i}/{len(rows)}  ({rate:.2f} s/clip)", flush=True)

    out_dir = Path(args.output_dir)
    write_predictions(out_dir / "predictions.jsonl", predictions)
    return predictions, out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="merged model folder or Hugging Face repo id")
    ap.add_argument("--manifest", help="evaluation manifest ({'audio_path', 'captions': [...]})")
    ap.add_argument("--output-dir", help="where to write predictions.jsonl and metrics.json")
    ap.add_argument("--predictions", help="score an existing predictions.jsonl instead of generating")
    ap.add_argument("--mode", default="merged", choices=("merged", "adapters"))
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N clips (0 = all)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.predictions:
        predictions = read_predictions(args.predictions)
        if any(p["references"] is None for p in predictions):
            ap.error("--predictions needs a predictions.jsonl written by evaluate.py")
        out_dir = Path(args.output_dir or Path(args.predictions).parent)
    elif args.model and args.manifest and args.output_dir:
        require_java()
        predictions, out_dir = generate(args)
    else:
        ap.error("give --model, --manifest and --output-dir, or --predictions")

    scores = caption_metrics([p["prediction"] for p in predictions], [p["references"] for p in predictions])
    result = {
        "num_clips": len(predictions),
        "model": args.model,
        "manifest": args.manifest,
        "decoding": {"num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens},
        "metrics": scores,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[eval] " + "  ".join(f"{LABELS[k]} {100 * scores[k]:.1f}" for k in PAPER_METRICS))
    print(f"[eval] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
