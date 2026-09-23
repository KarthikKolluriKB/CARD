"""Paired bootstrap significance test on per-clip CIDEr-D between two evaluated systems.

Both systems are scored on the same clips; the clip set is resampled 10,000 times to obtain a 95%
confidence interval on the CIDEr-D difference (B - A) and a two-sided p-value.

    python scripts/eval/bootstrap_significance.py \\
        --a results/no_distill_audiocaps/predictions.jsonl --label-a "No Distill" \\
        --b results/card_star_audiocaps/predictions.jsonl --label-b "CARD*"

Prediction files written by ``evaluate.py`` carry their references. For ``eval_results.json`` files
of the original training code, pass the evaluation manifest with ``--manifest``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from card.evaluation import load_references, paired_bootstrap, per_clip_cider_d, read_predictions  # noqa: E402


def as_mapping(rows, references):
    out = {}
    for row in rows:
        refs = row["references"] if row["references"] is not None else references.get(row["audio_path"])
        if refs is None:
            raise SystemExit(f"no references for {row['audio_path']}; pass --manifest")
        out[row["audio_path"]] = (row["prediction"], refs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="predictions of system A")
    ap.add_argument("--b", required=True, help="predictions of system B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--manifest", default=None, help="references for eval_results.json inputs")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    references = {}
    if args.manifest:
        for audio_path, resolved, refs in load_references(args.manifest):
            references[audio_path] = refs
            references[resolved] = refs
    a = as_mapping(read_predictions(args.a), references)
    b = as_mapping(read_predictions(args.b), references)
    clips = [k for k in a if k in b]
    if not clips:
        raise SystemExit("the two prediction files share no clips")
    refs = [a[k][1] for k in clips]
    scores_a = 100 * per_clip_cider_d([a[k][0] for k in clips], refs)
    scores_b = 100 * per_clip_cider_d([b[k][0] for k in clips], refs)
    r = paired_bootstrap(scores_a, scores_b, n_boot=args.n_boot, seed=args.seed)

    significant = r["ci_low"] > 0 or r["ci_high"] < 0
    print(f"{len(clips)} paired clips, per-clip CIDEr-D x100, {args.n_boot} resamples")
    print(f"  {args.label_a:<20} {r['mean_a']:7.2f}")
    print(f"  {args.label_b:<20} {r['mean_b']:7.2f}")
    print(f"  difference           {r['difference']:7.2f}   95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
    print(f"  p-value (two-sided)  {r['p_value']:.4f}   significant at 0.05: {'yes' if significant else 'no'}")


if __name__ == "__main__":
    main()
