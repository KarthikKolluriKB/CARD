"""Captioning metrics, prediction files and the paired bootstrap test.

Metrics follow the paper: ``aac_metrics.evaluate`` with its default metric set (BLEU-1..4, METEOR,
ROUGE-L, CIDEr-D, SPICE, SPIDEr) after its standard PTB tokenisation. METEOR and SPICE need Java 8+
and the files fetched by ``aac-metrics-download``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from card.data import load_manifest, resolve_audio_path

PAPER_METRICS = ("cider_d", "spider", "spice", "meteor")


def load_references(manifest_path: str | Path) -> List[Tuple[str, str, List[str]]]:
    """Rows of an evaluation manifest as (manifest audio_path, resolved path, references)."""
    manifest_path = Path(manifest_path)
    base = manifest_path.resolve().parent
    rows = []
    for row in load_manifest(manifest_path):
        if "captions" in row:
            refs = row["captions"] if isinstance(row["captions"], list) else [row["captions"]]
        else:
            refs = [row.get("caption", "")]
        rows.append((row["audio_path"], resolve_audio_path(row["audio_path"], base), refs))
    return rows


def require_java() -> None:
    if shutil.which("java") is None:
        raise RuntimeError("METEOR and SPICE need Java 8+ on PATH (and `aac-metrics-download` once)")


def caption_metrics(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Corpus-level scores in [0, 1] from ``aac_metrics.evaluate`` (default metric set)."""
    from aac_metrics import evaluate

    require_java()
    scores, _ = evaluate(list(predictions), [list(r) for r in references])
    return {k: float(v.item() if hasattr(v, "item") else v) for k, v in scores.items()}


def per_clip_cider_d(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> np.ndarray:
    """Per-clip CIDEr-D (IDF over the whole set, whitespace tokens), as used for the significance tests."""
    from aac_metrics.functional import cider_d

    _, per_clip = cider_d(list(predictions), [list(r) for r in references], return_all_scores=True)
    return np.asarray(per_clip["cider_d"].tolist(), dtype=float)


def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 10_000,
                     seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over clips for mean(scores_b - scores_a): 95% CI and two-sided p-value."""
    scores_a, scores_b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if scores_a.shape != scores_b.shape or scores_a.ndim != 1:
        raise ValueError("scores must be 1-D arrays over the same clips")
    diff = scores_b - scores_a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, max(2 * min((boot <= 0).mean(), (boot >= 0).mean()), 1.0 / n_boot))
    return {"mean_a": float(scores_a.mean()), "mean_b": float(scores_b.mean()),
            "difference": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "p_value": float(p)}


def write_predictions(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_predictions(path: str | Path) -> List[dict]:
    """Rows {"audio_path", "prediction", "references"} from ``evaluate.py``.

    Also reads the ``eval_results.json`` files of the original training code
    ({"results": [{"audio", "generated", ...}]}); those rows carry no references.
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [{"audio_path": r["audio"], "prediction": r["generated"], "references": None}
                for r in data["results"]]
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
