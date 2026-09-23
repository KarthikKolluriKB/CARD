"""Offline unit tests for the evaluation utilities (no model, no Java)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from card.evaluation import (  # noqa: E402
    load_references,
    paired_bootstrap,
    per_clip_cider_d,
    read_predictions,
    write_predictions,
)


def test_load_references_formats_and_paths(tmp_path):
    manifest = tmp_path / "set" / "test.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps([
        {"audio_path": "audio/a.wav", "captions": ["a dog barks", "a dog is barking"]},
        {"audio_path": "audio/b.wav", "caption": "rain falls"},
    ]), encoding="utf-8")
    rows = load_references(manifest)
    assert [r[0] for r in rows] == ["audio/a.wav", "audio/b.wav"]
    assert Path(rows[0][1]) == (tmp_path / "set" / "audio" / "a.wav").resolve()
    assert [r[2] for r in rows] == [["a dog barks", "a dog is barking"], ["rain falls"]]


def test_predictions_roundtrip_and_original_format(tmp_path):
    rows = [{"audio_path": "a.wav", "prediction": "a dog barks", "references": ["a dog barks"]}]
    write_predictions(tmp_path / "p.jsonl", rows)
    assert read_predictions(tmp_path / "p.jsonl") == rows

    original = {"metrics": {}, "results": [{"audio": "/x/a.wav", "reference": "r", "generated": "g"}]}
    (tmp_path / "eval_results.json").write_text(json.dumps(original), encoding="utf-8")
    assert read_predictions(tmp_path / "eval_results.json") == [
        {"audio_path": "/x/a.wav", "prediction": "g", "references": None}]


def test_per_clip_cider_d_orders_clips():
    pytest.importorskip("aac_metrics")
    refs = [["a dog barks loudly", "a dog is barking"], ["rain falls on a roof", "heavy rain falls"]]
    scores = per_clip_cider_d(["a dog barks loudly", "a dog barks loudly"], refs)
    assert scores.shape == (2,) and scores[0] > scores[1]


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    a = rng.normal(40, 20, size=500)
    same = paired_bootstrap(a, a.copy(), n_boot=2000)
    assert same["difference"] == 0 and same["p_value"] == 1.0

    b = a + 5 + rng.normal(0, 1, size=500)
    better = paired_bootstrap(a, b, n_boot=2000, seed=3)
    assert better["ci_low"] > 0 and better["p_value"] == 1 / 2000
    assert better == paired_bootstrap(a, b, n_boot=2000, seed=3)
    with pytest.raises(ValueError):
        paired_bootstrap(a, a[:10])


def test_evaluate_scores_existing_predictions(tmp_path, monkeypatch):
    import evaluate

    rows = [{"audio_path": "a.wav", "prediction": "a dog barks", "references": ["a dog barks"]}]
    write_predictions(tmp_path / "predictions.jsonl", rows)
    seen = {}

    def fake_metrics(predictions, references):
        seen["args"] = (predictions, references)
        return {"cider_d": 0.5, "spider": 0.3, "spice": 0.1, "meteor": 0.2}

    monkeypatch.setattr(evaluate, "caption_metrics", fake_metrics)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--predictions", str(tmp_path / "predictions.jsonl")])
    evaluate.main()
    assert seen["args"] == (["a dog barks"], [["a dog barks"]])
    result = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert result["num_clips"] == 1 and result["metrics"]["cider_d"] == 0.5
