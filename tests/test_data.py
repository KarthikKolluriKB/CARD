"""Offline unit tests for the training data pipeline (no downloads, stub tokenizer)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from card.data import (  # noqa: E402
    CAPTION_SYSTEM_PROMPT,
    CAPTION_USER_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    AudioCaptionDataset,
    HomogeneousBatchSampler,
    TextOnlyDataset,
    build_train_dataset,
    collate_mixed,
)


class StubTokenizer:
    """Character-level tokenizer with a Qwen3-like chat template."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=None):
        assert enable_thinking is False
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text, return_tensors="pt", add_special_tokens=False, max_length=None,
                 truncation=False):
        ids = [ord(c) % 1000 + 1 for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> Path:
    sf = pytest.importorskip("soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return path


def test_row_formats_prompts_and_label_masking(tmp_path):
    tok = StubTokenizer()
    manifest = _write(tmp_path / "set" / "m.json", [
        {"audio_path": "audio/a.wav", "caption": "a dog barks"},
        {"audio_path": "audio/a.wav", "captions": ["rain falls", "water drips"]},
        {"audio_path": "audio/a.wav", "prompt": "Summarize the audio.", "response": "birds sing"},
        {"audio_path": "audio/a.wav", "question": "Is it raining?", "answer": "yes"},
        {"audio_path": "audio/a.wav", "caption": ""},  # no target: dropped
        {"caption": "no audio"},                        # no audio: dropped
    ])
    _wav(tmp_path / "set" / "audio" / "a.wav")
    ds = AudioCaptionDataset(manifest, tok, max_audio_sec=10.0)
    assert len(ds) == 4
    assert [it["prompt"] for it in ds.items] == [CAPTION_USER_PROMPT, CAPTION_USER_PROMPT,
                                                 "Summarize the audio.", "Is it raining?"]
    assert [it["response"] for it in ds.items] == ["a dog barks", "rain falls", "birds sing", "yes"]
    assert [it["system"] for it in ds.items] == [None, None, None, QA_SYSTEM_PROMPT]
    assert Path(ds.items[0]["audio_path"]) == (tmp_path / "set" / "audio" / "a.wav").resolve()

    sample = ds[0]
    assert sample["waveform"].shape == (160000,)
    ids, labels = sample["text_ids"], sample["labels"]
    assert ids[-1].item() == tok.eos_token_id
    prompt_len = len(tok.apply_chat_template([{"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                                              {"role": "user", "content": CAPTION_USER_PROMPT}],
                                             enable_thinking=False))
    assert torch.all(labels[:prompt_len] == -100)
    assert torch.equal(labels[prompt_len:], ids[prompt_len:])
    assert labels.ne(-100).sum().item() == len("a dog barks") + 1  # response + EOS


def test_text_only_uses_general_prompt_and_truncates(tmp_path):
    tok = StubTokenizer()
    manifest = _write(tmp_path / "text.json", [
        {"instruction": "Say hello politely please", "response": "Hello there, friend"},
        {"instruction": "x" * 50, "response": "y" * 50},
        {"instruction": "", "response": "dropped"},
    ])
    ds = TextOnlyDataset(manifest, tok)
    assert len(ds) == 2
    s = ds[0]
    assert s["waveform"] is None and s["is_text_only"] is True
    assert s["text_ids"][-1].item() == tok.eos_token_id
    expected = tok.apply_chat_template([{"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                                        {"role": "user", "content": "Say hello politely please"}],
                                       enable_thinking=False)
    expected_ids = tok(expected + "Hello there, friend").input_ids[0]
    assert torch.equal(s["text_ids"][:-1], expected_ids)
    long = TextOnlyDataset(manifest, tok, max_length=64)[1]
    assert long["text_ids"].numel() == 64            # truncated, no room for EOS
    assert long["labels"].ne(-100).sum().item() >= 1  # at least one supervised token kept

    assert len(TextOnlyDataset(manifest, tok, max_n=1)) == 1


def test_build_train_dataset_requires_every_manifest(tmp_path):
    tok = StubTokenizer()
    audio = _write(tmp_path / "a.json", [{"audio_path": "x.wav", "caption": "c"}] * 3)
    text = _write(tmp_path / "t.json", [{"instruction": "i i i", "response": "r r r"}] * 5)
    ds = build_train_dataset([audio], tok, text_manifests=[text], text_max_n=2)
    assert len(ds) == 5
    with pytest.raises(FileNotFoundError):
        build_train_dataset([audio, tmp_path / "missing.json"], tok)


@pytest.mark.parametrize("world_size", [1, 2])
def test_sampler_batches_are_modality_pure_and_rank_aligned(tmp_path, world_size):
    tok = StubTokenizer()
    audio = _write(tmp_path / "a.json", [{"audio_path": f"{i}.wav", "caption": "c"} for i in range(37)])
    text = _write(tmp_path / "t.json", [{"instruction": "i i i", "response": "r r r"}] * 11)
    ds = build_train_dataset([audio], tok, text_manifests=[text])
    n_audio = 37

    per_rank = []
    for rank in range(world_size):
        sampler = HomogeneousBatchSampler(ds, batch_size=4, seed=7, rank=rank, world_size=world_size)
        sampler.set_epoch(3)
        batches = list(sampler)
        assert len(batches) == len(sampler)
        per_rank.append(batches)

    for step in zip(*per_rank, strict=True):
        kinds = {all(i < n_audio for i in b) for b in step}
        assert len(kinds) == 1                                   # same modality on every rank
        assert all(len(b) == 4 for b in step)
        for b in step:
            assert all(i < n_audio for i in b) or all(i >= n_audio for i in b)

    if world_size == 1:
        seen = {i for b in per_rank[0] for i in b}
        assert seen == set(range(len(ds)))                        # oversampling drops nothing
        s = HomogeneousBatchSampler(ds, batch_size=4, seed=7)
        s.set_epoch(3)
        assert list(s) == per_rank[0]                             # deterministic for (seed, epoch)


def test_collate_pads_audio_text_and_labels():
    batch = [
        {"waveform": torch.ones(5), "text_ids": torch.tensor([1, 2, 3]),
         "labels": torch.tensor([-100, 2, 3]), "is_text_only": False},
        {"waveform": torch.ones(3), "text_ids": torch.tensor([4]),
         "labels": torch.tensor([4]), "is_text_only": False},
    ]
    out = collate_mixed(batch, pad_token_id=9)
    assert out["waveform"].shape == (2, 5) and out["waveform"][1, 3:].eq(0).all()
    assert out["text_ids"].tolist() == [[1, 2, 3], [4, 9, 9]]
    assert out["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert out["is_text_only"].tolist() == [False, False]

    text = [{"waveform": None, "text_ids": torch.tensor([1]), "labels": torch.tensor([1]),
             "is_text_only": True}]
    assert collate_mixed(text)["waveform"] is None


def test_manifest_paths_are_written_relative(tmp_path):
    from _common import read_manifest, write_manifest

    audio = _wav(tmp_path / "ds" / "audio" / "clip.wav")
    write_manifest([{"audio_path": str(audio), "caption": "c"}], tmp_path / "ds" / "train.json")
    raw = json.loads((tmp_path / "ds" / "train.json").read_text())
    assert raw[0]["audio_path"] == "audio/clip.wav"
    assert Path(read_manifest(tmp_path / "ds" / "train.json")[0]["audio_path"]) == audio.resolve()


def test_phase2_expansion_counts_and_prompts():
    import prepare_audiocaps
    import prepare_clotho

    ac = prepare_audiocaps.expand_prompts([{"audio_path": "a", "caption": " c1 "},
                                           {"audio_path": "b", "caption": "c2"}])
    assert len(ac) == 10 and {r["response"] for r in ac} == {"c1", "c2"}
    assert {r["prompt"] for r in ac} == set(prepare_audiocaps.PROMPT_TEMPLATES)
    assert ac == prepare_audiocaps.expand_prompts([{"audio_path": "a", "caption": " c1 "},
                                                   {"audio_path": "b", "caption": "c2"}])

    cl = prepare_clotho.expand_prompts([{"audio_path": "a", "captions": ["c1", "c2", "c3", "c4", "c5"]}])
    assert len(cl) == 25
    assert [r["prompt"] for r in cl[:5]] == prepare_clotho.PROMPT_TEMPLATES
    assert [r["response"] for r in cl[::5]] == ["c1", "c2", "c3", "c4", "c5"]


def test_autoacd_youtube_id_parsing():
    import prepare_autoacd

    assert prepare_autoacd.youtube_id("audio/AudioSet_SL/Y--cB2ZVjpnA.flac") == "--cB2ZVjpnA"
    assert prepare_autoacd.youtube_id("Yb0RFKhbpFJA_30.000.flac") == "b0RFKhbpFJA"
    assert prepare_autoacd.youtube_id("YpItdNzDM0_8.flac") is None


def test_eval_reference_construction():
    import prepare_audiocaps
    import prepare_clotho

    joined = "A dog barks. Rain falls on a roof. Birds sing loudly and a car passes Wind blows."
    assert prepare_clotho.split_references(joined) == [
        "A dog barks", "Rain falls on a roof", "Birds sing loudly and a car passes Wind blows"]
    assert prepare_clotho.split_references("One long caption without a split.") == [
        "One long caption without a split."]

    rows = [{"audio_path": f"test_{i}.wav", "caption": f"c{i}"} for i in range(5)]
    audio = {"test_0.wav": "x", "test_1.wav": "y", "test_2.wav": "x", "test_3.wav": "z", "test_4.wav": "y"}
    groups = prepare_audiocaps.group_references(rows, audio_key=audio.__getitem__)
    assert groups == [{"audio_path": "test_2.wav", "captions": ["c0", "c2"]},
                      {"audio_path": "test_4.wav", "captions": ["c1", "c4"]},
                      {"audio_path": "test_3.wav", "captions": ["c3"]}]
