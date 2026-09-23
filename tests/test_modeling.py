"""Offline unit tests for the inference modules and the export layout (no model downloads)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from card_model.hub import CARDConfig  # noqa: E402
from card_model.modeling import AudioProjector, build_hybrid_attention_mask, load_audio  # noqa: E402


def test_projector_shape_and_downsampling():
    proj = AudioProjector(n_mels=64, hidden_dim=2560)
    x = torch.randn(2, 1001, 64)
    y = proj(x)
    assert y.shape == (2, 251, 2560)
    assert proj.n_tokens(1001) == 251
    assert abs(sum(p.numel() for p in proj.parameters()) / 1e6 - 13.2) < 0.2


def test_projector_state_dict_keys_match_training_checkpoints():
    keys = set(AudioProjector(n_mels=64).state_dict().keys())
    assert keys == {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
                    "proj.weight", "proj.bias", "norm.weight", "norm.bias", "pos_embed.weight"}
    shape = AudioProjector.infer_shape(AudioProjector(n_mels=80, hidden_dim=128,
                                                      conv_intermediate=32, max_audio_len=77).state_dict())
    assert shape == {"n_mels": 80, "conv_intermediate": 32, "hidden_dim": 128, "max_audio_len": 77}


def test_projector_roundtrip_through_safetensors(tmp_path):
    proj = AudioProjector(n_mels=64, hidden_dim=64, conv_intermediate=16, max_audio_len=32)
    save_file({k: v.contiguous() for k, v in proj.state_dict().items()}, str(tmp_path / "p.safetensors"))
    state = load_file(str(tmp_path / "p.safetensors"))
    proj2 = AudioProjector(**AudioProjector.infer_shape(state))
    proj2.load_state_dict(state)
    x = torch.randn(1, 40, 64)
    assert torch.equal(proj(x), proj2(x))


def test_hybrid_mask_semantics():
    m = build_hybrid_attention_mask(1, 3, 2, torch.device("cpu"), dtype=torch.float32)[0, 0]
    assert m.shape == (5, 5)
    assert torch.all(m[:3, :3] == 0)             # audio <-> audio
    assert torch.all(m[3:, :3] == 0)             # text -> audio
    assert torch.all(torch.isinf(m[:3, 3:]))     # audio -> text blocked
    assert m[3, 3] == 0 and torch.isinf(m[3, 4]) and m[4, 3] == 0 and m[4, 4] == 0  # causal text


def test_load_audio_pads_crops_and_downmixes():
    pytest.importorskip("torchaudio")
    sr = 8000
    short = np.random.randn(2, sr).astype(np.float32)  # 1 s stereo (C, T)
    w = load_audio(short, sr=sr, target_sr=16000, max_duration=10.0)
    assert w.shape == (160000,) and w.dtype == torch.float32
    assert torch.all(w[-1000:] == 0)             # zero padded
    long = np.random.randn(16000 * 12).astype(np.float32)
    with pytest.warns(UserWarning):
        w = load_audio(long, sr=16000, max_duration=10.0)
    assert w.shape == (160000,)
    with pytest.raises(ValueError):
        load_audio(long)


def test_config_roundtrip_and_partial_merge(tmp_path):
    cfg = CARDConfig.from_dict({"variant": "proj_early", "phase": 2, "finetune_dataset": "clotho",
                                "decoding": {"num_beams": 2}})
    assert cfg.decoding == {"num_beams": 2, "max_new_tokens": 40}
    assert cfg.frontend["type"] == "clap_processor"
    cfg.save(tmp_path / "card_config.json")
    back = CARDConfig.from_file(tmp_path / "card_config.json")
    assert back.to_dict() == cfg.to_dict()
    assert json.loads((tmp_path / "card_config.json").read_text())["lora"]["r"] == 16
