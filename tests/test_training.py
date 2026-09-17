"""Offline unit tests for the distillation heads, losses, configs and per-batch training losses."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from card.config import load_config, validate_config  # noqa: E402
from card.heads import LLMHeads, ProjectorHeads, allocate_blocks  # noqa: E402
from card.losses import llm_distill_loss, projector_distill_loss  # noqa: E402
from card.model import CARDStudent, init_projector_to_match_llm  # noqa: E402
from card.modeling import AudioProjector  # noqa: E402
from card.teacher import CLAP_STAGE_DEPTHS, CLAP_STAGE_DIMS  # noqa: E402
from card.training import batch_losses  # noqa: E402


@pytest.mark.parametrize("stages, ranges", [
    ([2, 3], [(0, 9), (9, 12)]),
    ([0, 1], [(0, 6), (6, 12)]),
    ([0, 1, 2, 3], [(0, 2), (2, 4), (4, 10), (10, 12)]),
])
def test_llm_block_allocation(stages, ranges):
    assert allocate_blocks([CLAP_STAGE_DEPTHS[s] for s in stages], 12) == ranges


def test_heads_shapes_and_published_key_names():
    proj_heads = ProjectorHeads([192, 384])
    out = proj_heads(torch.randn(2, 251, 2560))
    assert [o.shape for o in out] == [(2, 192), (2, 384)]
    assert sum(p.numel() for p in proj_heads.parameters()) == 4_376_640

    legacy = {k.replace("heads.", "head_s", 1): v for k, v in ProjectorHeads([192, 384]).state_dict().items()}
    assert all(k.startswith("head_s") for k in legacy)
    ProjectorHeads([192, 384]).load_state_dict(legacy, strict=True)

    llm_heads = LLMHeads([768, 768], [6, 2], llm_dim=32)
    hidden = [torch.randn(2, 5, 32) for _ in range(12)]
    out = llm_heads(hidden)
    assert [o.shape for o in out] == [(2, 5, 768), (2, 5, 768)]
    assert set(llm_heads.state_dict()) == {"heads.0.norm.weight", "heads.0.proj.weight", "heads.0.proj.bias",
                                           "heads.1.norm.weight", "heads.1.proj.weight", "heads.1.proj.bias"}


def test_distillation_losses():
    t = torch.randn(2, 16, 8)
    assert projector_distill_loss([t.mean(dim=1)], [t]).abs() < 1e-6
    assert torch.isclose(projector_distill_loss([-t.mean(dim=1)], [t]), torch.tensor(2.0))

    student = torch.randn(2, 4, 8)
    teacher = torch.nn.functional.interpolate(student.transpose(1, 2), size=9, mode="linear",
                                              align_corners=False).transpose(1, 2)
    assert llm_distill_loss([student], [student]).abs() < 1e-6
    loss = llm_distill_loss([student, student], [teacher, student])
    assert 0.0 <= loss.item() < 1.0


def test_configs_inherit_and_validate():
    star = validate_config(load_config(ROOT / "configs" / "card_star.yaml"))
    clotho = validate_config(load_config(ROOT / "configs" / "clotho" / "card_star.yaml"))
    assert star["distillation"]["projector_stages"] == [0, 1] and star["distillation"]["llm_stages"] == [2, 3]
    assert (star["lora"]["r"], star["lora"]["alpha"], star["lora"]["num_layers"]) == (16, 16, 36)
    p1, p2 = star["phase1"], star["phase2"]
    assert (p1["lr"], p1["warmup_steps"], p1["scheduler"], p1["batch_size"] * p1["grad_accum"] * 2) == \
        (2e-4, 200, "constant_with_warmup", 64)
    assert (p2["lr"], p2["warmup_steps"], p2["scheduler"]) == (2e-5, 50, "cosine")
    assert p1["num_text_samples"] == 50_000
    assert clotho["phase2"]["data"] == ["data/clotho/train_expanded.json"]
    assert clotho["phase2"]["phase1_checkpoint"] == "outputs/card_star/phase1/epoch-1"
    assert clotho["phase2"]["lr"] == p2["lr"] and clotho["phase1"] == p1
    assert clotho["validation"] is None
    with pytest.raises(ValueError):
        validate_config({**star, "distillation": {"projector_stages": [4], "llm_stages": []}})


class _Frontend(torch.nn.Module):
    n_mels = 16

    def forward(self, waves):
        return waves.view(waves.size(0), -1, 16)


class _Teacher:
    def __call__(self, waves):
        g = torch.Generator().manual_seed(0)
        return [torch.randn(waves.size(0), t, d, generator=g) for t, d in zip((64, 16, 4, 4), CLAP_STAGE_DIMS)]


@pytest.fixture(scope="module")
def tiny_llm():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    llm = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=12, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, max_position_embeddings=256))
    return peft.get_peft_model(llm, peft.LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"],
                                                    layers_to_transform=list(range(12)), task_type="CAUSAL_LM"))


@pytest.mark.parametrize("projector_stages, llm_stages",
                         [([0, 1], [2, 3]), ([], []), ([0, 1], []), ([], [0, 1, 2, 3])])
def test_batch_losses_routing(tiny_llm, projector_stages, llm_stages):
    projector = AudioProjector(n_mels=16, hidden_dim=32, conv_intermediate=8, max_audio_len=64)
    init_projector_to_match_llm(projector, tiny_llm.get_input_embeddings().weight.data)
    proj_heads = llm_heads = None
    if projector_stages:
        proj_heads = ProjectorHeads([CLAP_STAGE_DIMS[s] for s in projector_stages], input_dim=32)
    if llm_stages:
        llm_heads = LLMHeads([CLAP_STAGE_DIMS[s] for s in llm_stages],
                             [CLAP_STAGE_DEPTHS[s] for s in llm_stages], llm_dim=32)
    student = CARDStudent(tiny_llm, projector, proj_heads, llm_heads)
    teacher = _Teacher() if (projector_stages or llm_stages) else None
    distillation = {"projector_stages": projector_stages, "llm_stages": llm_stages,
                    "lambda_proj": 1.0, "lambda_llm": 1.0}

    ids = torch.randint(1, 128, (2, 6))
    labels = ids.clone()
    labels[:, :3] = -100
    cpu = torch.device("cpu")
    audio = {"waveform": torch.randn(2, 640), "text_ids": ids, "labels": labels,
             "is_text_only": torch.tensor([False, False])}
    out = batch_losses(student, student, audio, _Frontend(), teacher, distillation, pad_id=0, device=cpu)
    expected = out["loss_cap"] + out["loss_proj"] + out["loss_llm"]
    assert torch.allclose(out["total"], expected)
    assert (out["loss_proj"] > 0) == bool(projector_stages) and (out["loss_llm"] > 0) == bool(llm_stages)
    out["total"].backward()
    if proj_heads is not None:
        assert all(p.grad is not None for p in proj_heads.parameters())
    if llm_heads is not None:
        assert all(p.grad is not None for p in llm_heads.parameters())

    text = {"waveform": None, "text_ids": ids, "labels": labels, "is_text_only": torch.tensor([True, True])}
    out = batch_losses(student, student, text, _Frontend(), teacher, distillation, pad_id=0, device=cpu)
    assert out["loss_cap"] == 0 and out["loss_text"] > 0 and torch.equal(out["total"], out["loss_text"])


def test_resolve_checkpoint_layouts(tmp_path):
    from safetensors.torch import save_file

    from card.training import resolve_checkpoint

    state = {k: v.contiguous() for k, v in AudioProjector(n_mels=16, hidden_dim=32, conv_intermediate=8,
                                                           max_audio_len=64).state_dict().items()}
    training = tmp_path / "phase1" / "epoch-1"
    (training / "lora_adapter").mkdir(parents=True)
    save_file(state, str(training / "audio_projector.safetensors"))
    published = tmp_path / "published"
    (published / "adapters" / "phase1_lora").mkdir(parents=True)
    save_file(state, str(published / "audio_projector.safetensors"))

    layouts = ((training, training / "lora_adapter"), (published, published / "adapters" / "phase1_lora"))
    for root, adapter in layouts:
        got_adapter, got_state = resolve_checkpoint(str(root), phase=1)
        assert got_adapter == adapter and torch.equal(got_state["conv1.weight"], state["conv1.weight"])
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(published), phase=2)
