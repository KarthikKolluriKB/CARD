"""Phase 1 and Phase 2 training and adapter merging.

Phase 1: L_cap + lambda_proj * L_proj + lambda_llm * L_llm, with text-only batches interleaved.
Phase 2: merge the Phase 1 LoRA, train a new LoRA and the projector with L_cap only.
"""

from __future__ import annotations

import functools
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from card_model.data import (
    AudioCaptionDataset,
    HomogeneousBatchSampler,
    build_prompt,
    build_train_dataset,
    collate_mixed,
)
from card_model.heads import LLMHeads, ProjectorHeads
from card_model.hub import DEFAULT_USER_PROMPT, CARDConfig, merge_adapters
from card_model.losses import llm_distill_loss, projector_distill_loss
from card_model.model import CARDStudent, init_projector_to_match_llm
from card_model.modeling import AudioProjector, build_frontend, build_hybrid_attention_mask
from card_model.teacher import CLAP_STAGE_DEPTHS, CLAP_STAGE_DIMS, ClapTeacher

PROJECTOR_FILE = "audio_projector.safetensors"
PROJECTOR_HEADS_FILE = "projector_heads.safetensors"
LLM_HEADS_FILE = "llm_heads.safetensors"
ADAPTER_DIR = "lora_adapter"


# ----------------------------------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------------------------------

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return get_rank() == 0


def log(*args, **kwargs) -> None:
    if is_main():
        print(*args, **kwargs, flush=True)


def init_distributed() -> Optional[torch.device]:
    """Initialise the process group under torchrun and return this rank's device."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None
    backend = os.environ.get("DDP_BACKEND") or ("gloo" if sys.platform.startswith("win") else "nccl")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def _wandb_log(cfg: dict, payload: dict) -> None:
    if is_main() and (cfg.get("wandb") or {}).get("enabled"):
        import wandb

        wandb.log(payload)


# ----------------------------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------------------------

def lora_config(cfg: dict):
    from peft import LoraConfig

    lc = cfg["lora"]
    return LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        target_modules=list(lc["target_modules"]),
        layers_to_transform=list(range(int(lc["num_layers"]))),
        lora_dropout=float(lc.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["llm"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_llm(cfg: dict, device: torch.device, merge_adapter: Optional[Path] = None):
    """Frozen bf16 LLM (optionally with a previous adapter merged in) wrapped with a fresh LoRA."""
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    llm = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["llm"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    if merge_adapter is not None:
        log(f"  merging Phase 1 LoRA from {merge_adapter}")
        llm = PeftModel.from_pretrained(llm, str(merge_adapter)).merge_and_unload()
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in llm.parameters():
        p.requires_grad = False
    llm = get_peft_model(llm, lora_config(cfg))
    if is_main():
        llm.print_trainable_parameters()
    return llm


def build_projector(cfg: dict, frontend, llm, device: torch.device) -> AudioProjector:
    projector = AudioProjector(n_mels=frontend.n_mels, hidden_dim=llm.config.hidden_size).to(device)
    init_projector_to_match_llm(projector, llm.get_input_embeddings().weight.data)
    return projector


def build_phase1(cfg: dict, device: torch.device):
    """Student with distillation heads, the frontend, the teacher (or None) and the tokenizer."""
    d = cfg["distillation"]
    tokenizer = load_tokenizer(cfg)
    llm = build_llm(cfg, device)
    frontend = build_frontend(cfg["model"]["frontend"], cfg["model"]["teacher"]).to(device)
    projector = build_projector(cfg, frontend, llm, device)
    hidden = llm.config.hidden_size

    llm_heads = projector_heads = teacher = None
    if d["llm_stages"]:
        llm_heads = LLMHeads(
            stage_dims=[CLAP_STAGE_DIMS[s] for s in d["llm_stages"]],
            stage_depths=[CLAP_STAGE_DEPTHS[s] for s in d["llm_stages"]],
            llm_dim=hidden,
            num_blocks=int(d.get("llm_blocks", 12)),
        ).to(device)
    if d["projector_stages"]:
        projector_heads = ProjectorHeads(
            stage_dims=[CLAP_STAGE_DIMS[s] for s in d["projector_stages"]],
            input_dim=hidden,
            hidden_dim=int(d.get("projector_head_hidden", 768)),
        ).to(device)
    if llm_heads is not None or projector_heads is not None:
        teacher = ClapTeacher(cfg["model"]["teacher"]).to(device)

    log(f"  distillation: projector <- teacher stages {d['projector_stages'] or 'none'}, "
        f"LLM <- teacher stages {d['llm_stages'] or 'none'}"
        + (f" (LLM blocks {llm_heads.stage_block_ranges})" if llm_heads is not None else ""))
    return CARDStudent(llm, projector, projector_heads, llm_heads), frontend, teacher, tokenizer


def build_phase2(cfg: dict, device: torch.device, phase1_ckpt: Optional[str]):
    """Student with the Phase 1 LoRA merged, a fresh LoRA and the Phase 1 projector."""
    tokenizer = load_tokenizer(cfg)
    adapter, projector_state = (None, None)
    if phase1_ckpt:
        adapter, projector_state = resolve_checkpoint(phase1_ckpt, phase=1)
    else:
        log("  no Phase 1 checkpoint: Phase 2 starts from the base LLM and a new projector")
    llm = build_llm(cfg, device, merge_adapter=adapter)
    frontend = build_frontend(cfg["model"]["frontend"], cfg["model"]["teacher"]).to(device)
    projector = build_projector(cfg, frontend, llm, device)
    if projector_state is not None:
        projector.load_state_dict(projector_state)
        log("  loaded the Phase 1 audio projector")
    return CARDStudent(llm, projector), frontend, tokenizer


# ----------------------------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------------------------

def save_checkpoint(student: CARDStudent, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tensors = lambda m: {k: v.detach().contiguous().cpu() for k, v in m.state_dict().items()}  # noqa: E731
    save_file(tensors(student.audio_projector), str(ckpt_dir / PROJECTOR_FILE))
    student.llm.save_pretrained(str(ckpt_dir / ADAPTER_DIR))
    if student.projector_heads is not None:
        save_file(tensors(student.projector_heads), str(ckpt_dir / PROJECTOR_HEADS_FILE))
    if student.llm_heads is not None:
        save_file(tensors(student.llm_heads), str(ckpt_dir / LLM_HEADS_FILE))
    log(f"  saved checkpoint {ckpt_dir}")


def resolve_checkpoint(path_or_repo: str, phase: int) -> Tuple[Path, dict]:
    """(adapter dir, projector state) from a training checkpoint, a published folder or a Hub repo id."""
    root = Path(path_or_repo)
    hub_adapter = f"adapters/phase{phase}_lora"
    if not root.exists():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(str(path_or_repo),
                                      allow_patterns=[f"{hub_adapter}/*", PROJECTOR_FILE, "*.json"]))
    adapter = root / ADAPTER_DIR if (root / ADAPTER_DIR).exists() else root / hub_adapter
    if not adapter.exists():
        raise FileNotFoundError(f"no LoRA adapter under {root} ({ADAPTER_DIR}/ or {hub_adapter}/)")
    if (root / PROJECTOR_FILE).exists():
        state = load_file(str(root / PROJECTOR_FILE))
    elif (root / "audio_embedding.pt").exists():
        state = torch.load(root / "audio_embedding.pt", map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no audio projector weights under {root}")
    return adapter, state


# ----------------------------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------------------------

def build_scheduler(optimizer, name: str, warmup_steps: int, total_steps: int):
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    name = (name or "cosine").lower()
    if name in ("constant_with_warmup", "constant"):
        return get_constant_schedule_with_warmup(optimizer, warmup_steps)
    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    raise ValueError(f"unknown scheduler: {name}")


def batch_losses(model, student: CARDStudent, batch: dict, frontend, teacher, distillation: dict,
                 pad_id: int, device: torch.device) -> dict:
    """Losses for one batch; ``model`` is the student or its DDP wrapper."""
    text_ids = batch["text_ids"].to(device)
    labels = batch["labels"].to(device)
    is_text = batch["is_text_only"]
    audio_mask = ~is_text
    projector_stages = distillation["projector_stages"] if teacher is not None else []
    llm_stages = distillation["llm_stages"] if teacher is not None else []

    zero = lambda: torch.tensor(0.0, device=device)  # noqa: E731
    loss_cap, loss_proj, loss_llm, loss_text, loss_distill = zero(), zero(), zero(), zero(), zero()

    if audio_mask.any():
        idx = audio_mask.nonzero(as_tuple=True)[0]
        waves = batch["waveform"][idx].to(device)
        with torch.no_grad():
            log_mel = frontend(waves)
        audio_tokens = student.audio_projector(log_mel)
        out = model(audio_tokens, text_ids[idx], labels[idx],
                    output_llm_hidden=bool(llm_stages), run_projector_heads=bool(projector_stages))
        loss_cap = out["loss_cap"]
        if teacher is not None:
            with torch.no_grad():
                stages = teacher(waves)
            if llm_stages:
                loss_llm = llm_distill_loss(out["llm_outputs"], [stages[s] for s in llm_stages])
            if projector_stages:
                loss_proj = projector_distill_loss(out["projector_outputs"], [stages[s] for s in projector_stages])
            loss_distill = (float(distillation.get("lambda_proj", 1.0)) * loss_proj
                            + float(distillation.get("lambda_llm", 1.0)) * loss_llm)

    if is_text.any():
        idx = is_text.nonzero(as_tuple=True)[0]
        t_ids = text_ids[idx]
        loss_text = student.forward_text(t_ids, labels[idx], attention_mask=(t_ids != pad_id).long())

    total = audio_mask.float().mean().item() * loss_cap + is_text.float().mean().item() * loss_text + loss_distill
    return {"total": total, "loss_cap": loss_cap, "loss_proj": loss_proj, "loss_llm": loss_llm,
            "loss_text": loss_text}


def _run_phase(cfg: dict, phase: str, student: CARDStudent, frontend, teacher, tokenizer,
               device: torch.device) -> str:
    pc = cfg[phase]
    d = cfg["distillation"]

    train_set = build_train_dataset(
        pc["data"], tokenizer,
        max_audio_sec=float(cfg["model"]["max_audio_sec"]),
        text_manifests=pc.get("text_data"),
        text_max_n=pc.get("num_text_samples"),
    )
    pad_id = tokenizer.pad_token_id or 0
    sampler = HomogeneousBatchSampler(
        train_set, batch_size=int(pc["batch_size"]), shuffle=True, seed=int(pc["seed"]),
        oversample=True, rank=get_rank(), world_size=get_world_size(),
    )
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=int(cfg.get("num_workers", 4)),
                        pin_memory=True, collate_fn=functools.partial(collate_mixed, pad_token_id=pad_id))
    log(f"  {len(train_set):,} rows, {len(loader):,} batches per epoch per GPU (world size {get_world_size()})")

    lr = float(pc["lr"])
    groups = [
        {"params": student.audio_projector.parameters(), "lr": lr},
        {"params": [p for p in student.llm.parameters() if p.requires_grad], "lr": lr},
    ]
    for heads in (student.llm_heads, student.projector_heads):
        if heads is not None:
            groups.append({"params": heads.parameters(), "lr": lr})
    optimizer = torch.optim.AdamW(groups, betas=tuple(pc["betas"]), weight_decay=float(pc["weight_decay"]))
    grad_accum = int(pc["grad_accum"])
    total_steps = max(1, len(loader) * int(pc["epochs"]) // grad_accum)
    scheduler = build_scheduler(optimizer, pc["scheduler"], int(pc["warmup_steps"]), total_steps)

    model = student
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[device.index], find_unused_parameters=True)
    model.train()

    val = cfg.get("validation") or {}
    global_step, t0 = 0, time.time()
    for epoch in range(int(pc["epochs"])):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(loader):
            losses = batch_losses(model, student, batch, frontend, teacher, d, pad_id, device)
            loss = losses["total"] / grad_accum
            if not torch.isfinite(loss):
                log(f"  [warn] non-finite loss at batch {batch_idx}; skipping backward")
                optimizer.zero_grad()
                continue
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               float(pc["grad_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                every = int(val.get("every_steps", 0) or 0)
                if every and global_step % every == 0 and val.get("data") and Path(val["data"]).exists():
                    run_validation(cfg, student, frontend, tokenizer, device, phase, global_step)

            values = {k: v.item() for k, v in losses.items() if k != "total"}
            _wandb_log(cfg, {**{f"{phase}/{k}": v for k, v in values.items()},
                             f"{phase}/lr": scheduler.get_last_lr()[0], f"{phase}/step": global_step})
            if batch_idx % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (len(loader) - batch_idx) / max(1, batch_idx + 1)
                log(f"  {phase} epoch {epoch + 1} [{batch_idx + 1}/{len(loader)}] step {global_step} "
                    f"L_cap={values['loss_cap']:.3f} L_proj={values['loss_proj']:.3f} "
                    f"L_llm={values['loss_llm']:.3f} L_text={values['loss_text']:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.1e} eta={eta / 3600:.1f}h")

        if is_main():
            save_checkpoint(student, Path(cfg["output_dir"]) / phase / f"epoch-{epoch + 1}")
        if is_distributed():
            dist.barrier()

    log(f"  {phase} finished in {(time.time() - t0) / 60:.1f} min")
    return str(Path(cfg["output_dir"]) / phase / f"epoch-{pc['epochs']}")


def train_phase1(cfg: dict, device: torch.device) -> str:
    log("=" * 60 + "\nPHASE 1: captioning + cross-component distillation\n" + "=" * 60)
    student, frontend, teacher, tokenizer = build_phase1(cfg, device)
    return _run_phase(cfg, "phase1", student, frontend, teacher, tokenizer, device)


def train_phase2(cfg: dict, device: torch.device, phase1_ckpt: Optional[str]) -> str:
    log("=" * 60 + "\nPHASE 2: captioning fine-tuning\n" + "=" * 60)
    student, frontend, tokenizer = build_phase2(cfg, device, phase1_ckpt)
    return _run_phase(cfg, "phase2", student, frontend, None, tokenizer, device)


# ----------------------------------------------------------------------------------------------
# Validation during training
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def greedy_caption(student: CARDStudent, frontend, tokenizer, waveform: torch.Tensor,
                   device: torch.device, max_new_tokens: int = 40) -> str:
    audio = student.audio_projector(frontend(waveform.unsqueeze(0).to(device))).to(torch.bfloat16)
    prompt = build_prompt(tokenizer, DEFAULT_USER_PROMPT)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    embeds = torch.cat([audio, student.llm.get_input_embeddings()(ids)], dim=1)
    mask = build_hybrid_attention_mask(1, audio.size(1), ids.size(1), device, dtype=embeds.dtype)
    out = student.llm(inputs_embeds=embeds, attention_mask=mask, use_cache=True, return_dict=True)
    stop = {tokenizer.eos_token_id, *tokenizer.encode("<|im_end|>", add_special_tokens=False)[:1]}
    past, token, generated = out.past_key_values, out.logits[:, -1:].argmax(-1), []
    for _ in range(max_new_tokens):
        if token.item() in stop:
            break
        generated.append(token.item())
        out = student.llm(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        past, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def run_validation(cfg: dict, student: CARDStudent, frontend, tokenizer, device: torch.device,
                   phase: str, step: int) -> None:
    """Validation L_cap and a few greedy captions (main rank)."""
    if not is_main():
        if is_distributed():
            dist.barrier()
        return
    val = cfg["validation"]
    student.eval()
    dataset = AudioCaptionDataset(val["data"], tokenizer, max_n=int(val.get("num_samples", 50)),
                                  max_audio_sec=float(cfg["model"]["max_audio_sec"]))
    losses = []
    for i in range(len(dataset)):
        item = dataset[i]
        tokens = student.audio_projector(frontend(item["waveform"].unsqueeze(0).to(device)))
        out = student(tokens, item["text_ids"].unsqueeze(0).to(device), item["labels"].unsqueeze(0).to(device))
        if torch.isfinite(out["loss_cap"]):
            losses.append(out["loss_cap"].item())
    mean_loss = sum(losses) / max(1, len(losses))
    log(f"  [validation @ step {step}] L_cap={mean_loss:.4f} over {len(losses)} clips")
    for i in range(min(int(val.get("num_generations", 5)), len(dataset))):
        item = dataset[i]
        caption = greedy_caption(student, frontend, tokenizer, item["waveform"], device)
        log(f"    ref: {item['caption'][:80]}\n    gen: {caption[:80]}")
    _wandb_log(cfg, {f"{phase}/val_loss_cap": mean_loss, f"{phase}/step": step})
    student.train()
    if is_distributed():
        dist.barrier()


# ----------------------------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------------------------

def merge(cfg: dict, device: torch.device, phase1_ckpt: Optional[str], phase2_ckpt: Optional[str],
          out_dir: Optional[Path] = None) -> Optional[Path]:
    """Merge Phase 1 then Phase 2 adapters and write a folder readable by ``load_card``."""
    if not is_main():
        return None
    if not phase1_ckpt and not phase2_ckpt:
        raise ValueError("merge needs a Phase 1 and/or a Phase 2 checkpoint")
    adapters, projector_state = [], None
    if phase1_ckpt:
        adapter, projector_state = resolve_checkpoint(phase1_ckpt, phase=1)
        adapters.append(("phase1_lora", adapter))
    if phase2_ckpt:
        adapter, projector_state = resolve_checkpoint(phase2_ckpt, phase=2)
        adapters.append(("phase2_lora", adapter))
    phase = 2 if phase2_ckpt else 1
    out_dir = Path(out_dir or Path(cfg["output_dir"]) / ("merged" if phase == 2 else "merged_phase1"))
    log(f"  merging {', '.join(name for name, _ in adapters)} -> {out_dir}")

    llm = merge_adapters(cfg["model"]["llm"], [a for _, a in adapters], torch.bfloat16, device)
    llm.save_pretrained(str(out_dir / "llm"))
    load_tokenizer(cfg).save_pretrained(str(out_dir / "llm"))
    for name, adapter in adapters:
        shutil.copytree(adapter, out_dir / "adapters" / name, dirs_exist_ok=True)
    save_file({k: v.contiguous() for k, v in projector_state.items()}, str(out_dir / PROJECTOR_FILE))

    shape = AudioProjector.infer_shape(projector_state)
    frontend = {"type": cfg["model"]["frontend"], "clap_model": cfg["model"]["teacher"],
                "n_mels": shape["n_mels"], "max_audio_sec": float(cfg["model"]["max_audio_sec"])}
    if cfg["model"]["frontend"] == "clap_processor":
        frontend["sample_rate"] = 48000
    CARDConfig(
        variant=cfg["variant"], phase=phase, finetune_dataset=cfg["phase2"].get("dataset") if phase == 2 else None,
        base_llm=cfg["model"]["llm"], llm_hidden_dim=shape["hidden_dim"], frontend=frontend,
        projector={"type": "conv", **shape},
        lora={"r": int(cfg["lora"]["r"]), "alpha": int(cfg["lora"]["alpha"]),
              "target_modules": list(cfg["lora"]["target_modules"])},
    ).save(out_dir / "card_config.json")
    log(f"  wrote {out_dir}")
    return out_dir
