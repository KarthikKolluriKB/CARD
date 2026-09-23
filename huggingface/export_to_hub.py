#!/usr/bin/env python
"""Export a trained CARD run into a Hugging Face Hub repository folder.

Input: a training output directory laid out as the trainer writes it::

    <run>/stage1/epoch-N/{lora_adapter/, audio_embedding.pt, aux_heads.pt, patch_distill_head.pt}
    <run>/stage2/epoch-N/{lora_adapter/, audio_embedding.pt}
    <run>/merged/{llm/, audio_embedding.pt}

("stage1"/"stage2" are only the on-disk directory names the trainer uses for the
Phase 1 / Phase 2 checkpoints; everything else here says Phase.)

Output: a folder that ``card.hub.load_card`` and the model card expect::

    <out>/card_config.json
    <out>/generation_config.json
    <out>/audio_projector.safetensors
    <out>/adapters/phase1_lora/            (Phase 1 and Phase 2 exports)
    <out>/adapters/phase2_lora/            (Phase 2 exports)
    <out>/projector_heads.safetensors      (Phase 1 exports only: ProjectorHead MLPs, training-time)
    <out>/llm_heads.safetensors            (Phase 1 exports only: LLMHead linears, training-time)
    <out>/llm/                             (--include-merged: merged bf16 LLM + tokenizer)
    <out>/README.md
    <out>/MANIFEST.sha256

Examples::

    # headline CARD* AudioCaps, full merged model, verify, no upload
    python huggingface/export_to_hub.py \\
        --run outputs/card_star_audiocaps --out hub/CARD-Qwen3-4B-AudioCaps \\
        --variant card_star --phase 2 --dataset audiocaps --include-merged \\
        --readme huggingface/model_cards/CARD-Qwen3-4B-AudioCaps.md \\
        --check-merge --check-audio samples/*.wav --expected-captions samples/expected.json

    # Phase 2 fork whose Phase 1 checkpoint lives in another run
    python huggingface/export_to_hub.py --run outputs/card_star_clotho --phase1-run outputs/card_star_audiocaps \\
        --out hub/CARD-Qwen3-4B-Clotho --variant card_star --phase 2 --dataset clotho --include-merged

    # Phase 1 checkpoint only
    python huggingface/export_to_hub.py --run outputs/card_star_audiocaps --out hub/CARD-Qwen3-4B-Phase1 \\
        --variant card_star --phase 1

    # ablation bundle (adapters only) into a subfolder of the shared ablations repo
Nothing here talks to the Hub. Upload with ``upload_to_hub.py`` once the checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from card_model.hub import (  # noqa: E402
    CARD_CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    LLM_SUBDIR,
    PHASE1_ADAPTER,
    PHASE2_ADAPTER,
    PROJECTOR_NAME,
    CARDConfig,
    load_card,
)
from card_model.modeling import AudioProjector  # noqa: E402

VARIANT_LABELS = {
    "card_star": "CARD* (early stages -> projector, later stages -> LLM)",
    "card_diamond": "CARD-diamond (all stages -> projector, later stages -> LLM)",
    "proj_early": "Proj Early (early stages -> projector only)",
    "proj_full": "Proj Full (all stages -> projector only)",
    "reversed": "Reversed Routing (early stages -> LLM, later stages -> projector)",
    "llm_distill": "LLM Distill (teacher -> LLM only)",
    "no_distill": "No Distill (no teacher)",
}
DATASET_LABELS = {"audiocaps": "AudioCaps", "clotho": "Clotho"}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def latest_epoch_dir(phase_dir: Path, epoch: Optional[int]) -> Path:
    if epoch is not None:
        d = phase_dir / f"epoch-{epoch}"
        if not d.is_dir():
            raise FileNotFoundError(d)
        return d
    candidates = sorted(
        (p for p in phase_dir.glob("epoch-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No epoch-* directories under {phase_dir}")
    return candidates[-1]


def pt_to_safetensors(src: Path, dst: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{src} is not a state dict")
    state = {k: v.detach().contiguous() for k, v in state.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(dst))
    log(f"converted {src.name} -> {dst.name} ({sum(v.numel() for v in state.values()) / 1e6:.1f}M params)")
    return state


def copy_adapter(src: Path, dst: Path, expected_r: Optional[int], expected_alpha: Optional[int]) -> dict:
    if not (src / "adapter_config.json").exists():
        raise FileNotFoundError(f"{src} has no adapter_config.json")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "adapter_config.json", dst / "adapter_config.json")
    if (src / "adapter_model.safetensors").exists():
        shutil.copy2(src / "adapter_model.safetensors", dst / "adapter_model.safetensors")
    elif (src / "adapter_model.bin").exists():
        state = torch.load(src / "adapter_model.bin", map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in state.items()}, str(dst / "adapter_model.safetensors"))
        log(f"converted {src / 'adapter_model.bin'} to safetensors")
    else:
        raise FileNotFoundError(f"{src} has no adapter weights")
    with open(dst / "adapter_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if expected_r is not None and int(cfg.get("r", -1)) != expected_r:
        raise ValueError(f"{src}: adapter r={cfg.get('r')} but config says {expected_r}")
    if expected_alpha is not None and int(cfg.get("lora_alpha", -1)) != expected_alpha:
        raise ValueError(f"{src}: adapter alpha={cfg.get('lora_alpha')} but config says {expected_alpha}")
    return cfg


def copy_merged_llm(src: Path, dst: Path, base_llm: str, adapter_dirs: List[Path],
                    device: str) -> bool:
    """Copy the merged LLM into ``dst``. Returns True if it had to be rebuilt.

    Three cases: safetensors shards are copied as-is; pickle ``.bin`` shards are re-saved as
    safetensors; no shards at all (only config + tokenizer left behind) means the merged
    weights are rebuilt from ``base_llm`` + the Phase 1 and Phase 2 adapters, in that order,
    on ``device``. The trainer merged on the GPU in bf16, so rebuild there too: PEFT merges
    bf16 weights in fp32 on CPU but in bf16 on CUDA, and the two differ in the last bit.
    """
    dst.mkdir(parents=True, exist_ok=True)
    shards = list(src.glob("*.safetensors")) if src.is_dir() else []
    if shards:
        for p in src.iterdir():
            if p.is_file() and p.suffix not in (".bin", ".pt", ".pth"):
                shutil.copy2(p, dst / p.name)
        log(f"copied merged LLM ({len(shards)} safetensors shards, "
            f"{sum(p.stat().st_size for p in shards) / 1e9:.2f} GB)")
        return False

    bins = list(src.glob("*.bin")) if src.is_dir() else []
    if bins:
        log(f"merged LLM is stored as {len(bins)} pickle shard(s); re-saving as safetensors "
            f"(loads the model on CPU, needs about 8 GB RAM)")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(str(src), torch_dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True)
        rebuilt = False
    else:
        listing = ", ".join(sorted(p.name for p in src.iterdir())) if src.is_dir() else "(missing)"
        log(f"merged LLM has no weight shards under {src} (contents: {listing}); rebuilding "
            f"from {base_llm} + {len(adapter_dirs)} adapters on {device}")
        if device == "cpu":
            log("WARNING: rebuilding on CPU; PEFT merges bf16 in fp32 there, so the result can "
                "differ from the GPU-merged model in the last bit. Prefer --device cuda.")
        from card_model.hub import merge_adapters

        model = merge_adapters(base_llm, adapter_dirs, dtype=torch.bfloat16, device=device)
        rebuilt = True

    model.save_pretrained(str(dst), safe_serialization=True, max_shard_size="5GB")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # tokenizer and side files: from the merged dir if present, else from the base model
    side = [p for p in src.iterdir() if p.is_file() and p.suffix in (".json", ".txt", ".model", ".jinja")
            and p.name not in ("pytorch_model.bin.index.json", "model.safetensors.index.json",
                               "config.json", "generation_config.json")] if src.is_dir() else []
    if side:
        for p in side:
            shutil.copy2(p, dst / p.name)
    else:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(base_llm).save_pretrained(str(dst))
    new_shards = list(dst.glob("*.safetensors"))
    if not new_shards:
        raise RuntimeError(f"no safetensors shards were written to {dst}")
    log(f"{'rebuilt' if rebuilt else 're-saved'} merged LLM ({len(new_shards)} safetensors shards, "
        f"{sum(p.stat().st_size for p in new_shards) / 1e9:.2f} GB)")
    return rebuilt


def write_generation_config(path: Path, decoding: dict, eos_ids, pad_id, bos_id) -> None:
    gen = {
        "do_sample": False,
        "num_beams": int(decoding.get("num_beams", 4)),
        "max_new_tokens": int(decoding.get("max_new_tokens", 40)),
        "eos_token_id": eos_ids,
        "pad_token_id": pad_id,
        "bos_token_id": bos_id,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in gen.items() if v is not None}, f, indent=2)
        f.write("\n")


def read_yaml_config(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_card_config(args, proj_shape: dict, train_cfg: dict, adapter_cfg: Optional[dict]) -> CARDConfig:
    n_mels = proj_shape["n_mels"]
    mode = (train_cfg.get("audio_input_mode") or ("clap_processor" if n_mels == 64 else "torchaudio_16k")).lower()
    if mode in ("clap_processor", "clap"):
        frontend = {"type": "clap_processor", "clap_model": train_cfg.get("clap_model", "laion/clap-htsat-fused"),
                    "sample_rate": 48000, "n_mels": 64, "max_audio_sec": float(train_cfg.get("max_audio_sec", 10.0))}
    elif mode in ("torchaudio_16k", "torchaudio", "legacy"):
        frontend = {"type": "torchaudio_16k", "sample_rate": 16000, "n_mels": 80,
                    "max_audio_sec": float(train_cfg.get("max_audio_sec", 10.0))}
    else:
        raise ValueError(f"audio_input_mode {mode!r} is not exportable (encoder-kept variant)")
    if frontend["n_mels"] != n_mels:
        raise ValueError(f"front end {mode} gives {frontend['n_mels']} mel bands, projector has {n_mels}")

    lora = {
        "r": int(adapter_cfg["r"]) if adapter_cfg else int(train_cfg.get("lora_rank", 16)),
        "alpha": int(adapter_cfg["lora_alpha"]) if adapter_cfg else int(train_cfg.get("lora_alpha", 16)),
        "target_modules": sorted(adapter_cfg["target_modules"]) if adapter_cfg
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    paper = {"variant_label": VARIANT_LABELS.get(args.variant, args.variant)}
    if args.metrics:
        paper["metrics"] = json.loads(args.metrics)
    return CARDConfig(
        variant=args.variant,
        phase=args.phase,
        finetune_dataset=args.dataset if args.phase == 2 else None,
        base_llm=train_cfg.get("llm_model", "Qwen/Qwen3-4B"),
        llm_hidden_dim=proj_shape["hidden_dim"],
        frontend=frontend,
        projector={"type": "conv", **proj_shape},
        lora=lora,
        paper=paper,
    )


def render_stub_readme(out: Path, cfg: CARDConfig, args, included_merged: bool) -> None:
    """Minimal but valid model card. Replace with a full card via --readme before publishing."""
    name = out.name
    dataset = DATASET_LABELS.get(args.dataset or "", args.dataset or "")
    relation = "finetune" if included_merged else "adapter"
    lines = [
        "---",
        "license: apache-2.0",
        "language:",
        "  - en",
        "pipeline_tag: audio-text-to-text",
        f"base_model: {cfg.base_llm}",
        f"base_model_relation: {relation}",
        "tags:",
        "  - audio-captioning",
        "  - encoder-free",
        "  - knowledge-distillation",
        "  - lora",
        "  - qwen3",
        "  - slt-2026",
        "---",
        "",
        f"# {name}",
        "",
        "Encoder-free audio captioning model from \"CARD: Cross-component Audio Representation "
        "Distillation for Encoder-Free Audio Captioning\" (IEEE SLT 2026).",
        "Code: https://github.com/KarthikKolluriKB/CARD-Encoder-Free-Audio-Captioning",
        "",
        f"Variant: {cfg.paper.get('variant_label', cfg.variant)}. Phase {cfg.phase}"
        + (f", fine-tuned on {dataset}." if cfg.phase == 2 else " checkpoint (before Phase 2)."),
        "",
        "TODO: replace this stub with the full model card (docs/hf_model_card_template.md).",
        "",
        "```python",
        "from card_model import load_card",
        f"model = load_card(\"KarthikKB1998/{name}\"" + ("" if included_merged else ", mode=\"adapters\"") + ")",
        "print(model.caption(\"clip.wav\"))",
        "```",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(out: Path) -> None:
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rows.append(f"{h.hexdigest()}  {p.relative_to(out).as_posix()}")
    (out / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
    log(f"wrote MANIFEST.sha256 ({len(rows)} files)")


def scan_forbidden(out: Path) -> List[str]:
    bad = []
    for p in out.rglob("*"):
        if p.is_file() and p.suffix in (".pt", ".pth", ".bin", ".ckpt", ".pkl"):
            bad.append(str(p.relative_to(out)))
    for p in out.rglob("*.json"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"hf_[A-Za-z0-9]{20,}", text):
            bad.append(f"{p.relative_to(out)} (looks like a Hub token)")
    return bad


# ----------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------

def check_merge(out: Path, cfg: CARDConfig, device: str) -> bool:
    """Rebuild the LLM from base + adapters and compare with the exported merged shards."""
    from card_model.hub import merge_adapters

    log("check-merge: rebuilding LLM from base + adapters (this loads Qwen3-4B twice)")
    adapters = [out / p for p in (PHASE1_ADAPTER, PHASE2_ADAPTER) if (out / p).is_dir()]
    rebuilt = merge_adapters(cfg.base_llm, adapters, dtype=torch.bfloat16, device=device)
    rebuilt_sd = {k: v.detach().to("cpu") for k, v in rebuilt.state_dict().items()}
    del rebuilt

    exported: Dict[str, torch.Tensor] = {}
    for shard in sorted((out / LLM_SUBDIR).glob("*.safetensors")):
        exported.update(load_file(str(shard)))

    missing = sorted(set(rebuilt_sd) - set(exported))
    extra = sorted(set(exported) - set(rebuilt_sd))
    # tied embeddings: lm_head may be absent from the shards
    missing = [k for k in missing if k != "lm_head.weight"]
    worst = 0.0
    n_diff = 0
    for k, v in exported.items():
        if k not in rebuilt_sd:
            continue
        d = (v.float() - rebuilt_sd[k].float()).abs().max().item()
        worst = max(worst, d)
        if d > 0:
            n_diff += 1
    ok = not missing and not extra and worst == 0.0
    log(f"check-merge: {'PASS' if ok else 'FAIL'} tensors={len(exported)} differing={n_diff} "
        f"max_abs_diff={worst:.3e} missing={missing[:3]} extra={extra[:3]}")
    if not ok and worst > 0 and worst < 1e-2 and not missing and not extra:
        log("check-merge: differences are tiny; likely a peft/transformers version drift, "
            "re-run inside the pinned environment before publishing")
    return ok


def check_audio(out: Path, mode: str, audio_files: List[str], expected_path: Optional[str],
                device: str) -> bool:
    log(f"check-audio: loading exported model in mode={mode}")
    model = load_card(out, mode=mode, device=device)
    log(f"check-audio: {model}")
    expected = {}
    if expected_path:
        with open(expected_path, "r", encoding="utf-8") as f:
            expected = json.load(f)
    ok = True
    for f in audio_files:
        cap = model.caption(f)
        want = expected.get(f) or expected.get(Path(f).name)
        status = ""
        if want is not None:
            same = cap.strip() == want.strip()
            ok &= same
            status = " [match]" if same else f" [MISMATCH, expected: {want!r}]"
        log(f"  {Path(f).name}: {cap!r}{status}")
    log(f"check-audio: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="training output dir (contains stage1/, stage2/, merged/)")
    p.add_argument("--out", required=True, help="destination folder (becomes the Hub repo content)")
    p.add_argument("--variant", required=True, choices=sorted(VARIANT_LABELS), help="Table II row")
    p.add_argument("--phase", type=int, required=True, choices=(1, 2))
    p.add_argument("--dataset", choices=sorted(DATASET_LABELS), help="Phase 2 fine-tuning dataset")
    p.add_argument("--phase1-run", help="run dir holding the Phase 1 checkpoint if it differs from --run")
    p.add_argument("--phase1-epoch", type=int, help="Phase 1 epoch to export (default: latest)")
    p.add_argument("--phase2-epoch", type=int, help="Phase 2 epoch to export (default: latest)")
    p.add_argument("--train-config", type=Path, help="training YAML; fills base_llm, front end, LoRA")
    p.add_argument("--include-merged", action="store_true", help="copy merged/llm (about 8 GB)")
    p.add_argument("--readme", type=Path, help="model card to copy in as README.md")
    p.add_argument("--metrics", help='JSON dict stored under paper.metrics, e.g. \'{"cider_d": 55.4}\'')
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check-merge", action="store_true", help="rebuild from adapters and compare to llm/")
    p.add_argument("--check-audio", nargs="*", default=[], help="wav files to caption after export")
    p.add_argument("--check-mode", default=None, choices=("merged", "adapters"),
                   help="which loader path --check-audio uses (default: merged if exported)")
    p.add_argument("--expected-captions", help="JSON {filename: caption} for --check-audio")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run = Path(args.run)
    out = Path(args.out)
    if args.phase == 2 and not args.dataset:
        raise SystemExit("--dataset is required for --phase 2")
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} is not empty (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    train_cfg = read_yaml_config(args.train_config)
    phase1_root = Path(args.phase1_run) if args.phase1_run else run
    phase1_dir = latest_epoch_dir(phase1_root / "stage1", args.phase1_epoch)
    log(f"Phase 1 checkpoint: {phase1_dir}")
    phase2_dir = None
    if args.phase == 2:
        phase2_dir = latest_epoch_dir(run / "stage2", args.phase2_epoch)
        log(f"Phase 2 checkpoint: {phase2_dir}")

    # ---- projector (the Phase 2 projector for Phase 2 exports, else the Phase 1 one)
    proj_src = (phase2_dir or phase1_dir) / "audio_embedding.pt"
    if args.phase == 2 and (run / "merged" / "audio_embedding.pt").exists():
        proj_src = run / "merged" / "audio_embedding.pt"
    proj_state = pt_to_safetensors(proj_src, out / PROJECTOR_NAME)
    if "linear1.weight" in proj_state:
        raise SystemExit("encoder-kept projector found; this exporter only handles the conv projector")
    proj_shape = AudioProjector.infer_shape(proj_state)
    log(f"projector shape: {proj_shape}")

    # ---- adapters
    lora_r = int(train_cfg["lora_rank"]) if "lora_rank" in train_cfg else None
    lora_a = int(train_cfg["lora_alpha"]) if "lora_alpha" in train_cfg else None
    adapter_cfg = copy_adapter(phase1_dir / "lora_adapter", out / PHASE1_ADAPTER, lora_r, lora_a)
    log("copied Phase 1 adapter")
    if phase2_dir is not None:
        copy_adapter(phase2_dir / "lora_adapter", out / PHASE2_ADAPTER, lora_r, lora_a)
        log("copied Phase 2 adapter")
    else:
        # training-time distillation heads, renamed to the paper's names
        for src_name, dst_name in (("patch_distill_head.pt", "projector_heads.safetensors"),
                                   ("aux_heads.pt", "llm_heads.safetensors")):
            src = phase1_dir / src_name
            if src.exists():
                pt_to_safetensors(src, out / dst_name)

    # ---- config
    cfg = build_card_config(args, proj_shape, train_cfg, adapter_cfg)
    cfg.save(out / CARD_CONFIG_NAME)
    log(f"wrote {CARD_CONFIG_NAME}")

    # ---- merged LLM + generation config
    included_merged = False
    rebuilt = False
    eos_ids, pad_id, bos_id = None, None, None
    if args.include_merged:
        if args.phase != 2:
            raise SystemExit("--include-merged only makes sense for Phase 2 exports")
        adapter_dirs = [out / PHASE1_ADAPTER, out / PHASE2_ADAPTER]
        rebuilt = copy_merged_llm(run / "merged" / "llm", out / LLM_SUBDIR, cfg.base_llm,
                                  adapter_dirs, args.device)
        included_merged = True
        if rebuilt:
            cfg.paper["merged_llm_provenance"] = (
                "rebuilt at export from the base model and the Phase 1 + Phase 2 adapters; "
                "the original merged shards were not available")
            cfg.save(out / CARD_CONFIG_NAME)
        base_gen = out / LLM_SUBDIR / GENERATION_CONFIG_NAME
        if base_gen.exists():
            with open(base_gen, "r", encoding="utf-8") as f:
                g = json.load(f)
            eos_ids, pad_id, bos_id = g.get("eos_token_id"), g.get("pad_token_id"), g.get("bos_token_id")
        write_generation_config(base_gen, cfg.decoding, eos_ids, pad_id, bos_id)
    write_generation_config(out / GENERATION_CONFIG_NAME, cfg.decoding, eos_ids, pad_id, bos_id)
    log(f"wrote {GENERATION_CONFIG_NAME} (beam {cfg.decoding['num_beams']}, "
        f"{cfg.decoding['max_new_tokens']} tokens, no sampling)")

    # ---- README
    if args.readme:
        shutil.copy2(args.readme, out / "README.md")
        log(f"copied model card from {args.readme}")
    else:
        render_stub_readme(out, cfg, args, included_merged)
        log("wrote stub README.md (replace before publishing)")

    # ---- hygiene
    bad = scan_forbidden(out)
    if bad:
        raise SystemExit("forbidden files in export: " + ", ".join(bad))
    write_manifest(out)

    # ---- checks
    ok = True
    if args.check_merge:
        if not included_merged:
            raise SystemExit("--check-merge needs --include-merged")
        if rebuilt:
            log("check-merge: skipped, the merged LLM was itself rebuilt from the adapters; "
                "verify with --check-audio or by re-running the evaluation instead")
        else:
            ok &= check_merge(out, cfg, args.device)
    if args.check_audio:
        mode = args.check_mode or ("merged" if included_merged else "adapters")
        ok &= check_audio(out, mode, args.check_audio, args.expected_captions, args.device)

    size_gb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
    log(f"done: {out} ({size_gb:.2f} GB) {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
