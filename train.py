"""Train CARD.

    bash scripts/launch_ddp.sh configs/card_star.yaml
    bash scripts/launch_ddp.sh configs/clotho/card_star.yaml --phase 2
    python train.py --config configs/card_star.yaml --phase 1
    python train.py --config configs/card_star.yaml --merge-only --phase1-ckpt outputs/card_star/phase1/epoch-1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from card.config import load_config, validate_config
from card.training import (
    get_world_size,
    init_distributed,
    is_distributed,
    is_main,
    log,
    merge,
    train_phase1,
    train_phase2,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="training config (YAML)")
    ap.add_argument("--phase", choices=["all", "1", "2"], default="all",
                    help="all: Phase 1, Phase 2 and merge (default); 1: Phase 1 only; 2: Phase 2 and merge")
    ap.add_argument("--phase1-ckpt", default=None,
                    help="Phase 1 checkpoint for Phase 2 or merging (default: phase2.phase1_checkpoint, "
                         "else <output_dir>/phase1/epoch-N); a published repo id also works")
    ap.add_argument("--phase2-ckpt", default=None, help="Phase 2 checkpoint to merge (with --merge-only)")
    ap.add_argument("--merge-only", action="store_true", help="only merge existing checkpoints")
    ap.add_argument("--device", default="cuda:0", help="device when not launched with torchrun")
    args = ap.parse_args()

    cfg = validate_config(load_config(args.config))
    device = init_distributed() or torch.device(args.device)

    if is_main():
        world = get_world_size()
        log("=" * 60)
        log(f"CARD  variant={cfg['variant']}  config={args.config}  output={cfg['output_dir']}")
        log(f"LLM {cfg['model']['llm']} | LoRA r={cfg['lora']['r']} alpha={cfg['lora']['alpha']} "
            f"on {cfg['lora']['num_layers']} blocks | frontend {cfg['model']['frontend']} | world size {world}")
        for phase in ("phase1", "phase2"):
            pc = cfg[phase]
            log(f"{phase}: {pc['epochs']} epoch(s), batch {pc['batch_size']} x grad_accum {pc['grad_accum']} "
                f"x {world} GPU(s) = {pc['batch_size'] * pc['grad_accum'] * world}, lr {pc['lr']}, "
                f"{pc['scheduler']} ({pc['warmup_steps']} warmup steps)")
        log("=" * 60)
        if (cfg.get("wandb") or {}).get("enabled"):
            import wandb

            wandb.init(project=cfg["wandb"].get("project", "card"),
                       name=cfg["wandb"].get("run_name", cfg["variant"]), config=cfg)

    # phase2.phase1_checkpoint: null starts Phase 2 from the base model.
    if args.phase1_ckpt:
        phase1_ckpt = args.phase1_ckpt
    elif "phase1_checkpoint" in cfg["phase2"]:
        phase1_ckpt = cfg["phase2"]["phase1_checkpoint"]
    else:
        phase1_ckpt = str(Path(cfg["output_dir"]) / "phase1" / f"epoch-{cfg['phase1']['epochs']}")

    if args.merge_only:
        merge(cfg, device, phase1_ckpt, args.phase2_ckpt)
    else:
        if args.phase in ("all", "1"):
            phase1_ckpt = train_phase1(cfg, device)
        if args.phase in ("all", "2"):
            phase2_ckpt = train_phase2(cfg, device, phase1_ckpt)
            merge(cfg, device, phase1_ckpt, phase2_ckpt)

    if is_main() and (cfg.get("wandb") or {}).get("enabled"):
        import wandb

        wandb.finish()
    if is_distributed():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
