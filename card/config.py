"""YAML training configs. ``base: <path>`` inherits from another config (mappings are merged)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Union

import yaml

REQUIRED = ("variant", "output_dir", "model", "lora", "distillation", "phase1", "phase2")


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Union[str, Path]) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("base", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return cfg


def validate_config(cfg: dict) -> dict:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config is missing: {', '.join(missing)}")
    d = cfg["distillation"]
    for key in ("projector_stages", "llm_stages"):
        stages = list(d.get(key) or [])
        bad = [s for s in stages if not 0 <= int(s) < 4]
        if bad:
            raise ValueError(f"distillation.{key} has invalid teacher stages {bad}; valid: 0-3")
        d[key] = [int(s) for s in stages]
    return cfg
