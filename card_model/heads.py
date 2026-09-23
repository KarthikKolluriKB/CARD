"""Distillation heads, used in Phase 1 only.

Projector heads: one MLP per stage (2560 -> 768 -> d_i, GELU), mean-pooled over time.
LLM heads: RMSNorm + linear per stage, applied to the average hidden state of its LLM blocks.
For CARD* (t2, t3) the blocks are 0-8 and 9-11.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class ProjectorHeads(nn.Module):
    """One MLP per teacher stage routed to the audio projector; outputs are mean-pooled over time."""

    def __init__(self, stage_dims: Sequence[int], input_dim: int = 2560, hidden_dim: int = 768):
        super().__init__()
        if not stage_dims:
            raise ValueError("ProjectorHeads needs at least one stage")
        self.stage_dims = tuple(int(d) for d in stage_dims)
        self.hidden_dim = hidden_dim
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d))
            for d in self.stage_dims
        ])
        self._register_load_state_dict_pre_hook(self._rename_legacy_keys)

    @staticmethod
    def _rename_legacy_keys(state_dict, prefix, *args):
        for key in [k for k in state_dict if k.startswith(prefix + "head_s")]:
            index, rest = key[len(prefix) + len("head_s"):].split(".", 1)
            state_dict[f"{prefix}heads.{index}.{rest}"] = state_dict.pop(key)

    def forward(self, audio_tokens: torch.Tensor) -> List[torch.Tensor]:
        """(B, T, input_dim) -> one (B, d_i) tensor per stage."""
        return [head(audio_tokens).mean(dim=1) for head in self.heads]


class LLMHead(nn.Module):
    """RMSNorm followed by a linear projection to one teacher stage dimension."""

    def __init__(self, llm_dim: int = 2560, stage_dim: int = 768):
        super().__init__()
        self.norm = nn.RMSNorm(llm_dim)
        self.proj = nn.Linear(llm_dim, stage_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden))


def allocate_blocks(stage_depths: Sequence[int], num_blocks: int) -> List[Tuple[int, int]]:
    """Contiguous LLM block range per stage, proportional to the stage depths."""
    total = sum(stage_depths)
    n = len(stage_depths)
    ranges, start = [], 0
    if num_blocks == total:
        for depth in stage_depths:
            ranges.append((start, start + depth))
            start += depth
        return ranges
    remaining = num_blocks
    for i, depth in enumerate(stage_depths):
        if i == n - 1:
            count = max(1, remaining)
        else:
            count = max(1, round(num_blocks * depth / total))
            count = min(count, remaining - (n - 1 - i))
        ranges.append((start, start + count))
        start += count
        remaining -= count
    return ranges


class LLMHeads(nn.Module):
    """One ``LLMHead`` per teacher stage routed to the LLM, each over its own range of LLM blocks."""

    def __init__(self, stage_dims: Sequence[int], stage_depths: Sequence[int],
                 llm_dim: int = 2560, num_blocks: int = 12):
        super().__init__()
        if len(stage_dims) != len(stage_depths) or not stage_dims:
            raise ValueError("stage_dims and stage_depths must be non-empty and the same length")
        self.num_blocks = num_blocks
        self.stage_block_ranges = allocate_blocks(stage_depths, num_blocks)
        self.heads = nn.ModuleList([LLMHead(llm_dim, d) for d in stage_dims])

    def forward(self, block_hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """``num_blocks`` tensors (B, T_audio, llm_dim) -> one (B, T_audio, d_i) tensor per stage."""
        return [
            head(torch.stack(block_hidden_states[start:end]).mean(dim=0))
            for head, (start, end) in zip(self.heads, self.stage_block_ranges)
        ]
