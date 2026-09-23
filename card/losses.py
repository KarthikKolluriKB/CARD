"""Distillation losses, l_distill(u, t) = 1 - cos(u, t), averaged over the routed stages.

L_proj compares time-pooled vectors. L_llm interpolates the teacher stage to the audio-token
length and averages the cosine over positions.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def projector_distill_loss(student: List[torch.Tensor], teacher: List[torch.Tensor]) -> torch.Tensor:
    """L_proj. ``student``: pooled (B, d_i) head outputs; ``teacher``: (B, T_i, d_i) stage outputs."""
    losses = []
    for u, t in zip(student, teacher, strict=True):
        cos = F.cosine_similarity(u.float(), t.float().mean(dim=1), dim=-1)
        losses.append((1.0 - cos).mean())
    return sum(losses) / len(losses)


def llm_distill_loss(student: List[torch.Tensor], teacher: List[torch.Tensor]) -> torch.Tensor:
    """L_llm. ``student``: (B, T_audio, d_i) head outputs; ``teacher``: (B, T_i, d_i) stage outputs."""
    total = 0.0
    for u, t in zip(student, teacher, strict=True):
        num_tokens = u.size(1)
        if t.size(1) != num_tokens:
            t = F.interpolate(t.transpose(1, 2), size=num_tokens, mode="linear",
                              align_corners=False).transpose(1, 2)
        cos = (F.normalize(u.float(), dim=-1) * F.normalize(t.float(), dim=-1)).sum(dim=-1)
        total = total + (1.0 - cos).mean()
    return total / len(student)
