"""CARD student: audio projector and LoRA-adapted LLM, with the Phase 1 distillation heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from card_model.heads import LLMHeads, ProjectorHeads
from card_model.modeling import AudioProjector, build_hybrid_attention_mask


def init_projector_to_match_llm(projector: AudioProjector, llm_embed_table: torch.Tensor) -> None:
    """Initialise the projector so its output scale matches the LLM token embeddings."""
    target_std = llm_embed_table.float().std().item()
    nn.init.normal_(projector.conv1.weight, std=0.01)
    nn.init.zeros_(projector.conv1.bias)
    nn.init.normal_(projector.conv2.weight, std=0.01)
    nn.init.zeros_(projector.conv2.bias)
    nn.init.normal_(projector.proj.weight, std=target_std / (projector.hidden_dim ** 0.5))
    nn.init.zeros_(projector.proj.bias)
    nn.init.normal_(projector.pos_embed.weight, std=target_std * 0.02)


class CARDStudent(nn.Module):
    """LLM (PEFT-wrapped during training), audio projector and optional distillation heads."""

    def __init__(self, llm: nn.Module, audio_projector: AudioProjector,
                 projector_heads: Optional[ProjectorHeads] = None,
                 llm_heads: Optional[LLMHeads] = None):
        super().__init__()
        self.llm = llm
        self.audio_projector = audio_projector
        self.projector_heads = projector_heads
        self.llm_heads = llm_heads

    def forward(self, audio_tokens: torch.Tensor, text_ids: torch.Tensor, labels: torch.Tensor,
                output_llm_hidden: bool = False, run_projector_heads: bool = False) -> dict:
        """Captioning loss on [audio tokens | text], plus the requested head outputs."""
        batch, num_audio, _ = audio_tokens.shape

        projector_outputs = None
        if run_projector_heads and self.projector_heads is not None:
            projector_outputs = self.projector_heads(audio_tokens)

        text_embeds = self.llm.get_input_embeddings()(text_ids)
        embeds = torch.cat([audio_tokens.to(dtype=text_embeds.dtype), text_embeds], dim=1)
        full_labels = torch.full((batch, embeds.size(1)), -100, dtype=torch.long, device=embeds.device)
        full_labels[:, num_audio:] = labels
        mask = build_hybrid_attention_mask(batch, num_audio, text_embeds.size(1), embeds.device,
                                           dtype=embeds.dtype)

        out = self.llm(inputs_embeds=embeds, attention_mask=mask, labels=full_labels,
                       output_hidden_states=output_llm_hidden, return_dict=True)

        llm_outputs = None
        if output_llm_hidden and self.llm_heads is not None:
            head_dtype = next(self.llm_heads.parameters()).dtype
            blocks = [out.hidden_states[i + 1][:, :num_audio, :].to(dtype=head_dtype)
                      for i in range(self.llm_heads.num_blocks)]
            llm_outputs = self.llm_heads(blocks)

        return {
            "loss_cap": out.loss,
            "projector_outputs": projector_outputs,
            "llm_outputs": llm_outputs,
            "num_audio_tokens": num_audio,
        }

    def forward_text(self, input_ids: torch.Tensor, labels: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """LM loss on a text-only batch."""
        out = self.llm(input_ids=input_ids, labels=labels, attention_mask=attention_mask, return_dict=True)
        return out.loss
