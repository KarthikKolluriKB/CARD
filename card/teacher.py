"""Frozen CLAP-HTSAT teacher. Its four stage outputs t0..t3 (192, 384, 768, 768 dims) are the
distillation targets in Phase 1."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

CLAP_STAGE_DIMS = (192, 384, 768, 768)
CLAP_STAGE_DEPTHS = (2, 2, 6, 2)


class ClapTeacher(nn.Module):
    """Takes the student's 16 kHz waveform; the spectrogram matches ``ClapMelFrontend``."""

    def __init__(self, model_name: str = "laion/clap-htsat-fused",
                 torch_dtype: torch.dtype = torch.float32,
                 source_sr: int = 16000, target_sr: int = 48000):
        super().__init__()
        from transformers import ClapModel, ClapProcessor

        self.model_name = model_name
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.clap_model = ClapModel.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.clap_model.eval()
        for p in self.clap_model.parameters():
            p.requires_grad = False
        self.source_sr = source_sr
        self.target_sr = target_sr
        self.stage_dims = CLAP_STAGE_DIMS
        self.stage_depths = CLAP_STAGE_DEPTHS

    def train(self, mode: bool = True):
        super().train(mode)
        self.clap_model.eval()
        return self

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """(B, T) waveform at ``source_sr`` -> [t0, t1, t2, t3], each (B, T_i, C_i)."""
        import torchaudio

        resample = torchaudio.transforms.Resample(self.source_sr, self.target_sr).to(waveform.device)
        wave = resample(waveform)
        inputs = self.processor(
            audios=list(wave.cpu().numpy()),
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        device = self.clap_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.clap_model.audio_model(**inputs, output_hidden_states=True)

        stages = []
        for feat in outputs.hidden_states[1:len(self.stage_dims) + 1]:
            b, c, h, w = feat.shape
            stages.append(feat.permute(0, 2, 3, 1).reshape(b, h * w, c))  # (B, H*W, C)
        return stages
