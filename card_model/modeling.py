"""Inference-time modules of CARD.

Everything that survives training lives here:

  * ``AudioProjector``      the 13.2 M-parameter convolutional projector (log-Mel -> audio tokens)
  * ``ClapMelFrontend``     the parameter-free CLAP-matched log-Mel front end (48 kHz, 64 bands)
  * ``LegacyMelFrontend``   the 16 kHz / 80-band front end used by the "LLM Distill" ablation only
  * ``build_hybrid_attention_mask``  bidirectional audio / causal text mask used in training
  * ``load_audio``          file or array -> mono 16 kHz waveform cropped or zero-padded to 10 s

Parameter names of ``AudioProjector`` (conv1, conv2, norm, proj, pos_embed) match the training
checkpoints one to one, so ``load_state_dict`` needs no key remapping.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AudioLike = Union[str, Path, np.ndarray, torch.Tensor]


# ----------------------------------------------------------------------------------------------
# Audio projector
# ----------------------------------------------------------------------------------------------

class AudioProjector(nn.Module):
    """Log-Mel spectrogram -> sequence of audio tokens at the LLM hidden size.

    Conv1d(n_mels -> 512, k3 s2 p1) + GELU
    Conv1d(512 -> hidden, k3 s2 p1) + GELU
    LayerNorm(hidden) -> Linear(hidden -> hidden) -> + learned positional embeddings

    Two stride-2 convolutions give 4x temporal downsampling: 1001 CLAP frames (10 s) become
    251 audio tokens. ``max_audio_len`` bounds the positional table (1024 >= 251).
    """

    def __init__(
        self,
        n_mels: int = 64,
        hidden_dim: int = 2560,
        conv_intermediate: int = 512,
        max_audio_len: int = 1024,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(n_mels, conv_intermediate, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(conv_intermediate, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pos_embed = nn.Embedding(max_audio_len, hidden_dim)
        self.act = nn.GELU()

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """(B, T_frames, n_mels) -> (B, T_frames / 4, hidden_dim)."""
        x = log_mel.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.proj(x)
        pos_ids = torch.arange(x.size(1), device=x.device)
        return x + self.pos_embed(pos_ids).unsqueeze(0)

    def n_tokens(self, mel_frames: int) -> int:
        after_conv1 = (mel_frames - 1) // 2 + 1
        return (after_conv1 - 1) // 2 + 1

    @staticmethod
    def infer_shape(state_dict: dict) -> dict:
        """Read (n_mels, hidden_dim, conv_intermediate, max_audio_len) off a checkpoint."""
        w1 = state_dict["conv1.weight"]
        w2 = state_dict["conv2.weight"]
        pe = state_dict["pos_embed.weight"]
        return {
            "n_mels": int(w1.shape[1]),
            "conv_intermediate": int(w1.shape[0]),
            "hidden_dim": int(w2.shape[0]),
            "max_audio_len": int(pe.shape[0]),
        }


# ----------------------------------------------------------------------------------------------
# Front ends (no learned parameters)
# ----------------------------------------------------------------------------------------------

class ClapMelFrontend(nn.Module):
    """CLAP-matched log-Mel spectrogram, identical to what the CLAP-HTSAT teacher consumes.

    waveform 16 kHz (B, T) -> torchaudio resample to 48 kHz -> ``ClapProcessor`` (64 mel bands,
    n_fft 1024, hop 480, 50-14000 Hz) -> (B, T_frames, 64). For a 10 s clip T_frames is 1001.

    The 16 kHz input contract is deliberate: training and evaluation loaded audio at 16 kHz,
    zero-padded it to 10 s, and only then resampled to 48 kHz. Feeding native 48 kHz audio
    straight in would change the spectrogram slightly and drift from the published numbers.
    """

    n_mels = 64

    def __init__(
        self,
        clap_model: str = "laion/clap-htsat-fused",
        source_sr: int = 16000,
        target_sr: int = 48000,
    ):
        super().__init__()
        from transformers import ClapProcessor

        self.processor = ClapProcessor.from_pretrained(clap_model)
        self.source_sr = source_sr
        self.target_sr = target_sr
        self._resampler = None
        self._resampler_device = None

    def _ensure_resampler(self, device: torch.device) -> None:
        import torchaudio

        if self._resampler is None or self._resampler_device != device:
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=self.source_sr, new_freq=self.target_sr
            ).to(device)
            self._resampler_device = device

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        self._ensure_resampler(waveform.device)
        wave_48k = self._resampler(waveform)
        inputs = self.processor(
            audios=list(wave_48k.detach().cpu().numpy()),
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        feats = inputs["input_features"].to(waveform.device)
        if feats.dim() == 4:  # (B, 1, T, 64) for <= 10 s clips; first chunk otherwise
            feats = feats[:, 0]
        return feats.contiguous()


class LegacyMelFrontend(nn.Module):
    """16 kHz / 80-band torchaudio log-Mel with per-utterance normalisation.

    Used only by the "LLM Distill" row of Table II, which predates the CLAP-matched front end.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        import torchaudio

        self.n_mels = n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        self.mel = self.mel.to(waveform.device)
        self.amp_to_db = self.amp_to_db.to(waveform.device)
        log_mel = self.amp_to_db(self.mel(waveform)).transpose(1, 2)
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True) + 1e-5
        return (log_mel - mean) / std


def build_frontend(frontend_type: str, clap_model: str = "laion/clap-htsat-fused") -> nn.Module:
    if frontend_type in ("clap_processor", "clap"):
        return ClapMelFrontend(clap_model=clap_model)
    if frontend_type in ("torchaudio_16k", "legacy"):
        return LegacyMelFrontend()
    raise ValueError(
        f"Unsupported frontend type {frontend_type!r}. Encoder-kept variants (SLAM-AAC "
        "baselines, upper bound) are not part of this loader."
    )


# ----------------------------------------------------------------------------------------------
# Attention mask
# ----------------------------------------------------------------------------------------------

def build_hybrid_attention_mask(
    batch_size: int,
    num_audio_tokens: int,
    num_text_tokens: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """(B, 1, L, L) additive mask: audio<->audio bidirectional, text->audio allowed,
    text->text causal, audio->text blocked. 0 = attend, -inf = blocked."""
    total = num_audio_tokens + num_text_tokens
    t_a = num_audio_tokens
    mask = torch.full((total, total), float("-inf"), device=device, dtype=dtype)
    mask[:t_a, :t_a] = 0.0
    mask[t_a:, :t_a] = 0.0
    mask[t_a:, t_a:] = torch.triu(
        torch.full((num_text_tokens, num_text_tokens), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, total, total)


# ----------------------------------------------------------------------------------------------
# Audio loading
# ----------------------------------------------------------------------------------------------

def load_audio(
    audio: AudioLike,
    sr: Optional[int] = None,
    target_sr: int = 16000,
    max_duration: float = 10.0,
) -> torch.Tensor:
    """Return a mono ``(T,)`` float32 waveform at ``target_sr``, cropped or zero-padded to
    ``max_duration`` seconds. Mirrors the training-time loader exactly.

    ``audio`` may be a file path (anything torchaudio or soundfile can read), or an array /
    tensor shaped ``(T,)``, ``(C, T)`` or ``(T, C)`` together with its sample rate ``sr``.
    """
    if isinstance(audio, (str, Path)):
        waveform, sr = _read_file(Path(audio))
    else:
        if sr is None:
            raise ValueError("sr is required when passing an array or tensor")
        waveform = torch.as_tensor(np.asarray(audio) if isinstance(audio, np.ndarray) else audio)
        waveform = waveform.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.t()  # (T, C) -> (C, T)

    if sr != target_sr:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, int(sr), target_sr)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    target_samples = int(max_duration * target_sr)
    if waveform.size(0) > target_samples:
        warnings.warn(
            f"Audio is {waveform.size(0) / target_sr:.1f} s; CARD was trained on <= "
            f"{max_duration:.0f} s clips, so only the first {max_duration:.0f} s are used."
        )
        waveform = waveform[:target_samples]
    elif waveform.size(0) < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.size(0)))
    return waveform.contiguous()


def _read_file(path: Path):
    try:
        import torchaudio

        waveform, sr = torchaudio.load(str(path))
        return waveform.float(), int(sr)
    except Exception:  # torchaudio backend missing or codec unsupported
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
        return torch.from_numpy(data.T.copy()), int(sr)
