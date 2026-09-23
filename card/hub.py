"""Load CARD models from the Hugging Face Hub (or a local export folder).

Repository layout (see ``huggingface/export_to_hub.py``)::

    card_config.json               model hyper-parameters, prompt and decoding settings
    generation_config.json         beam search 4, 40 new tokens, no sampling
    audio_projector.safetensors    AudioProjector weights (fp32)
    llm/                           merged Qwen3-4B (bf16 safetensors) + tokenizer   [optional]
    adapters/phase1_lora/          PEFT adapter trained in Phase 1                  [optional]
    adapters/phase2_lora/          PEFT adapter trained in Phase 2                  [optional]

``mode="merged"`` loads ``llm/`` directly. ``mode="adapters"`` rebuilds the same weights from the
base model: Phase 1 LoRA is merged first, then Phase 2 LoRA is applied on top and merged. The
order matters: Phase 2 was trained on the Phase-1-merged backbone.

Decoding reproduces the paper's evaluation: beam search (4 beams, 40 new tokens) over
``[audio tokens | chat prompt]`` with the standard causal mask. The greedy path (``num_beams=1``)
uses the training-time hybrid mask for the prefill, as the reference evaluation script did.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from safetensors.torch import load_file

from card.modeling import (
    AudioLike,
    AudioProjector,
    build_frontend,
    build_hybrid_attention_mask,
    load_audio,
)

CARD_CONFIG_NAME = "card_config.json"
GENERATION_CONFIG_NAME = "generation_config.json"
PROJECTOR_NAME = "audio_projector.safetensors"
LLM_SUBDIR = "llm"
PHASE1_ADAPTER = "adapters/phase1_lora"
PHASE2_ADAPTER = "adapters/phase2_lora"

DEFAULT_SYSTEM_PROMPT = (
    "You are an audio captioning assistant. Given an audio clip, "
    "respond with a concise single-sentence caption describing the "
    "audio content. Do not add explanations or preamble."
)
DEFAULT_USER_PROMPT = "Describe this audio."
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------

@dataclass
class CARDConfig:
    """Contents of ``card_config.json``. Every field has the CARD* default."""

    model_type: str = "card"
    variant: str = "card_star"
    phase: int = 2
    finetune_dataset: Optional[str] = None
    base_llm: str = "Qwen/Qwen3-4B"
    llm_hidden_dim: int = 2560
    frontend: dict = field(default_factory=lambda: {
        "type": "clap_processor",
        "clap_model": "laion/clap-htsat-fused",
        "sample_rate": 48000,
        "n_mels": 64,
        "max_audio_sec": 10.0,
    })
    projector: dict = field(default_factory=lambda: {
        "type": "conv",
        "n_mels": 64,
        "conv_intermediate": 512,
        "hidden_dim": 2560,
        "max_audio_len": 1024,
    })
    lora: dict = field(default_factory=lambda: {
        "r": 16,
        "alpha": 16,
        "target_modules": list(DEFAULT_TARGET_MODULES),
    })
    prompt: dict = field(default_factory=lambda: {
        "system": DEFAULT_SYSTEM_PROMPT,
        "user": DEFAULT_USER_PROMPT,
        "enable_thinking": False,
    })
    decoding: dict = field(default_factory=lambda: {"num_beams": 4, "max_new_tokens": 40})
    paper: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CARDConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        # merge nested dicts over the defaults so partial configs stay valid
        for key in ("frontend", "projector", "lora", "prompt", "decoding"):
            base = getattr(cls(), key)
            base.update(d.get(key, {}) or {})
            setattr(cfg, key, base)
        return cfg

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "CARDConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------

class CARDModel:
    """Merged LLM + audio projector + parameter-free front end, with a ``caption`` method."""

    def __init__(self, llm, tokenizer, projector: AudioProjector, frontend, config: CARDConfig,
                 device: torch.device):
        self.llm = llm.eval()
        self.tokenizer = tokenizer
        self.projector = projector.eval()
        self.frontend = frontend.eval()
        self.config = config
        self.device = device
        self._stop_ids = self._collect_stop_ids()

    # ---- prompt -------------------------------------------------------------------------------

    def build_prompt(self, user_prompt: Optional[str] = None,
                     system_prompt: Optional[str] = None) -> str:
        p = self.config.prompt
        messages = [
            {"role": "system", "content": system_prompt or p["system"]},
            {"role": "user", "content": user_prompt or p["user"]},
        ]
        kwargs = {}
        if p.get("enable_thinking") is not None:
            kwargs["enable_thinking"] = bool(p["enable_thinking"])
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def _collect_stop_ids(self) -> set:
        stop = {self.tokenizer.eos_token_id}
        im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if im_end:
            stop.add(im_end[0])
        return stop

    # ---- audio ---------------------------------------------------------------------------------

    @torch.no_grad()
    def embed_audio(self, waveform_16k: torch.Tensor) -> torch.Tensor:
        """(B, T) 16 kHz waveform -> (B, T_audio, hidden) audio tokens in the LLM dtype."""
        log_mel = self.frontend(waveform_16k.to(self.device))
        tokens = self.projector(log_mel.float())
        return tokens.to(dtype=self.llm.dtype)

    # ---- generation ----------------------------------------------------------------------------

    @torch.no_grad()
    def caption(
        self,
        audio: AudioLike,
        sr: Optional[int] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Caption one clip. Defaults reproduce the paper (beam 4, 40 tokens)."""
        num_beams = num_beams or int(self.config.decoding.get("num_beams", 4))
        max_new_tokens = max_new_tokens or int(self.config.decoding.get("max_new_tokens", 40))
        max_sec = float(self.config.frontend.get("max_audio_sec", 10.0))

        wave = load_audio(audio, sr=sr, max_duration=max_sec).unsqueeze(0)
        audio_embeds = self.embed_audio(wave)

        prompt_ids = self.tokenizer(
            self.build_prompt(user_prompt, system_prompt),
            return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([audio_embeds, prompt_embeds], dim=1)

        if num_beams > 1:
            ids = self._generate_beam(inputs_embeds, num_beams, max_new_tokens)
        else:
            ids = self._generate_greedy(inputs_embeds, audio_embeds.size(1),
                                        prompt_embeds.size(1), max_new_tokens)
        return self._clean(self.tokenizer.decode(ids, skip_special_tokens=True))

    def caption_batch(self, audios: Sequence[AudioLike], **kwargs) -> List[str]:
        """Caption several clips one at a time (batch 1 keeps numerics identical to the paper)."""
        return [self.caption(a, **kwargs) for a in audios]

    def _generate_beam(self, inputs_embeds: torch.Tensor, num_beams: int,
                       max_new_tokens: int) -> torch.Tensor:
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        return out[0]

    def _generate_greedy(self, inputs_embeds: torch.Tensor, n_audio: int, n_text: int,
                         max_new_tokens: int) -> torch.Tensor:
        mask = build_hybrid_attention_mask(1, n_audio, n_text, self.device, dtype=inputs_embeds.dtype)
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token.item()]
        for _ in range(max_new_tokens - 1):
            if next_token.item() in self._stop_ids:
                break
            out = self.llm(input_ids=next_token, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
        return torch.tensor(generated, device=self.device)

    @staticmethod
    def _clean(text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        for sep in ("<|im_end|>", "<|im_start|>", "<think>", "\n"):
            if sep in text:
                text = text.split(sep, 1)[0]
        return text.strip()

    # ---- info ----------------------------------------------------------------------------------

    @property
    def n_projector_params(self) -> int:
        return sum(p.numel() for p in self.projector.parameters())

    def __repr__(self) -> str:
        c = self.config
        return (f"CARDModel(variant={c.variant!r}, phase={c.phase}, dataset={c.finetune_dataset!r}, "
                f"base_llm={c.base_llm!r}, projector={self.n_projector_params / 1e6:.1f}M, "
                f"device={self.device})")


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

def resolve_model_dir(
    repo_or_path: Union[str, Path],
    mode: str = "merged",
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Local directory as-is; otherwise ``snapshot_download`` with the patterns ``mode`` needs."""
    local = Path(repo_or_path)
    if local.is_dir():
        return local / subfolder if subfolder else local

    from huggingface_hub import snapshot_download

    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    if mode == "merged":
        allow = [f"{prefix}*.json", f"{prefix}*.md", f"{prefix}{PROJECTOR_NAME}", f"{prefix}{LLM_SUBDIR}/*"]
    else:
        allow = [f"{prefix}*.json", f"{prefix}*.md", f"{prefix}{PROJECTOR_NAME}",
                 f"{prefix}adapters/*", f"{prefix}{LLM_SUBDIR}/*.json", f"{prefix}{LLM_SUBDIR}/*.txt"]
    root = snapshot_download(
        repo_id=str(repo_or_path), repo_type="model", revision=revision, token=token,
        cache_dir=cache_dir, allow_patterns=allow,
    )
    return Path(root) / subfolder if subfolder else Path(root)


def merge_adapters(base_llm: str, adapter_dirs: Sequence[Path], dtype: torch.dtype,
                   device: Union[str, torch.device], token: Optional[str] = None):
    """base -> merge adapter_dirs[0] -> merge adapter_dirs[1] -> ... (Phase 1 first)."""
    from transformers import AutoModelForCausalLM

    try:
        from peft import PeftModel
    except ImportError as e:  # pragma: no cover
        raise ImportError("mode='adapters' needs peft: pip install peft==0.18.1") from e

    llm = AutoModelForCausalLM.from_pretrained(base_llm, torch_dtype=dtype, token=token).to(device)
    for adapter in adapter_dirs:
        llm = PeftModel.from_pretrained(llm, str(adapter))
        llm = llm.merge_and_unload()
    return llm


def _apply_generation_config(llm, model_dir: Path, config: CARDConfig) -> None:
    """Make ``llm.generation_config`` match the paper decoding and silence sampling defaults."""
    from transformers import GenerationConfig

    gen_path = model_dir / GENERATION_CONFIG_NAME
    if gen_path.exists():
        gen = GenerationConfig.from_pretrained(str(model_dir))
    else:
        gen = GenerationConfig(
            num_beams=int(config.decoding.get("num_beams", 4)),
            max_new_tokens=int(config.decoding.get("max_new_tokens", 40)),
            do_sample=False,
        )
    gen.do_sample = False
    gen.temperature = None
    gen.top_p = None
    gen.top_k = None
    gen.eos_token_id = llm.generation_config.eos_token_id
    gen.pad_token_id = llm.generation_config.pad_token_id
    gen.bos_token_id = llm.generation_config.bos_token_id
    llm.generation_config = gen


def load_card(
    repo_or_path: Union[str, Path],
    mode: str = "merged",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.bfloat16,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    base_llm: Optional[str] = None,
) -> CARDModel:
    """Load a CARD model.

    Args:
        repo_or_path: Hub repo id (``KarthikKB1998/CARD-Qwen3-4B-AudioCaps``) or a local export dir.
        mode: ``"merged"`` (download ``llm/``) or ``"adapters"`` (rebuild from ``base_llm`` +
            the PEFT adapters found in the repo). A repo without ``llm/`` falls back to adapters.
        device: defaults to CUDA when available.
        dtype: LLM dtype; the projector always runs in fp32 and casts its output.
        subfolder: for a model stored in a subfolder of the repo (``"proj_early/audiocaps"``).
        base_llm: override the base model id stored in ``card_config.json``.
    """
    if mode not in ("merged", "adapters"):
        raise ValueError("mode must be 'merged' or 'adapters'")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_dir = resolve_model_dir(repo_or_path, mode=mode, subfolder=subfolder, revision=revision,
                                  token=token, cache_dir=cache_dir)
    config = CARDConfig.from_file(model_dir / CARD_CONFIG_NAME)
    base_llm = base_llm or config.base_llm

    # ---- projector + front end
    state = load_file(str(model_dir / PROJECTOR_NAME))
    if "linear1.weight" in state:
        raise ValueError("This repo holds an encoder-kept projector, which load_card does not support.")
    shape = AudioProjector.infer_shape(state)
    projector = AudioProjector(**shape)
    projector.load_state_dict(state)
    projector.to(device)
    frontend = build_frontend(config.frontend.get("type", "clap_processor"),
                              clap_model=config.frontend.get("clap_model", "laion/clap-htsat-fused"))
    frontend.to(device)
    if getattr(frontend, "n_mels", shape["n_mels"]) != shape["n_mels"]:
        raise ValueError(f"Front end produces {frontend.n_mels} mel bands but the projector expects "
                         f"{shape['n_mels']}; card_config.json and the checkpoint disagree.")

    # ---- LLM
    llm_dir = model_dir / LLM_SUBDIR
    has_merged = (llm_dir / "config.json").exists() and any(llm_dir.glob("*.safetensors"))
    if mode == "merged" and not has_merged:
        warnings.warn(f"No merged LLM under {llm_dir}; falling back to mode='adapters'.")
        mode = "adapters"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if mode == "merged":
        llm = AutoModelForCausalLM.from_pretrained(str(llm_dir), torch_dtype=dtype).to(device)
    else:
        adapters = [model_dir / p for p in (PHASE1_ADAPTER, PHASE2_ADAPTER)
                    if (model_dir / p / "adapter_config.json").exists()]
        if not adapters:
            raise FileNotFoundError(f"No PEFT adapters found under {model_dir / 'adapters'}")
        if config.phase == 2 and len(adapters) != 2:
            raise FileNotFoundError("Phase 2 model needs both phase1_lora and phase2_lora adapters")
        llm = merge_adapters(base_llm, adapters, dtype=dtype, device=device, token=token)

    tok_src = str(llm_dir) if (llm_dir / "tokenizer_config.json").exists() else base_llm
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    _apply_generation_config(llm, model_dir, config)

    return CARDModel(llm, tokenizer, projector, frontend, config, device)
