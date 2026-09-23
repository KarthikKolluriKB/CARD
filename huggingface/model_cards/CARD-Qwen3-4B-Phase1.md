---
license: apache-2.0
language:
  - en
pipeline_tag: audio-text-to-text
base_model: Qwen/Qwen3-4B
base_model_relation: adapter
tags:
  - audio-captioning
  - encoder-free
  - knowledge-distillation
  - lora
  - qwen3
  - clap
  - slt-2026
datasets:
  - cvssp/WavCaps
  - Loie/Auto-ACD
  - OpenSound/AudioCaps
  - confit/clotho
  - BAAI/Infinity-Instruct
---

# CARD-Qwen3-4B-Phase1

The **Phase 1 checkpoint** of CARD\* from the paper
**"CARD: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning"**
(IEEE SLT 2026). Code: https://github.com/KarthikKolluriKB/CARD-Encoder-Free-Audio-Captioning

Phase 1 is where the encoder-free student learns to hear: a frozen Qwen3-4B with a rank-16 LoRA
and a 13.2 M-parameter audio projector are trained on about 276 K audio-caption pairs under the
captioning loss plus cross-component distillation from a frozen CLAP-HTSAT teacher (early teacher
stages into the projector, later stages into the LLM). This checkpoint is the common parent of the
two deployed models:

- `KarthikKB1998/CARD-Qwen3-4B-AudioCaps` (Phase 2 on AudioCaps, CIDEr-D 55.4)
- `KarthikKB1998/CARD-Qwen3-4B-Clotho` (Phase 2 on Clotho, CIDEr-D 27.5)

Use this repo as a starting point for your own Phase 2 fine-tuning or to study the distillation
heads. For plain captioning, use one of the Phase 2 repos: the Phase 1 model alone has not been
adapted to the caption format and scores only 10.3 CIDEr-D on AudioCaps (paper Table IV).

## What is in this repo

| Path | What | Size |
|---|---|---|
| `adapters/phase1_lora/` | Phase 1 LoRA adapter (PEFT format, r=16, alpha=16, q/k/v/o/gate/up/down x 36 blocks) on `Qwen/Qwen3-4B` | 66 MB |
| `audio_projector.safetensors` | The audio projector after Phase 1 (Conv1d 64->512 s2, Conv1d 512->2560 s2, LayerNorm, Linear 2560->2560, learned positions), fp32 | 53 MB |
| `projector_heads.safetensors` | The two ProjectorHead MLPs (2560->768->192 and 2560->768->384) used to distil CLAP stages 0 and 1 into the projector. Training-time only | |
| `llm_heads.safetensors` | The two LLMHead projections (RMSNorm + Linear 2560->768) used to distil CLAP stages 2 and 3 into LLM blocks 0-8 and 9-11. Training-time only | |
| `card_config.json` | Front end, projector, LoRA, prompt and decoding settings read by the loader | |
| `generation_config.json` | Beam search 4, max 40 new tokens, no sampling | |

There is no merged `llm/` here; the loader rebuilds the Phase 1 model by merging the adapter into
`Qwen/Qwen3-4B`.

## Quick start

```bash
pip install git+https://github.com/KarthikKolluriKB/CARD-Encoder-Free-Audio-Captioning
```

```python
from card import load_card

model = load_card("KarthikKB1998/CARD-Qwen3-4B-Phase1", mode="adapters", device="cuda")
print(model.caption("clip.wav"))   # works, but see the note above: Phase 2 models caption far better
```

Training code for running Phase 2 from this checkpoint will be released in the code repository.

Requirements: `torch==2.7.1`, `transformers==4.53.1`, `peft==0.18.1`, `torchaudio`, `soundfile`.

## Results

Paper Section IV-C, "Generalization beyond captioning": binary Clotho-AQA zero-shot accuracy (%).
No checkpoint sees QA data at any stage.

| Model | Clotho-AQA (binary, zero-shot) |
|---|---|
| **Phase 1, CARD\* (this repo)** | **76.98** |
| Phase 1, No Distill | 60.47 |
| Phase 2 (Clotho) of this checkpoint | 78.64 |
| Phase 2 (AudioCaps) of this checkpoint | 75.10 |
| Phase 2 fine-tuned on Clotho-AQA (reference) | 80.72 |

Captioning with this checkpoint alone (paper Table IV, "Phase 1 Only"): AudioCaps CIDEr-D 10.3,
Clotho 6.3. Phase 1 transfers acoustic knowledge from the teacher, while Phase 2 adapts the model
for caption generation.

## Training summary

- **Backbone**: Qwen3-4B, frozen. LoRA r=16, alpha=16, dropout 0, on all 7 linear projections of
  all 36 blocks. Audio tokens attend bidirectionally; text stays causal.
- **Teacher**: `laion/clap-htsat-fused`, frozen. Its four HTSAT stage outputs (192-d, 384-d,
  768-d, 768-d) are the targets. Stages 0-1 -> projector through the ProjectorHead MLPs
  (mean-pooled cosine). Stages 2-3 -> the audio-token hidden states of LLM blocks 0-8 and 9-11
  through the LLMHead projections (token-level cosine, teacher interpolated along time).
  Loss: L_cap + L_proj + L_llm with lambda_proj = lambda_llm = 1.
- **Data** (1 epoch): WavCaps 140,750; Auto-ACD 82,268; AudioCaps 45,178; Clotho 3,839;
  MACS 3,930; plus 50,000 Infinity-Instruct text-only examples interleaved as modality-pure
  batches. Audio datasets sampled in proportion to size.
- **Optimisation**: AdamW, betas (0.9, 0.95), effective batch 64, lr 2e-4 constant with 200
  warmup steps, grad clip 1.0. 2 x A6000 48 GB.

Model tree: `Qwen/Qwen3-4B` -> **`CARD-Qwen3-4B-Phase1`** (this repo) -> `CARD-Qwen3-4B-AudioCaps`,
`CARD-Qwen3-4B-Clotho`.

## Intended use and limitations

- A research checkpoint: the starting point for Phase 2 fine-tuning and for analysis of what
  cross-component distillation puts into the projector and the LLM. Not a ready-to-use captioner.
- English only; 10 s audio windows; environmental sounds, not speech transcription.
- Inherits the biases of the caption datasets and of Qwen3-4B.

## License

Apache-2.0, as are `Qwen/Qwen3-4B` and `laion/clap-htsat-fused`. No training audio is redistributed.

## Citation

```bibtex
@inproceedings{kolluri2026card,
  title     = {{CARD}: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning},
  author    = {Kolluri, Ganesh Pavan Kartikeya Bharadwaj and Zhang, Yuchen and Kampouridis, Michael and Shekhar, Ravi},
  booktitle = {Proceedings of the IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```

Contact: karthik.kolluri@essex.ac.uk
