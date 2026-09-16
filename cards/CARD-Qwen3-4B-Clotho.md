---
license: apache-2.0
language:
  - en
pipeline_tag: audio-text-to-text
base_model: KarthikKB1998/CARD-Qwen3-4B-Phase1
base_model_relation: finetune
tags:
  - audio-captioning
  - automated-audio-captioning
  - encoder-free
  - knowledge-distillation
  - lora
  - qwen3
  - clap
  - slt-2026
datasets:
  - confit/clotho
  - OpenSound/AudioCaps
  - cvssp/WavCaps
  - Loie/Auto-ACD
  - BAAI/Infinity-Instruct
metrics:
  - cider
  - spice
  - meteor
model-index:
  - name: CARD-Qwen3-4B-Clotho
    results:
      - task:
          type: audio-captioning
          name: Automated Audio Captioning
        dataset:
          type: clotho
          name: Clotho evaluation
          split: evaluation
        metrics:
          - type: cider
            name: CIDEr-D
            value: 27.5
          - type: spider
            name: SPIDEr
            value: 18.8
          - type: spice
            name: SPICE
            value: 10.1
          - type: meteor
            name: METEOR
            value: 14.2
---

# CARD-Qwen3-4B-Clotho

Encoder-free audio captioning model from the paper
**"CARD: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning"**
(IEEE SLT 2026). Code: https://github.com/KarthikKolluriKB/CARD-Encoder-Free-Audio-Captioning

CARD removes the audio encoder at inference. A 13.2 M-parameter convolutional projector turns a
log-Mel spectrogram into about 250 audio tokens, which a Qwen3-4B language model (with LoRA
adapters merged into its weights) reads to write the caption. During training a frozen CLAP-HTSAT
teacher was distilled into the model *by component*: its early, perceptual stages into the projector
and its later, semantic stages into the LLM. The teacher and all distillation heads are gone at
inference.

This repo is the **CARD\* model fine-tuned (Phase 2) on Clotho**. It shares its Phase 1 checkpoint
with the AudioCaps model. Related repos:

- `KarthikKB1998/CARD-Qwen3-4B-AudioCaps`: the same Phase 1 checkpoint fine-tuned on AudioCaps.
- `KarthikKB1998/CARD-Qwen3-4B-Phase1`: the shared Phase 1 checkpoint (before Phase 2).
- `KarthikKB1998/CARD-Ablations`: every encoder-free ablation of Table II as adapter bundles.

## What is in this repo

| Path | What | Size |
|---|---|---|
| `llm/` | Qwen3-4B with the Phase 1 and Phase 2 LoRA updates merged in (bf16, safetensors) + tokenizer | ~8.0 GB |
| `audio_projector.safetensors` | The audio projector (Conv1d 64->512 s2, Conv1d 512->2560 s2, LayerNorm, Linear 2560->2560, learned positions), fp32 | 53 MB |
| `adapters/phase1_lora/` | Phase 1 LoRA adapter (PEFT format, r=16, alpha=16, all 7 linear projections x 36 blocks) | 66 MB |
| `adapters/phase2_lora/` | Phase 2 LoRA adapter trained on top of the Phase-1-merged backbone | 66 MB |
| `card_config.json` | Front end, projector, LoRA, prompt and decoding settings read by the loader | |
| `generation_config.json` | Beam search 4, max 40 new tokens, no sampling (overrides Qwen3 defaults) | |

You need only `llm/` and `audio_projector.safetensors` to run the model. The adapters are provided
so the merged weights can be rebuilt from `Qwen/Qwen3-4B` (Phase 1 merged first, then Phase 2) and
so the two phases can be studied separately.

## Quick start

```bash
pip install git+https://github.com/KarthikKolluriKB/CARD-Encoder-Free-Audio-Captioning
```

```python
from card import load_card

model = load_card("KarthikKB1998/CARD-Qwen3-4B-Clotho", device="cuda")
print(model.caption("rain_and_thunder.wav"))
```

To rebuild the merged weights from the base model and the two adapters instead of downloading them:

```python
model = load_card("KarthikKB1998/CARD-Qwen3-4B-Clotho", mode="adapters", device="cuda")
```

Input contract: any sample rate, mono (stereo is averaged), any length. The front end resamples to
16 kHz, crops or zero-pads to 10 s, then resamples to 48 kHz for the CLAP log-Mel spectrogram,
exactly as in training; clips longer than 10 s are truncated (Clotho clips are 15-30 s, so only
their first 10 s are captioned). Output is one English sentence. Decoding is beam search (4 beams,
40 tokens) with the prompt "Describe this audio." under the captioning system prompt stored in
`card_config.json`; `enable_thinking=False` is applied to the Qwen3 chat template.

Requirements: `torch==2.7.1`, `transformers==4.53.1`, `peft==0.18.1` (only for `mode="adapters"`),
`torchaudio`, `soundfile`. About 9 GB of GPU memory in bf16.

## Results

Clotho evaluation split (1,045 clips), beam 4, metrics x100 (paper Table II). Results are
Phase-2-fine-tuned on the Clotho training set, not zero-shot. All rows share the frozen Qwen3-4B
backbone, the CLAP-HTSAT teacher, the Phase 1 data and the r=16 all-linear LoRA; only the audio
pathway and the distillation routing differ.

| Model | Encoder at inference | CIDEr-D | SPIDEr | SPICE | METEOR |
|---|---|---|---|---|---|
| SLAM-AAC (CLAP encoder + LoRA), encoder-kept reference | yes | 39.0 | 25.8 | 12.7 | 16.2 |
| **CARD\* (this repo)** | **no** | **27.5** | **18.8** | **10.1** | **14.2** |
| Proj Early (early stages -> projector only) | no | 24.3 | 16.6 | 8.9 | 13.1 |
| LLM Distill (teacher -> LLM only) | no | 22.5 | 15.1 | 8.4 | 13.0 |
| No Distill | no | 22.3 | 15.5 | 8.6 | 13.0 |

Differences between CARD\* and the encoder-free ablations are significant under a paired bootstrap
over per-clip CIDEr-D (10,000 resamples). Metrics computed with `aac-metrics==0.5.4`.

## Training summary

- **Backbone**: Qwen3-4B, frozen. LoRA r=16, alpha=16, dropout 0, on q/k/v/o/gate/up/down of all
  36 blocks. Audio tokens attend to each other bidirectionally during training; text stays causal.
- **Teacher (Phase 1 only)**: `laion/clap-htsat-fused`, frozen. Its four HTSAT stage outputs
  (192-d, 384-d, 768-d, 768-d) are the distillation targets. Stages 0-1 supervise the projector
  through two small MLP heads (mean-pooled cosine loss). Stages 2-3 supervise the LLM's audio-token
  hidden states from blocks 0-8 and 9-11 through RMSNorm + linear heads (token-level cosine loss,
  teacher interpolated along time). Loss: L_cap + L_proj + L_llm with both weights 1.
- **Phase 1** (1 epoch): WavCaps, Auto-ACD, AudioCaps, Clotho, MACS captions (about 276 K) sampled
  by size, plus 50 K Infinity-Instruct text-only batches. AdamW, betas (0.9, 0.95), effective batch
  64, lr 2e-4 constant with 200 warmup steps, grad clip 1.0.
- **Phase 2** (1 epoch): teacher and heads removed, Phase 1 LoRA merged, fresh LoRA (r=16, alpha=16)
  and the projector fine-tuned on the prompt-expanded Clotho training set (95,975 pairs) with the
  captioning loss only, lr 2e-5 cosine with 50 warmup steps. Then the Phase 2 LoRA is merged.
- Hardware: 2 x A6000 48 GB.

Model tree: `Qwen/Qwen3-4B` -> `CARD-Qwen3-4B-Phase1` -> **`CARD-Qwen3-4B-Clotho`** (this repo).

## Intended use and limitations

- Intended for research on encoder-free audio-language models and for captioning short
  environmental-sound clips in English. Not a speech recogniser: speech is described ("a man
  speaks"), not transcribed.
- Trained on captions of 10 s windows; Clotho clips are longer and only their first 10 s are seen,
  which is one reason Clotho scores are lower than AudioCaps scores for every model in the paper.
- The model inherits the biases of its caption datasets (Freesound-sourced Clotho, YouTube-sourced
  AudioCaps) and of the Qwen3-4B backbone. Captions can be wrong or generic; do not use for
  safety-critical monitoring.
- Performance is below encoder-kept systems (39.0 vs 27.5 CIDEr-D); the contribution is the
  encoder-free operating point and the distillation placement result, not state of the art.

## License

Weights and code are released under Apache-2.0. The backbone `Qwen/Qwen3-4B` and the teacher
`laion/clap-htsat-fused` are also Apache-2.0. No training audio is redistributed here.

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
