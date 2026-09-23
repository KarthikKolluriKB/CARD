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
  - OpenSound/AudioCaps
  - confit/clotho
  - cvssp/WavCaps
  - Loie/Auto-ACD
  - BAAI/Infinity-Instruct
metrics:
  - cider
  - spice
  - meteor
model-index:
  - name: CARD-Qwen3-4B-AudioCaps
    results:
      - task:
          type: audio-captioning
          name: Automated Audio Captioning
        dataset:
          type: audiocaps
          name: AudioCaps test
          split: test
        metrics:
          - type: cider
            name: CIDEr-D
            value: 55.4
          - type: spider
            name: SPIDEr
            value: 35.2
          - type: spice
            name: SPICE
            value: 15.1
          - type: meteor
            name: METEOR
            value: 21.2
---

![CARD: encoder-free audio captioning](assets/card-banner.png)

# CARD-Qwen3-4B-AudioCaps

Encoder-free audio captioning model from the paper
**"CARD: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning"**
(IEEE SLT 2026). Code: https://github.com/KarthikKolluriKB/CARD

CARD removes the audio encoder at inference. A 13.2 M-parameter convolutional projector turns a
log-Mel spectrogram into about 250 audio tokens, which a Qwen3-4B language model (with LoRA
adapters merged into its weights) reads to write the caption. During training a frozen CLAP-HTSAT
teacher was distilled into the model *by component*: its early, perceptual stages into the projector
and its later, semantic stages into the LLM. The teacher and all distillation heads are gone at
inference.

This repo is the **CARD\* model fine-tuned (Phase 2) on AudioCaps**, the headline row of
Table II in the paper. Related repos:

- `KarthikKB1998/CARD-Qwen3-4B-Clotho`: the same Phase 1 checkpoint fine-tuned on Clotho.
- `KarthikKB1998/CARD-Qwen3-4B-Phase1`: the shared Phase 1 checkpoint (before Phase 2).

![CARD training and inference pipelines](assets/card_architecture.png)

*CARD training (left) and inference (right) pipelines (paper Fig. 1). A frozen CLAP teacher's early,
perceptual stages are distilled into the audio projector and its later, semantic stages into the
LLM's LoRA adapters; at inference only the projector and the merged LLM remain.*

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

Try it without installing anything: listen to sample clips on the [demo page](https://huggingface.co/spaces/KarthikKB1998/CARD-Audio-Captioning), or caption
your own audio in the [Colab notebook](https://colab.research.google.com/github/KarthikKolluriKB/CARD/blob/main/notebooks/CARD_demo.ipynb).

```bash
pip install git+https://github.com/KarthikKolluriKB/CARD
```

```python
from card import load_card

model = load_card("KarthikKB1998/CARD-Qwen3-4B-AudioCaps", device="cuda")
print(model.caption("dog_barking.wav"))
```

To rebuild the merged weights from the base model and the two adapters instead of downloading them:

```python
model = load_card("KarthikKB1998/CARD-Qwen3-4B-AudioCaps", mode="adapters", device="cuda")
```

Input contract: any sample rate, mono (stereo is averaged), any length. The front end resamples to
16 kHz, crops or zero-pads to 10 s, then resamples to 48 kHz for the CLAP log-Mel spectrogram,
exactly as in training; clips longer than 10 s are truncated. Output is one English sentence.
Decoding is beam search (4 beams, 40 tokens) with the prompt "Describe this audio." under the
captioning system prompt stored in `card_config.json`; `enable_thinking=False` is applied to the
Qwen3 chat template.

Requirements: `torch==2.7.1`, `transformers==4.53.1`, `peft==0.18.1` (only for `mode="adapters"`),
`torchaudio`, `soundfile`. Peak GPU memory is 8.4 GB at batch 1 with beam 4 (paper Section IV-B).

## Results

AudioCaps test, beam 4, all values in % (paper Table II). SLAM-AAC keeps the frozen CLAP encoder at
inference; the CARD variants remove it and differ only in where the teacher is distilled.

| Model | Encoder at inference | CIDEr-D | SPIDEr | SPICE | METEOR |
|---|---|---|---|---|---|
| SLAM-AAC (CLAP encoder + LoRA), encoder-kept reference | yes | 66.4 | 41.9 | 17.3 | 23.3 |
| **CARD\* (this repo)** | **no** | **55.4** | **35.2** | **15.1** | **21.2** |
| Proj Early (early stages -> projector only) | no | 52.5 | 33.3 | 14.1 | 20.1 |
| LLM Distill (teacher -> LLM only) | no | 43.5 | 27.8 | 12.2 | 18.4 |
| No Distill | no | 43.2 | 27.9 | 12.6 | 18.4 |

All numbers are as reported in the paper. CARD\*'s gains over no distillation, LLM-only
distillation and mismatched projector supervision are statistically significant on both datasets
(paired bootstrap over per-clip CIDEr-D, 10,000 resamples).

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
  and the projector fine-tuned on the prompt-expanded AudioCaps training set (225,890 pairs) with
  the captioning loss only, lr 2e-5 cosine with 50 warmup steps. Then the Phase 2 LoRA is merged.
- Hardware: 2 x A6000 48 GB.

Model tree: `Qwen/Qwen3-4B` -> `CARD-Qwen3-4B-Phase1` -> **`CARD-Qwen3-4B-AudioCaps`** (this repo).

## Intended use and limitations

- Intended for research on encoder-free audio-language models and for captioning short
  environmental-sound clips in English. Not a speech recogniser: speech is described ("a man
  speaks"), not transcribed.
- Trained on captions of 10 s clips; behaviour on longer or very quiet audio is untested and longer
  clips are truncated.
- The model inherits the biases of its caption datasets (YouTube-sourced AudioCaps, Freesound-sourced
  Clotho) and of the Qwen3-4B backbone. Captions can be wrong or generic; do not use for safety-critical
  monitoring.
- Performance is below encoder-kept systems (66.4 vs 55.4 CIDEr-D); the contribution is the
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
