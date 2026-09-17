# CARD: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning

Official code and models for the IEEE SLT 2026 paper
**"CARD: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning"**
by Ganesh Pavan Kartikeya Bharadwaj Kolluri, Yuchen Zhang, Michael Kampouridis and Ravi Shekhar
(University of Essex).

Models: [Hugging Face collection](https://huggingface.co/collections/KarthikKB1998/card-encoder-free-audio-captioning-ieee-slt-2026-6aab0d5d46f574fd376ff1d7)

## What CARD is

Most audio captioning systems run a frozen audio encoder (CLAP, EAT, CED, ...) in front of a
language model and train a projector between the two. The encoder stays in the pipeline forever:
every clip pays for its forward pass, and the language model only ever sees the encoder's fixed
features.

CARD removes the encoder at inference. The deployed model is just two parts:

- an **audio projector** (13.2 M parameters): two stride-2 convolutions over a log-Mel
  spectrogram, a LayerNorm, a linear layer and learned positions, producing about 250 audio tokens
  at the LLM hidden size;
- **Qwen3-4B** with LoRA adapters merged into its weights.

The encoder is used only as a **teacher during training**, and the point of the paper is *where*
its knowledge goes. CLAP-HTSAT is a hierarchical encoder with four stages; CARD routes them by
role:

| Teacher stage | Character | Distilled into | Head (training only) |
|---|---|---|---|
| 0 (192-d), 1 (384-d) | perceptual, low-level | the audio projector | two-layer MLP per stage, mean-pooled cosine loss |
| 2 (768-d), 3 (768-d) | semantic | LLM blocks 0-8 and 9-11 (audio-token positions) | RMSNorm + linear per stage, token-level cosine loss |

Training runs in two phases:

1. **Phase 1**: projector, LoRA and heads are trained together on about 276 K audio-caption pairs
   (WavCaps, Auto-ACD, AudioCaps, Clotho, MACS) plus interleaved text-only instruction batches,
   with `L = L_cap + lambda_proj * L_proj + lambda_llm * L_llm` (both lambdas 1). Teacher frozen.
2. **Phase 2**: teacher and heads removed, Phase 1 LoRA merged into the backbone, a fresh LoRA and
   the projector fine-tuned on the target dataset with the captioning loss only, then merged.

At inference nothing of the teacher remains.

## Results

Paper Table II: CIDEr-D / SPIDEr / SPICE / METEOR, all values in %, beam search 4. SLAM-AAC keeps
the frozen CLAP encoder at inference; every CARD variant removes it.

| Model | Variant | Encoder at inference | AudioCaps | Clotho |
|---|---|---|---|---|
| SLAM-AAC | No LoRA | yes | 59.8 / 38.7 / 17.5 / 23.7 | 32.0 / 22.3 / 12.2 / 16.4 |
| SLAM-AAC | LoRA | yes | 66.4 / 41.9 / 17.3 / 23.3 | 39.0 / 25.8 / 12.7 / 16.2 |
| CARD | No Distill (no teacher) | no | 43.2 / 27.9 / 12.6 / 18.4 | 22.3 / 15.5 / 8.6 / 13.0 |
| CARD | LLM Distill (LLM only) | no | 43.5 / 27.8 / 12.2 / 18.4 | 22.5 / 15.1 / 8.4 / 13.0 |
| CARD | Proj Full (all stages -> projector) | no | 40.7 / 26.4 / 12.1 / 17.8 | 21.2 / 14.6 / 8.1 / 12.7 |
| CARD | Proj Early (early stages -> projector) | no | 52.5 / 33.3 / 14.1 / 20.1 | 24.3 / 16.6 / 8.9 / 13.1 |
| CARD | Reversed Routing (early stages -> LLM, later stages -> projector) | no | 50.0 / 31.8 / 13.5 / 19.3 | 22.8 / 15.9 / 9.0 / 13.2 |
| CARD | CARD-diamond (projector, all stages + LLM) | no | 49.9 / 32 / 14.1 / 20.3 | 24.8 / 17.1 / 9.4 / 13.1 |
| CARD | **CARD\*** (projector, early stages + LLM) | no | **55.4 / 35.2 / 15.1 / 21.2** | **27.5 / 18.8 / 10.1 / 14.2** |

LLM-only distillation gives negligible gains over the non-distilled model, and projector-only
distillation with the full teacher hierarchy degrades performance. Supervising the projector with the
early teacher stages substantially improves it. Compared with No Distill, CARD\* improves CIDEr-D by
12.2 percentage points on AudioCaps and 5.2 on Clotho, with no encoder at inference. Its gains over
no distillation, LLM-only distillation and mismatched projector supervision are statistically
significant on both datasets (paired bootstrap over per-clip CIDEr-D, 10,000 resamples).

## Pretrained models

Three models from the paper are released on the Hugging Face Hub under
[KarthikKB1998](https://huggingface.co/KarthikKB1998).

| Repo | What it is | Contents | Size |
|---|---|---|---|
| [`CARD-Qwen3-4B-AudioCaps`](https://huggingface.co/KarthikKB1998/CARD-Qwen3-4B-AudioCaps) | CARD\* after Phase 2 on AudioCaps (CIDEr-D 55.4) | merged LLM + projector + Phase 1 and Phase 2 LoRA adapters | 8.4 GB |
| [`CARD-Qwen3-4B-Clotho`](https://huggingface.co/KarthikKB1998/CARD-Qwen3-4B-Clotho) | CARD\* after Phase 2 on Clotho (CIDEr-D 27.5) | merged LLM + projector + Phase 1 and Phase 2 LoRA adapters | 8.4 GB |
| [`CARD-Qwen3-4B-Phase1`](https://huggingface.co/KarthikKB1998/CARD-Qwen3-4B-Phase1) | the shared Phase 1 checkpoint both models start from | Phase 1 LoRA adapter + projector + distillation heads | 0.2 GB |

Weights are being uploaded; a repo that is not yet visible will appear shortly.

## Quick start

```bash
pip install git+https://github.com/KarthikKolluriKB/CARD-Encoder-Free-Audio-Captioning
```

```python
from card import load_card

model = load_card("KarthikKB1998/CARD-Qwen3-4B-AudioCaps", device="cuda")
print(model.caption("dog_barking.wav"))
```

`load_card` downloads the merged LLM and the projector and returns a captioner. Any sample rate or
channel count is accepted; audio is converted to mono 16 kHz, cropped or zero-padded to 10 s, and
turned into the CLAP-matched 64-band log-Mel spectrogram the model was trained on. Decoding
reproduces the paper (beam 4, 40 new tokens, no sampling).

Rebuild the merged weights from `Qwen/Qwen3-4B` and the two adapters instead of downloading them
(needs `peft`):

```python
model = load_card("KarthikKB1998/CARD-Qwen3-4B-AudioCaps", mode="adapters", device="cuda")
```

Requirements: Python 3.10+, `torch==2.7.1`, `transformers==4.53.1`, `peft==0.18.1` (adapters mode
only), `torchaudio`, `soundfile`. The pinned environment is in `pyproject.toml` / `uv.lock`
(`uv sync`). Peak GPU memory is 8.4 GB at batch 1 with beam 4.

## Repository layout

```
card/
  modeling.py        AudioProjector, CLAP-matched and legacy mel front ends, hybrid attention mask, audio loading
  hub.py             load_card(): merged or adapter-rebuilt models from the Hub or a local folder
scripts/
  export_to_hub.py   training output -> Hub repo folder (safetensors, configs, model card, checksums, self-checks)
  upload_to_hub.py   push an exported folder to the Hub
cards/               model cards published with each Hub repo
space/               Gradio demo Space: app, sample-clip builder, deploy script
tests/               offline unit tests (python -m pytest -q)
```

Training and evaluation code (Phase 1 cross-component distillation, Phase 2 fine-tuning, LoRA
merging, AudioCaps / Clotho evaluation with `aac-metrics`) is being cleaned up for release and will
be added here.

## Citation

```bibtex
@inproceedings{kolluri2026card,
  title     = {{CARD}: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning},
  author    = {Kolluri, Ganesh Pavan Kartikeya Bharadwaj and Zhang, Yuchen and Kampouridis, Michael and Shekhar, Ravi},
  booktitle = {Proceedings of the IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```

## License

Apache-2.0 (see `LICENSE`). The backbone `Qwen/Qwen3-4B` and the teacher `laion/clap-htsat-fused`
are also Apache-2.0. No training audio is redistributed.

Contact: karthik.kolluri@essex.ac.uk
