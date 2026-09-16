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
  - ablation
datasets:
  - OpenSound/AudioCaps
  - confit/clotho
  - cvssp/WavCaps
  - Loie/Auto-ACD
  - BAAI/Infinity-Instruct
---

# CARD-Ablations

Every encoder-free ablation of Table II from
**"CARD: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning"**
(IEEE SLT 2026), packaged as small adapter bundles.
Code: https://github.com/KarthikKolluriKB/CARD-Encoder-Free-Audio-Captioning

All rows share the frozen Qwen3-4B backbone, the CLAP-HTSAT teacher, the Phase 1 data mixture, the
two-phase recipe and the rank-16 all-linear LoRA. They differ only in **which teacher stages are
routed to which student component** during Phase 1. The headline model CARD\* is published on its
own (`KarthikKB1998/CARD-Qwen3-4B-AudioCaps`, `KarthikKB1998/CARD-Qwen3-4B-Clotho`); the rows
below are the controls that show the routing is what matters.

## Folder map and results (CIDEr-D / SPIDEr / SPICE / METEOR, x100)

| Folder | Table II row | Routing in Phase 1 | AudioCaps test | Clotho evaluation |
|---|---|---|---|---|
| `no_distill/` | No Distill | no teacher | 43.2 / 27.9 / 12.6 / 18.4 | 22.3 / 15.5 / 8.6 / 13.0 |
| `llm_distill/` | LLM Distill | all stages -> LLM only | 43.5 / 27.8 / 12.2 / 18.4 | 22.5 / 15.1 / 8.4 / 13.0 |
| `proj_full/` | Proj Full | all stages -> projector only | 40.7 / 26.4 / 12.1 / 17.8 | 21.2 / 14.6 / 8.1 / 12.7 |
| `proj_early/` | Proj Early | early stages (0, 1) -> projector only | 52.5 / 33.3 / 14.1 / 20.1 | 24.3 / 16.6 / 8.9 / 13.1 |
| `reversed/` | Reversed Routing | early stages -> LLM, later stages -> projector | 50.0 / 31.8 / 13.5 / 19.3 | 22.8 / 15.9 / 9.0 / 13.2 |
| `card_diamond/` | CARD-diamond | all stages -> projector, later stages -> LLM | 49.9 / 32.0 / 14.1 / 20.3 | 24.8 / 17.1 / 9.4 / 13.1 |
| (separate repos) | CARD\* | early stages -> projector, later stages -> LLM | 55.4 / 35.2 / 15.1 / 21.2 | 27.5 / 18.8 / 10.1 / 14.2 |

Each folder has two Phase 2 fine-tunes, `audiocaps/` and `clotho/`:

```
<variant>/<dataset>/
    card_config.json
    generation_config.json
    audio_projector.safetensors     projector after Phase 2 (fp32, 53 MB)
    adapters/phase1_lora/           Phase 1 LoRA on Qwen/Qwen3-4B (r=16, alpha=16)
    adapters/phase2_lora/           Phase 2 LoRA on the Phase-1-merged backbone
```

No merged LLM weights are stored here (that would be twelve near-identical 8 GB copies of
Qwen3-4B); the loader rebuilds each model by merging Phase 1 then Phase 2 into `Qwen/Qwen3-4B`.

Note on `llm_distill/`: this row predates the CLAP-matched front end and uses a 16 kHz, 80-band
torchaudio log-Mel input (its `card_config.json` says `"type": "torchaudio_16k"`); the loader
handles it automatically. Its AudioCaps number is reported as indicative in the paper.

## Quick start

```bash
pip install git+https://github.com/KarthikKolluriKB/CARD-Encoder-Free-Audio-Captioning
```

```python
from card import load_card

model = load_card("KarthikKB1998/CARD-Ablations", subfolder="proj_early/audiocaps",
                  mode="adapters", device="cuda")
print(model.caption("clip.wav"))
```

Decoding, prompt and input handling are identical to the CARD\* repos: 16 kHz load, crop or zero-pad
to 10 s, CLAP 64-band log-Mel, beam search with 4 beams and 40 new tokens.

Requirements: `torch==2.7.1`, `transformers==4.53.1`, `peft==0.18.1`, `torchaudio`, `soundfile`.

## What the rows show

- Distilling the teacher into the LLM alone (`llm_distill`) gains nothing over no teacher at all.
- Distilling into the projector helps only when the teacher stages match the projector's role:
  early, perceptual stages (`proj_early`, +9.3 CIDEr-D on AudioCaps) rather than all stages
  (`proj_full`, -2.5).
- Swapping the routing (`reversed`) keeps every other setting fixed and loses 5.4 CIDEr-D against
  CARD\*, so direction matters, not just the presence of projector supervision.
- Combining early-stage projector supervision with later-stage LLM supervision (CARD\*) is best on
  both datasets. Differences are significant under a paired bootstrap over per-clip CIDEr-D.

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
