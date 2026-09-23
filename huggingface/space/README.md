---
title: CARD Audio Captioning
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
python_version: "3.10"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Encoder-free audio captioning demo (CARD, IEEE SLT 2026)
models:
  - KarthikKB1998/CARD-Qwen3-4B-AudioCaps
---

# CARD audio captioning demo

Interactive demo of **"CARD: Cross-component Audio Representation Distillation for Encoder-Free
Audio Captioning"** (IEEE SLT 2026).

Pick one of the sample clips, listen to it, and press **Generate caption**. The model,
[`KarthikKB1998/CARD-Qwen3-4B-AudioCaps`](https://huggingface.co/KarthikKB1998/CARD-Qwen3-4B-AudioCaps),
writes a caption live, and the human reference captions from the test set are shown next to it.

CARD has no audio encoder at inference. A 13.2 M-parameter projector turns the log-Mel spectrogram
into audio tokens, which a Qwen3-4B language model with merged LoRA adapters reads to write the
caption. During training a frozen CLAP-HTSAT teacher was distilled into the model by component,
its early stages into the projector and its later stages into the language model.

- Code: https://github.com/KarthikKolluriKB/CARD
- Decoding follows the paper: beam search with 4 beams and at most 40 new tokens.
- The source and license of each sample clip are shown under it in the app.

```bibtex
@inproceedings{kolluri2026card,
  title     = {{CARD}: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning},
  author    = {Kolluri, Ganesh Pavan Kartikeya Bharadwaj and Zhang, Yuchen and Kampouridis, Michael and Shekhar, Ravi},
  booktitle = {Proceedings of the IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```
