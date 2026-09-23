---
title: CARD Audio Captioning
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Encoder-free audio captioning demo (CARD, IEEE SLT 2026)
models:
  - KarthikKB1998/CARD-Qwen3-4B-AudioCaps
---

# CARD audio captioning demo

Interactive demo of **"CARD: Cross-component Audio Representation Distillation for Encoder-Free
Audio Captioning"** (IEEE SLT 2026).

Pick a sample clip, listen to it, and reveal the caption written by
[`KarthikKB1998/CARD-Qwen3-4B-AudioCaps`](https://huggingface.co/KarthikKB1998/CARD-Qwen3-4B-AudioCaps)
next to the human reference captions from the test set.

This is a static page. The captions were generated beforehand by the released model with the
paper's decoding settings (beam search, 4 beams, at most 40 new tokens). Beam search is
deterministic, so they are the captions the model produces live for these clips. To caption your
own audio, run the model in the
[Colab notebook](https://colab.research.google.com/github/KarthikKolluriKB/CARD/blob/main/notebooks/CARD_demo.ipynb).

CARD has no audio encoder at inference. A 13.2 M-parameter projector turns the log-Mel spectrogram
into audio tokens, which a Qwen3-4B language model with merged LoRA adapters reads to write the
caption. During training a frozen CLAP-HTSAT teacher was distilled into the model by component,
its early stages into the projector and its later stages into the language model.

- Code: https://github.com/KarthikKolluriKB/CARD
- Each sample clip keeps its original source and license, shown with its caption.

```bibtex
@inproceedings{kolluri2026card,
  title     = {{CARD}: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning},
  author    = {Kolluri, Ganesh Pavan Kartikeya Bharadwaj and Zhang, Yuchen and Kampouridis, Michael and Shekhar, Ravi},
  booktitle = {Proceedings of the IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```
