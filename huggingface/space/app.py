"""CARD audio captioning demo (Hugging Face Space).

Pick one of the bundled sample clips, press Generate, and the encoder-free CARD model writes a
caption live; the human reference captions from the test set are shown next to it.

Runs unchanged on CPU, on a dedicated GPU, and on ZeroGPU hardware. Configuration:

    CARD_MODEL_ID   Hub repo to load (default: KarthikKB1998/CARD-Qwen3-4B-AudioCaps)
    HF_TOKEN        only needed while the model repo is private (add it as a Space secret)

Local checks without the web UI:

    python app.py --selftest     caption every sample and compare with samples.json
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path

try:  # ZeroGPU needs `spaces` imported before torch touches CUDA
    import spaces

    gpu = spaces.GPU(duration=120)
except Exception:  # plain CPU / GPU hardware, or running locally
    def gpu(fn):
        return fn

HERE = Path(__file__).resolve().parent
MODEL_ID = os.environ.get("CARD_MODEL_ID", "KarthikKB1998/CARD-Qwen3-4B-AudioCaps")
SAMPLES_JSON = HERE / "samples" / "samples.json"
CODE_URL = "https://github.com/KarthikKolluriKB/CARD"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"

MODEL = None
MODEL_ERROR = None
DEVICE = "cpu"


# ----------------------------------------------------------------------------------------------
# samples and model
# ----------------------------------------------------------------------------------------------

def load_samples(path: Path = SAMPLES_JSON) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for s in data.get("samples", []):
        audio = path.parent / s["file"]
        if audio.exists():
            samples.append({**s, "path": str(audio)})
        else:
            print(f"[demo] sample file missing, skipped: {audio}", flush=True)
    return samples


def load_model() -> None:
    global MODEL, MODEL_ERROR, DEVICE
    try:
        import torch

        from card import load_card

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[demo] loading {MODEL_ID} on {DEVICE}", flush=True)
        MODEL = load_card(MODEL_ID, device=DEVICE, token=os.environ.get("HF_TOKEN"))
        print(f"[demo] ready: {MODEL}", flush=True)
    except Exception as e:  # surface the reason in the UI instead of crashing the Space
        MODEL_ERROR = f"{type(e).__name__}: {e}"
        print(f"[demo] model failed to load: {MODEL_ERROR}", flush=True)


@gpu
def run_caption(audio_path: str) -> tuple[str, float]:
    t0 = time.time()
    caption = MODEL.caption(audio_path)
    return caption, time.time() - t0


# ----------------------------------------------------------------------------------------------
# presentation
# ----------------------------------------------------------------------------------------------

ARROW = ('<svg class="arrow" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h13m-5-6 6 6-6 6" '
         'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
         'stroke-linejoin="round"/></svg>')

HERO = f"""
<div class="hero">
  <div class="eyebrow">IEEE SLT 2026</div>
  <h1 class="hero-title">CARD <span>audio captioning</span></h1>
  <p class="hero-sub">Cross-component Audio Representation Distillation for Encoder-Free Audio
  Captioning. Listen to a clip, press Generate, and compare the model's caption with what human
  annotators wrote.</p>
  <div class="links">
    <a class="pill-link primary" href="{MODEL_URL}" target="_blank" rel="noopener">Model on the Hub</a>
    <a class="pill-link" href="{CODE_URL}" target="_blank" rel="noopener">Code on GitHub</a>
  </div>
</div>
"""

PIPELINE = f"""
<div class="pipeline">
  <div class="steps">
    <div class="step"><div class="step-k">Input</div><div class="step-v">Audio clip</div></div>
    {ARROW}
    <div class="step"><div class="step-k">Front end, no parameters</div><div class="step-v">Log-Mel spectrogram</div></div>
    {ARROW}
    <div class="step accent"><div class="step-k">13.2 M parameters</div><div class="step-v">Audio projector</div></div>
    {ARROW}
    <div class="step accent"><div class="step-k">LoRA merged</div><div class="step-v">Qwen3-4B</div></div>
    {ARROW}
    <div class="step"><div class="step-k">Output</div><div class="step-v">Caption</div></div>
  </div>
  <div class="teacher">Training only: a frozen CLAP-HTSAT teacher is distilled by component, its early
  stages into the projector and its later stages into the language model. The teacher is removed, so
  there is no audio encoder at inference.</div>
</div>
"""

STATS = """
<div class="stats">
  <div class="stat"><div class="stat-v">55.4</div><div class="stat-k">CIDEr-D on AudioCaps</div></div>
  <div class="stat"><div class="stat-v">13.2 M</div><div class="stat-k">audio interface parameters, 34.4 M with the encoder</div></div>
  <div class="stat"><div class="stat-v">0.4 ms</div><div class="stat-k">audio interface latency, 11.1 ms with the encoder</div></div>
  <div class="stat"><div class="stat-v">0</div><div class="stat-k">audio encoders at inference</div></div>
</div>
<p class="stats-note">Numbers from the paper (Table II and Section IV-B).</p>
"""

CITATION = """```bibtex
@inproceedings{kolluri2026card,
  title     = {{CARD}: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning},
  author    = {Kolluri, Ganesh Pavan Kartikeya Bharadwaj and Zhang, Yuchen and Kampouridis, Michael and Shekhar, Ravi},
  booktitle = {Proceedings of the IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```"""

STOPWORDS = {
    "a", "an", "the", "and", "or", "is", "are", "be", "being", "in", "on", "of", "with", "as", "to",
    "by", "while", "then", "some", "it", "its", "at", "from", "for", "into", "followed", "there",
    "this", "that", "several", "multiple", "times", "time",
}


def _stem(word: str) -> str:
    w = word.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def highlight_caption(caption: str, references: list[str]) -> str:
    ref_words = {_stem(w) for r in references for w in re.findall(r"[A-Za-z]+", r)}
    out = []
    for part in re.split(r"([A-Za-z]+)", caption):
        if re.fullmatch(r"[A-Za-z]+", part) and part.lower() not in STOPWORDS and _stem(part) in ref_words:
            out.append(f"<mark>{html.escape(part)}</mark>")
        else:
            out.append(html.escape(part))
    return "".join(out)


def source_line(sample: dict) -> str:
    bits = []
    if sample.get("dataset"):
        bits.append(html.escape(sample["dataset"]))
    if sample.get("source"):
        src = sample["source"]
        if src.startswith("http"):
            bits.append(f'<a href="{html.escape(src)}" target="_blank" rel="noopener">source</a>')
        else:
            bits.append(html.escape(src))
    if sample.get("license"):
        bits.append(f"license: {html.escape(sample['license'])}")
    return " &middot; ".join(bits)


def placeholder_html(sample: dict | None) -> str:
    name = html.escape(sample["name"]) if sample else "No clip selected"
    src = source_line(sample) if sample else ""
    return f"""
<div class="result empty">
  <div class="result-k">Selected clip</div>
  <div class="result-name">{name}</div>
  <div class="result-src">{src}</div>
  <div class="empty-hint">Press <b>Generate caption</b> to hear what CARD writes for this clip.</div>
</div>
"""


def result_html(sample: dict, caption: str, seconds: float, device: str) -> str:
    refs = sample.get("references") or []
    items = "".join(f"<li>{html.escape(r)}</li>" for r in refs) or "<li>No reference captions.</li>"
    return f"""
<div class="result">
  <div class="result-k">CARD caption</div>
  <div class="caption">{highlight_caption(caption, refs)}</div>
  <div class="chips">
    <span class="chip">{seconds:.1f} s</span>
    <span class="chip">{html.escape(device.upper())}</span>
    <span class="chip">beam search, 4 beams</span>
  </div>
  <div class="refs-k">Human reference captions</div>
  <ol class="refs">{items}</ol>
  <div class="legend"><mark>highlighted</mark> words also appear in a human reference.</div>
  <div class="result-src">{html.escape(sample['name'])} &middot; {source_line(sample)}</div>
</div>
"""


def notice_html(text: str, kind: str = "info") -> str:
    return f'<div class="notice {kind}">{text}</div>'


CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
footer { opacity: .7; }

.hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 34px 34px 28px;
  border: 1px solid var(--border-color-primary);
  background:
    radial-gradient(700px 260px at 0% 0%, rgba(99, 102, 241, .22), transparent 70%),
    radial-gradient(600px 240px at 100% 0%, rgba(236, 72, 153, .14), transparent 70%),
    var(--block-background-fill);
}
.eyebrow {
  display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; padding: 4px 10px; border-radius: 999px;
  color: #6366f1; background: rgba(99, 102, 241, .12); border: 1px solid rgba(99, 102, 241, .3);
}
.hero-title { margin: 14px 0 8px !important; font-size: 2.6rem !important; line-height: 1.05; font-weight: 800; letter-spacing: -.02em; }
.hero-title span {
  background: linear-gradient(90deg, #6366f1, #8b5cf6 55%, #ec4899);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { max-width: 760px; margin: 0 0 18px !important; color: var(--body-text-color-subdued); font-size: 1.02rem; line-height: 1.55; }
.links { display: flex; flex-wrap: wrap; gap: 10px; }
.pill-link {
  display: inline-flex; align-items: center; padding: 9px 16px; border-radius: 999px; font-weight: 600;
  font-size: .9rem; text-decoration: none !important; color: var(--body-text-color) !important;
  border: 1px solid var(--border-color-primary); background: var(--background-fill-primary);
  transition: transform .15s ease, box-shadow .15s ease;
}
.pill-link:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(99, 102, 241, .18); }
.pill-link.primary { color: #fff !important; border-color: transparent; background: linear-gradient(90deg, #6366f1, #8b5cf6); }

.pipeline { border-radius: 18px; padding: 18px 20px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.steps { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.step { flex: 1 1 130px; padding: 10px 12px; border-radius: 12px; border: 1px solid var(--border-color-primary); background: var(--background-fill-primary); }
.step.accent { border-color: rgba(99, 102, 241, .55); background: rgba(99, 102, 241, .08); }
.step-k { font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued); }
.step-v { font-weight: 700; margin-top: 2px; }
.arrow { width: 18px; height: 18px; flex: 0 0 18px; color: #8b5cf6; }
.teacher { margin-top: 12px; padding: 10px 12px; border-radius: 12px; font-size: .88rem; color: var(--body-text-color-subdued); border: 1px dashed var(--border-color-primary); }

.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat { padding: 16px; border-radius: 16px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.stat-v { font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em; background: linear-gradient(90deg, #6366f1, #8b5cf6); -webkit-background-clip: text; background-clip: text; color: transparent; }
.stat-k { font-size: .82rem; color: var(--body-text-color-subdued); margin-top: 2px; line-height: 1.35; }
.stats-note { font-size: .75rem; color: var(--body-text-color-subdued); margin: 6px 2px 0 !important; }

.section-k { font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--body-text-color-subdued); margin: 4px 2px 8px; }

#clip-picker .wrap { display: grid !important; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
#clip-picker label {
  position: relative; margin: 0 !important; padding: 14px 14px 14px 42px !important; border-radius: 14px !important;
  border: 1px solid var(--border-color-primary) !important; background: var(--background-fill-primary) !important;
  font-weight: 600; cursor: pointer; transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
}
#clip-picker label::before {
  content: ""; position: absolute; left: 14px; top: 50%; width: 16px; height: 16px; margin-top: -8px;
  border-radius: 50%; border: 2px solid var(--border-color-primary);
}
#clip-picker label:hover { transform: translateY(-1px); border-color: rgba(99, 102, 241, .6) !important; }
#clip-picker label.selected { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99, 102, 241, .18); background: rgba(99, 102, 241, .08) !important; }
#clip-picker label, #clip-picker label span, #clip-picker label.selected, #clip-picker label.selected span { color: var(--body-text-color) !important; }
#clip-picker label.selected::before { border-color: #6366f1; background: radial-gradient(circle, #6366f1 45%, transparent 50%); }
#clip-picker input[type="radio"] { position: absolute; opacity: 0; pointer-events: none; }

#generate-btn { font-size: 1.05rem !important; padding: 14px !important; border-radius: 14px !important; }

.result { border-radius: 18px; padding: 20px 22px; min-height: 330px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill); position: relative; }
.result::before { content: ""; position: absolute; left: 0; top: 18px; bottom: 18px; width: 4px; border-radius: 0 4px 4px 0; background: linear-gradient(180deg, #6366f1, #ec4899); }
.result-k, .refs-k { font-size: .74rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--body-text-color-subdued); }
.refs-k { margin-top: 18px; }
.caption { font-size: 1.45rem; line-height: 1.35; font-weight: 700; margin: 8px 0 12px; letter-spacing: -.01em; }
.caption mark, .legend mark { background: rgba(139, 92, 246, .22); color: inherit; border-radius: 5px; padding: 0 4px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font-size: .76rem; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
.refs { margin: 8px 0 0 !important; padding-left: 20px !important; }
.refs li { margin: 4px 0; line-height: 1.45; }
.legend { margin-top: 12px; font-size: .78rem; color: var(--body-text-color-subdued); }
.result-src { margin-top: 10px; font-size: .78rem; color: var(--body-text-color-subdued); }
.result-name { font-size: 1.35rem; font-weight: 700; margin-top: 6px; }
.result.empty { display: flex; flex-direction: column; }
.empty-hint { margin-top: auto; padding: 16px; border-radius: 14px; text-align: center; color: var(--body-text-color-subdued); border: 1px dashed var(--border-color-primary); }

.notice { padding: 10px 14px; border-radius: 12px; font-size: .9rem; border: 1px solid; }
.notice.info { border-color: rgba(99, 102, 241, .35); background: rgba(99, 102, 241, .08); }
.notice.warn { border-color: rgba(234, 88, 12, .45); background: rgba(234, 88, 12, .08); }

@media (max-width: 720px) {
  .hero { padding: 24px 20px; }
  .hero-title { font-size: 2rem !important; }
  .stats { grid-template-columns: repeat(2, 1fr); }
  .arrow { transform: rotate(90deg); }
  .steps { flex-direction: column; align-items: stretch; }
  .step { flex: 0 0 auto; }
  .steps .arrow { align-self: center; }
}
"""


def make_theme():
    import gradio as gr

    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
        button_primary_background_fill_hover="linear-gradient(90deg, *primary_600, *secondary_600)",
        button_primary_background_fill_dark="linear-gradient(90deg, *primary_500, *secondary_500)",
        button_primary_background_fill_hover_dark="linear-gradient(90deg, *primary_600, *secondary_600)",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        block_radius="16px",
    )


# ----------------------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------------------

def build_demo(samples: list[dict], caption_fn, status: str | None = None, device: str = "cpu"):
    import gradio as gr

    by_name = {s["name"]: s for s in samples}
    names = list(by_name)
    first = by_name[names[0]] if names else None

    def on_select(name):
        s = by_name.get(name)
        return (s["path"] if s else None), placeholder_html(s)

    def on_generate(name):
        s = by_name.get(name)
        if s is None:
            raise gr.Error("Choose a sample clip first.")
        if status is not None:
            raise gr.Error(status)
        caption, seconds = caption_fn(s["path"])
        return result_html(s, caption, seconds, device)

    with gr.Blocks(title="CARD audio captioning") as demo:
        gr.HTML(HERO)
        if status is not None:
            gr.HTML(notice_html(f"<b>The model is not available right now.</b> {html.escape(status)}", "warn"))
        elif device == "cpu":
            gr.HTML(notice_html("This Space runs on CPU, so a caption can take a while. "
                                "Keep the page open after pressing Generate."))
        if not names:
            gr.HTML(notice_html("<b>No sample clips are installed yet.</b>", "warn"))

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.HTML('<div class="section-k">1. Choose a clip</div>')
                clip = gr.Radio(choices=names, value=names[0] if names else None, show_label=False,
                                container=False, elem_id="clip-picker")
                audio = gr.Audio(
                    value=first["path"] if first else None, label="Listen", type="filepath",
                    interactive=False, elem_id="player",
                    waveform_options=gr.WaveformOptions(waveform_color="#a5b4fc",
                                                        waveform_progress_color="#6366f1"),
                )
                button = gr.Button("Generate caption", variant="primary", elem_id="generate-btn")
            with gr.Column(scale=6):
                gr.HTML('<div class="section-k">2. Compare with the human captions</div>')
                result = gr.HTML(placeholder_html(first))

        gr.HTML('<div class="section-k" style="margin-top:18px">How CARD works</div>')
        gr.HTML(PIPELINE)
        gr.HTML(STATS)
        with gr.Accordion("Cite this work", open=False):
            gr.Markdown(CITATION)

        clip.change(on_select, inputs=clip, outputs=[audio, result])
        button.click(on_generate, inputs=clip, outputs=result)
    return demo


def launch_demo(demo, **kwargs):
    return demo.queue(max_size=16).launch(theme=make_theme(), css=CSS, **kwargs)


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------

def selftest(samples: list[dict]) -> int:
    load_model()
    if MODEL is None:
        print(f"[selftest] FAIL: {MODEL_ERROR}")
        return 1
    ok = True
    for s in samples:
        caption, seconds = run_caption(s["path"])
        expected = s.get("expected_caption")
        verdict = "n/a " if expected is None else ("MATCH" if caption == expected else "DIFF ")
        ok &= expected is None or caption == expected
        print(f"[selftest] {verdict} {s['name']} ({seconds:.1f} s): {caption}")
        if expected is not None and caption != expected:
            print(f"           expected: {expected}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'} on {len(samples)} samples")
    return 0 if ok else 1


def main() -> int:
    samples = load_samples()
    if "--selftest" in sys.argv:
        return selftest(samples)
    load_model()
    status = None if MODEL is not None else (
        f"Loading {MODEL_ID} failed ({MODEL_ERROR}). If the model repo is private, add a read token "
        "as the Space secret HF_TOKEN and restart the Space.")
    launch_demo(build_demo(samples, run_caption, status=status, device=DEVICE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
