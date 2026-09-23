"""CARD: Cross-component Audio Representation Distillation for Encoder-Free Audio Captioning.

Public entry points:

    from card_model import load_card
    model = load_card("KarthikKB1998/CARD-Qwen3-4B-AudioCaps")
    model.caption("clip.wav")
"""

__version__ = "1.0.0"

from card_model.hub import CARDConfig, CARDModel, load_card  # noqa: E402

__all__ = ["CARDConfig", "CARDModel", "load_card", "__version__"]
