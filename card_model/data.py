"""Training data for CARD: manifests, datasets, the modality-pure batch sampler and collation.

Manifests are JSON lists built by ``scripts/data/``. Four row formats are understood:

    {"audio_path": ..., "caption": "..."}                 Phase 1 (AudioCaps, WavCaps, Auto-ACD, MACS)
    {"audio_path": ..., "captions": ["...", ...]}         Phase 1 (Clotho, first caption) and evaluation
    {"audio_path": ..., "prompt": "...", "response": ...} Phase 2 prompt-expanded AudioCaps / Clotho
    {"audio_path": ..., "question": ..., "answer": ...}   audio question answering (Clotho-AQA)

and, for the text-only instruction mix, ``{"instruction": ..., "response": ...}``.

A relative ``audio_path`` is resolved against the directory that holds the manifest, so a data
folder can be moved as a whole.

Phase 1 caption rows are trained with the fixed user prompt ``"Caption:"``; Phase 2 rows carry their
own prompt. Every sample is rendered with the Qwen3 chat template (``enable_thinking=False``) under a
system prompt chosen by modality, and the loss is taken on the response tokens plus EOS only.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Sampler

from card_model.hub import DEFAULT_SYSTEM_PROMPT
from card_model.modeling import load_audio

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# System prompts. Audio samples use the captioning persona; text-only samples use a generic one so
# the instruction mix is not conditioned on audio captioning.
CAPTION_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
GENERAL_SYSTEM_PROMPT = "You are a helpful assistant."
QA_SYSTEM_PROMPT = (
    "You are an audio question answering assistant. Given an audio clip and a "
    "question about it, answer the question directly. Do not add explanations "
    "or preamble."
)

# User prompt for plain caption rows (Phase 1). Prompt-expanded rows carry their own prompt.
CAPTION_USER_PROMPT = "Caption:"


def build_prompt(
    tokenizer,
    user_prompt: str,
    response: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Render one turn with the Qwen3 chat template.

    Without ``response`` this is the generation prefix (it ends with the assistant turn start);
    with ``response`` the response text is appended. EOS is added by the dataset, not here.
    """
    if system_prompt is None:
        system_prompt = CAPTION_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if response is None:
        return prefix
    return prefix + response


def load_manifest(path: PathLike, max_n: Optional[int] = None) -> List[dict]:
    """Read a JSON-list manifest, keeping the first ``max_n`` rows when given."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of rows")
    return rows[:max_n] if max_n else rows


def resolve_audio_path(audio_path: str, manifest_dir: Path) -> str:
    p = Path(audio_path)
    return str(p if p.is_absolute() else (manifest_dir / p))


def _tokenize_pair(tokenizer, prompt_only: str, full_no_eos: str,
                   max_length: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids for prompt+response(+EOS) and labels with the prompt masked to -100."""
    prompt_ids = tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False).input_ids[0]
    kwargs = {"max_length": max_length, "truncation": True} if max_length else {}
    full_ids = tokenizer(full_no_eos, return_tensors="pt", add_special_tokens=False, **kwargs).input_ids[0]

    eos_id = tokenizer.eos_token_id
    room_for_eos = max_length is None or len(full_ids) < max_length
    if eos_id is not None and room_for_eos and (len(full_ids) == 0 or full_ids[-1].item() != eos_id):
        full_ids = torch.cat([full_ids, torch.tensor([eos_id], dtype=full_ids.dtype)])

    labels = full_ids.clone()
    # Keep at least one supervised token: a long prompt can exceed a truncated sequence.
    n_mask = max(min(len(prompt_ids), len(full_ids) - 1), 0)
    labels[:n_mask] = -100
    return full_ids, labels


class AudioCaptionDataset(Dataset):
    """Audio + prompt + response samples read from one manifest."""

    def __init__(self, manifest_path: PathLike, tokenizer, max_n: Optional[int] = None,
                 max_audio_sec: float = 10.0):
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.max_audio_sec = max_audio_sec
        manifest_dir = self.manifest_path.resolve().parent

        self.items = []
        for row in load_manifest(self.manifest_path, max_n):
            audio_path = row.get("audio_path")
            if not audio_path:
                continue
            system_prompt = None
            if "prompt" in row and "response" in row:
                prompt, response = row["prompt"], row["response"]
            elif "question" in row and "answer" in row:
                prompt, response = row["question"], str(row["answer"])
                system_prompt = QA_SYSTEM_PROMPT
            elif "captions" in row:
                caps = row["captions"] if isinstance(row["captions"], list) else [row["captions"]]
                prompt, response = CAPTION_USER_PROMPT, (caps[0] if caps else "")
            elif "caption" in row:
                prompt, response = CAPTION_USER_PROMPT, row["caption"]
            else:
                continue
            if not response:
                continue
            self.items.append({
                "audio_path": resolve_audio_path(audio_path, manifest_dir),
                "prompt": prompt,
                "response": response,
                "system": system_prompt,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int, _depth: int = 0) -> dict:
        item = self.items[idx]
        try:
            with warnings.catch_warnings():
                # Clips longer than max_audio_sec are cropped by design; do not warn per sample.
                warnings.simplefilter("ignore")
                waveform = load_audio(item["audio_path"], max_duration=self.max_audio_sec)
        except Exception as e:  # unreadable file: log it and move on to the next index
            name = str(item["audio_path"]).encode("ascii", errors="replace").decode()
            sys.stderr.write(f"[warn] audio load failed at idx={idx} ({type(e).__name__}): {name}\n")
            if _depth >= 16:
                raise RuntimeError(f"too many consecutive audio-load failures (last: {name})") from e
            return self.__getitem__((idx + 1) % len(self), _depth=_depth + 1)

        system_prompt = item["system"]
        text_ids, labels = _tokenize_pair(
            self.tokenizer,
            build_prompt(self.tokenizer, item["prompt"], system_prompt=system_prompt),
            build_prompt(self.tokenizer, item["prompt"], response=item["response"],
                         system_prompt=system_prompt),
        )
        return {
            "waveform": waveform,
            "text_ids": text_ids,
            "labels": labels,
            "caption": item["response"],
            "is_text_only": False,
        }


class TextOnlyDataset(Dataset):
    """Text-only instruction samples, interleaved in Phase 1 to preserve the LLM's instruction following."""

    def __init__(self, manifest_path: PathLike, tokenizer, max_n: Optional[int] = None,
                 max_length: int = 512):
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        for row in load_manifest(self.manifest_path, max_n):
            instruction = row.get("instruction", row.get("input", ""))
            response = row.get("response", row.get("output", ""))
            if instruction and response:
                self.items.append({"instruction": instruction, "response": response})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        text_ids, labels = _tokenize_pair(
            self.tokenizer,
            build_prompt(self.tokenizer, item["instruction"], system_prompt=GENERAL_SYSTEM_PROMPT),
            build_prompt(self.tokenizer, item["instruction"], response=item["response"],
                         system_prompt=GENERAL_SYSTEM_PROMPT),
            max_length=self.max_length,
        )
        return {"waveform": None, "text_ids": text_ids, "labels": labels, "is_text_only": True}


def build_train_dataset(
    audio_manifests: Sequence[PathLike],
    tokenizer,
    max_audio_sec: float = 10.0,
    max_n: Optional[int] = None,
    text_manifests: Optional[Sequence[PathLike]] = None,
    text_max_n: Optional[int] = None,
) -> Dataset:
    """Concatenate audio manifests and optional text-only manifests.

    There are no sampling weights: every row is seen once per epoch, so datasets are drawn in
    proportion to their size. ``max_n`` / ``text_max_n`` keep the first N rows of each manifest
    (Phase 1 uses ``text_max_n=50000``).
    """
    datasets: List[Dataset] = []
    for path in audio_manifests:
        if not Path(path).exists():
            raise FileNotFoundError(f"audio manifest not found: {path}")
        ds = AudioCaptionDataset(path, tokenizer, max_n=max_n, max_audio_sec=max_audio_sec)
        logger.info("[audio] %s: %s rows", path, f"{len(ds):,}")
        datasets.append(ds)
    for path in text_manifests or []:
        if not Path(path).exists():
            raise FileNotFoundError(f"text manifest not found: {path}")
        ds = TextOnlyDataset(path, tokenizer, max_n=text_max_n)
        logger.info("[text ] %s: %s rows", path, f"{len(ds):,}")
        datasets.append(ds)
    if not datasets:
        raise ValueError("no manifests given")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


class HomogeneousBatchSampler(Sampler):
    """Batches that are all-audio or all-text, identical in modality across DDP ranks at every step.

    Each modality is shuffled and cut into batches on its own; a short tail batch is filled by
    re-sampling from the same modality (``oversample=True``) so no row is dropped. Batches are
    grouped into super-batches of ``world_size`` consecutive same-modality batches, the super-batches
    of both modalities are shuffled together, and rank ``r`` takes the ``r``-th batch of each. The
    audio:text batch ratio therefore follows the relative dataset sizes, in random order.

    Datasets are split by type: ``TextOnlyDataset`` rows are text, everything else is audio.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int = 42,
                 oversample: bool = True, rank: int = 0, world_size: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.oversample = oversample
        self.rank = rank
        self.world_size = max(1, world_size)
        self.epoch = 0
        self.audio_idx, self.text_idx = self._split_indices(dataset)

    @staticmethod
    def _split_indices(dataset: Dataset) -> tuple[List[int], List[int]]:
        audio_idx: List[int] = []
        text_idx: List[int] = []
        parts: Iterable[Dataset] = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
        offset = 0
        for ds in parts:
            n = len(ds)
            (text_idx if isinstance(ds, TextOnlyDataset) else audio_idx).extend(range(offset, offset + n))
            offset += n
        return audio_idx, text_idx

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _make_batches(self, indices: List[int], rng: random.Random) -> List[List[int]]:
        idx = indices[:]
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for i in range(0, len(idx), self.batch_size):
            b = idx[i:i + self.batch_size]
            if len(b) < self.batch_size:
                if not self.oversample:
                    continue
                b = b + rng.choices(indices, k=self.batch_size - len(b))
            batches.append(b)
        return batches

    def _super_batches(self, batches: List[List[int]]) -> List[List[List[int]]]:
        usable = len(batches) - len(batches) % self.world_size
        return [batches[i:i + self.world_size] for i in range(0, usable, self.world_size)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        # Audio batches are drawn from the RNG before text batches; keep this order for reproducibility.
        audio_batches = self._make_batches(self.audio_idx, rng) if self.audio_idx else []
        text_batches = self._make_batches(self.text_idx, rng) if self.text_idx else []
        supers = self._super_batches(audio_batches) + self._super_batches(text_batches)
        if self.shuffle:
            rng.shuffle(supers)
        for super_batch in supers:
            yield super_batch[self.rank]

    def __len__(self) -> int:
        def n_batches(indices: List[int]) -> int:
            if not indices:
                return 0
            if self.oversample:
                return (len(indices) + self.batch_size - 1) // self.batch_size
            return len(indices) // self.batch_size

        return (n_batches(self.audio_idx) // self.world_size
                + n_batches(self.text_idx) // self.world_size)


def collate_mixed(batch: List[dict], pad_token_id: int = 0) -> dict:
    """Pad waveforms with zeros, ``text_ids`` with ``pad_token_id`` and ``labels`` with -100."""
    audio = [b for b in batch if b.get("waveform") is not None]
    waveforms = None
    if audio:
        max_wave = max(b["waveform"].size(0) for b in audio)
        waves = []
        for b in batch:
            w = b.get("waveform")
            if w is None:
                waves.append(torch.zeros(max_wave))
            else:
                waves.append(F.pad(w, (0, max_wave - w.size(0))) if w.size(0) < max_wave else w)
        waveforms = torch.stack(waves)

    max_text = max(b["text_ids"].size(0) for b in batch)
    text_ids = torch.full((len(batch), max_text), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_text), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["text_ids"].size(0)
        text_ids[i, :n] = b["text_ids"]
        labels[i, :n] = b["labels"]

    return {
        "waveform": waveforms,
        "text_ids": text_ids,
        "labels": labels,
        "is_text_only": torch.tensor([bool(b.get("is_text_only", False)) for b in batch], dtype=torch.bool),
    }
