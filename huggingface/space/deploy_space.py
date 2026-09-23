#!/usr/bin/env python
"""Assemble the demo Space and upload it to the Hub.

Two kinds of Space share the same sample clips (``huggingface/space/samples``, built by build_samples.py):

    static  (default, free for every account)  static/index.html + README.md. Shows the caption
            the released model produced for each clip (``expected_caption`` in samples.json).
    gradio  (needs a PRO account on Hugging Face)  app.py + the ``card`` package; generates
            captions live.

    python huggingface/space/deploy_space.py --space-id KarthikKB1998/CARD-Audio-Captioning --private
    python huggingface/space/deploy_space.py --space-id KarthikKB1998/CARD-Audio-Captioning --dry-run
    python huggingface/space/deploy_space.py --kind gradio --space-id KarthikKB1998/CARD-Audio-Captioning-Live

If ``samples/samples.json`` exists locally, the samples folder is uploaded and replaces the samples
already in the Space. Without a local samples folder the Space's existing samples are left alone, so
the page can be redeployed from a machine that does not hold the audio.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def stage_code(dest: Path, kind: str) -> None:
    if kind == "static":
        for name in ("index.html", "README.md"):
            shutil.copy2(HERE / "static" / name, dest / name)
    else:
        for name in ("app.py", "requirements.txt", "README.md"):
            shutil.copy2(HERE / name, dest / name)
        shutil.copytree(REPO_ROOT / "card", dest / "card", ignore=shutil.ignore_patterns("__pycache__"))
    if (REPO_ROOT / "LICENSE").exists():
        shutil.copy2(REPO_ROOT / "LICENSE", dest / "LICENSE")


def stage_samples(dest: Path, samples_dir: Path, kind: str) -> bool:
    manifest = samples_dir / "samples.json"
    if not manifest.exists():
        return False
    with open(manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    (dest / "samples").mkdir()
    shutil.copy2(manifest, dest / "samples" / "samples.json")
    for s in data["samples"]:
        src = samples_dir / s["file"]
        if not src.exists():
            raise SystemExit(f"samples.json lists {s['file']} but {src} does not exist")
        if kind == "static" and not (s.get("caption") or s.get("expected_caption")):
            raise SystemExit(f"sample {s['name']!r} has no caption; the static page needs expected_caption "
                             "(run build_samples.py with --eval-results)")
        shutil.copy2(src, dest / "samples" / s["file"])
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space-id", required=True)
    p.add_argument("--kind", choices=("static", "gradio"), default="static")
    p.add_argument("--samples-dir", type=Path, default=HERE / "samples")
    p.add_argument("--private", action="store_true", help="create the Space private (flip later in Settings)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        stage_code(dest, args.kind)
        with_samples = stage_samples(dest, args.samples_dir, args.kind)
        files = sorted(q for q in dest.rglob("*") if q.is_file())
        print(f"[space] {args.kind} Space, {len(files)} files, {sum(q.stat().st_size for q in files) / 1e6:.1f} MB, "
              f"samples {'included' if with_samples else 'not included'}")
        for q in files:
            print(f"  {q.stat().st_size / 1e3:9.1f} kB  {q.relative_to(dest).as_posix()}")
        if args.dry_run:
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.space_id, repo_type="space", space_sdk=args.kind,
                        private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.space_id, repo_type="space", folder_path=str(dest),
            commit_message=f"Deploy CARD {args.kind} demo" + (" with sample clips" if with_samples else ""),
            delete_patterns=["samples/*"] if with_samples else None,
        )
    print(f"[space] done: https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
