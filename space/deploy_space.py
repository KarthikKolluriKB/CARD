#!/usr/bin/env python
"""Assemble the demo Space (app, card package, samples) and upload it to the Hub.

    python space/deploy_space.py --space-id KarthikKB1998/CARD-Audio-Captioning --private
    python space/deploy_space.py --space-id KarthikKB1998/CARD-Audio-Captioning --dry-run

Uploads app.py, requirements.txt, README.md and a copy of the ``card`` package. If
``space/samples/samples.json`` exists, the samples folder is uploaded too and replaces any samples
already in the Space. Existing samples are left alone when there is no local samples folder, so the
code can be redeployed from a machine that does not hold the audio.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def stage(dest: Path, samples_dir: Path) -> bool:
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(HERE / name, dest / name)
    shutil.copytree(REPO_ROOT / "card", dest / "card", ignore=shutil.ignore_patterns("__pycache__"))
    if (REPO_ROOT / "LICENSE").exists():
        shutil.copy2(REPO_ROOT / "LICENSE", dest / "LICENSE")

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
        shutil.copy2(src, dest / "samples" / s["file"])
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space-id", required=True)
    p.add_argument("--samples-dir", type=Path, default=HERE / "samples")
    p.add_argument("--private", action="store_true", help="create the Space private (flip later in Settings)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        with_samples = stage(dest, args.samples_dir)
        files = sorted(q for q in dest.rglob("*") if q.is_file())
        print(f"[space] {len(files)} files, {sum(q.stat().st_size for q in files) / 1e6:.1f} MB, "
              f"samples {'included' if with_samples else 'not included'}")
        for q in files:
            print(f"  {q.stat().st_size / 1e3:9.1f} kB  {q.relative_to(dest).as_posix()}")
        if args.dry_run:
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio",
                        private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.space_id, repo_type="space", folder_path=str(dest),
            commit_message="Deploy CARD demo" + (" with sample clips" if with_samples else ""),
            delete_patterns=["samples/*"] if with_samples else None,
        )
    print(f"[space] done: https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
