#!/usr/bin/env python
"""Upload an exported CARD folder to the Hugging Face Hub.

    python scripts/upload_to_hub.py --folder hub/CARD-Qwen3-4B-AudioCaps \\
        --repo KarthikKB1998/CARD-Qwen3-4B-AudioCaps [--private] [--dry-run]

Authentication: ``hf auth login`` beforehand, or export ``HF_TOKEN``. The token is never
written anywhere by this script.

Folders containing ``llm/`` (multi-GB shards) go through ``upload_large_folder`` (resumable,
parallel, Xet-backed); small folders use ``upload_folder`` in a single commit. Refuses to
upload pickles or a folder without ``MANIFEST.sha256`` (run ``export_to_hub.py`` first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True, type=Path, help="export folder from export_to_hub.py")
    p.add_argument("--repo", required=True, help="repo id, e.g. KarthikKB1998/CARD-Qwen3-4B-AudioCaps")
    p.add_argument("--private", action="store_true", help="create the repo private (flip public later)")
    p.add_argument("--path-in-repo", default=None, help="upload under this subfolder (ablation bundles)")
    p.add_argument("--message", default="Upload CARD model", help="commit message")
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded and exit")
    args = p.parse_args(argv)

    folder = args.folder
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a directory")
    if not (folder / "MANIFEST.sha256").exists():
        raise SystemExit("MANIFEST.sha256 missing: run scripts/export_to_hub.py first")
    pickles = [str(q.relative_to(folder)) for q in folder.rglob("*")
               if q.is_file() and q.suffix in (".pt", ".pth", ".bin", ".ckpt", ".pkl")]
    if pickles:
        raise SystemExit("refusing to upload pickle files: " + ", ".join(pickles))

    files = [q for q in folder.rglob("*") if q.is_file()]
    total_gb = sum(q.stat().st_size for q in files) / 1e9
    large = (folder / "llm").is_dir()
    print(f"[upload] {len(files)} files, {total_gb:.2f} GB -> https://huggingface.co/{args.repo}"
          + (f"/tree/main/{args.path_in_repo}" if args.path_in_repo else ""))
    print(f"[upload] method: {'upload_large_folder' if large else 'upload_folder'}"
          f"{' (private)' if args.private else ''}")
    if args.dry_run:
        for q in sorted(files):
            print(f"  {q.stat().st_size / 1e6:9.1f} MB  {q.relative_to(folder).as_posix()}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"[upload] authenticated as {who.get('name')}")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    if large and not args.path_in_repo:
        api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=str(folder),
                                print_report=True)
    else:
        api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(folder),
                          path_in_repo=args.path_in_repo, commit_message=args.message)
    print(f"[upload] done: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
