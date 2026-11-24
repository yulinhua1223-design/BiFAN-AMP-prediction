#!/usr/bin/env python3
"""Download the ESM-2 checkpoint used by BiFAN.

The default download contains the configuration, tokenizer and safetensors
weights (about 2.7 GB). Model weights are not redistributed with this
repository and remain subject to the upstream model licence.

Usage:
    python src/download_esm.py
    python src/download_esm.py --all-files

Optional mirror settings are environment variables, for example:
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HUB_DISABLE_XET=1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REQUIRED_FILES = (
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
NEEDED_PATTERNS = ("*.safetensors", "*.json", "*.txt")


def repo_root() -> Path:
    root = Path(__file__).resolve().parent
    return root.parent if root.name == "src" else root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--target", default="models/esm2_t33_650M_UR50D",
                        help="path relative to the repository root")
    parser.add_argument("--endpoint", default=None,
                        help="optional Hugging Face endpoint or mirror")
    parser.add_argument("--disable-xet", action="store_true",
                        help="use the traditional HTTP download path")
    parser.add_argument("--all-files", action="store_true",
                        help="download every file in the model repository")
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    target = Path(args.target)
    if not target.is_absolute():
        target = repo_root() / target
    target.mkdir(parents=True, exist_ok=True)
    allow_patterns = None if args.all_files else list(NEEDED_PATTERNS)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("missing huggingface_hub; run: pip install -r requirements.txt")
        return 1

    print(f"model      : {args.model}")
    print(f"target dir : {target}")
    print(f"endpoint   : {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"xet        : {'disabled' if args.disable_xet else 'default'}")
    scope = "all files" if allow_patterns is None else "configuration + safetensors (~2.7 GB)"
    print(f"scope      : {scope}")
    print("downloading; interrupted files are resumed on the next run")

    snapshot_download(repo_id=args.model, local_dir=str(target),
                      allow_patterns=allow_patterns)

    names = {path.name for path in target.iterdir() if path.is_file()}
    missing = [name for name in REQUIRED_FILES if name not in names]
    if not any(name in names for name in WEIGHT_FILES):
        missing.append("weight file (model.safetensors)")
    if missing:
        print("download is incomplete; missing: " + ", ".join(missing))
        print("re-run with --all-files if the upstream repository layout differs")
        return 1

    total_bytes = sum(
        path.stat().st_size for path in target.rglob("*") if path.is_file()
    )
    print(f"done: {total_bytes / 1024 ** 3:.2f} GB")
    print(f"next: python src/main1.py --esm-path {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
