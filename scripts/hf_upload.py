"""
Initialize a private Hugging Face Hub model repository (if it does not exist)
and upload an ONNX model + sanitized config.json into a task subfolder.

Requires only `huggingface_hub`:
    pip install huggingface_hub

Usage:
    export HF_TOKEN="hf_your_write_token"
    python scipts/hf_upload.py \
        --repo-id khushalR/doc-models-private \
        --subpath id-field-detector \
        --model-dir ./model
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


PRIVATE_REPO_README = """---
private: true
---

# Private Document AI Models

Central private model registry organized by capability subfolder.
"""


def sanitize_config(raw: dict) -> dict:
    """Keep only runtime inference parameters; strip proprietary model/variant names."""
    return {
        "idx2label": raw.get("idx2label", {}),
        "conf_thre": float(raw.get("conf_thre", 0.5)),
        "nms_thre": float(raw.get("nms_thre", 0.45)),
        "test_size": [int(x) for x in raw.get("test_size", [640, 640])],
    }


def upload_model(
    repo_id: str,
    subpath: str,
    model_dir: Path,
    onnx_name: str,
    config_name: str,
    revision: str,
    public: bool,
) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Export a Hugging Face token with write permissions before running this script."
        )

    onnx_path = model_dir / onnx_name
    config_path = model_dir / config_name
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    api = HfApi(token=token)

    # 1. Create or ensure the private model repository exists
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=not public,
        exist_ok=True,
    )
    print(f"Repository ready: {repo_url} (private={not public})")

    # 2. Initialize a minimal README.md if the repo does not have one yet
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    except Exception:
        files = []

    if "README.md" not in files:
        api.upload_file(
            path_or_fileobj=PRIVATE_REPO_README.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            commit_message="Initialize private model registry README",
        )
        print("Initialized README.md in repository root.")

    # 3. Sanitize config.json in memory before uploading so no model/variant IP leaks
    with open(config_path, "r", encoding="utf-8") as f:
        sanitized = sanitize_config(json.load(f))
    sanitized_bytes = json.dumps(sanitized, indent=2).encode("utf-8")

    prefix = subpath.strip("/")
    onnx_repo_path = f"{prefix}/{onnx_name}" if prefix else onnx_name
    cfg_repo_path = f"{prefix}/{config_name}" if prefix else config_name

    # 4. Upload config.json and model.onnx (HF Hub automatically stores large .onnx files via Git LFS / Xet)
    api.upload_file(
        path_or_fileobj=sanitized_bytes,
        path_in_repo=cfg_repo_path,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=f"Upload {cfg_repo_path}",
    )
    print(f"Uploaded {cfg_repo_path}")

    api.upload_file(
        path_or_fileobj=str(onnx_path),
        path_in_repo=onnx_repo_path,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=f"Upload {onnx_repo_path}",
    )
    print(f"Uploaded {onnx_repo_path}")
    print(f"Done! Update config.ini with source = huggingface, hf_repo = {repo_id}, hf_subpath = {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a private Hugging Face Hub repository and upload an ONNX model + config.json."
    )
    parser.add_argument("--repo-id", required=True, help="HF repo ID (e.g. khushalR/doc-models-private)")
    parser.add_argument("--subpath", default="id-field-detector", help="Subfolder inside the repo for this model")
    parser.add_argument("--model-dir", default=Path("model"), type=Path, help="Local directory containing model.onnx and config.json")
    parser.add_argument("--onnx-name", default="model.onnx", help="ONNX filename inside --model-dir")
    parser.add_argument("--config-name", default="config.json", help="Config filename inside --model-dir")
    parser.add_argument("--revision", default="main", help="Target branch or revision")
    parser.add_argument("--public", action="store_true", help="Create as public repo (default is private)")
    args = parser.parse_args()

    upload_model(
        repo_id=args.repo_id,
        subpath=args.subpath,
        model_dir=args.model_dir,
        onnx_name=args.onnx_name,
        config_name=args.config_name,
        revision=args.revision,
        public=args.public,
    )


if __name__ == "__main__":
    main()
