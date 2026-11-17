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
import configparser
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


def parse_config_ini(config_path: Path) -> dict:
    """Read the project config.ini and return the model settings."""
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    if "model" not in parser:
        raise ValueError(f"Missing [model] section in {config_path}")

    model_cfg = parser["model"]
    return {
        "source": model_cfg.get("source", "local").strip(),
        "dir": model_cfg.get("dir", "model").strip(),
        "onnx_file": model_cfg.get("onnx_file", "model.onnx").strip(),
        "config_file": model_cfg.get("config_file", "config.json").strip(),
        "hf_repo": model_cfg.get("hf_repo", "").strip(),
        "hf_subpath": model_cfg.get("hf_subpath", "").strip(),
        "hf_revision": model_cfg.get("hf_revision", "main").strip(),
    }


def relevant_model_files(model_dir: Path, onnx_name: str, config_name: str) -> list[Path]:
    """Return all model artifacts that should be uploaded with the ONNX bundle."""
    files: list[Path] = []

    config_path = model_dir / config_name
    if config_path.exists():
        files.append(config_path)

    onnx_path = model_dir / onnx_name
    if onnx_path.exists():
        files.append(onnx_path)

    onnx_stem = Path(onnx_name).stem
    for child in sorted(model_dir.iterdir()):
        if not child.is_file():
            continue
        name = child.name
        if name == config_name:
            continue
        if name == onnx_name:
            continue
        if name.startswith(f"{onnx_stem}.onnx") or name.startswith(f"{onnx_name}."):
            files.append(child)

    # Upload the sanitized config first, then the ONNX payloads.
    files = list(dict.fromkeys(files))
    return files


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

    model_dir = model_dir.resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = relevant_model_files(model_dir, onnx_name, config_name)
    if not files:
        raise FileNotFoundError(f"No model artifacts found in {model_dir}")

    api = HfApi(token=token)

    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=not public,
        exist_ok=True,
    )
    print(f"Repository ready: {repo_url} (private={not public})")

    try:
        uploaded_files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    except Exception:
        uploaded_files = []

    if "README.md" not in uploaded_files:
        api.upload_file(
            path_or_fileobj=PRIVATE_REPO_README.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            commit_message="Initialize private model registry README",
        )
        print("Initialized README.md in repository root.")

    prefix = subpath.strip("/")
    uploaded_targets: list[str] = []

    for file_path in files:
        rel_name = file_path.name
        repo_path = f"{prefix}/{rel_name}" if prefix else rel_name

        if file_path.name == config_name:
            with open(file_path, "r", encoding="utf-8") as f:
                sanitized = sanitize_config(json.load(f))
            payload = json.dumps(sanitized, indent=2).encode("utf-8")
            api.upload_file(
                path_or_fileobj=payload,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                commit_message=f"Upload {repo_path}",
            )
            print(f"Uploaded {repo_path}")
            uploaded_targets.append(repo_path)
            continue

        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            commit_message=f"Upload {repo_path}",
        )
        print(f"Uploaded {repo_path}")
        uploaded_targets.append(repo_path)

    print(f"Done! Uploaded {len(uploaded_targets)} file(s) to {repo_id}/{prefix or '.'}")
    print(f"Update config.ini with source = huggingface, hf_repo = {repo_id}, hf_subpath = {prefix}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "config.ini"

    parser = argparse.ArgumentParser(
        description="Initialize a private Hugging Face Hub repository and upload model artifacts from config.ini."
    )
    parser.add_argument("--repo-id", help="HF repo ID (e.g. khushalR/doc-models-private). Overrides config.ini value.")
    parser.add_argument("--subpath", help="Subfolder inside the repo for this model. Overrides config.ini value.")
    parser.add_argument("--model-dir", type=Path, help="Local directory containing model artifacts. Overrides config.ini model.dir.")
    parser.add_argument("--onnx-name", default="model.onnx", help="ONNX filename inside --model-dir")
    parser.add_argument("--config-name", default="config.json", help="Config filename inside --model-dir")
    parser.add_argument("--revision", default="main", help="Target branch or revision")
    parser.add_argument("--public", action="store_true", help="Create as public repo (default is private)")
    parser.add_argument("--config", type=Path, default=default_config, help="Path to config.ini used for repo metadata")
    args = parser.parse_args()

    cfg = parse_config_ini(args.config)
    repo_id = args.repo_id or cfg["hf_repo"]
    subpath = args.subpath or cfg["hf_subpath"]
    model_dir = args.model_dir or Path(cfg["dir"]).expanduser()
    if not repo_id:
        raise ValueError("No Hugging Face repo configured. Set hf_repo in config.ini or pass --repo-id.")

    upload_model(
        repo_id=repo_id,
        subpath=subpath,
        model_dir=model_dir,
        onnx_name=args.onnx_name,
        config_name=args.config_name,
        revision=args.revision or cfg["hf_revision"],
        public=args.public,
    )


if __name__ == "__main__":
    main()
