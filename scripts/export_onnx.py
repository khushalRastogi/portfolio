"""
Export a trained PyTorch detector checkpoint (.pth) + config.json to ONNX
and optionally upload the artifacts to a private Hugging Face Hub repository.

Usage:
    python scripts/export_onnx.py \
        --model-dir /path/to/checkpoint_dir \
        --checkpoint model_weights.pth \
        --output-dir ./model

    python scripts/export_onnx.py \
        --model-dir /home/khushal/Downloads/IDX_MODELS/IDX/Indian_IDX/YoloX-Medium/Iteration-4-3rd_Plus_Aadhaar_VID_plus_VoterID_Addition \
        --checkpoint yoloX_M_iter4.pth \
        --output-dir /home/khushal/Downloads/IDX_MODELS/IDX/Indian_IDX/YoloX-Medium-ONNX/Iteration-4-3rd_Plus_Aadhaar_VID_plus_VoterID_Addition
    
    python scripts/export_onnx.py \
        --model-dir /home/khushal/Downloads/IDX_MODELS/IDX/Indian_IDX/YoloX-Small/Iteration-6-5th_iter_plus_New_VoterID_Samples_Addition \
        --checkpoint yoloX_S_iter6.pth \
        --output-dir /home/khushal/Downloads/IDX_MODELS/IDX/Indian_IDX/YoloX-Small-ONNX/Iteration-6-5th_iter_plus_New_VoterID_Samples_Addition
    
    
    
    # Optional: export and push directly to a private Hugging Face Hub repo
    HF_TOKEN=hf_xxx python scripts/export_onnx.py \
        --model-dir /path/to/checkpoint_dir \
        --checkpoint model_weights.pth \
        --output-dir ./model \
        --hf-repo khushalR/doc-models-private \
        --hf-subpath id-field-detector
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import torch
from yolox.exp import Exp, get_exp


VARIANT_SCALES = {
    "yolox-nano": (0.33, 0.25),
    "yolox-t": (0.33, 0.375),
    "yolox-tiny": (0.33, 0.375),
    "yolox-s": (0.33, 0.50),
    "yolox-m": (0.67, 0.75),
    "yolox-l": (1.00, 1.00),
    "yolox-x": (1.33, 1.25),
}

WIDTH_TO_VARIANT = {
    96: "yolox-t",
    128: "yolox-s",
    192: "yolox-m",
    256: "yolox-l",
}


def infer_config_from_state_dict(state_dict: dict) -> dict:
    inferred = {}
    cls_key = "head.cls_preds.0.weight"
    if cls_key in state_dict:
        weight = state_dict[cls_key]
        inferred["num_classes"] = int(weight.shape[0])
        in_ch = int(weight.shape[1])
        if in_ch in WIDTH_TO_VARIANT:
            inferred["variant"] = WIDTH_TO_VARIANT[in_ch]
    return inferred


def build_exp(variant: str, num_classes: int, test_size: tuple[int, int], exp_file: str | None) -> Exp:
    if exp_file:
        exp = get_exp(exp_file, None)
        if hasattr(exp, "set_model_scale"):
            exp.set_model_scale(variant)
    else:
        exp = Exp()
        if variant not in VARIANT_SCALES:
            raise ValueError(f"Unsupported variant '{variant}'. Expected one of: {list(VARIANT_SCALES)}")
        exp.depth, exp.width = VARIANT_SCALES[variant]

    exp.num_classes = num_classes
    exp.test_size = test_size
    return exp


def export_model(
    model_dir: Path,
    checkpoint_name: str,
    output_dir: Path,
    onnx_name: str,
    exp_file: str | None,
    opset: int,
) -> tuple[Path, Path]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        parent_config = model_dir.parent / "config.json"
        if parent_config.exists():
            config_path = parent_config

    meta = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    ckpt_path = model_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt)
    elif hasattr(ckpt, "state_dict"):
        state_dict = ckpt.state_dict()
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    inferred = infer_config_from_state_dict(state_dict)
    variant = meta.get("variant") or inferred.get("variant")
    if not variant:
        raise ValueError("Model variant could not be inferred; specify 'variant' in config.json.")

    idx2label = meta.get("idx2label")
    if idx2label:
        num_classes = len(idx2label)
    else:
        num_classes = inferred.get("num_classes")
        if not num_classes:
            raise ValueError("num_classes could not be determined; provide 'idx2label' in config.json.")
        idx2label = {str(i): f"class_{i}" for i in range(num_classes)}

    test_size = tuple(meta.get("test_size") or (640, 640))
    exp = build_exp(variant, num_classes, test_size, exp_file)

    model = exp.get_model()
    model.load_state_dict(state_dict)
    model.eval()
    model.head.decode_in_inference = True

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / onnx_name
    dummy_input = torch.randn(1, 3, test_size[0], test_size[1])

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["images"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )

    # Write sanitized runtime config.json (omit variant/model identifiers)
    runtime_config = {
        "idx2label": idx2label,
        "conf_thre": float(meta.get("conf_thre", 0.5)),
        "nms_thre": float(meta.get("nms_thre", 0.45)),
        "test_size": [int(test_size[0]), int(test_size[1])],
    }
    out_config_path = output_dir / "config.json"
    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(runtime_config, f, indent=2)

    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"images": np.zeros((1, 3, test_size[0], test_size[1]), dtype=np.float32)})
        print(f"Verified ONNX export: output shape {out[0].shape}")
    except ImportError:
        print("onnxruntime not installed in Python env; skipped verification pass.")

    print(f"Exported ONNX model to {onnx_path}")
    print(f"Wrote runtime config to {out_config_path}")
    return onnx_path, out_config_path


def upload_to_hf(
    onnx_path: Path,
    config_path: Path,
    repo_id: str,
    subpath: str,
    revision: str,
    private: bool,
) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is required to upload to Hugging Face Hub.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    prefix = subpath.strip("/")
    for local_file in (onnx_path, config_path):
        path_in_repo = f"{prefix}/{local_file.name}" if prefix else local_file.name
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            commit_message=f"Update {path_in_repo}",
        )
        print(f"Uploaded {path_in_repo} -> hf://{repo_id}@{revision}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX and optionally upload to Hugging Face Hub.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory containing .pth checkpoint and config.json")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint filename inside --model-dir")
    parser.add_argument("--output-dir", default=Path("model"), type=Path, help="Output directory for model.onnx and config.json")
    parser.add_argument("--onnx-name", default="model.onnx", help="Output ONNX filename")
    parser.add_argument("--exp-file", default=None, help="Optional custom experiment definition file")
    parser.add_argument("--opset", default=12, type=int, help="ONNX opset version")
    parser.add_argument("--hf-repo", default=None, help="Optional HF Hub repo ID (e.g. khushalR/doc-models-private)")
    parser.add_argument("--hf-subpath", default="id-field-detector", help="Subfolder inside the HF Hub repo")
    parser.add_argument("--hf-revision", default="main", help="Branch or tag on HF Hub")
    parser.add_argument("--public", action="store_true", help="Create HF repo as public (default: private)")
    args = parser.parse_args()

    onnx_path, out_config_path = export_model(
        model_dir=args.model_dir,
        checkpoint_name=args.checkpoint,
        output_dir=args.output_dir,
        onnx_name=args.onnx_name,
        exp_file=args.exp_file,
        opset=args.opset,
    )

    if args.hf_repo:
        upload_to_hf(
            onnx_path=onnx_path,
            config_path=out_config_path,
            repo_id=args.hf_repo,
            subpath=args.hf_subpath,
            revision=args.hf_revision,
            private=not args.public,
        )


if __name__ == "__main__":
    main()
