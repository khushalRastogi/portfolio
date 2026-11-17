# Model Export & Private Hugging Face Hub Hosting Guide

Keep all trained weights out of your portfolio git repository and maintain every model in **one private Hugging Face Hub repository** organized by task subfolder (e.g., `id-field-detector/`, `invoice-fields/`, `table-structure/`).

---

## Step 1: Create and Set `HF_TOKEN`

1. Open **Hugging Face → Settings → Access Tokens** (`https://huggingface.co/settings/tokens`).
2. Create two tokens following least privilege:
   - **Upload / Write token** (for your local machine when pushing models): Fine-grained token with *Write access to contents/settings of your personal repos*.
   - **Server / Read token** (for `server.js` in production): Fine-grained token with *Read access to contents of your private model repo only*.
3. Export `HF_TOKEN` in your shell before running upload commands:

   **Linux / macOS:**
   ```bash
   export HF_TOKEN="hf_your_write_token"
   ```

   **Windows PowerShell:**
   ```powershell
   $env:HF_TOKEN="hf_your_write_token"
   ```

   **Local Node.js Server (`.env`, gitignored):**
   ```env
   HF_TOKEN=hf_your_read_only_token
   PORT=3000
   ```
   *(On Render, Railway, Cloud Run, or Fly.io, add `HF_TOKEN` in the service's Environment Variables dashboard—never commit `.env`.)*

---

## Step 2: Convert `.pth` Checkpoint to ONNX (if needed)

Run inside your Python training environment (where `torch`, `yolox`, and optionally `onnxruntime` are installed):

```bash
python scripts/export_onnx.py \
  --model-dir /path/to/your/checkpoint_dir \
  --checkpoint your_weights.pth \
  --output-dir ./model
```

This creates:
- `model/model.onnx` (with decoded inference outputs)
- `model/config.json` (sanitized to only `idx2label`, `conf_thre`, `nms_thre`, and `test_size`—proprietary variant/model names are stripped)

---

## Step 3: Initialize Private Repo & Upload ONNX Model

### Option A: Upload an existing `./model` folder (`scripts/hf_upload.py`)

Requires only `huggingface_hub` (`pip install huggingface_hub`—no PyTorch needed):

```bash
export HF_TOKEN="hf_your_write_token"

python scripts/hf_upload.py \
  --repo-id khushalR/doc-models-private \
  --subpath id-field-detector \
  --model-dir ./model
```

This script:
1. Creates `khushalR/doc-models-private` as a **private** model repo (`exist_ok=True`).
2. Initializes a root `README.md` if absent.
3. Strips any proprietary metadata from `config.json` and uploads both `id-field-detector/config.json` and `id-field-detector/model.onnx` (large binaries are handled automatically by HF storage).

### Option B: Export and upload in a single command

```bash
export HF_TOKEN="hf_your_write_token"

python scripts/export_onnx.py \
  --model-dir /path/to/your/checkpoint_dir \
  --checkpoint your_weights.pth \
  --output-dir ./model \
  --hf-repo khushalR/doc-models-private \
  --hf-subpath id-field-detector
```

---

## Step 4: Configure the Web App to Fetch from Hugging Face Hub

Update `config.ini` in the project root:

```ini
[model]
source = huggingface
onnx_file = model.onnx
config_file = config.json

hf_repo = khushalR/doc-models-private
hf_subpath = id-field-detector
hf_revision = main

[samples]
dir = samples
```

When the Node.js server starts or handles its first `/api/workbench-config` or `/api/detect` request, it authenticates using `process.env.HF_TOKEN`, downloads `config.json` + `model.onnx` into `.model-cache/` (gitignored), and runs live ONNX inference on port `3000`.
