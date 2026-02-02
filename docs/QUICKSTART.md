# Quick Start Guide - Disk Space Constrained Setup

## ⚠️ Disk Space Issue Detected

Your installation failed due to insufficient disk space. Here's how to proceed:

### Option 1: Free Up Disk Space (Recommended)

**Quick Wins:**
```powershell
# Clean pip cache
pip cache purge

# Clean temp files
Remove-Item -Path $env:TEMP\* -Recurse -Force -ErrorAction SilentlyContinue

# Check disk space
Get-PSDrive C
```

**Then install minimal requirements:**
```powershell
pip install -r requirements-minimal.txt
```

### Option 2: Use Minimal Installation (Current Solution)

I've created `requirements-minimal.txt` which excludes:
- ❌ `gym` (RL Phase 4 - optional)
- ❌ `wandb`, `tensorboard` (logging - optional)
- ❌ `pytest`, `black`, `isort` (dev tools - optional)
- ❌ `onnx`, `tensorrt` (deployment - install later)

**Install:**
```powershell
cd "d:\github pipeline\project-aegis"
pip install -r requirements-minimal.txt
```

**This gives you:**
- ✅ V-JEPA (core model)
- ✅ VLM with Llama 3.1
- ✅ Training capability (console logging)
- ✅ Inference

### Option 3: Use Google Colab (Zero Local Storage)

**Best option if disk space is very limited!**

1. Upload project to Google Drive
2. Open Colab notebook
3. Install everything in Colab environment (free 100GB+ storage)

**Colab Install:**
```python
!pip install -r requirements.txt  # Full install works in Colab
```

---

## After Installing Minimal Requirements

### Test Installation

```powershell
python -c "from models.vjepa import VJEPAModel; from models.vlm import AEGISModel; print('✅ Installation successful!')"
```

### Run Training (Without wandb)

```powershell
python scripts/train_vlm.py --config configs/vlm_config.yaml
```

**Note:** Training will log to console instead of wandb. This is fine for learning!

---

## If You Want Full Features Later

**After freeing disk space, install optional packages:**

```powershell
# Logging (recommended for experiments)
pip install wandb tensorboard

# Development tools (for contributing)
pip install pytest black isort flake8 mypy

# Deployment (when ready to deploy)
pip install onnx onnxruntime-gpu

# RL Phase 4 (advanced - requires significant space)
pip install gym stable-baselines3
```

---

## Disk Space Requirements

| Component | Size | Priority |
|-----------|------|----------|
| **Minimal requirements** | ~2GB | Required |
| **Full requirements** | ~4GB | Nice-to-have |
| **Datasets** (Kinetics subset) | ~20GB | Required for training |
| **Checkpoints** (V-JEPA + VLM) | ~5GB | Required |
| **Total (minimal setup)** | **~27GB** | |
| **Total (full setup)** | **~29GB** | |

---

## Current Status

✅ **Fixed Issues:**
- Created `requirements-minimal.txt`
- Made wandb optional in training script
- Training will work with console logging

**Next Step:**
```powershell
pip install -r requirements-minimal.txt
```

Then try training again!
