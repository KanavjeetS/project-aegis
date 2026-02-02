# Project A.E.G.I.S. - Setup Guide

## Prerequisites

- Python 3.9 or higher
- CUDA 11.7+ (for GPU acceleration)
- 40GB free disk space (for datasets)
- 8GB+ VRAM (RTX 2060 or better) OR Google Colab

---

## Option 1: Local Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/project-aegis.git
cd project-aegis
```

### Step 2: Create Environment

```bash
# Using conda (recommended)
conda create -n aegis python=3.9
conda activate aegis

# OR using venv
python -m venv aegis-env
source aegis-env/bin/activate  # Linux/Mac
# aegis-env\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with `decord`, install it separately:
```bash
pip install decord -f https://github.com/dmlc/decord/releases
```

### Step 4: Download Pre-trained Checkpoints

```bash
python scripts/download_checkpoints.py
```

This will download:
- V-JEPA ViT-L/16 pre-trained weights (~900MB)
- (Optional) Fine-tuned VLM checkpoint

### Step 5: Verify Installation

```bash
python -c "import torch; from models.vjepa import VJEPAModel; print('Installation successful!')"
```

---

## Option 2: Docker Installation

```bash
# Build Docker image
docker build -t aegis:latest .

# Run container
docker run -it --gpus all -v $(pwd):/workspace aegis:latest

# Inside container
python scripts/inference_vlm.py --video samples/flood.mp4 --checkpoint checkpoints/aegis_vlm.pth
```

---

## Option 3: Google Colab (Free Tier Compatible)

1. Open the [Quick Start Notebook](https://colab.research.google.com/github/yourusername/project-aegis/blob/main/notebooks/01_quick_start.ipynb)
2. Runtime → Change runtime type → T4 GPU
3. Run all cells

**No installation needed!** Everything runs in browser.

---

## Dataset Setup (Optional - for Training)

### Download Datasets

```bash
# Download Kinetics subset (20GB)
python data/downloaders/kinetics_downloader.py --subset --output data/kin etics

# Download Ego4D subset (10GB)
python data/downloaders/ego4d_downloader.py --subset --output data/ego4d

# Download LADI disaster images (5GB)
python data/downloaders/ladi_converter.py --output data/ladi

# Download MADOS marine dataset (2GB)
python data/downloaders/mados_downloader.py --output data/mados
```

**Total:** ~40GB

---

## Common Issues & Solutions

### Issue: CUDA Out of Memory

**Solution:**
- Reduce batch size in `configs/vjepa_config.yaml`
- Enable gradient checkpointing
- Use 4-bit quantization for VLM

```yaml
training:
  batch_size: 4  # Reduce from 16
```

### Issue: `decord` Installation Fails

**Solution:**
```bash
conda install -c conda-forge decord
```

### Issue: Slow Video Loading

**Solution:**
- Install OpenCV with FFmpeg support
- Use SSD storage for datasets

### Issue: Hugging Face Model Download Stalls

**Solution:**
```bash
export HF_HOME=/path/to/large/disk
huggingface-cli login
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 2060 (8GB VRAM) | RTX 3090 (24GB VRAM) |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 60GB SSD | 200GB+ NVMe SSD |
| **CPU** | 4 cores | 8+ cores |

**For Free Colab:**
- T4 GPU (16GB VRAM) - sufficient for inference
- P100/V100 - sufficient for training Q-Former

---

## Next Steps

After installation:

1. **Test Inference:** `python scripts/inference_vlm.py --video samples/test.mp4`
2. **Extract Embeddings:** `python scripts/extract_embeddings.py`
3. **Fine-tune (optional):** See [TRAINING.md](TRAINING.md)
4. **Deploy to Edge:** See [DEPLOYMENT.md](DEPLOYMENT.md)

---

## Troubleshooting

If you encounter any issues:

1. Check [GitHub Issues](https://github.com/yourusername/project-aegis/issues)
2. Join [Discussions](https://github.com/yourusername/project-aegis/discussions)
3. Read full documentation in `docs/`

---

**Ready to predict disasters! 🌍**
