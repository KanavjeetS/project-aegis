# Project A.E.G.I.S. - Training Guide

## Overview

Three training phases:
1. **Phase 1:** V-JEPA pre-training (optional - use pretrained)
2. **Phase 2:** Q-Former training (~6-8 hours on T4)
3. **Phase 3:** RL Agent training (optional - architecture ready)

---

## Phase 1: V-JEPA Pre-training (Optional)

**Recommended:** Use pre-trained checkpoint to save compute.

If you want to train from scratch:

```bash
python scripts/train_vjepa.py --config configs/vjepa_config.yaml
```

**Hardware:** 4x RTX 3090 (or equivalent), 2-3 days

**Innovation:** Our training includes physics-aware temporal causality loss:
```python
# Penalizes physically impossible transitions
causality_loss = torch.norm(acceleration, p=2).mean()
```

---

## Phase 2: Q-Former Training (Recommended)

**Goal:** Train connector between V-JEPA and Llama 3.1

### Step 1: Prepare Video-Caption Dataset

Option A: Use Existing (Kinetics + Auto-captions)
```bash
# Generate captions for Kinetics using GPT-4
python data/caption_generator.py \
  --videos data/kinetics \
  --output data/kinetics_captions.json \
  --model gpt-4
```

Option B: Custom Disaster Dataset
```bash
# Create your own video-caption pairs
# Format: JSON file with {"video_path": "...", "caption": "..."}
```

### Step 2: Update Config

```yaml
# configs/vlm_config.yaml
data:
  video_caption_pairs: "data/kinetics_captions.json"
  # OR
  video_caption_pairs: "data/custom_captions.json"
```

### Step 3: Train Q-Former

**Local GPU (RTX 2060+):**
```bash
python scripts/train_vlm.py --config configs/vlm_config.yaml
```

**Free Colab T4:**
```python
# In Colab notebook
!python scripts/train_vlm.py --config configs/vlm_config.yaml
```

**Training Time:**
- T4 (Colab): ~8 hours
- RTX 3090: ~3 hours
- A100: ~1.5 hours

**Checkpoints:** Saved every epoch to `checkpoints/`

---

## Phase 3: RL Agent Training (Advanced - Optional)

**Status:** Architecture implemented, training optional

**Requirements:**
- Habitat-Sim environment setup
- Multi-GPU for faster training
- 1-2 weeks compute time

**Command:**
```bash
python scripts/train_rl.py --config configs/rl_config.yaml
```

**Future Work:** We provide the architecture; full training left as extension.

---

## Training Tips

### For Limited Compute (RTX 2060, 8GB VRAM)

1. **Reduce Batch Size:**
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch = 16
```

2. **Enable Gradient Checkpointing:**
```python
model.qformer.gradient_checkpointing_enable()
```

3. **Use Mixed Precision:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(video, caption)
scaler.scale(loss).backward()
```

### For Free Colab

**Problem:** 12-hour session limit

**Solution:** Checkpoint-based training
```python
# Auto-resume from last checkpoint
if Path("checkpoints/latest.pth").exists():
    checkpoint = torch.load("checkpoints/latest.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

### Monitoring Training

**Weights & Biases:**
```bash
wandb login
# Training automatically logs to W&B
```

**TensorBoard:**
```bash
tensorboard --logdir runs/
```

**Key Metrics:**
- `train/loss`: Should decrease steadily
- `train/reconstruction_loss`: V-JEPA embedding prediction quality
- `train/causality_loss`: Our novel physics-aware loss

---

## Expected Results

### V-JEPA (if training from scratch)
- **Epoch 50:** Embeddings capture object motion
- **Epoch 100:** Understands basic physics (gravity, collisions)

### VLM (Q-Former)
- **Epoch 1-2:** Random captions
- **Epoch 3-5:** Generic descriptions ("A person is moving")
- **Epoch 6-10:** Disaster-specific language ("Water level rising rapidly")

**Validation:**
```bash
python scripts/inference_vlm.py \
  --video samples/flood_test.mp4 \
  --checkpoint checkpoints/aegis_vlm_epoch_10.pth
```

---

## Fine-tuning on Custom Data

### Step 1: Collect Data

Need 500-1000 video-caption pairs for good results.

**Sources:**
- YouTube disaster videos (check copyright!)
- News footage with captions
- Simulation engines (Blender, Unity)

### Step 2: Preprocess

```bash
python data/preprocess_custom.py \
  --input_dir raw_videos/ \
  --output_dir data/custom/ \
  --create_captions  # Auto-generate with GPT-4
```

### Step 3: Train

```bash
python scripts/train_vlm.py \
  --config configs/vlm_custom.yaml \
  --data data/custom/captions.json
```

---

## Troubleshooting

### Loss Not Decreasing

**Causes:**
- Learning rate too high/low
- Frozen LLM not loading properly
- Data quality issues

**Solutions:**
```yaml
training:
  learning_rate: 0.00005  # Try lower LR
```

### CUDA Out of Memory

```bash
# Reduce batch size or video length
training:
  batch_size: 1
data:
  video_length: 8  # From 16
```

### Training Too Slow

**Use Distributed Training:**
```bash
torchrun --nproc_per_node=4 scripts/train_vlm.py --config configs/vlm_config.yaml
```

---

## Advanced: Multi-GPU Training

```python
# Modify train_vlm.py
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```

**Command:**
```bash
torchrun --nproc_per_node=4 scripts/train_vlm.py
```

---

## Hyperparameter Tuning

**Key Hyperparameters:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `learning_rate` | 1e-4 | 1e-5 to 5e-4 | Convergence speed |
| `num_query_tokens` | 32 | 16-64 | Compression ratio |
| `qformer_depth` | 6 | 4-12 | Model capacity |
| `mask_ratio` | 0.8 | 0.6-0.9 | V-JEPA difficulty |

**Recommended Sweep (W&B):**
```yaml
sweep_config:
  parameters:
    learning_rate:
      values: [1e-5, 5e-5, 1e-4, 5e-4]
    num_query_tokens:
      values: [16, 32, 48, 64]
```

---

**Happy Training! 🚀**
