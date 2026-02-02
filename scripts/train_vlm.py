import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# Optional: wandb for logging (disable if not installed)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Logging to console only.")

from omegaconf import OmegaConf
from tqdm import tqdm

from models.vlm import AEGISModel


def train_one_epoch(model, dataloader, optimizer, device, epoch, config):
    """Train Q-Former for one epoch"""
    model.train()
    
    # Freeze V-JEPA and LLM (only train Q-Former)
    model.vjepa.eval()
    model.llm.eval()
    
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (videos, captions) in enumerate(pbar):
        videos = videos.to(device)
        
        # Forward pass
        loss = model(videos, text=captions, return_loss=True)
        
        # Gradient accumulation
        loss = loss / config.training.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.qformer.parameters(),
                config.training.max_grad_norm
            )
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item() * config.training.gradient_accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if WANDB_AVAILABLE and config.logging.wandb and batch_idx % config.logging.log_freq == 0:
            wandb.log({
                'train/loss': loss.item() * config.training.gradient_accumulation_steps,
                'train/epoch': epoch,
            })
    
    return total_loss / len(dataloader)


def main(args):
    # Load config
    config = OmegaConf.load(args.config)
    
    # Initialize wandb
    if WANDB_AVAILABLE and config.logging.wandb:
        wandb.init(project=config.logging.project_name, config=OmegaConf.to_container(config))
    else:
        print("📊 Logging to console only (wandb not available)")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = AEGISModel(
        vjepa_checkpoint=config.model.vjepa_checkpoint,
        llm_model_name=config.model.llm.model_name,
        num_query_tokens=config.model.qformer.num_query_tokens,
        quantization=config.model.llm.quantization,
    )
    print(f"VLM initialized")
    print(f"Trainable parameters (Q-Former only): {sum(p.numel() for p in model.qformer.parameters() if p.requires_grad)/1e6:.2f}M")
    
    # Dataset
    # Placeholder - implement video-caption dataset
    print("Note: Using dummy dataset. Implement video-caption dataset for real training.")
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(50, 16, 3, 224, 224),  # Videos
        ["A disaster scene with water flooding."] * 50  # Captions (dummy)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    
    # Optimizer (only for Q-Former parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint.save_freq == 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'vjepa_checkpoint': config.model.vjepa_checkpoint,
                    'llm_model_name': config.model.llm.model_name,
                    'num_query_tokens': config.model.qformer.num_query_tokens,
                }
            }, checkpoint_dir / f"aegis_vlm_epoch_{epoch}.pth")
            print(f"Checkpoint saved: aegis_vlm_epoch_{epoch}.pth")
    
    if WANDB_AVAILABLE and config.logging.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vlm_config.yaml")
    args = parser.parse_args()
    
    main(args)
