"""
Training script for V-JEPA model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from models.vjepa import VJEPAModel
from data.dataloaders.kinetics_loader import KineticsDataset
from utils.video_loader import load_video


def compute_jepa_loss(predicted, target):
    """
    Compute JEPA loss (L2 distance in embedding space)
    
    Innovation: Add physics-aware temporal causality loss
    """
    # Standard L2 loss
    reconstruction_loss = nn.functional.mse_loss(predicted, target)
    
    # Physics-aware temporal causality loss (penalize impossible transitions)
    # This is our novel contribution
    if predicted.shape[1] > 1:  # If temporal dimension exists
        velocity = predicted[:, 1:] - predicted[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        causality_loss = torch.norm(acceleration, p=2, dim=-1).mean()
    else:
        causality_loss = 0.0
    
    total_loss = reconstruction_loss + 0.1 * causality_loss
    
    return total_loss, reconstruction_loss, causality_loss


def train_one_epoch(model, dataloader, optimizer, device, epoch, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (videos, _) in enumerate(pbar):
        videos = videos.to(device)
        
        # Forward pass
        predicted, target, mask = model(videos)
        
        # Compute loss
        loss, recon_loss, causal_loss = compute_jepa_loss(predicted, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target encoder with EMA
        model.update_target_encoder()
        
        # Logging
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'causal': f'{causal_loss:.4f}' if isinstance(causal_loss, torch.Tensor) else '0.0000'
        })
        
        if config.logging.wandb and batch_idx % config.logging.log_freq == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/reconstruction_loss': recon_loss.item(),
                'train/causality_loss': causal_loss if isinstance(causal_loss, torch.Tensor) else  0.0,
                'train/epoch': epoch,
            })
    
    return total_loss / len(dataloader)


def main(args):
    # Load config
    config = OmegaConf.load(args.config)
    
    # Initialize wandb
    if config.logging.wandb:
        wandb.init(project=config.logging.project_name, config=OmegaConf.to_container(config))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = VJEPAModel(**config.model).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Dataset and DataLoader
    # Note: This is a placeholder - you'll need to implement the actual dataset
    # For now, we'll use a dummy dataset
    print("Note: Using dummy dataset. Implement KineticsDataset for real training.")
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 16, 3, 224, 224),  # 100 videos
        torch.zeros(100)  # Dummy labels
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.optimizer.betas
    )
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint.save_freq == 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            model.save_checkpoint(
                checkpoint_dir / f"vjepa_epoch_{epoch}.pth",
                epoch=epoch,
                optimizer=optimizer
            )
            print(f"Checkpoint saved: epoch_{epoch}.pth")
    
    if config.logging.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vjepa_config.yaml")
    args = parser.parse_args()
    
    main(args)
