"""
V-JEPA (Vision Joint Embedding Predictive Architecture) Core Implementation
Based on Meta FAIR's V-JEPA paper and codebase
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange

from .backbone import VisionTransformer
from .predictor import JEPAPredictor
from .encoder import ContextEncoder


class VJEPAModel(nn.Module):
    """
    V-JEPA: Self-supervised learning through predicting representations of masked patches

    Architecture:
        1. Context Encoder: Processes visible patches → context embeddings
        2. Target Encoder: Processes ALL patches (including masked) → target embeddings
        3. Predictor: Predicts target embeddings of masked patches given context

    Key Innovation: No pixel reconstruction - predicts in abstract embedding space
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        predictor_depth: int = 6,
        predictor_embed_dim: int = 384,
        mask_ratio: float = 0.8,
        **kwargs
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Context Encoder (processes visible patches)
        self.context_encoder = ContextEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Target Encoder (EMA of context encoder, processes all patches)
        self.target_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        
        # Initialize target encoder as copy of context encoder
        for param_c, param_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False  # No gradients for target encoder

        # Predictor (predicts target embeddings from context embeddings)
        self.predictor = JEPAPredictor(
            context_dim=embed_dim,
            target_dim=embed_dim,
            pred_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=num_heads,
        )

        # EMA momentum for target encoder
        self.momentum = 0.996

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training

        Args:
            x: Input video/image tensor [B, T, C, H, W] or [B, C, H, W]
            mask: Binary mask indicating visible patches [B, num_patches]

        Returns:
            predicted_embeddings: Predicted embeddings for masked patches
            target_embeddings: True embeddings for masked patches (from target encoder)
            mask: The mask tensor used
        """
        B = x.shape[0]
        
        # Handle video input
        if x.ndim == 5:  # [B, T, C, H, W]
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            video_mode = True
        else:
            video_mode = False

        # Generate mask if not provided
        if mask is None:
            mask = self._generate_mask(B, self.context_encoder.num_patches)

        # Context encoder: process only visible patches
        context_embeddings = self.context_encoder(x, mask)

        # Target encoder: process all patches (no gradients)
        with torch.no_grad():
            target_embeddings = self.target_encoder(x)

        # Predictor: predict masked patch embeddings from context
        predicted_embeddings = self.predictor(context_embeddings, mask)

        # Extract only embeddings of masked patches for loss calculation
        masked_target = target_embeddings[~mask]
        masked_predicted = predicted_embeddings[~mask]

        return masked_predicted, masked_target, mask

    @torch.no_grad()
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for inference (no masking)

        Args:
            x: Input tensor [B, T, C, H, W] or [B, C, H, W]

        Returns:
            embeddings: [B, T, embed_dim] or [B, embed_dim]
        """
        original_shape = x.shape
        video_mode = x.ndim == 5

        if video_mode:  # Video input
            B, T = x.shape[:2]
            x = rearrange(x, 'b t c h w -> (b t) c h w')

        # Use target encoder (stable, EMA-updated)
        embeddings = self.target_encoder(x)  # [B*T, num_patches, embed_dim]
        
        # Global average pooling over patches
        embeddings = embeddings.mean(dim=1)  # [B*T, embed_dim]

        if video_mode:
            embeddings = rearrange(embeddings, '(b t) d -> b t d', b=B, t=T)

        return embeddings

    def update_target_encoder(self):
        """Update target encoder using exponential moving average (EMA)"""
        with torch.no_grad():
            for param_c, param_t in zip(
                self.context_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                param_t.data = (
                    self.momentum * param_t.data + 
                    (1 - self.momentum) * param_c.data
                )

    def _generate_mask(self, batch_size: int, num_patches: int) -> torch.Tensor:
        """
        Generate random mask for patches

        Args:
            batch_size: Batch size
            num_patches: Total number of patches

        Returns:
            mask: Boolean tensor [B, num_patches], True = visible, False = masked
        """
        num_masked = int(num_patches * self.mask_ratio)
        
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
        for i in range(batch_size):
            # Randomly select patches to keep visible
            visible_indices = torch.randperm(num_patches)[:num_patches - num_masked]
            mask[i, visible_indices] = True
        
        return mask.to(next(self.parameters()).device)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        """Load pre-trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract config from checkpoint if available
        config = checkpoint.get('config', {})
        config.update(kwargs)
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        return model

    def save_checkpoint(self, path: str, epoch: int, optimizer=None, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': {
                'img_size': self.img_size,
                'patch_size': self.patch_size,
                'mask_ratio': self.mask_ratio,
            },
            **kwargs
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
