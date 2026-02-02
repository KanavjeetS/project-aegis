"""
JEPA Predictor: Predicts target embeddings of masked patches from context
"""

import torch
import torch.nn as nn
from .backbone import Block


class JEPAPredictor(nn.Module):
    """
    Predictor network that predicts target encoder embeddings from context embeddings
    
    Key insight: Learns to predict abstract representations, not pixels
    """
    
    def __init__(
        self,
        context_dim=768,
        target_dim=768,
        pred_dim=384,
        depth=6,
        num_heads=12,
    ):
        super().__init__()
        
        # Project context embeddings to predictor dimension
        self.context_proj = nn.Linear(context_dim, pred_dim)
        
        # Learnable mask tokens (for masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        
        # Transformer blocks for prediction
        self.blocks = nn.ModuleList([
            Block(pred_dim, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        
        # Project to target dimension
        self.target_proj = nn.Linear(pred_dim, target_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, context_embeddings, mask):
        """
        Args:
            context_embeddings: [B, num_visible_patches, context_dim]
            mask: [B, num_patches] - True for visible, False for masked
        
        Returns:
            predictions: [B, num_patches, target_dim]
        """
        B, num_visible, _ = context_embeddings.shape
        num_patches = mask.shape[1]
        
        # Project context embeddings
        x = self.context_proj(context_embeddings)  # [B, num_visible, pred_dim]
        
        # Create full sequence with mask tokens for masked patches
        full_sequence = self.mask_token.expand(B, num_patches, -1).clone()
        
        # Fill in visible patch embeddings
        for i in range(B):
            visible_idx = torch.where(mask[i])[0]
            full_sequence[i, visible_idx] = x[i]
        
        # Pass through transformer blocks
        for blk in self.blocks:
            full_sequence = blk(full_sequence)
        
        # Project to target dimension
        predictions = self.target_proj(full_sequence)  # [B, num_patches, target_dim]
        
        return predictions
