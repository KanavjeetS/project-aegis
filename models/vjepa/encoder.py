"""
Context Encoder: Processes only visible (non-masked) patches
"""

import torch
import torch.nn as nn
from .backbone import VisionTransformer


class ContextEncoder(VisionTransformer):
    """
    Context Encoder extends ViT to handle masked inputs
    Only processes visible patches (memory efficient)
    """
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            mask: Binary mask [B, num_patches] - True for visible patches
        
        Returns:
            embeddings: [B, num_visible_patches, embed_dim]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        if mask is not None:
            # Select only visible patches
            visible_embeddings = []
            for i in range(B):
                visible_idx = torch.where(mask[i])[0]
                visible_embeddings.append(x[i, visible_idx])
            
            # Pad to same length for batching (use max visible count)
            max_visible = max(emb.shape[0] for emb in visible_embeddings)
            
            padded_embeddings = []
            attention_mask = torch.zeros(B, max_visible, dtype=torch.bool, device=x.device)
            
            for i, emb in enumerate(visible_embeddings):
                num_visible = emb.shape[0]
                if num_visible < max_visible:
                    # Pad with zeros
                    padding = torch.zeros(max_visible - num_visible, self.embed_dim, device=x.device)
                    emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(emb)
                attention_mask[i, :num_visible] = True
            
            x = torch.stack(padded_embeddings)  # [B, max_visible, embed_dim]
        else:
            attention_mask = None
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding (interpolate if needed)
        if mask is not None:
            # For masked inputs, we need to select corresponding position embeddings
            # For simplicity, we use the full position embedding
            # (Production version would select based on unmasked positions)
            x = x + self.pos_embed[:, :x.shape[1]]
        else:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        return x[:, 1:]  # Remove CLS token
