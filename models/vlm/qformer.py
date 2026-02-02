"""
Q-Former: Querying Transformer from BLIP-2
Bridges vision and language modalities via learnable queries
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class QFormer(nn.Module):
    """
    Q-Former with cross-attention to visual embeddings
    
    Key Innovation: Uses learnable query tokens that attend to visual features,
    compressing variable-length video into fixed-length representation for LLM
    """
    
    def __init__(
        self,
        encoder_hidden_size=768, 
        num_query_tokens=32,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
    ):
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_size))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Q-Former encoder (BERT-style)
        self.qformer_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            add_cross_attention=True,  # Key: cross-attention to visual features
        )
        
        self.qformer = BertModel(self.qformer_config, add_pooling_layer=False)
        
        # Projection layer for visual features (if dimensions don't match)
        if encoder_hidden_size != hidden_size:
            self.visual_proj = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.visual_proj = nn.Identity()
    
    def forward(self, visual_embeds):
        """
        Args:
            visual_embeds: [B, num_visual_tokens, encoder_hidden_size]
                          e.g., [B, T, 768] for video
        
        Returns:
            output: [B, num_query_tokens, hidden_size]
        """
        B = visual_embeds.shape[0]
        
        # Project visual embeddings if needed
        visual_embeds = self.visual_proj(visual_embeds)
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # Create attention mask for visual embeddings
        visual_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device)
        
        # Q-Former forward
        # Query tokens attend to themselves (self-attention)
        # AND to visual embeddings (cross-attention)
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=visual_embeds,
            encoder_attention_mask=visual_atts,
            return_dict=True,
        )
        
        return query_output.last_hidden_state  # [B, num_query_tokens, hidden_size]
