"""
Project A.E.G.I.S. - VLM (Vision-Language Model) Implementation
Combines V-JEPA with Llama 3.1 via Q-Former
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional

from ..vjepa import VJEPAModel
from .qformer import QFormer


class AEGISModel(nn.Module):
    """
    A.E.G.I.S. Vision-Language Model
    
    Architecture:
        Video → V-JEPA → Embeddings → Q-Former → LLM → Text Description
    
    Training Strategy:
        - Freeze V-JEPA (already trained on visual understanding)
        - Freeze LLM (already trained on language)
        - Train ONLY Q-Former (32M params) to bridge modalities
    """
    
    def __init__(
        self,
        vjepa_checkpoint: str,
        llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        num_query_tokens: int = 32,
        quantization: str = "4bit",
        freeze_vjepa: bool = True,
        freeze_llm: bool = True,
    ):
        super().__init__()
        
        # 1. V-JEPA Encoder (Observer)
        self.vjepa = VJEPAModel.from_pretrained(vjepa_checkpoint)
        if freeze_vjepa:
            for param in self.vjepa.parameters():
                param.requires_grad = False
            self.vjepa.eval()
        
        # 2. Q-Former (Connector)
        self.qformer = QFormer(
            encoder_hidden_size=768,  # V-JEPA embedding dim
            num_query_tokens=num_query_tokens,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
        )
        
        # 3. LLM (Analyst)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()
        
        # Projection from Q-Former to LLM input space
        self.llm_proj = nn.Linear(768, self.llm.config.hidden_size)
    
    def forward(
        self,
        video: torch.Tensor,
        text: Optional[str] = None,
        return_loss: bool = True
    ):
        """
        Forward pass
        
        Args:
            video: [B, T, C, H, W] video tensor
            text: Text description (for training)
            return_loss: Whether to compute loss
        
        Returns:
            If return_loss: loss tensor
            Else: generated text
        """
        # Extract V-JEPA embeddings
        with torch.no_grad():
            video_embeds = self.vjepa.extract_embeddings(video)  # [B, T, 768]
        
        # Q-Former: compress video embeddings to query tokens
        query_output = self.qformer(video_embeds)  # [B, num_queries, 768]
        
        # Project to LLM input space
        inputs_llm = self.llm_proj(query_output)  # [B, num_queries, llm_hidden_size]
        
        # Prepare inputs for LLM
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=inputs_llm.device)
        
        if return_loss and text is not None:
            # Training mode
            # Tokenize text
            text_tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=512,
            ).to(inputs_llm.device)
            
            # Embed text tokens
            text_embeds = self.llm.get_input_embeddings()(text_tokens.input_ids)
            
            # Concatenate video and text embeddings
            inputs_embeds = torch.cat([inputs_llm, text_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, text_tokens.attention_mask], dim=1)
            
            # Forward through LLM
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=text_tokens.input_ids,
            )
            
            return outputs.loss
        
        else:
            # Inference mode
            outputs = self.llm.generate(
                inputs_embeds=inputs_llm,
                attention_mask=atts_llm,
                max_new_tokens=256,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
    
    @torch.no_grad()
    def predict(self, video_path: str) -> str:
        """
        Generate description for a video 
        
        Args:
            video_path: Path to video file
        
        Returns:
            description: Generated text description
        """
        from utils.video_loader import load_video
        
        # Load video
        video = load_video(video_path, device=next(self.parameters()).device)
        video = video.unsqueeze(0)  # Add batch dimension [1, T, C, H, W]
        
        # Generate description
        description = self.forward(video, return_loss=False)
        
        return description
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str):
        """Load pretrained VLM model"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        return model
