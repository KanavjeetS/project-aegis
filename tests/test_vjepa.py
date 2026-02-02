"""
Unit tests for V-JEPA model
"""

import torch
import pytest
from models.vjepa import VJEPAModel


def test_vjepa_initialization():
    """Test model initializes correctly"""
    model = VJEPAModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
    )
    
    assert model is not None
    assert model.img_size == 224
    assert model.patch_size == 16


def test_vjepa_forward_pass():
    """Test forward pass works"""
    model = VJEPAModel()
    
    # Create dummy input [B, C, H, W]
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    predicted, target, mask = model(x)
    
    # Check outputs
    assert predicted.shape[-1] == 768  # embed_dim
    assert target.shape[-1] == 768
    assert mask.shape[0] == 2  # batch size


def test_vjepa_video_input():
    """Test video input [B, T, C, H, W]"""
    model = VJEPAModel()
    
    # Video input
    x = torch.randn(2, 16, 3, 224, 224)  # 2 videos, 16 frames
    
    predicted, target, mask = model(x)
    
    assert predicted is not None
    assert target is not None


def test_embedding_extraction():
    """Test embedding extraction for inference"""
    model = VJEPAModel()
    model.eval()
    
    # Single image
    x = torch.randn(1, 3, 224, 224)
    embeddings = model.extract_embeddings(x)
    
    assert embeddings.shape == (1, 768)  # [B, embed_dim]
    
    # Video
    x = torch.randn(1, 16, 3, 224, 224)
    embeddings = model.extract_embeddings(x)
    
    assert embeddings.shape == (1, 16, 768)  # [B, T, embed_dim]


def test_ema_update():
    """Test EMA update for target encoder"""
    model = VJEPAModel()
    
    # Get initial target encoder params
    initial_params = [p.clone() for p in model.target_encoder.parameters()]
    
    # Update context encoder (simulate gradient update)
    for p in model.context_encoder.parameters():
        p.data += 0.1
    
    # EMA update
    model.update_target_encoder()
    
    # Check target encoder changed
    for p_init, p_current in zip(initial_params, model.target_encoder.parameters()):
        assert not torch.equal(p_init, p_current)


def test_checkpoint_save_load():
    """Test saving and loading checkpoints"""
    model1 = VJEPAModel()
    
    # Save
    model1.save_checkpoint("test_checkpoint.pth", epoch=0)
    
    # Load
    model2 = VJEPAModel.from_pretrained("test_checkpoint.pth")
    
    # Check weights match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
    
    # Cleanup
    import os
    os.remove("test_checkpoint.pth")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
