"""
Tests for video loading utilities
"""

import torch
import pytest
import numpy as np
from utils.video_loader import temporal_crop, spatial_crop


def test_temporal_crop():
    """Test temporal cropping"""
    # Create dummy video
    video = torch.randn(32, 3, 224, 224)  # 32 frames
    
    # Crop to 16 frames
    cropped = temporal_crop(video, 16)
    
    assert cropped.shape == (16, 3, 224, 224)


def test_temporal_crop_padding():
    """Test temporal crop with padding for short videos"""
    # Short video
    video = torch.randn(8, 3, 224, 224)
    
    # Request more frames than available
    cropped = temporal_crop(video, 16)
    
    assert cropped.shape == (16, 3, 224, 224)
    # First 8 frames should be original, rest padded


def test_spatial_crop():
    """Test spatial cropping"""
    video = torch.randn(16, 3, 256, 256)
    
    cropped = spatial_crop(video, 224)
    
    assert cropped.shape == (16, 3, 224, 224)


def test_spatial_crop_no_op():
    """Test spatial crop when size already matches"""
    video = torch.randn(16, 3, 224, 224)
    
    cropped = spatial_crop(video, 224)
    
    assert torch.equal(video, cropped)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
