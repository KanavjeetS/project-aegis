"""
Utility functions for video processing
"""

import torch
import decord
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import cv2


decord.bridge.set_bridge('torch')


def load_video(
    video_path: Union[str, Path],
    num_frames: int = 16,
    sample_rate: int = 2,
    target_size: Tuple[int, int] = (224, 224),
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load video and sample frames
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        sample_rate: Frame sampling rate
        target_size: Target spatial resolution (H, W)
        device: Device to load tensor to
    
    Returns:
        video_tensor: [T, C, H, W] tensor
    """
    # Load video with decord (GPU-accelerated)
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    
    # Calculate frame indices
    total_frames = len(vr)
    frame_indices = list(range(0, total_frames, sample_rate))[:num_frames]
    
    # Pad if not enough frames
    if len(frame_indices) < num_frames:
        frame_indices += [frame_indices[-1]] * (num_frames - len(frame_indices))
    
    # Load frames
    frames = vr.get_batch(frame_indices).asnumpy()  # [T, H, W, C]
    
    # Resize frames
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        resized_frames.append(resized)
    
    frames = np.stack(resized_frames, axis=0)  # [T, H, W, C]
    
    # Convert to tensor and normalize
    frames = torch.from_numpy(frames).float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    
    return frames.to(device)


def save_video_frames(
    frames: torch.Tensor,
    output_path: Union[str, Path],
    fps: int = 8
):
    """
    Save video frames to file
    
    Args:
        frames: [T, C, H, W] tensor
        output_path: Output path
        fps: Frames per second
    """
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = frames * std + mean
    frames = (frames * 255).clamp(0, 255).byte()
    
    # Convert to numpy
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
    
    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    writer.release()


def temporal_crop(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Randomly crop temporal dimension
    
    Args:
        video: [T, C, H, W]
        num_frames: Target number of frames
    
    Returns:
        cropped: [num_frames, C, H, W]
    """
    T = video.shape[0]
    
    if T <= num_frames:
        # Pad if too short
        padding = torch.zeros(num_frames - T, *video.shape[1:], device=video.device)
        return torch.cat([video, padding], dim=0)
    
    # Random temporal crop
    start_idx = torch.randint(0, T - num_frames + 1, (1,)).item()
    return video[start_idx:start_idx + num_frames]


def spatial_crop(video: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Random spatial crop
    
    Args:
        video: [T, C, H, W]
        crop_size: Size of crop
    
    Returns:
        cropped: [T, C, crop_size, crop_size]
    """
    T, C, H, W = video.shape
    
    if H == crop_size and W == crop_size:
        return video
    
    # Random crop
    top = torch.randint(0, H - crop_size + 1, (1,)).item()
    left = torch.randint(0, W - crop_size + 1, (1,)).item()
    
    return video[:, :, top:top+crop_size, left:left+crop_size]
