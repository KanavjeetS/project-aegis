"""
Inference script for A.E.G.I.S. VLM
Generate disaster descriptions from video
"""

import torch
import argparse
from pathlib import Path

from models.vlm import AEGISModel
from utils.video_loader import load_video


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = AEGISModel.from_pretrained(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load video
    print(f"Loading video: {args.video}")
    video = load_video(args.video, device=device)
    video = video.unsqueeze(0)  # Add batch dimension
    
    # Generate description
    print("Generating description...")
    with torch.no_grad():
        description = model.forward(video, return_loss=False)
    
    # Output
    print("\n" + "="*60)
    print("DISASTER ANALYSIS:")
    print("="*60)
    print(description)
    print("="*60 + "\n")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Video: {args.video}\n")
            f.write(f"Description: {description}\n")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A.E.G.I.S. Video-to-Text Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output file for description")
    
    args = parser.parse_args()
    
    main(args)
