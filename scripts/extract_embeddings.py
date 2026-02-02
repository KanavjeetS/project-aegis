"""
Extract embeddings from videos using V-JEPA
"""

import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from models.vjepa import VJEPAModel
from utils.video_loader import load_video


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading V-JEPA from {args.checkpoint}...")
    model = VJEPAModel.from_pretrained(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Get video files
    video_dir = Path(args.video_dir)
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    print(f"Found {len(video_files)} videos")
    
    # Extract embeddings
    all_embeddings = []
    all_filenames = []
    
    with torch.no_grad():
        for video_path in tqdm(video_files, desc="Extracting embeddings"):
            try:
                video = load_video(str(video_path), device=device)
                video = video.unsqueeze(0)  # Add batch dim
                
                embeddings = model.extract_embeddings(video)  # [1, T, embed_dim]
                embeddings = embeddings.cpu().numpy()[0]  # Remove batch dim
                
                all_embeddings.append(embeddings)
                all_filenames.append(video_path.name)
            
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
                continue
    
    # Save embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "embeddings.npy", np.array(all_embeddings, dtype=object))
    
    with open(output_dir / "filenames.txt", 'w') as f:
        f.write('\n'.join(all_filenames))
    
    print(f"\nExtracted embeddings for {len(all_embeddings)} videos")
    print(f"Saved to {output_dir}")
    print(f"Embedding shape per video: {all_embeddings[0].shape}")


if __name__ == "__main__":
    parser = argparser.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="embeddings")
    
    args = parser.parse_args()
    main(args)
