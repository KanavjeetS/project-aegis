"""
Synthetic Disaster Dataset Generator
Creates realistic disaster scenarios with captions for demo/testing
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import random
import json


class DisasterScenarioGenerator:
    """Generate synthetic disaster videos and captions"""
    
    DISASTER_TYPES = [
        "flood", "wildfire", "earthquake", "tsunami", 
        "hurricane", "tornado", "landslide", "avalanche"
    ]
    
    SEVERITY_LEVELS = ["minor", "moderate", "severe", "catastrophic"]
    
    CAPTION_TEMPLATES = {
        "flood": [
            "Rising water levels threatening residential area, estimated depth {depth}m",
            "Flood waters advancing at {speed} km/h, immediate evacuation recommended",
            "Urban flooding detected, {buildings} structures at risk"
        ],
        "wildfire": [
            "Active wildfire spreading {direction}, wind speed {wind} km/h",
            "Forest fire detected, estimated area {area} hectares",
            "Smoke plume visible, fire intensity {severity}"
        ],
        "earthquake": [
            "Seismic activity detected, magnitude {magnitude}, depth {depth}km",
            "Structural damage observed, {buildings} buildings affected",
            "Ground displacement visible, aftershocks expected"
        ],
        "tsunami": [
            "Tsunami wave approaching coastline, estimated height {height}m",
            "Rapid coastal water recession detected, tsunami imminent",
            "Ocean surge detected, coastal evacuation critical"
        ],
        "hurricane": [
            "Hurricane force winds {speed} km/h, Category {category}",
            "Storm surge expected {height}m, coastal flooding likely",
            "Tropical cyclone approaching, landfall in {hours} hours"
        ],
        "tornado": [
            "Tornado funnel cloud detected, moving {direction}",
            "Debris field visible, EF{rating} classification estimated",
            "Rotating wall cloud, tornado formation imminent"
        ]
    }
    
    def __init__(self, output_dir: str = "data/synthetic", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_flood_frame(self, severity: str, time_step: int) -> np.ndarray:
        """Generate a single flood scenario frame"""
        # Create base image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 135  # Gray background
        
        # Water level based on severity
        water_levels = {"minor": 50, "moderate": 100, "severe": 150, "catastrophic": 200}
        water_y = 224 - water_levels[severity] - (time_step * 2)  # Rising water
        
        # Draw water
        img[max(0, water_y):, :] = [65, 105, 225]  # Blue water
        
        # Add ripples
        for i in range(0, 224, 20):
            ripple_offset = int(5 * np.sin(time_step * 0.1 + i * 0.1))
            cv2.line(img, (0, water_y + i + ripple_offset), 
                    (224, water_y + i + ripple_offset), (85, 125, 255), 1)
        
        # Add buildings (darker as they flood)
        for x in [40, 100, 160]:
            building_top = water_y - 60
            submerged = max(0, water_y - building_top)
            darkness = min(100, submerged)
            cv2.rectangle(img, (x, building_top), (x+30, 224), 
                         (100-darkness, 100-darkness, 100-darkness), -1)
        
        return img
    
    def generate_wildfire_frame(self, severity: str, time_step: int) -> np.ndarray:
        """Generate wildfire scenario frame"""
        img = np.ones((224, 224, 3), dtype=np.uint8)
        img[:, :] = [34, 139, 34]  # Green forest background
        
        # Fire spread based on severity
        fire_spread = {"minor": 30, "moderate": 60, "severe": 120, "catastrophic": 180}
        fire_size = min(224, fire_spread[severity] + time_step * 3)
        
        # Draw fire gradient
        for y in range(224):
            for x in range(fire_size):
                if random.random() > 0.3:
                    # Fire colors
                    r = 255
                    g = max(0, 255 - int(y * 1.5))
                    b = 0
                    img[y, x] = [b, g, r]
        
        # Add smoke
        smoke_overlay = np.zeros_like(img)
        cv2.circle(smoke_overlay, (fire_size-20, 50), 60, (128, 128, 128), -1)
        img = cv2.addWeighted(img, 0.7, smoke_overlay, 0.3, 0)
        
        return img
    
    def generate_earthquake_frame(self, severity: str, time_step: int) -> np.ndarray:
        """Generate earthquake scenario frame"""
        img = np.ones((224, 224, 3), dtype=np.uint8) * 100
        
        # Shake intensity
        shake_intensity = {"minor": 2, "moderate": 5, "severe": 10, "catastrophic": 15}
        shake = shake_intensity[severity]
        
        # Buildings with damage
        for x in [40, 100, 160]:
            height = random.randint(80, 120)
            tilt = int(shake * np.sin(time_step * 0.5))
            
            # Draw tilted building
            pts = np.array([
                [x+tilt, 224-height],
                [x+30+tilt, 224-height],
                [x+30, 224],
                [x, 224]
            ], np.int32)
            cv2.fillPoly(img, [pts], (80, 80, 80))
            
            # Cracks
            if severity in ["severe", "catastrophic"]:
                cv2.line(img, (x+15, 224-height), (x+15+tilt, 224), (30, 30, 30), 2)
        
        # Ground shake blur
        if time_step % 5 < 2:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        
        return img
    
    def generate_caption(self, disaster_type: str, severity: str, metadata: Dict) -> str:
        """Generate realistic caption for disaster scenario"""
        template = random.choice(self.CAPTION_TEMPLATES[disaster_type])
        
        # Fill template with realistic values
        params = {
            "depth": f"{random.uniform(1.5, 8.0):.1f}",
            "speed": random.randint(5, 50),
            "buildings": random.randint(10, 500),
            "direction": random.choice(["northeast", "southwest", "northwest", "southeast"]),
            "wind": random.randint(40, 200),
            "area": random.randint(50, 5000),
            "severity": severity,
            "magnitude": f"{random.uniform(4.0, 8.5):.1f}",
            "height": f"{random.uniform(2.0, 15.0):.1f}",
            "category": random.randint(1, 5),
            "hours": random.randint(2, 24),
            "rating": random.randint(0, 5),
        }
        
        caption = template.format(**{k: v for k, v in params.items() if f"{{{k}}}" in template})
        return caption
    
    def generate_video(
        self, 
        disaster_type: str, 
        severity: str, 
        num_frames: int = 16
    ) -> Tuple[np.ndarray, str, Dict]:
        """Generate complete disaster video with caption"""
        
        frames = []
        generator_map = {
            "flood": self.generate_flood_frame,
            "wildfire": self.generate_wildfire_frame,
            "earthquake": self.generate_earthquake_frame,
        }
        
        # Use available generator or default
        generator = generator_map.get(disaster_type, self.generate_flood_frame)
        
        for t in range(num_frames):
            frame = generator(severity, t)
            frames.append(frame)
        
        video = np.stack(frames)  # [T, H, W, C]
        
        metadata = {
            "disaster_type": disaster_type,
            "severity": severity,
            "num_frames": num_frames,
            "resolution": "224x224"
        }
        
        caption = self.generate_caption(disaster_type, severity, metadata)
        
        return video, caption, metadata
    
    def generate_dataset(self, num_samples: int = 100) -> List[Dict]:
        """Generate full synthetic dataset"""
        dataset = []
        
        for i in range(num_samples):
            disaster_type = random.choice(self.DISASTER_TYPES[:3])  # Use implemented types
            severity = random.choice(self.SEVERITY_LEVELS)
            
            video, caption, metadata = self.generate_video(disaster_type, severity)
            
            # Save video as numpy array
            video_path = self.output_dir / f"video_{i:04d}.npy"
            np.save(video_path, video)
            
            # Create dataset entry
            entry = {
                "video_id": f"synthetic_{i:04d}",
                "video_path": str(video_path),
                "caption": caption,
                "disaster_type": disaster_type,
                "severity": severity,
                **metadata
            }
            dataset.append(entry)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i+1}/{num_samples} samples")
        
        # Save metadata
        metadata_path = self.output_dir / "dataset.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Dataset saved to {self.output_dir}")
        print(f"  - {num_samples} video samples")
        print(f"  - Metadata: {metadata_path}")
        
        return dataset


if __name__ == "__main__":
    generator = DisasterScenarioGenerator()
    dataset = generator.generate_dataset(num_samples=50)
    
    print("\nSample entries:")
    for entry in dataset[:3]:
        print(f"  {entry['video_id']}: {entry['caption']}")
