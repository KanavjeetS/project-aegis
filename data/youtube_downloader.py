"""
YouTube Disaster Video Downloader
Downloads disaster-related videos for training dataset
"""

import yt_dlp
from pathlib import Path
from typing import List, Dict
import json
import time


class DisasterVideoDownloader:
    """Download disaster videos from YouTube"""
    
    DISASTER_QUERIES = [
        "flood disaster caught on camera",
        "wildfire footage",
        "earthquake damage",
        "tsunami waves",
        "hurricane landfall",
        "tornado footage",
        "landslide caught on camera",
        "volcanic eruption"
    ]
    
    def __init__(self, output_dir: str = "data/youtube"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ydl_opts = {
            'format': 'best[height<=480]',  # Lower quality for space
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
    
    def search_videos(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search YouTube for disaster videos"""
        
        search_opts = {
            **self.ydl_opts,
            'extract_flat': 'in_playlist',
            'playlistend': max_results,
        }
        
        with yt_dlp.YoutubeDL(search_opts) as ydl:
            try:
                search_url = f"ytsearch{max_results}:{query}"
                info = ydl.extract_info(search_url, download=False)
                
                videos = []
                if 'entries' in info:
                    for entry in info['entries']:
                        videos.append({
                            'id': entry.get('id'),
                            'title': entry.get('title'),
                            'duration': entry.get('duration'),
                            'url': entry.get('webpage_url'),
                            'description': entry.get('description', '')
                        })
                
                return videos
                
            except Exception as e:
                print(f"❌ Search failed for '{query}': {e}")
                return []
    
    def download_video(self, video_info: Dict) -> str:
        """Download a single video"""
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([video_info['url']])
                
                # Find downloaded file
                video_id = video_info['id']
                video_files = list(self.output_dir.glob(f"{video_id}.*"))
                
                if video_files:
                    return str(video_files[0])
                else:
                    return None
                    
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def build_dataset(self, videos_per_query: int = 5) -> List[Dict]:
        """Build complete disaster dataset"""
        
        dataset = []
        
        for query in self.DISASTER_QUERIES:
            print(f"\n🔍 Searching: {query}")
            
            videos = self.search_videos(query, max_results=videos_per_query)
            print(f"  Found {len(videos)} videos")
            
            for i, video in enumerate(videos):
                print(f"  Downloading {i+1}/{len(videos)}: {video['title'][:50]}...")
                
                video_path = self.download_video(video)
                
                if video_path:
                    dataset.append({
                        'video_id': video['id'],
                        'video_path': video_path,
                        'title': video['title'],
                        'description': video['description'],
                        'duration': video['duration'],
                        'query': query,
                        'source': 'youtube'
                    })
                    
                    print(f"    ✅ Saved to {video_path}")
                
                time.sleep(1)  # Rate limiting
        
        # Save metadata
        metadata_path = self.output_dir / "dataset.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✅ Downloaded {len(dataset)} videos")
        print(f"   Metadata: {metadata_path}")
        
        return dataset


if __name__ == "__main__":
    downloader = DisasterVideoDownloader()
    dataset = downloader.build_dataset(videos_per_query=3)  # Start small
