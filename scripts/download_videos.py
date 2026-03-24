import json
import os
import sys
import argparse
import yt_dlp
from tqdm import tqdm
from colorama import Fore, init

# Resolve Base Directory to find data folders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

init(autoreset=True)

def parse_arguments():
    default_json = os.path.join(BASE_DIR, 'data', 'ground_truth', 'val_1.json')
    default_out = os.path.join(BASE_DIR, 'data', 'videos')

    parser = argparse.ArgumentParser(description="Compact Video Downloader for ActivityNet")
    parser.add_argument("--json", type=str, default=default_json, help="Path to ground truth JSON")
    parser.add_argument("--videos", type=str, default=default_out, help="Output directory for videos")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of videos to download")
    return parser.parse_args()

def download_videos(json_path, output_dir, target_count):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(Fore.CYAN + f"📂 Reading: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(Fore.RED + f"❌ Error: File not found {json_path}")
        return

    print(Fore.YELLOW + f"🔍 JSON loaded ({len(data)} records).")
    print(Fore.CYAN + f"🚀 Starting download... (Target: {target_count} videos)")

    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
    }

    downloaded_count = 0
    
    # Progress bar setup
    with tqdm(total=target_count, unit="vid", desc="Downloading", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for vid_key in data.keys():
                if downloaded_count >= target_count:
                    break
                
                # Normalize ID (remove 'v_' prefix if present)
                youtube_id = vid_key[2:] if vid_key.startswith("v_") else vid_key
                save_path = os.path.join(output_dir, f"{youtube_id}.mp4")
                url = f"https://www.youtube.com/watch?v={youtube_id}"

                # Skip if already exists
                if os.path.exists(save_path):
                    downloaded_count += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"Found: {youtube_id}", refresh=False)
                    continue

                try:
                    pbar.set_postfix_str(f"Trying: {youtube_id}", refresh=True)
                    ydl.download([url])
                    
                    if os.path.exists(save_path):
                        downloaded_count += 1
                        pbar.update(1)
                except Exception:
                    # Silently fail for unavailable videos (common in ActivityNet)
                    pass

    print(Fore.GREEN + f"\n✅ Done! {downloaded_count}/{target_count} videos are available in: {output_dir}")

def main():
    args = parse_arguments()
    download_videos(args.json, args.videos, args.limit)

if __name__ == "__main__":
    main()