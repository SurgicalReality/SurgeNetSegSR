"""
Video Cropping Tool for Surgical Video Preprocessing

This script crops videos by removing specified pixels from each side.
Useful for removing black borders and bottom bars from surgical video footage.

Usage:
    python crop_video.py <video_path> --left <px> --right <px> --top <px> --bottom <px> [--output <path>]
    python crop_video.py <video_path> --config <crop_config.json> [--output <path>]

Examples:
    python crop_video.py video.mp4 --left 100 --right 100 --top 50 --bottom 80
    python crop_video.py video.mp4 --config crop_config.json --output cropped_video.mp4
"""

import argparse
from hmac import new
import json
import os
from pathlib import Path
from tracemalloc import start
from av import video
import cv2
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description='Crop videos by removing pixels from each side.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the input video file'
    )
    
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to crop config JSON file (overrides individual parameters)'
    )
    
    args = parser.parse_args()
    video_prefix = None
    # Load crop config
    with open(args.config, 'r') as f:
        config = json.load(f)
    video_name = Path(args.video_path).stem
    videos = config.get('videos', {})
    for key in videos:
        if key in video_name:
            video_prefix = key
            crop_params = videos[key].get('crop_params', {})

    if video_prefix is None:
        raise ValueError(f"No crop config found for video '{video_name}' in {args.config}")

    video_path = Path(args.video_path)
    video_name = video_path.stem

    # refactor video name
    name_parts = video_name.split('.')

    # get name part after vieo prefix
    postfix = name_parts[0].split(video_prefix +"_")[-1]
    print(f"  Prefix: {video_prefix}, Postfix: {postfix}")

    start_frames = postfix.split('_')[0]
    end_frames = postfix.split('_')[1]
    print(f"  Start frames: {start_frames}, End frames: {end_frames}")

    sart_sec = int(start_frames) // 30
    end_sec = int(end_frames) // 30
    print(f"  Start sec: {sart_sec}, End sec: {end_sec}")

    new_video_name = f"{video_prefix}_{sart_sec:04}_{end_sec:04}_sec_cropped.mp4"

    import shutil

    new_video_path = video_path.parent / new_video_name
    shutil.copytree(video_path, new_video_path)



if __name__ == "__main__":
    exit(main())
