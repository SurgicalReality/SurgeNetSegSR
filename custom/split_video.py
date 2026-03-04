"""
Video Splitting Tool for Surgical Video Datasets

This script splits a video file into multiple clips of specified duration.
Output files are named: [original_name]_[start_frame]_[end_frame].mp4

Usage:
    python split_video.py <video_path> <clip_duration_seconds> [--output_dir <dir>]

Example:
    python split_video.py surgery_video.mp4 5
    python split_video.py surgery_video.mp4 10 --output_dir ./custom/data/output_clips
"""

import argparse
import os
from pathlib import Path
try:
    from moviepy import VideoFileClip
except ImportError:
    print("Problem with import.")
    print("Please install it using: pip install moviepy")
    exit(1)
import json


def split_video_into_clips(video_path, clip_duration_seconds, output_dir=None, out_frame_rate=None):
    """
    Split a video into multiple clips of specified duration.
    
    Args:
        video_path (str): Path to the input video file
        clip_duration_seconds (float): Duration of each clip in seconds
        output_dir (str, optional): Output directory for clips. If None, uses video's directory
        frame_rate (float, optional): Frame rate to use for calculations. If None, uses video's native fps
    
    Returns:
        list: List of paths to the created clip files
    """
    # Load the video
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Loading video: {video_path}")
    video = VideoFileClip(str(video_path))
    
    # Get video properties
    total_duration = video.duration
    fps = out_frame_rate if out_frame_rate is not None else video.fps
    total_frames = int(total_duration * fps)
    
    print(f"Video duration: {total_duration:.2f} seconds")
    print(f"Video native frame rate: {video.fps} fps")
    print(f"Frame rate for calculations: {fps} fps")
    print(f"Total frames: {total_frames}")
    print(f"Clip duration: {clip_duration_seconds} seconds")
    
    # Setup output directory
    if output_dir is None:
        output_dir = video_path.parent / "clips"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get base name without extension
    base_name = video_path.stem
    
    # Calculate number of clips
    num_clips = int(total_duration / clip_duration_seconds)
    if total_duration % clip_duration_seconds > 0:
        num_clips += 1
    
    print(f"\nCreating {num_clips} clips...")
    
    created_files = []
    
    # Split video into clips
    for i in range(num_clips):
        start_time = i * clip_duration_seconds
        end_time = min((i + 1) * clip_duration_seconds, total_duration)
        
        # Calculate frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Create clip filename
        clip_filename = f"{base_name}_{start_frame:06d}_{end_frame:06d}.mp4"
        clip_path = output_dir / clip_filename
        
        print(f"  Clip {i+1}/{num_clips}: {clip_filename} ({start_time:.2f}s - {end_time:.2f}s)")
        
        # Extract and save clip
        clip = video.subclipped(start_time, end_time)
        clip.write_videofile(
            str(clip_path),
            codec='libx264',
            audio=False,
            remove_temp=True,
            logger=None  # Suppress moviepy progress bars for cleaner output
        )
        clip.close()
        
        created_files.append(str(clip_path))
    
    # Close the original video
    video.close()
    
    print(f"\n✓ Successfully created {len(created_files)} clips in {output_dir}")
    return created_files


def prime_crop_config(video_path, config_path="custom/crop_config.json"):
    """Prime crop_config.json with a blank entry for the video prefix."""
    from pathlib import Path
    video_prefix = Path(video_path).stem  # Use full filename without extension as prefix
    try:
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_configs = {}
    if 'videos' not in all_configs:
        all_configs['videos'] = {}
    if video_prefix not in all_configs['videos']:
        all_configs['videos'][video_prefix] = {
            "crop_params": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "original_resolution": None,
            "new_resolution": None,
            "video_prefix": video_prefix
        }
        with open(config_path, 'w') as f:
            json.dump(all_configs, f, indent=4)
        print(f"Primed crop_config.json for video prefix: {video_prefix}")
    else:
        print(f"Entry for video prefix {video_prefix} already exists in crop_config.json")


def main():
    parser = argparse.ArgumentParser(
        description='Split a video into multiple clips of specified duration.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the input video file (.mp4)'
    )
    
    parser.add_argument(
        '--clip_duration',
        type=float,
        help='Duration of each clip in seconds',
        default=30,
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for clips (default: creates "clips" subdirectory next to video)'
    )
    
    parser.add_argument(
        '--out_frame_rate',
        type=float,
        default=None,
        help='Frame rate to use for frame number calculations (default: use video\'s native fps)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.clip_duration <= 0:
        print("Error: Clip duration must be positive")
        return 1
    
    # Prime crop config for this video
    prime_crop_config(args.video_path)
    
    try:
        split_video_into_clips(
            args.video_path,
            args.clip_duration,
            args.output_dir,
            args.out_frame_rate
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
