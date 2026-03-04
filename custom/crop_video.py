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
import json
from pathlib import Path
import cv2
from tqdm import tqdm


def crop_video(video_path, left=0, right=0, top=0, bottom=0, output_path=None):
    """
    Crop a video by removing specified pixels from each side.
    
    Args:
        video_path (str): Path to the input video file
        left (int): Pixels to remove from left side
        right (int): Pixels to remove from right side
        top (int): Pixels to remove from top
        bottom (int): Pixels to remove from bottom
        output_path (str, optional): Output path. If None, creates "_cropped" version
    
    Returns:
        str: Path to the cropped video file
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate new dimensions
    new_width = width - left - right
    new_height = height - top - bottom
    
    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"Invalid crop parameters. New dimensions would be {new_width}x{new_height}")
    
    print(f"Processing video: {video_path.name}")
    print(f"  Original: {width}x{height}")
    print(f"  Cropped:  {new_width}x{new_height}")
    print(f"  Crop: L={left}, R={right}, T={top}, B={bottom}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {frame_count}")
    
    # Setup output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_cropped{video_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))
    
    # Process frames
    print(f"\nCropping video...")
    with tqdm(total=frame_count, unit='frames') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame
            cropped_frame = frame[top:height-bottom, left:width-right]
            out.write(cropped_frame)
            
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✓ Cropped video saved to: {output_path}")
    return str(output_path)


def load_crop_config(config_path):
    """Load crop parameters from JSON config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    video_name = Path(load_crop_config.video_path).stem
    videos = config.get('videos', {})
    for key in videos:
        if key in video_name:
            video_prefix = key
            crop_params = videos[key].get('crop_params', {})
            return {
                'left': crop_params.get('left', 0),
                'right': crop_params.get('right', 0),
                'top': crop_params.get('top', 0),
                'bottom': crop_params.get('bottom', 0)
            }
    raise ValueError(f"No crop config found for video prefix '{video_prefix}' in {config_path}")


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
        '--left',
        type=int,
        default=0,
        help='Pixels to remove from left side'
    )
    
    parser.add_argument(
        '--right',
        type=int,
        default=0,
        help='Pixels to remove from right side'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=0,
        help='Pixels to remove from top'
    )
    
    parser.add_argument(
        '--bottom',
        type=int,
        default=0,
        help='Pixels to remove from bottom'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to crop config JSON file (overrides individual parameters)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path (default: adds "_cropped" to input filename)'
    )
    
    args = parser.parse_args()
    
    # Load crop parameters
    if args.config:
        print(f"Loading crop parameters from: {args.config}")
        load_crop_config.video_path = args.video_path
        try:
            crop_params = load_crop_config(args.config)
        except Exception as e:
            print(f"\nError: {e}")
            print("Aborting: No valid crop config found for this video.")
            return 1
    else:
        crop_params = {
            'left': args.left,
            'right': args.right,
            'top': args.top,
            'bottom': args.bottom
        }

    # Validate parameters
    if all(v == 0 for v in crop_params.values()):
        print("Warning: All crop parameters are 0. No cropping will be performed.")

    try:
        crop_video(
            args.video_path,
            output_path=args.output,
            **crop_params
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
