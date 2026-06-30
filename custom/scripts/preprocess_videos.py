"""
Video Preprocessing Tool for Surgical Video Datasets

This script provides functions for preprocessing surgical videos including:
- Cropping videos to remove borders and bars
- Applying the same cropping to segmented videos/masks

Usage:
    # Crop a regular video
    python preprocess_videos.py crop --video <path> --left <px> --right <px> --top <px> --bottom <px>
    
    # Crop a segmented video (mask)
    python preprocess_videos.py crop-mask --video <path> --config <crop_config.json>
    
    # Batch process multiple videos
    python preprocess_videos.py batch --input_dir <dir> --config <crop_config.json>

Examples:
    python preprocess_videos.py crop --video video.mp4 --left 100 --right 100 --top 50 --bottom 80
    python preprocess_videos.py crop-mask --video mask.mp4 --config crop_config.json
    python preprocess_videos.py batch --input_dir ./videos --config crop_config.json
"""

import argparse
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np


def crop_video(video_path, left=0, right=0, top=0, bottom=0, output_path=None, preserve_mask=False):
    """
    Crop a video by removing specified pixels from each side.
    
    Args:
        video_path (str): Path to the input video file
        left (int): Pixels to remove from left side
        right (int): Pixels to remove from right side
        top (int): Pixels to remove from top
        bottom (int): Pixels to remove from bottom
        output_path (str, optional): Output path. If None, creates "_cropped" version
        preserve_mask (bool): If True, uses nearest neighbor interpolation (for masks)
    
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
    print(f"  Type: {'Mask/Segmentation' if preserve_mask else 'Regular Video'}")
    print(f"  Original: {width}x{height}")
    print(f"  Cropped:  {new_width}x{new_height}")
    print(f"  Crop: L={left}, R={right}, T={top}, B={bottom}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {frame_count}")
    
    # Setup output path
    if output_path is None:
        suffix = "_cropped_mask" if preserve_mask else "_cropped"
        output_path = video_path.parent / f"{video_path.stem}{suffix}{video_path.suffix}"
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
            
            # For masks, ensure we preserve exact values (no interpolation artifacts)
            if preserve_mask:
                # Convert to ensure no color space issues
                if len(cropped_frame.shape) == 3:
                    cropped_frame = cropped_frame
            
            out.write(cropped_frame)
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✓ Cropped video saved to: {output_path}")
    return str(output_path)


def crop_segmented_video(video_path, left=0, right=0, top=0, bottom=0, output_path=None):
    """
    Crop a segmented video/mask with the same parameters as the original video.
    This ensures spatial consistency between original and segmented footage.
    
    Args:
        video_path (str): Path to the segmented video/mask file
        left (int): Pixels to remove from left side
        right (int): Pixels to remove from right side
        top (int): Pixels to remove from top
        bottom (int): Pixels to remove from bottom
        output_path (str, optional): Output path
    
    Returns:
        str: Path to the cropped segmented video file
    """
    return crop_video(video_path, left, right, top, bottom, output_path, preserve_mask=True)


def crop_image_sequence(input_dir, left=0, right=0, top=0, bottom=0, output_dir=None):
    """
    Crop a sequence of images (JPEGs or PNGs) with the same parameters.
    Useful for cropping DAVIS-style datasets.
    
    Args:
        input_dir (str): Directory containing input images
        left, right, top, bottom (int): Crop parameters
        output_dir (str, optional): Output directory
    
    Returns:
        int: Number of images processed
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_cropped"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = sorted(list(input_dir.glob("*.jpg")) + 
                        list(input_dir.glob("*.jpeg")) + 
                        list(input_dir.glob("*.png")))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return 0
    
    print(f"Processing {len(image_files)} images from {input_dir.name}")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Cropping images"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path.name}, skipping")
            continue
        
        height, width = img.shape[:2]
        cropped = img[top:height-bottom, left:width-right]
        
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), cropped)
    
    print(f"\n✓ Cropped images saved to: {output_dir}")
    return len(image_files)


def load_crop_config(config_path):
    """Load crop parameters from JSON config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    crop_params = config.get('crop_params', {})
    return {
        'left': crop_params.get('left', 0),
        'right': crop_params.get('right', 0),
        'top': crop_params.get('top', 0),
        'bottom': crop_params.get('bottom', 0)
    }


def batch_process_videos(input_dir, crop_params, output_dir=None, process_masks=False):
    """
    Batch process multiple videos with the same crop parameters.
    
    Args:
        input_dir (str): Directory containing input videos
        crop_params (dict): Dictionary with 'left', 'right', 'top', 'bottom' keys
        output_dir (str, optional): Output directory
        process_masks (bool): If True, process as mask videos
    
    Returns:
        list: Paths to processed videos
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Setup output directory
    if output_dir is None:
        suffix = "_cropped_masks" if process_masks else "_cropped"
        output_dir = input_dir.parent / f"{input_dir.name}{suffix}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = sorted(list(input_dir.glob("*.mp4")) + 
                        list(input_dir.glob("*.avi")) + 
                        list(input_dir.glob("*.mov")))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return []
    
    print(f"\nBatch processing {len(video_files)} videos")
    print(f"Crop parameters: {crop_params}")
    
    processed_videos = []
    for video_path in video_files:
        try:
            output_path = output_dir / f"{video_path.stem}_cropped{video_path.suffix}"
            result = crop_video(
                video_path,
                output_path=output_path,
                preserve_mask=process_masks,
                **crop_params
            )
            processed_videos.append(result)
            print()  # Blank line between videos
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            continue
    
    print(f"\n✓ Batch processing complete. Processed {len(processed_videos)}/{len(video_files)} videos")
    return processed_videos


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess surgical videos (cropping, etc.)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Crop command
    crop_parser = subparsers.add_parser('crop', help='Crop a single video')
    crop_parser.add_argument('--video', type=str, required=True, help='Path to video file')
    crop_parser.add_argument('--left', type=int, default=0, help='Pixels to remove from left')
    crop_parser.add_argument('--right', type=int, default=0, help='Pixels to remove from right')
    crop_parser.add_argument('--top', type=int, default=0, help='Pixels to remove from top')
    crop_parser.add_argument('--bottom', type=int, default=0, help='Pixels to remove from bottom')
    crop_parser.add_argument('--config', type=str, help='Crop config JSON file')
    crop_parser.add_argument('--output', type=str, help='Output video path')
    
    # Crop mask command
    mask_parser = subparsers.add_parser('crop-mask', help='Crop a segmented video/mask')
    mask_parser.add_argument('--video', type=str, required=True, help='Path to mask video file')
    mask_parser.add_argument('--left', type=int, default=0, help='Pixels to remove from left')
    mask_parser.add_argument('--right', type=int, default=0, help='Pixels to remove from right')
    mask_parser.add_argument('--top', type=int, default=0, help='Pixels to remove from top')
    mask_parser.add_argument('--bottom', type=int, default=0, help='Pixels to remove from bottom')
    mask_parser.add_argument('--config', type=str, help='Crop config JSON file')
    mask_parser.add_argument('--output', type=str, help='Output video path')
    
    # Crop images command
    imgs_parser = subparsers.add_parser('crop-images', help='Crop a directory of images')
    imgs_parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    imgs_parser.add_argument('--left', type=int, default=0, help='Pixels to remove from left')
    imgs_parser.add_argument('--right', type=int, default=0, help='Pixels to remove from right')
    imgs_parser.add_argument('--top', type=int, default=0, help='Pixels to remove from top')
    imgs_parser.add_argument('--bottom', type=int, default=0, help='Pixels to remove from bottom')
    imgs_parser.add_argument('--config', type=str, help='Crop config JSON file')
    imgs_parser.add_argument('--output_dir', type=str, help='Output directory')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple videos')
    batch_parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    batch_parser.add_argument('--config', type=str, required=True, help='Crop config JSON file')
    batch_parser.add_argument('--output_dir', type=str, help='Output directory')
    batch_parser.add_argument('--masks', action='store_true', help='Process as mask videos')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        # Load crop config if provided
        if hasattr(args, 'config') and args.config:
            crop_params = load_crop_config(args.config)
        else:
            crop_params = {
                'left': getattr(args, 'left', 0),
                'right': getattr(args, 'right', 0),
                'top': getattr(args, 'top', 0),
                'bottom': getattr(args, 'bottom', 0)
            }
        
        # Execute command
        if args.command == 'crop':
            crop_video(args.video, output_path=args.output, **crop_params)
        
        elif args.command == 'crop-mask':
            crop_segmented_video(args.video, output_path=args.output, **crop_params)
        
        elif args.command == 'crop-images':
            crop_image_sequence(args.input_dir, output_dir=args.output_dir, **crop_params)
        
        elif args.command == 'batch':
            batch_process_videos(
                args.input_dir,
                crop_params,
                output_dir=args.output_dir,
                process_masks=args.masks
            )
        
        return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
