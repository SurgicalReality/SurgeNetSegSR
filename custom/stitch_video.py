"""
Stitch images and mask overlays back into a video.

Takes a folder from the workspace directory and combines the original images
with colored mask overlays at a configurable transparency level.

Usage:
    python stitch_video.py <folder_name> --alpha 0.5 --fps 30 --output output.mp4
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def import_palette(palette_path):
    """Import color palette from the palette.py module."""
    spec = importlib.util.spec_from_file_location("palette", palette_path)
    palette_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(palette_module)
    return palette_module.color_palette


def colorize_mask(mask_path, color_palette):
    """
    Convert an indexed mask to an RGB image using the color palette.
    
    Args:
        mask_path: Path to the indexed PNG mask
        color_palette: Dictionary mapping index to (R, G, B) tuples
    
    Returns:
        RGB numpy array of the colorized mask
    """
    mask = np.array(Image.open(mask_path))
    h, w = mask.shape
    
    # Create RGB output
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply colors for each unique index
    unique_indices = np.unique(mask)
    for idx in unique_indices:
        if idx == 0:
            continue  # Skip background (index 0)
        if idx in color_palette:
            color = color_palette[idx]
            rgb_mask[mask == idx] = color
    
    return rgb_mask


def blend_image_and_mask(image, mask_rgb, alpha):
    """
    Blend the original image with the colored mask overlay.
    
    Args:
        image: Original RGB image (numpy array)
        mask_rgb: Colorized mask (numpy array)
        alpha: Transparency of the mask overlay (0.0 = invisible, 1.0 = opaque)
    
    Returns:
        Blended image as numpy array
    """
    # Create a mask of where there are non-zero values (actual mask content)
    mask_presence = np.any(mask_rgb > 0, axis=2)
    
    # Blend only where mask is present
    blended = image.copy().astype(np.float32)
    mask_rgb_float = mask_rgb.astype(np.float32)
    
    blended[mask_presence] = (
        (1 - alpha) * blended[mask_presence] + alpha * mask_rgb_float[mask_presence]
    )
    
    return blended.astype(np.uint8)


def process_frame(args):
    """
    Process a single frame: read image, apply mask overlay, return blended result.
    
    Args:
        args: Tuple of (frame_idx, image_file, masks_dir, color_palette, alpha)
    
    Returns:
        Tuple of (frame_idx, blended_bgr_image)
    """
    frame_idx, image_file, masks_dir, color_palette, alpha = args
    
    # Read image (BGR format)
    image_bgr = cv2.imread(str(image_file))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Find corresponding mask
    mask_file = masks_dir / (image_file.stem + ".png")
    
    if mask_file.exists():
        # Colorize mask and blend
        mask_rgb = colorize_mask(mask_file, color_palette)
        blended_rgb = blend_image_and_mask(image_rgb, mask_rgb, alpha)
    else:
        # No mask, use original image
        blended_rgb = image_rgb
    
    # Convert back to BGR for OpenCV
    blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)
    
    return frame_idx, blended_bgr


def stitch_video(folder_path, output_path, alpha, fps, color_palette, num_workers=8):
    """
    Create video from images with mask overlays.
    
    Args:
        folder_path: Path to the workspace folder containing images/ and masks/
        output_path: Path for the output video file
        alpha: Mask overlay transparency (0.0-1.0)
        fps: Frames per second for output video
        color_palette: Dictionary mapping mask indices to RGB colors
        num_workers: Number of worker threads for parallel processing
    """
    images_dir = folder_path / "images"
    masks_dir = folder_path / "masks"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Get sorted list of image files
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    height, width = first_image.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Creating video: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Alpha: {alpha}")
    print(f"  Frames: {len(image_files)}")
    print(f"  Workers: {num_workers}")
    
    # Prepare arguments for parallel processing
    task_args = [
        (idx, image_file, masks_dir, color_palette, alpha)
        for idx, image_file in enumerate(image_files)
    ]
    
    # Process frames in parallel and collect results
    processed_frames = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_frame, args): args[0] for args in task_args}
        
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Processing frames"):
            frame_idx, blended_bgr = future.result()
            processed_frames[frame_idx] = blended_bgr
    
    # Write frames to video in order
    print("Writing video...")
    for idx in tqdm(range(len(image_files)), desc="Writing frames"):
        video_writer.write(processed_frames[idx])
        # Free memory as we go
        del processed_frames[idx]
    
    video_writer.release()
    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch images and mask overlays into a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stitch_video.py s8-s10_LLL_0570_0600_sec_cropped.mp4 --alpha 0.5
    python stitch_video.py RLL_S9_0060_0090_sec_cropped.mp4 --alpha 0.3 --fps 30
    python stitch_video.py my_video.mp4 --alpha 0.7 --output custom_output.mp4
    python stitch_video.py my_video.mp4 --alpha 0.5 --workers 16
        """
    )
    
    parser.add_argument(
        "folder_name",
        type=str,
        help="Name of the folder in the workspace directory"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency of mask overlay (0.0 = invisible, 1.0 = opaque). Default: 0.5"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for output video. Default: 30"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path. Default: <folder_name>_overlay.mp4 in workspace folder"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Path to workspace directory. Default: ../workspace (relative to this script)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing. Default: 8"
    )
    
    args = parser.parse_args()
    
    # Validate alpha
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("Alpha must be between 0.0 and 1.0")
    
    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    
    if args.workspace:
        workspace_dir = Path(args.workspace).resolve()
    else:
        workspace_dir = (script_dir.parent / "workspace").resolve()
    
    folder_path = workspace_dir / args.folder_name
    
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        print(f"\nAvailable folders in {workspace_dir}:")
        if workspace_dir.exists():
            for f in sorted(workspace_dir.iterdir()):
                if f.is_dir():
                    print(f"  - {f.name}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        # Remove .mp4 suffix if present in folder name, then add _overlay.mp4
        base_name = args.folder_name
        if base_name.endswith('.mp4'):
            base_name = base_name[:-4]
        output_path = workspace_dir / f"{base_name}_overlay_{args.alpha}.mp4"
    
    # Import color palette
    palette_path = script_dir.parent / "gui" / "cutie" / "utils" / "palette.py"
    if not palette_path.exists():
        print(f"Error: Palette file not found: {palette_path}")
        sys.exit(1)
    
    color_palette = import_palette(str(palette_path))
    
    # Create video
    stitch_video(folder_path, output_path, args.alpha, args.fps, color_palette, args.workers)


if __name__ == "__main__":
    main()
