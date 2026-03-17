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
import os
from pathlib import Path
from tracemalloc import start
import cv2
from tqdm import tqdm


def crop_images(crop_params, video_path):
    """
    Crop a video by removing specified pixels from each side.
    """
    video_path = Path(video_path)
    print(crop_params)
    video_name = crop_params["video_name"]
    images_dir = os.path.join(video_path, "images")
    annotations_dir = os.path.join(video_path, "masks")
    
    # get video name by extracting the video path directory name
    video_name = video_path.stem
    print(f"Processing video: {video_name}")

    # check if directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    

    # loop through images and crop them
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') ])
    annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.png')])

    # double check both are there
    if len(image_files) != len(annotation_files):
        raise ValueError(f"Number of images and annotations do not match: {len(image_files)} vs {len(annotation_files)}")

    # get width and height of the first image to calculate new dimensions
    sample_image_path = os.path.join(images_dir, image_files[0])
    sample_image = cv2.imread(sample_image_path)
    height, width, _ = sample_image.shape

    # Calculate new dimensions
    left = crop_params["left"]
    right = crop_params["right"]
    top = crop_params["top"]
    bottom = crop_params["bottom"]
    new_width = width - left - right
    new_height = height - top - bottom  

    # refactor video name
    name_parts = video_name.split('.')
    prefix = crop_params["video_prefix"]
    # get name part after vieo prefix
    postfix = name_parts[0].split(prefix +"_")[-1]
    print(f"  Prefix: {prefix}, Postfix: {postfix}")

    start_frames = postfix.split('_')[0]
    end_frames = postfix.split('_')[1]
    print(f"  Start frames: {start_frames}, End frames: {end_frames}")

    sart_sec = int(start_frames) // 30
    end_sec = int(end_frames) // 30
    print(f"  Start sec: {sart_sec}, End sec: {end_sec}")

    new_video_name = f"{prefix}_{sart_sec:04}_{end_sec:04}_sec_cropped.mp4"

    # output paths
    output_path = video_path.parent / new_video_name    
    output_path.mkdir(parents=True, exist_ok=True)

    images_output_dir = os.path.join(output_path, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    annotations_output_dir = os.path.join(output_path, "masks")
    os.makedirs(annotations_output_dir, exist_ok=True)

    # copy old soft_masks and visualization folders if they exist
    import shutil
    shutil.copytree(os.path.join(video_path, "soft_masks"), os.path.join(output_path, "soft_masks"))
    shutil.copytree(os.path.join(video_path, "visualization"), os.path.join(output_path, "visualization"))

    # Use PIL for mask cropping to preserve palette
    from PIL import Image

    # Loop through images and crop them
    for img_file, ann_file in tqdm(zip(image_files, annotation_files), total=len(image_files), desc="Cropping images"):
        img_path = os.path.join(images_dir, img_file)
        ann_path = os.path.join(annotations_dir, ann_file)

        # Read image
        img = cv2.imread(img_path)
        cropped_img = img[top:height-bottom, left:width-right]
        img_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(images_output_dir, img_name), cropped_img)

        # Read annotation with PIL
        ann = Image.open(ann_path)
        # Crop annotation
        cropped_ann = ann.crop((left, top, width - right, height - bottom))
        # Preserve palette if present
        if ann.mode == 'P':
            cropped_ann.putpalette(ann.getpalette())
        ann_name = os.path.basename(ann_path)
        cropped_ann.save(os.path.join(annotations_output_dir, ann_name))

    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"Invalid crop parameters. New dimensions would be {new_width}x{new_height}")

    print(f"Processing video: {video_name}")
    print(f"  Original: {width}x{height}")
    print(f"  Cropped:  {new_width}x{new_height}")
    print(f"  Crop: L={left}, R={right}, T={top}, B={bottom}")
    print(f"  Frames: {len(image_files)}")

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
                'video_prefix': video_prefix,
                'video_name': video_name,
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
            print(crop_params)
        except Exception as e:
            print(f"\nError: {e}")
            print("Aborting: No valid crop config found for this video.")
            return 1
    else:
        print("Error: No crop config provided.")
        return 1


    # Validate parameters
    if all(v == 0 for v in crop_params.values()):
        print("Warning: All crop parameters are 0. No cropping will be performed.")

    try:
        crop_images(
            crop_params,
            args.video_path
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
