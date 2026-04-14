# This script converts the output from the SurgNetSeg workspace into the DAVIS style dataset that SAM2 expects. It assumes that the output from SurgNetSeg is in the format of clips created by split_video.py, and that the corresponding masks are named with a "_mask" suffix before the file extension.
"""
Input: path to workspace, video_name


Output: 
JPEGImages/
  video1/
    00000.jpg
    00001.jpg
    00002.jpg
	video2/
		...
Annotations/
	video1/
		00000.png
    00001.png
    00002.png
ImageSets/
	Set1/
		train.txt # this contains the video ids used for training
		val.txt # opt.?
Usage:
    python convert_to_DAVIS_dataset.py --workspace_path <path_to_workspace> --out_path <output_davis_path>

Video names are automatically extracted from subdirectory names in the workspace_path.
"""

import os
import re
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from tqdm import tqdm

# =============================================================================
# LABEL CONFIGURATION
# =============================================================================
# Set which labels to KEEP (True) or REMOVE (False)
# Labels set to True will be remapped to consecutive values starting from 1
# Labels set to False will be set to 0 (background)
LABEL_CONFIG = {
    1: True,   # Surgical Instruments
    2: True,   # Vein (major)
    3: True,   # Artery (major)
    4: True,   # Right Superior (Upper) Lobe
    5: True,   # Right Middle Lobe
    6: True,   # Right Inferior (Lower) Lobe
    7: True,   # Left Superior (Upper) Lobe
    8: True,   # Left Inferior (Lower) Lobe
    9: True,   # Bronchus
    10: False, # Right Horizontal Fissure
    11: False, # Right Oblique Fissure
    12: False, # Left Oblique Fissure
    13: False, # Phrenic Nerve
    14: True,  # Aorta
    15: False, # Esophagus
    16: False, # Lymph Nodes
    17: True,  # Cotton Swab
}

# Build the remapping dictionary automatically from LABEL_CONFIG
# Maps original label -> new consecutive label (or 0 if disabled)
def build_label_remap():
    """Build label remapping from config. Enabled labels get consecutive IDs starting from 1."""
    remap = {}
    new_id = 1
    for orig_id in sorted(LABEL_CONFIG.keys()):
        if LABEL_CONFIG[orig_id]:
            remap[orig_id] = new_id
            new_id += 1
        else:
            remap[orig_id] = 0
    return remap

LABEL_REMAP = build_label_remap()

def remap_mask_labels(mask_path, output_path):
    """Load a mask, remap labels according to LABEL_REMAP, and save."""
    # Load mask as grayscale
    mask = np.array(Image.open(mask_path))
    
    # Create output mask initialized to 0 (background)
    remapped_mask = np.zeros_like(mask)
    
    # Apply remapping
    for orig_label, new_label in LABEL_REMAP.items():
        if new_label > 0:  # Only remap if the label is enabled
            remapped_mask[mask == orig_label] = new_label
    
    # Any labels not in LABEL_REMAP are already 0 (background)
    
    # Save the remapped mask
    Image.fromarray(remapped_mask.astype(np.uint8)).save(output_path)


def copy_image_task(src_path, dst_path):
    """Task for copying a single image file."""
    shutil.copy(src_path, dst_path)
    return 1


def remap_mask_task(mask_path, output_path):
    """Task for remapping and saving a single mask file."""
    remap_mask_labels(mask_path, output_path)
    return 1


def main(args):
    workspace_path = args.workspace_path
    out_path = args.out_path
    count = 0

    # Dynamically extract video names from subdirectories in workspace_path
    video_names = [
        d for d in os.listdir(workspace_path)
        if os.path.isdir(os.path.join(workspace_path, d))
    ]
    
    if not video_names:
        print(f"No subdirectories found in workspace path: {workspace_path}")
        return
    
    print(f"Found {len(video_names)} video folders: {video_names}")

    # Create output path if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    
    # For logging
    video_frame_counts = {}
    total_frames = 0
    start_time = datetime.now()

    # Use thread pool for parallel processing
    num_workers = args.threads if args.threads else min(8, os.cpu_count() or 4)
    print(f"Using {num_workers} worker threads")
    
    for video_name in tqdm(video_names, desc="Processing videos", unit="video"):
        if args.snippet_length > 0:
            snippet_names = []
        video_folder = os.path.join(workspace_path, video_name)

        if not os.path.exists(video_folder):
            tqdm.write(f"Folder for video '{video_name}' not found in workspace. Skipping.")
            continue
        
        # Track frames for this video
        video_frames = 0
        
        # images folder -> JPEGImages/video_name
        image_folder = os.path.join(video_folder, "images")
        if args.snippet_length > 0:
            snippet_name = f"{video_name.replace('.', '_')}_frames_0-{args.snippet_length}"
            snippet_names.append(snippet_name)
            images_out_folder = os.path.join(out_path, "JPEGImages", snippet_name)
            
        else:
            images_out_folder = os.path.join(out_path, "JPEGImages", video_name.replace(".", "_"))
        os.makedirs(images_out_folder, exist_ok=True)

        # Prepare image copy tasks
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        # sort files by frame number (assuming filename format is something like "00000.jpg", "00001.jpg", etc.)
        image_files.sort(key=lambda x: int(x.rsplit(".", 1)[0]))

        image_tasks = []
        snippet_frame_counter = 0
        for filename in image_files:
            file_idx = int(filename.rsplit(".", 1)[0]) # get file which contains the current frame_number
            # if snippet length is met, create new snippet (reset counter) - this will create multiple snippets for a video if snippet_length is set
            if args.snippet_length > 0 and snippet_frame_counter == args.snippet_length:
                snippet_frame_counter = 0
                snippet_name = f"{video_name.replace('.', '_')}_frames_{file_idx}-{file_idx + args.snippet_length}"
                snippet_names.append(snippet_name)
                images_out_folder = os.path.join(out_path, "JPEGImages", snippet_name)
                os.makedirs(images_out_folder, exist_ok=True)
            file_path = os.path.join(image_folder, filename)
            if (args.snippet_length > 0):
                new_filename = f"{(file_idx % args.snippet_length):05d}.jpg"
            else:
                new_filename = f"{file_idx:05d}.jpg"
            dst_path = os.path.join(images_out_folder, new_filename)
            image_tasks.append((file_path, dst_path))
            snippet_frame_counter += 1
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(copy_image_task, src, dst) for src, dst in image_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc=f"  Copying images ({video_name})", unit="img", leave=False):
                count += future.result()
                video_frames += 1
        
        # Store frame count for this video
        video_frame_counts[video_name.replace(".", "_")] = video_frames
        total_frames += video_frames
        
        # masks folder -> Annotations/video_name
        mask_folder = os.path.join(video_folder, "masks")
        if args.snippet_length > 0:
            masks_out_folder = os.path.join(out_path, "Annotations", video_name.replace(".", "_") + f"_frames_0-{args.snippet_length}")
        else:
            masks_out_folder = os.path.join(out_path, "Annotations", video_name.replace(".", "_"))
        os.makedirs(masks_out_folder, exist_ok=True)

        # Prepare mask remap tasks
        mask_files = [f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))]
        mask_files.sort(key=lambda x: int(x.rsplit(".", 1)[0]))
        mask_tasks = []
        mask_snippet_frame_counter = 0
        for filename in mask_files:
            mask_idx = int(filename.rsplit(".", 1)[0])
            # if snippet length is met, create new snippet (reset counter) - this will create multiple snippets for a video if snippet_length is set
            if args.snippet_length > 0 and mask_snippet_frame_counter == args.snippet_length:
                mask_snippet_frame_counter = 0
                masks_out_folder = os.path.join(out_path, "Annotations", video_name.replace(".", "_") + f"_frames_{mask_idx}-{mask_idx + args.snippet_length}")
                os.makedirs(masks_out_folder, exist_ok=True)
            file_path = os.path.join(mask_folder, filename)
            if (args.snippet_length > 0):
                new_filename = f"{(mask_idx % args.snippet_length):05d}.jpg"
            else:
                new_filename = f"{mask_idx:05d}.jpg"
            
            output_path = os.path.join(masks_out_folder, new_filename)
            mask_tasks.append((file_path, output_path))
            mask_snippet_frame_counter += 1
        
        # Process masks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(remap_mask_task, src, dst) for src, dst in mask_tasks]
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"  Remapping masks ({video_name})", unit="mask", leave=False):
                count += future.result()

    train_txt_path = os.path.join(out_path, "training_list.txt")
    with open(train_txt_path, "w") as f:
        if args.snippet_length > 0:
            for snippet_name in snippet_names:
                f.write(f"{snippet_name.replace('.', '_')}\n")
        else:
            for video_name in video_names:
                f.write(f"{video_name.replace('.', '_')}\n")
    print(f"Created training list at '{train_txt_path}' with {len(video_names)} videos.")

    # Create empty val_list.txt
    val_txt_path = os.path.join(out_path, "val_list.txt")
    with open(val_txt_path, "w") as f:
        pass  # Create empty file
    print(f"Created empty validation list at '{val_txt_path}'.")

    print(f"Total files processed: {count}")
    
    # Write log file
    log_path = os.path.join(out_path, "conversion_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  # create parent directories if they don't exist
    with open(log_path, "w") as log_file:
        log_file.write("=" * 70 + "\n")
        log_file.write("DAVIS DATASET CONVERSION LOG\n")
        log_file.write("=" * 70 + "\n\n")
        
        log_file.write(f"Conversion Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Workspace Path: {workspace_path}\n")
        log_file.write(f"Output Path: {out_path}\n\n")
        
        log_file.write("-" * 70 + "\n")
        log_file.write("LABEL REMAPPING:\n")
        log_file.write("-" * 70 + "\n\n")
        
        label_names = {
            1: "Surgical Instruments",
            2: "Vein (major)",
            3: "Artery (major)",
            4: "Right Superior (Upper) Lobe",
            5: "Right Middle Lobe",
            6: "Right Inferior (Lower) Lobe",
            7: "Left Superior (Upper) Lobe",
            8: "Left Inferior (Lower) Lobe",
            9: "Bronchus",
            10: "Right Horizontal Fissure",
            11: "Right Oblique Fissure",
            12: "Left Oblique Fissure",
            13: "Phrenic Nerve",
            14: "Aorta",
            15: "Esophagus",
            16: "Lymph Nodes",
            17: "Cotton Swab",
        }
        
        log_file.write("  Original -> New  | Status   | Label Name\n")
        log_file.write("  " + "-" * 55 + "\n")
        for orig_id in sorted(LABEL_CONFIG.keys()):
            new_id = LABEL_REMAP.get(orig_id, 0)
            status = "KEPT" if LABEL_CONFIG[orig_id] else "REMOVED"
            name = label_names.get(orig_id, "Unknown")
            if LABEL_CONFIG[orig_id]:
                log_file.write(f"  {orig_id:>8} -> {new_id:<4} | {status:<8} | {name}\n")
            else:
                log_file.write(f"  {orig_id:>8} -> 0    | {status:<8} | {name}\n")
        
        log_file.write("\n" + "-" * 70 + "\n")
        log_file.write("VIDEO CLIPS PROCESSED:\n")
        log_file.write("-" * 70 + "\n\n")
        
        for video_name, frame_count in video_frame_counts.items():
            log_file.write(f"  {video_name:<50} {frame_count:>6} frames\n")
        
        log_file.write("\n" + "-" * 70 + "\n")
        log_file.write("SUMMARY:\n")
        log_file.write("-" * 70 + "\n\n")
        log_file.write(f"  Total Video Clips: {len(video_frame_counts)}\n")
        log_file.write(f"  Total Frames: {total_frames}\n")
        log_file.write(f"  Total Files Processed: {count}\n")
        log_file.write(f"\n  Training List: {train_txt_path}\n")
        
        log_file.write("\n" + "=" * 70 + "\n")
    
    print(f"\n✓ Conversion log saved to: {log_path}")

if __name__ == "__main__":
    import argparse

    # arguments
    parser = argparse.ArgumentParser(description="Convert SurgNetSeg output to DAVIS dataset format.")
    parser.add_argument("--snippet_length", type=int, default=-1, help="Number of frames for each video snippet (default: -1, meaning no snippet, 1->1 translation)")
    parser.add_argument("--workspace_path", type=str, default="./workspace/", help="Path to the SurgNetSeg workspace (subdirectories will be used as video names)")
    parser.add_argument("--out_path", type=str, required=True, help="Path to the output DAVIS dataset")
    parser.add_argument("--threads", type=int, default=8, help="Number of worker threads (default: auto-detect, max 8)")
    args = parser.parse_args()

    main(args)