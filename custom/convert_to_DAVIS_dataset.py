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
    python convert_to_DAVIS_dataset.py --workspace_path <path_to_workspace> --video_names <video1,video2,...> --out_path <output_davis_path>

"""

import os
import re
import shutil
from datetime import datetime


def main(args):
    workspace_path = args.workspace_path
    video_names = args.video_names.split(",")
    out_path = args.out_path
    count = 0
    
    # For logging
    video_frame_counts = {}
    total_frames = 0
    start_time = datetime.now()

    for video_name in video_names:
        video_folder = os.path.join(workspace_path, video_name)
        if not os.path.exists(video_folder):
            print(f"Folder for video '{video_name}' not found in workspace. Skipping.")
            continue
        
        # Track frames for this video
        video_frames = 0
        
        # images folder -> JPEGImages/video_name
        image_folder = os.path.join(video_folder, "images")
        images_out_folder = os.path.join(out_path, "JPEGImages", video_name.replace(".", "_"))
        os.makedirs(images_out_folder, exist_ok=True)

        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            if os.path.isfile(file_path):
                # clean filename to be compatible with SAM 2.1
                '''
                new_filename = filename.replace(".", "_", filename.count(".") - 1)
                if not re.search(r"_\d+\.\w+$", new_filename):
                    new_filename = new_filename.replace(".", "_1.")
                '''
                file_idx = int(filename.rsplit(".", 1)[0])  # remove extension
                new_filename = f"{file_idx:05d}.jpg"

                # save copy to output folder
                shutil.copy(file_path, os.path.join(images_out_folder, new_filename))
                count += 1
                video_frames += 1
        
        # Store frame count for this video
        video_frame_counts[video_name.replace(".", "_")] = video_frames
        total_frames += video_frames
        
        # masks folder -> Annotations/video_name
        mask_folder = os.path.join(video_folder, "masks")
        masks_out_folder = os.path.join(out_path, "Annotations", video_name.replace(".", "_"))
        os.makedirs(masks_out_folder, exist_ok=True)

        for filename in os.listdir(mask_folder):
            file_path = os.path.join(mask_folder, filename)
            if os.path.isfile(file_path):
                '''
                # clean filename to be compatible with SAM 2.1
                new_filename = filename.replace(".", "_", filename.count(".") - 1)
                #
                if not re.search(r"_\d+\.\w+$", new_filename):
                    new_filename = new_filename.replace(".", "_1.")
                '''
                mask_idx = int(filename.rsplit(".", 1)[0])  # remove extension

                # save copy to output folder
                new_filename = f"{mask_idx:05d}.png"  # ensure .png extension for masks
                shutil.copy(file_path, os.path.join(masks_out_folder, new_filename))

                count += 1

        print(f"Processed video '{video_name}' and saved to DAVIS format in '{out_path}'.")

    train_txt_path = os.path.join(out_path, "training_list.txt")
    with open(train_txt_path, "w") as f:
        for video_name in video_names:
            f.write(f"{video_name.replace('.', '_')}\n")

        print(f"Created training list at '{train_txt_path}' with {len(video_names)} videos.")

    print(f"Total files processed: {count}")
    
    # Write log file
    log_path = os.path.join(out_path, "conversion_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write("=" * 70 + "\n")
        log_file.write("DAVIS DATASET CONVERSION LOG\n")
        log_file.write("=" * 70 + "\n\n")
        
        log_file.write(f"Conversion Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Workspace Path: {workspace_path}\n")
        log_file.write(f"Output Path: {out_path}\n\n")
        
        log_file.write("-" * 70 + "\n")
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
    parser.add_argument("--workspace_path", type=str, help="Path to the SurgNetSeg workspace")
    parser.add_argument("--video_names", type=str, help="Comma-separated list of video names (without extension)")
    parser.add_argument("--out_path", type=str, help="Path to the output DAVIS dataset")
    args = parser.parse_args()

    main(args)