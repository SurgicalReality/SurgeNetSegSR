
import os
import argparse
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import threading
import json
import re
from datetime import timedelta
# Import color_palette and custom_names from palette.py
import sys
import importlib.util
import csv

color_palette = {
    1: (255, 255, 255),  # Surgical Instruments - White
    2: (0, 0, 255),      # Vein (major) - Blue
    3: (255, 0, 0),      # Artery (major) - Red
    4: (255, 255, 0),    # Right Superior (Upper) Lobe - Yellow 
    5: (0, 255, 0),      # Right Middle Lobe - Green
    6: (150, 0, 100),    # Right Inferior (Lower) Lobe - Dark Purlple
    7: (200, 150, 100),  # Left Superior (Upper) Lobe - Beige
    8: (150, 100, 50),  # Left Inferior (Lower) Lobe - levender
    9: (0, 200, 100),  # Bronchus - Dark Green
    10: (0, 200, 255),   # Aorta - Teal
    11: (200, 100, 200), # Cotton swab - Light Purple

}

custom_names = {
    1: "Surgical Instruments",
    2: "Vein (major)",
    3: "Artery (major)",
    4: "Right Superior (Upper) Lobe",
    5: "Right Middle Lobe",
    6: "Right Inferior (Lower) Lobe",
    7: "Left Superior (Upper) Lobe",
    8: "Left Inferior (Lower) Lobe",
    9: "Bronchus",
    10: "Aorta",
    11: "Cotton Swab",
}



# =============================================================================
# LABEL DISPLAY SETTINGS
# Set to False to exclude a label from all figures
# =============================================================================


def time_to_seconds(time_str):
    """Convert time string 'HH:MM:SS' to seconds."""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def frame_to_seconds(frame_num, fps=30):
    """Convert frame number to time in seconds."""
    return frame_num / fps


def load_annotation_data(annotation_path):
    """Load video annotation data from JSON file."""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def parse_clip_info(clip_dirname):
    """
    Parse video name and time range from clip directory name.
    Example: 's8-s10_LLL_00600_0090_cropped.mp4' 
    Returns: (video_name, start_secs, end_secs)
    """
    # Remove the .mp4/ suffix if present
    name = clip_dirname.rstrip('/')
    if name.endswith('.mp4'):
        name = name[:-4]
    
    # Match pattern: video_name_XXXXXX_XXXXXX
    match = re.match(r'(.+?)_(\d+)_(\d+)', name)
    if match:
        video_name = match.group(1)
        start_secs = int(match.group(2))
        end_secs = int(match.group(3))
        return video_name, start_secs, end_secs
    return None, None, None


def get_frame_phase(frame_num, fps, video_phases):
    """
    Determine which phase and view a frame belongs to.
    Returns: (phase_name, view) or (None, None) if not in any phase
    """
    frame_time = frame_to_seconds(frame_num, fps)
    
    for phase_name, phase_info in video_phases.items():
        start_time = time_to_seconds(phase_info['start_time'])
        end_time = time_to_seconds(phase_info['end_time'])
        if start_time <= frame_time < end_time:
            return phase_name, phase_info['view']
    
    return None, None


def load_split_config(config_path):
    """Load the train/val/test split configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def collect_split_data(mask_dir, annotation_path=None, split_name=None, split_clips=None):
    """Collect mask statistics for a split. Returns a data dict."""
    label_counts = Counter()  # Number of frames where label occurs
    label_areas = defaultdict(int)  # Total area (pixels) for each label
    label_frames = defaultdict(set)  # Set of frames (file paths) where label occurs
    objects_per_frame = Counter()  # Count of how many objects visible per frame
    
    # Phase analysis variables
    phase_counts = Counter()  # Number of frames in each phase
    view_counts = Counter()  # Number of frames annotated in each view
    frame_phase_mapping = {}  # Map frame path to (phase, view)
    
    mask_dir_path = os.path.join(mask_dir, "Annotations/")
    mask_dir_path = Path(mask_dir_path)
    masks_dirs = []
    masks_dirs = [p for p in mask_dir_path.iterdir() if p.is_dir()]

    if not masks_dirs:
        print(f"No 'Annotations' subfolders found in {mask_dir}")
        return
    print(f"Found {len(masks_dirs)} 'Annotations' subfolders.")
    
    # Filter masks_dirs by split_clips if provided
    if split_clips is not None:
        filtered_dirs = []
        for masks_dir in masks_dirs:
            # Extract clip name from path
            # clip_dir = os.path.dirname(masks_dir)
            clip_name = os.path.basename(masks_dir)
            if clip_name in split_clips:
                filtered_dirs.append(masks_dir)
        masks_dirs = filtered_dirs
        print(f"Filtered to {len(masks_dirs)} 'masks' subfolders for {split_name} split.")
    
    # Load annotation data if provided
    annotation_data = None if annotation_path is None else load_annotation_data(annotation_path)
    
    lock = threading.Lock()

    def process_masks_dir(masks_dir):
        local_counts = Counter()
        local_areas = defaultdict(int)
        local_phase_counts = Counter()
        local_view_counts = Counter()
        local_frame_phase_mapping = {}
        local_objects_per_frame = Counter()
        
        # Try to extract video info from the workspace directory structure
        video_name = None
        start_secs = None
        video_phases = None
        
        # Get the clip directory path from masks_dir
        # masks_dir is like: .../workspace/Annotations/s8-s10_LLL_001800_002700_cropped.mp4

        # New mask_dir is like .../DAVIS_1.2.1/Annotations/s8-s10_LLL_001800_002700_cropped.mp4/
        clip_dirname = os.path.basename(masks_dir)
        # TODO continue here
        
        if annotation_data:
            video_name, start_secs, end_secs = parse_clip_info(clip_dirname)
            if video_name and video_name in annotation_data['videos']:
                video_phases = annotation_data['videos'][video_name]['phases']
        
        for fname in os.listdir(masks_dir):
            if fname.lower().endswith('.png'):
                fpath = os.path.join(masks_dir, fname)
                
                # Extract frame number from filename
                frame_name = fname.replace('.png', '')
                try:
                    # 30 frames per second
                    frame_offset = int(frame_name)
                    frame_num = start_secs * 30 + frame_offset if start_secs is not None else frame_offset
                except ValueError:
                    frame_num = None
                
                # Determine phase if we have annotation data
                if video_phases and frame_num is not None:
                    phase_name, view = get_frame_phase(frame_num, fps=30, video_phases=video_phases)
                    if phase_name:
                        local_phase_counts[phase_name] += 1
                        local_view_counts[view] += 1
                        local_frame_phase_mapping[fpath] = (phase_name, view)
                
                mask = np.array(Image.open(fpath))
                unique, counts = np.unique(mask, return_counts=True)
                # Count number of objects (non-zero labels) in this frame
                num_objects = len([l for l in unique if l != 0])
                local_objects_per_frame[num_objects] += 1
                # For area
                for label, area in zip(unique, counts):
                    local_areas[label] += area
                # For frame count: only count once per frame if label is present
                for label in unique:
                    label_frames[label].add(fpath)
        
        with lock:
            for k, v in local_areas.items():
                label_areas[k] += v
            for k, v in local_phase_counts.items():
                phase_counts[k] += v
            for k, v in local_view_counts.items():
                view_counts[k] += v
            for k, v in local_objects_per_frame.items():
                objects_per_frame[k] += v
            frame_phase_mapping.update(local_frame_phase_mapping)

    threads = []
    for masks_dir in masks_dirs:
        t = threading.Thread(target=process_masks_dir, args=(masks_dir,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # After all threads, count number of frames for each label
    for label, frames in label_frames.items():
        label_counts[label] = len(frames)

    return {
        'label_counts': label_counts,
        'label_areas': label_areas,
        'objects_per_frame': objects_per_frame,
        'view_counts': view_counts,
        'phase_counts': phase_counts,
    }


def plot_combined(split_data, save_path=None):
    """
    Plot all splits in a single figure with 4 subplots.
    split_data: dict of {split_name: data_dict} from collect_split_data()
    Bars use the same label colors; splits are distinguished by hatch pattern.
    """
    SPLIT_HATCHES = {'train_list': '', 'val_list': '///', 'test_list': 'xxx'}
    SPLIT_EDGE_COLORS = {'train_list': 'none', 'val_list': '#333333', 'test_list': '#333333'}

    label_ids = sorted(custom_names.keys())
    label_names = [custom_names[label_id] for label_id in label_ids]
    bar_colors = []
    for l in label_ids:
        if l == 1:
            bar_colors.append((0.7, 0.7, 0.7))
        else:
            bar_colors.append(tuple(np.array(color_palette.get(l, (128, 128, 128))) / 255.0))

    splits = list(split_data.keys())
    n_splits = len(splits)
    n_labels = len(label_ids)
    width = 0.8 / n_splits
    x = np.arange(n_labels)

    fig, axes = plt.subplots(2, 2, figsize=(max(16, n_labels * 1.2), 13))
    fig.suptitle("Dataset Split Analysis", fontsize=17, fontweight='bold')
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # ---- Plot 1: Frames per label ----
    for i, split in enumerate(splits):
        data = split_data[split]
        vals = [data['label_counts'].get(l, 0) for l in label_ids]
        total = sum(data['label_counts'].values()) or 1
        pcts = [v / total * 100 for v in vals]
        offset = (i - n_splits / 2 + 0.5) * width
        bars = ax1.bar(x + offset, vals, width=width, color=bar_colors,
                       hatch=SPLIT_HATCHES[split], edgecolor=SPLIT_EDGE_COLORS[split],
                       linewidth=0.8, label=split.upper(), alpha=0.7)
        for bar, pct, n in zip(bars, pcts, vals):
            if n > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                         f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)
    ax1.set_ylabel("Number of Frames")
    ax1.set_title("Frames Where Label Occurs")
    ax1.set_xticks(x)
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    ax1.legend()

    # ---- Plot 2: Total area per label ----
    for i, split in enumerate(splits):
        data = split_data[split]
        vals = [data['label_areas'].get(l, 0) for l in label_ids]
        total = sum(data['label_areas'].values()) or 1
        pcts = [v / total * 100 for v in vals]
        offset = (i - n_splits / 2 + 0.5) * width
        bars = ax2.bar(x + offset, vals, width=width, color=bar_colors,
                       hatch=SPLIT_HATCHES[split], edgecolor=SPLIT_EDGE_COLORS[split],
                       linewidth=0.8, label=split.upper(), alpha=0.7)
        for bar, pct, n in zip(bars, pcts, vals):
            if n > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                         f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)
    ax2.set_ylabel("Total Area (pixels)")
    ax2.set_title("Total Area Covered by Each Label")
    ax2.set_xticks(x)
    ax2.set_xticklabels(label_names, rotation=45, ha='right')
    ax2.legend()

    # ---- Plot 3: View distribution ----
    all_views = sorted({v for data in split_data.values() for v in data['view_counts']})
    if all_views:
        colors_views = plt.cm.Set3(np.linspace(0, 1, len(all_views)))
        view_color_map = {v: colors_views[i] for i, v in enumerate(all_views)}
        x_views = np.arange(len(all_views))
        for i, split in enumerate(splits):
            vals = [split_data[split]['view_counts'].get(v, 0) for v in all_views]
            total = sum(split_data[split]['view_counts'].values()) or 1
            pcts = [v / total * 100 for v in vals]
            offset = (i - n_splits / 2 + 0.5) * width
            bars =ax3.bar(x_views + offset, vals, width=width,
                    color=[view_color_map[v] for v in all_views],
                    hatch=SPLIT_HATCHES[split], edgecolor=SPLIT_EDGE_COLORS[split],
                    linewidth=0.8, label=split.upper(), alpha=0.7)
            for bar, pct, n in zip(bars, pcts, vals):
                if n > 0:
                    ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                             f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)
        ax3.set_ylabel("Number of Annotated Frames")
        ax3.set_title("Frames by Surgical View")
        ax3.set_xticks(x_views)
        ax3.set_xticklabels(all_views, rotation=45, ha='right')
        ax3.legend()
    else:
        ax3.set_visible(False)

    # ---- Plot 4: Objects per frame distribution ----
    all_obj_counts = sorted({k for data in split_data.values() for k in data['objects_per_frame']})
    if all_obj_counts:
        x_obj = np.arange(len(all_obj_counts))
        colors_obj = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_obj_counts)))
        for i, split in enumerate(splits):
            data = split_data[split]
            total = sum(data['objects_per_frame'].values()) or 1
            vals = [data['objects_per_frame'].get(k, 0) for k in all_obj_counts]
            pcts = [v / total * 100 for v in vals]
            offset = (i - n_splits / 2 + 0.5) * width
            bars = ax4.bar(x_obj + offset, vals, width=width,
                           color=colors_obj,
                           hatch=SPLIT_HATCHES[split], edgecolor=SPLIT_EDGE_COLORS[split],
                           linewidth=0.8, label=split.upper())
            for bar, pct, n in zip(bars, pcts, vals):
                if n > 0:
                    ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                             f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)
        ax4.set_xlabel("Number of Objects Visible")
        ax4.set_ylabel("Number of Frames")
        ax4.set_title("Objects Visible Per Frame")
        ax4.set_xticks(x_obj)
        ax4.set_xticklabels([str(k) for k in all_obj_counts])
        ax4.legend()
    else:
        ax4.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Combined plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution in mask PNGs and surgical phase distribution by split.")
    parser.add_argument("--mask_dir", type=str, default="./custom/data/debug2", help="Directory containing mask .png files.")
    parser.add_argument("--annotation", type=str, default="./custom/view_annotation.json", 
                        help="Path to view_annotation.json for phase/view analysis (optional).")
    parser.add_argument("--save", type=str, default=None, help="Path to save the plots (optional, will be suffixed with _train/_val/_test).")
    
    parser.add_argument("--train_list", type=str, default="training_list.txt", help="Path to text file listing clips for training split (one clip name per line).")
    parser.add_argument("--val_list", type=str, default="val_list.txt", help="Path to text file listing clips for validation split (one clip name per line).")
    parser.add_argument("--test_list", type=str, default="test_list.txt", help="Path to text file listing clips for test split (one clip name per line).")  


    args = parser.parse_args()
    
    # check if train/val/test list files exist
    for split_arg in ['train_list', 'val_list', 'test_list']:
        split_path = os.path.join(args.mask_dir, getattr(args, split_arg))
        if not os.path.isfile(split_path):
            print(f"Warning: {split_arg} file '{split_path}' not found. This split will be skipped.")
    
    snippet_names = {}
    for split_arg in ['train_list', 'val_list', 'test_list']:
        snippet_names[split_arg] = []
        with open(os.path.join(args.mask_dir, getattr(args, split_arg)), newline="\n") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                snippet_names[split_arg].append(row[0])
    

    
    # Collect data for each split
    split_data = {}
    for split in ['train_list', 'val_list', 'test_list']:
        split_clips = snippet_names.get(split, [])
        if not split_clips:
            print(f"No clips found for {split} split. Skipping.")
            continue
        
        print(f"\n{'='*70}")
        print(f"Collecting data for {split} split ({len(split_clips)} clips)")
        print(f"{'='*70}")
        
        # collect data for this split
        data = collect_split_data(args.mask_dir, annotation_path=args.annotation,
                                  split_name=split, split_clips=split_clips)
        if data:
            split_data[split] = data
    
    if not split_data:
        print("No data collected. Exiting.")
        return
    
    plot_combined(split_data, save_path=args.save)
    plt.show()

if __name__ == "__main__":
    main()
