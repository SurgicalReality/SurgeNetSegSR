
import os
import argparse
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

# Dynamically import palette.py
palette_path = os.path.join(os.path.dirname(__file__), '../gui/cutie/utils/palette.py')
spec = importlib.util.spec_from_file_location('palette', palette_path)
palette = importlib.util.module_from_spec(spec)
sys.modules['palette'] = palette
spec.loader.exec_module(palette)


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


def analyze_masks(mask_dir, annotation_path=None, save_path=None):
    label_counts = Counter()  # Number of frames where label occurs
    label_areas = defaultdict(int)  # Total area (pixels) for each label
    label_frames = defaultdict(set)  # Set of frames (file paths) where label occurs
    objects_per_frame = Counter()  # Count of how many objects visible per frame
    
    # Phase analysis variables
    phase_counts = Counter()  # Number of frames in each phase
    view_counts = Counter()  # Number of frames annotated in each view
    frame_phase_mapping = {}  # Map frame path to (phase, view)
    
    masks_dirs = []
    for root, dirs, files in os.walk(mask_dir):
        if os.path.basename(root) == "masks":
            masks_dirs.append(root)
    if not masks_dirs:
        print(f"No 'masks' subfolders found in {mask_dir}")
        return
    print(f"Found {len(masks_dirs)} 'masks' subfolders.")
    
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
        # masks_dir is like: .../workspace/s8-s10_LLL_001800_002700_cropped.mp4/masks
        clip_dir = os.path.dirname(masks_dir)
        clip_dirname = os.path.basename(clip_dir)
        
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

    # Exclude label 0 (background) from the plot
    filtered_labels = [l for l in sorted(label_counts.keys()) if l != 0]

    # Prepare data for both plots
    values_frames = [label_counts[l] for l in filtered_labels]
    values_area = [label_areas[l] for l in filtered_labels]
    label_names = [palette.custom_names.get(l, str(l)) for l in filtered_labels]
    bar_colors = []
    for l in filtered_labels:
        if l == 1:
            bar_colors.append((0.7, 0.7, 0.7))
        else:
            bar_colors.append(tuple(np.array(palette.color_palette.get(l, (128, 128, 128))) / 255.0))

    # Create figure with 2x2 grid layout (3 plots used)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # Plot 1 (top-left): Number of frames where label occurs
    bars1 = ax1.bar(label_names, values_frames, color=bar_colors)
    ax1.set_ylabel("Number of Frames")
    ax1.set_title("Number of Frames Where Label Occurs")
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    y_max1 = max(values_frames) if values_frames else 1
    ax1.set_ylim(0, y_max1 * 1.1)

    # Plot 2 (top-right): Total area (pixels) for each label
    bars2 = ax2.bar(label_names, values_area, color=bar_colors)
    ax2.set_ylabel("Total Area (pixels)")
    ax2.set_title("Total Area Covered by Each Label")
    ax2.set_xticklabels(label_names, rotation=45, ha='right')
    y_max2 = max(values_area) if values_area else 1
    ax2.set_ylim(0, y_max2 * 1.1)

    # Plot 3 (bottom-left): Distribution of frames by view
    if view_counts:
        sorted_views = sorted(view_counts.keys())
        view_values = [view_counts[v] for v in sorted_views]
        colors_views = plt.cm.Set3(np.linspace(0, 1, len(sorted_views)))
        bars3 = ax3.bar(sorted_views, view_values, color=colors_views)
        ax3.set_ylabel("Number of Annotated Frames")
        ax3.set_title("Distribution of Annotated Frames by Surgical View")
        ax3.set_xticklabels(sorted_views, rotation=45, ha='right')
        y_max3 = max(view_values) if view_values else 1
        ax3.set_ylim(0, y_max3 * 1.1)
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    else:
        ax3.set_visible(False)

    # Plot 4 (bottom-right): Distribution of objects per frame
    if objects_per_frame:
        sorted_obj_counts = sorted(objects_per_frame.keys())
        obj_values = [objects_per_frame[k] for k in sorted_obj_counts]
        total_frames = sum(obj_values)
        percentages = [v / total_frames * 100 for v in obj_values]
        
        colors_obj = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_obj_counts)))
        bars4 = ax4.bar([str(k) for k in sorted_obj_counts], obj_values, color=colors_obj)
        ax4.set_xlabel("Number of Objects Visible")
        ax4.set_ylabel("Number of Frames")
        ax4.set_title("Distribution of Objects Visible Per Frame")
        y_max4 = max(obj_values) if obj_values else 1
        ax4.set_ylim(0, y_max4 * 1.2)
        
        # Add percentage and n labels on bars
        for bar, pct, n in zip(bars4, percentages, obj_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%\n(n={n})',
                    ha='center', va='bottom', fontsize=9)
    else:
        ax4.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution in mask PNGs and surgical phase distribution.")
    parser.add_argument("--mask_dir", type=str, default="./workspace", help="Directory containing mask .png files.")
    parser.add_argument("--annotation", type=str, default="./custom/view_annotation.json", 
                        help="Path to view_annotation.json for phase/view analysis (optional).")

    parser.add_argument("--save", type=str, default=None, help="Path to save the plot (optional).")
    args = parser.parse_args()
    analyze_masks(args.mask_dir, annotation_path=args.annotation, save_path=args.save)

if __name__ == "__main__":
    main()
