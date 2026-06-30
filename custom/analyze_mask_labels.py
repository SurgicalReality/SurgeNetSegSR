
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


# =============================================================================
# LABEL DISPLAY SETTINGS
# Set to False to exclude a label from all figures
# =============================================================================
LABEL_DISPLAY = {
    1:  True,   # Surgical Instruments
    2:  True,   # Vein (major)
    3:  True,   # Artery (major)
    4:  True,   # Right Superior (Upper) Lobe
    5:  True,   # Right Middle Lobe
    6:  True,   # Right Inferior (Lower) Lobe
    7:  True,   # Left Superior (Upper) Lobe
    8:  True,   # Left Inferior (Lower) Lobe
    9:  True,   # Bronchus
    10: True,   # Right Horizontal Fissure
    11: True,   # Right Oblique Fissure
    12: True,   # Left Oblique Fissure
    13: True,   # Phrenic Nerve
    14: True,   # Aorta
    15: True,   # Esophagus
    16: False,  # Lymph Nodes
    17: True,   # Cotton Swab
}


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


def collect_split_data(mask_dir, annotation_path=None):
    """Collect mask statistics for a split. Returns a data dict."""
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
    hatch_cycle = ['', '///', 'xxx', '\\\\\\', '...']

    # Gather all labels present across all splits (filtered by LABEL_DISPLAY)
    all_labels = sorted(
        {l for data in split_data.values() for l in data['label_counts'] if l != 0 and LABEL_DISPLAY.get(l, True)}
    )
    label_names = [palette.custom_names.get(l, str(l)) for l in all_labels]
    bar_colors = []
    for l in all_labels:
        if l == 1:
            bar_colors.append((0.7, 0.7, 0.7))
        else:
            bar_colors.append(tuple(np.array(palette.color_palette.get(l, (128, 128, 128))) / 255.0))

    splits = list(split_data.keys())
    n_splits = len(splits)
    split_hatches = {s: hatch_cycle[i % len(hatch_cycle)] for i, s in enumerate(splits)}
    split_edge_colors = {s: ('none' if split_hatches[s] == '' else '#333333') for s in splits}
    n_labels = len(all_labels)
    width = 0.8 / n_splits
    x = np.arange(n_labels)

    fig, axes = plt.subplots(2, 2, figsize=(max(16, n_labels * 1.2), 13))
    fig.suptitle("Dataset Analysis", fontsize=17, fontweight='bold')
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # ---- Plot 1: Frames per label ----
    for i, split in enumerate(splits):
        data = split_data[split]
        vals = [data['label_counts'].get(l, 0) for l in all_labels]
        offset = (i - n_splits / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width=width, color=bar_colors,
            hatch=split_hatches[split], edgecolor=split_edge_colors[split],
                linewidth=0.8, label=split.upper())
    ax1.set_ylabel("Number of Frames")
    ax1.set_title("Frames Where Label Occurs")
    ax1.set_xticks(x)
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    ax1.legend()

    # ---- Plot 2: Total area per label ----
    for i, split in enumerate(splits):
        data = split_data[split]
        vals = [data['label_areas'].get(l, 0) for l in all_labels]
        offset = (i - n_splits / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width=width, color=bar_colors,
            hatch=split_hatches[split], edgecolor=split_edge_colors[split],
                linewidth=0.8, label=split.upper())
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
            offset = (i - n_splits / 2 + 0.5) * width
            ax3.bar(x_views + offset, vals, width=width,
                    color=[view_color_map[v] for v in all_views],
                    hatch=split_hatches[split], edgecolor=split_edge_colors[split],
                    linewidth=0.8, label=split.upper())
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
                           hatch=split_hatches[split], edgecolor=split_edge_colors[split],
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
    parser = argparse.ArgumentParser(description="Analyze label distribution in mask PNGs and surgical phase distribution.")
    parser.add_argument("--mask_dir", type=str, default="./workspace", help="Directory containing mask .png files.")
    parser.add_argument("--annotation", type=str, default="./custom/view_annotation.json", 
                        help="Path to view_annotation.json for phase/view analysis (optional).")
    parser.add_argument("--save", type=str, default=None, help="Path to save the combined plot (optional).")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Collecting data from workspace: {args.mask_dir}")
    print(f"{'='*70}")

    data = collect_split_data(args.mask_dir, annotation_path=args.annotation)
    if not data:
        print("No data collected. Exiting.")
        return

    split_data = {'workspace': data}
    plot_combined(split_data, save_path=args.save)
    plt.show()

if __name__ == "__main__":
    main()
