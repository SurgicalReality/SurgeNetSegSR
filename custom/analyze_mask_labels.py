
import os
import argparse
import numpy as np
from collections import Counter, defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import threading
# Import color_palette and custom_names from palette.py
import sys
import importlib.util

# Dynamically import palette.py
palette_path = os.path.join(os.path.dirname(__file__), '../gui/cutie/utils/palette.py')
spec = importlib.util.spec_from_file_location('palette', palette_path)
palette = importlib.util.module_from_spec(spec)
sys.modules['palette'] = palette
spec.loader.exec_module(palette)


def analyze_masks(mask_dir, plot_area=False, save_path=None):
    label_counts = Counter()  # Number of frames where label occurs
    label_areas = defaultdict(int)  # Total area (pixels) for each label
    label_frames = defaultdict(set)  # Set of frames (file paths) where label occurs
    masks_dirs = []
    for root, dirs, files in os.walk(mask_dir):
        if os.path.basename(root) == "masks":
            masks_dirs.append(root)
    if not masks_dirs:
        print(f"No 'masks' subfolders found in {mask_dir}")
        return
    print(f"Found {len(masks_dirs)} 'masks' subfolders.")

    lock = threading.Lock()

    def process_masks_dir(masks_dir):
        local_counts = Counter()
        local_areas = defaultdict(int)
        for fname in os.listdir(masks_dir):
            if fname.lower().endswith('.png'):
                fpath = os.path.join(masks_dir, fname)
                mask = np.array(Image.open(fpath))
                unique, counts = np.unique(mask, return_counts=True)
                # For area
                for label, area in zip(unique, counts):
                    local_areas[label] += area
                # For frame count: only count once per frame if label is present
                for label in unique:
                    label_frames[label].add(fpath)
        with lock:
            for k, v in local_areas.items():
                label_areas[k] += v

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(filtered_labels)), 12))

    # Plot 1: Number of frames where label occurs
    bars1 = ax1.bar(label_names, values_frames, color=bar_colors)
    ax1.set_ylabel("Number of Frames")
    ax1.set_title("Number of Frames Where Label Occurs")
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    y_max1 = max(values_frames) if values_frames else 1
    ax1.set_ylim(0, y_max1 * 1.1)

    # Plot 2: Total area (pixels) for each label
    bars2 = ax2.bar(label_names, values_area, color=bar_colors)
    ax2.set_ylabel("Total Area (pixels)")
    ax2.set_title("Total Area Covered by Each Label")
    ax2.set_xticklabels(label_names, rotation=45, ha='right')
    y_max2 = max(values_area) if values_area else 1
    ax2.set_ylim(0, y_max2 * 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution in mask PNGs.")
    parser.add_argument("mask_dir", type=str, help="Directory containing mask .png files.")
    parser.add_argument("--plot_area", action="store_true", help="Plot total area covered by each label instead of count.")
    parser.add_argument("--save", type=str, default=None, help="Path to save the plot (optional).")
    args = parser.parse_args()
    analyze_masks(args.mask_dir, plot_area=args.plot_area, save_path=args.save)

if __name__ == "__main__":
    main()
