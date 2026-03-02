import os
from PIL import Image
import importlib.util
import argparse

# Import color_palette from the repo
def import_palette(palette_path):
    spec = importlib.util.spec_from_file_location("palette", palette_path)
    palette_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(palette_module)
    color_palette = palette_module.color_palette
    return color_palette

def build_flat_palette(color_palette):
    # PIL expects a flat list of 256*3 values
    palette = [(0, 0, 0)] * 256
    for idx, color in color_palette.items():
        palette[idx] = color
    flat_palette = [c for rgb in palette for c in rgb]
    return flat_palette

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def update_masks_palette(mask_dirs, flat_palette, num_workers=8):
    mask_files = []

    for mask_dir in mask_dirs:
        for root, _, files in os.walk(mask_dir):
            for fname in files:
                if fname.endswith('.png'):
                    mask_files.append(os.path.join(root, fname))

    def process_file(fpath):
        img = Image.open(fpath)
        img.putpalette(flat_palette)
        img.save(fpath)
        return fpath

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in mask_files}
        for _ in tqdm(as_completed(futures), total=len(mask_files), desc="Updating palettes"):
            pass

if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser(description="Update the palette of the workspace or the DAVIS dataset")
    parser.add_argument("--num_workers", type=int, help="Number of worker threads to use", default=8)
    args = parser.parse_args()

    # Set these paths as needed
    palette_path = "../gui/cutie/utils/palette.py"  # relative to repo root

    # Import palette
    color_palette = import_palette(palette_path)
    flat_palette = build_flat_palette(color_palette)

    # Choose target directory
    mask_dirs = ["data/DAVIS/Annotations", "../workspace"]


    update_masks_palette(mask_dirs, flat_palette, args.num_workers)
    print("Palette update complete.")