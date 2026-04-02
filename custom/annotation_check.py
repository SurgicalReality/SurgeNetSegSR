"""
Check annotation completeness by comparing image and mask counts in workspace subfolders.
"""

import os
from pathlib import Path

# Default workspace path (relative to script location)
WORKSPACE_DIR = Path(__file__).parent.parent / "workspace"


def count_files(folder: Path, extensions: tuple = (".png", ".jpg", ".jpeg")) -> int:
    """Count image files in a folder."""
    if not folder.exists():
        return -1  # Folder doesn't exist
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in extensions)


def check_annotations(workspace_dir: Path = WORKSPACE_DIR) -> None:
    """Check all snippet folders for matching image and mask counts."""
    
    if not workspace_dir.exists():
        print(f"Error: Workspace directory not found: {workspace_dir}")
        return
    
    print(f"Checking annotations in: {workspace_dir}\n")
    print(f"{'Snippet Name':<50} {'Images':>8} {'Masks':>8} {'Status':>12}")
    print("-" * 80)
    
    complete_count = 0
    incomplete_count = 0
    missing_count = 0
    
    # Get all subdirectories (skip files like overlay videos)
    snippets = sorted([d for d in workspace_dir.iterdir() if d.is_dir()])
    
    for snippet in snippets:
        images_dir = snippet / "images"
        masks_dir = snippet / "masks"
        
        image_count = count_files(images_dir)
        mask_count = count_files(masks_dir)
        
        # Determine status
        if image_count == -1 or mask_count == -1:
            status = "MISSING FOLDER"
            missing_count += 1
        elif image_count == mask_count:
            status = "OK"
            complete_count += 1
        else:
            status = f"MISMATCH ({mask_count - image_count:+d})"
            incomplete_count += 1
        
        # Format counts for display
        img_str = str(image_count) if image_count >= 0 else "N/A"
        mask_str = str(mask_count) if mask_count >= 0 else "N/A"
        
        print(f"{snippet.name:<50} {img_str:>8} {mask_str:>8} {status:>12}")
    
    # Summary
    print("-" * 80)
    print(f"\nSummary:")
    print(f"  Complete:   {complete_count}")
    print(f"  Incomplete: {incomplete_count}")
    print(f"  Missing:    {missing_count}")
    print(f"  Total:      {len(snippets)}")
    
    if incomplete_count == 0 and missing_count == 0:
        print("\n✓ All annotations are complete!")
    else:
        print(f"\n⚠ {incomplete_count + missing_count} snippet(s) need attention.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check annotation completeness")
    parser.add_argument(
        "--workspace", "-w",
        type=Path,
        default=WORKSPACE_DIR,
        help="Path to workspace directory"
    )
    args = parser.parse_args()
    
    check_annotations(args.workspace)
