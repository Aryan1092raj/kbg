"""
reorganize_multiclass.py — Download and reorganize the Bone Break Classifier
dataset into a stratified train/val/test split for multi-class fracture
classification.

Source: Kaggle – amohankumar/bone-break-classifier-dataset (12 fracture types)
Split : 70 % train  /  10 % val  /  20 % test  (stratified)

Usage:
    python scripts/reorganize_multiclass.py            # auto-download via kagglehub
    python scripts/reorganize_multiclass.py --src DIR  # use a local copy
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ── Constants ──────────────────────────────────────────────────────────────
FRACTURE_CLASSES = [
    "Avulsion fracture",
    "Comminuted fracture",
    "Compression-Crush fracture",
    "Fracture Dislocation",
    "Greenstick fracture",
    "Hairline Fracture",
    "Impacted fracture",
    "Intra-articular fracture",
    "Longitudinal fracture",
    "Oblique fracture",
    "Pathological fracture",
    "Spiral Fracture",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


# ── Helpers ────────────────────────────────────────────────────────────────
def collect_images(src_dir: Path) -> dict:
    """Walk source directory and collect all images grouped by class."""
    class_images = defaultdict(list)
    for cls_name in FRACTURE_CLASSES:
        cls_dir = src_dir / cls_name
        if not cls_dir.exists():
            print(f"  [WARN] Missing class directory: {cls_dir}")
            continue
        for root, _, files in os.walk(cls_dir):
            for fname in files:
                if Path(fname).suffix.lower() in IMAGE_EXTS:
                    class_images[cls_name].append(Path(root) / fname)
    return class_images


def split_list(items, train_r, val_r):
    """Split a list into train / val / test with given ratios."""
    n = len(items)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def copy_images(file_list, dest_dir: Path, cls_name: str):
    """Copy images to dest_dir/cls_name/, renaming to avoid collisions."""
    cls_dir = dest_dir / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i, src_path in enumerate(file_list):
        ext = src_path.suffix.lower()
        # Use deterministic naming: classname_index.ext
        dst_name = f"{cls_name}_{i:04d}{ext}"
        dst_path = cls_dir / dst_name
        shutil.copy2(src_path, dst_path)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Reorganize dataset for multi-class fracture classification")
    parser.add_argument("--src", type=str, default=None,
                        help="Path to source dataset with class subdirectories. If omitted, downloads via kagglehub.")
    parser.add_argument("--dst", type=str, default=None,
                        help="Destination data directory. Defaults to <project>/data")
    args = parser.parse_args()

    # Determine project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent

    # ── Source directory ──
    if args.src:
        src_dir = Path(args.src)
    else:
        try:
            import kagglehub
            print("[INFO] Downloading dataset via kagglehub ...")
            src_dir = Path(kagglehub.dataset_download("amohankumar/bone-break-classifier-dataset"))
            print(f"[INFO] Dataset at: {src_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to download dataset: {e}")
            print("Install kagglehub (pip install kagglehub) or pass --src manually.")
            sys.exit(1)

    if not src_dir.exists():
        print(f"[ERROR] Source directory does not exist: {src_dir}")
        sys.exit(1)

    # ── Destination directory ──
    dst_dir = Path(args.dst) if args.dst else (project_root / "data")
    
    # Back up old data if it exists
    if dst_dir.exists():
        backup = dst_dir.parent / "data_binary_backup"
        if not backup.exists():
            print(f"[INFO] Backing up existing data/ to {backup}")
            shutil.copytree(dst_dir, backup)
        # Remove old splits
        for split in ["train", "val", "test"]:
            split_dir = dst_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    # ── Collect and split ──
    print("[INFO] Collecting images ...")
    class_images = collect_images(src_dir)

    random.seed(SEED)
    total = {"train": 0, "val": 0, "test": 0}

    for cls_name, images in sorted(class_images.items()):
        random.shuffle(images)
        train, val, test = split_list(images, TRAIN_RATIO, VAL_RATIO)

        copy_images(train, dst_dir / "train", cls_name)
        copy_images(val, dst_dir / "val", cls_name)
        copy_images(test, dst_dir / "test", cls_name)

        total["train"] += len(train)
        total["val"] += len(val)
        total["test"] += len(test)

        print(f"  {cls_name:35s} → train={len(train):4d}  val={len(val):3d}  test={len(test):3d}")

    print(f"\n[DONE] Split complete:")
    print(f"  Train: {total['train']:5d}")
    print(f"  Val:   {total['val']:5d}")
    print(f"  Test:  {total['test']:5d}")
    print(f"  Total: {sum(total.values()):5d}")
    print(f"  Output: {dst_dir}")


if __name__ == "__main__":
    main()
