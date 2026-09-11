"""
Reorganise SkatingVerse train_val Dataset

The downloaded train_val/ folder has this structure:
    train_val/
        train/
            0/   
            1/
            2/
            ...
        val/
            0/
            1/
            ...

This script reorganises it into human-readable class names:
    dataset/
        train/
            Jump/
            Spin/
            Other/
        val/
            Jump/
            Spin/
            Other/

Usage:
    python reorganise_dataset.py --input_dir train_val --output_dir dataset --granularity coarse
    python reorganise_dataset.py --input_dir train_val --output_dir dataset --granularity coarse --max_per_class 50

"""

import os
import shutil
import random
import argparse
from collections import defaultdict
from pathlib import Path


ELEMENT_NAMES = {
    "0":  "3Toeloop",
    "1":  "3Loop",
    "2":  "2Axel",
    "3":  "CamelSpin",
    "4":  "SitSpin",
    "5":  "UprightSpin",
    "6":  "2Salchow",
    "7":  "2Toeloop",
    "8":  "3Salchow",
    "9":  "3Axel",
    "10": "3Flip",
    "11": "3Lutz",
    "12": "NoBasic",
    "13": "2Lutz",
    "14": "4Salchow",
    "15": "4Flip",
    "16": "4Toeloop",
    "17": "4Lutz",
    "18": "4Loop",
    "19": "2Flip",
    "20": "2Loop",
    "21": "1Axel",
    "22": "1Loop",
    "23": "1Salchow",
    "24": "1Toeloop",
    "25": "1Flip",
    "26": "1Lutz",
    "27": "Sequence",
}

SET_LEVEL = {
    "0": "Jump", "1": "Jump", "2": "Jump", "6": "Jump", "7": "Jump",
    "8": "Jump", "9": "Jump", "10": "Jump", "11": "Jump", "13": "Jump",
    "14": "Jump", "15": "Jump", "16": "Jump", "17": "Jump", "18": "Jump",
    "19": "Jump", "20": "Jump", "21": "Jump", "22": "Jump", "23": "Jump",
    "24": "Jump", "25": "Jump", "26": "Jump",
    "3": "CamelSpin",
    "4": "SitSpin",
    "5": "UprightSpin",
    "12": "NoBasic",
    "27": "Sequence",
}

COARSE = {
    "0": "Jump", "1": "Jump", "2": "Jump", "6": "Jump", "7": "Jump",
    "8": "Jump", "9": "Jump", "10": "Jump", "11": "Jump", "13": "Jump",
    "14": "Jump", "15": "Jump", "16": "Jump", "17": "Jump", "18": "Jump",
    "19": "Jump", "20": "Jump", "21": "Jump", "22": "Jump", "23": "Jump",
    "24": "Jump", "25": "Jump", "26": "Jump",
    "3": "Spin", "4": "Spin", "5": "Spin",
    "12": "Other", "27": "Other",
}


def get_mapping(granularity: str) -> dict:
    if granularity == "coarse":
        return COARSE
    elif granularity == "set":
        return SET_LEVEL
    elif granularity == "element":
        return ELEMENT_NAMES
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def reorganise(input_dir, output_dir, granularity, max_per_class, seed):
    random.seed(seed)
    mapping = get_mapping(granularity)

    for split in ["train", "val"]:
        src_split = os.path.join(input_dir, split)
        dst_split = os.path.join(output_dir, split)

        if not os.path.isdir(src_split):
            print(f"[Warning] '{src_split}' not found, skipping.")
            continue

        class_videos = defaultdict(list)

        for numbered_folder in sorted(os.listdir(src_split)):
            folder_path = os.path.join(src_split, numbered_folder)
            if not os.path.isdir(folder_path):
                continue

            class_name = mapping.get(numbered_folder)
            if class_name is None:
                print(f"  [Warning] Unknown class folder '{numbered_folder}', skipping.")
                continue

            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath):
                    class_videos[class_name].append(fpath)

        if max_per_class is not None:
            for cls_name in class_videos:
                videos = class_videos[cls_name]
                if len(videos) > max_per_class:
                    random.shuffle(videos)
                    class_videos[cls_name] = videos[:max_per_class]

        print(f"\n[{split.upper()}]")
        split_total = 0

        for cls_name in sorted(class_videos.keys()):
            dst_class = os.path.join(dst_split, cls_name)
            os.makedirs(dst_class, exist_ok=True)

            videos = class_videos[cls_name]
            for src_path in videos:
                dst_path = os.path.join(dst_class, os.path.basename(src_path))
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

            split_total += len(videos)
            print(f"  {cls_name}: {len(videos)} videos")

        print(f"  Total: {split_total} videos")

    print(f"\n{'='*55}")
    print(f" DONE — Dataset ready in '{output_dir}/'")
    print(f"{'='*55}")
    print(f"\n  {output_dir}/")
    print(f"    train/")
    train_path = os.path.join(output_dir, "train")
    if os.path.isdir(train_path):
        for d in sorted(os.listdir(train_path)):
            count = len(os.listdir(os.path.join(train_path, d)))
            print(f"      {d}/  ({count} videos)")
    print(f"    val/")
    val_path = os.path.join(output_dir, "val")
    if os.path.isdir(val_path):
        for d in sorted(os.listdir(val_path)):
            count = len(os.listdir(os.path.join(val_path, d)))
            print(f"      {d}/  ({count} videos)")

    print(f"\nNext step:")
    print(f"  python skating_action_recognition.py --data_dir {output_dir} --epochs 25")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganise SkatingVerse train_val into named class folders",
    )
    parser.add_argument(
        "--input_dir", type=str, default="train_val",
        help="Path to the downloaded train_val folder",
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset",
        help="Output directory (default: 'dataset')",
    )
    parser.add_argument(
        "--granularity", type=str, default="coarse",
        choices=["element", "set", "coarse"],
        help="'coarse' (3 classes: Jump/Spin/Other), 'set' (7 classes), 'element' (28 classes)",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=None,
        help="Limit videos per class (e.g. 50 for quick testing)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    print(f"{'='*55}")
    print(f" Reorganise SkatingVerse Dataset")
    print(f"{'='*55}")
    print(f"  Input:       {args.input_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Granularity: {args.granularity}")
    print(f"  Max/class:   {args.max_per_class or 'all'}")

    reorganise(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        granularity=args.granularity,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()