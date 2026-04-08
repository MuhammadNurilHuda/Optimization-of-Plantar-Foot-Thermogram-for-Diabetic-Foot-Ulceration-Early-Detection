"""Dataset preprocessing entrypoint for the DFU thermogram pipeline.

This script creates the `data/processed` artifacts expected by the training
pipeline:

- `data/processed/[Preprocessed]Plantar Thermogram Data Analysis.csv`
- `data/processed/images_per_part/...`
- `data/processed/resized_images/...`
- `data/processed/image enhancement/...`
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.apply_image_enhancement import process_images
from src.data.tabular_preprocess import create_labels, convert_gender_to_numeric


RAW_TABULAR_PATH = Path("data/external/Plantar Thermogram Data Analysis.csv")
RAW_IMAGE_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
PROCESSED_TABULAR_PATH = PROCESSED_ROOT / "[Preprocessed]Plantar Thermogram Data Analysis.csv"
IMAGES_PER_PART_DIR = PROCESSED_ROOT / "images_per_part"
RESIZED_IMAGES_DIR = PROCESSED_ROOT / "resized_images"
ENHANCED_IMAGES_DIR = PROCESSED_ROOT / "image enhancement"

TABULAR_FEATURES = [
    "General_right",
    "LCA_right",
    "LPA_right",
    "MCA_right",
    "MPA_right",
    "TCI_right",
    "General_left",
    "LCA_left",
    "LPA_left",
    "MCA_left",
    "MPA_left",
    "TCI_left",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the DFU thermogram dataset.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing generated directories under data/processed before regenerating.",
    )
    return parser.parse_args()


def maybe_clean_outputs(clean: bool) -> None:
    if not clean:
        return

    for path in [IMAGES_PER_PART_DIR, RESIZED_IMAGES_DIR, ENHANCED_IMAGES_DIR, PROCESSED_TABULAR_PATH]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def preprocess_tabular_csv() -> None:
    df = pd.read_csv(RAW_TABULAR_PATH, sep=";")
    df = create_labels(df)
    df = convert_gender_to_numeric(df)

    # Match the repository's current processed CSV convention:
    # keep Gender as 0/1, standardize the temperature features only.
    feature_values = df[TABULAR_FEATURES].to_numpy(dtype=float)
    means = feature_values.mean(axis=0)
    stds = feature_values.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    df.loc[:, TABULAR_FEATURES] = (feature_values - means) / stds

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_TABULAR_PATH, index=False)


def _target_group_folder(subject_id: str, side: str) -> Path:
    if subject_id.startswith("DM"):
        label = "DM"
    elif subject_id.startswith("CG"):
        label = "CG"
    else:
        raise ValueError(f"Unsupported subject prefix for {subject_id}")

    return IMAGES_PER_PART_DIR / side / f"{label} {side}"


def extract_images_per_part() -> None:
    for image_path in RAW_IMAGE_ROOT.rglob("*.png"):
        path_str = str(image_path)
        if "Angiosoms" in path_str:
            continue

        stem = image_path.stem
        if not stem.endswith(("_L", "_R")):
            continue

        side = "Left" if stem.endswith("_L") else "Right"
        subject_id = stem.split("_")[0]
        target_dir = _target_group_folder(subject_id, side)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_dir / image_path.name)


def compute_average_image_size() -> tuple[int, int]:
    widths = []
    heights = []

    for image_path in IMAGES_PER_PART_DIR.rglob("*.png"):
        with Image.open(image_path) as img:
            widths.append(img.width)
            heights.append(img.height)

    if not widths or not heights:
        raise RuntimeError("No images found in data/processed/images_per_part after extraction.")

    return int(np.mean(widths)), int(np.mean(heights))


def resize_images(target_size: tuple[int, int]) -> None:
    for image_path in IMAGES_PER_PART_DIR.rglob("*.png"):
        relative_path = image_path.relative_to(IMAGES_PER_PART_DIR)
        output_path = RESIZED_IMAGES_DIR / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as img:
            resized = img.resize(target_size, Image.LANCZOS)
            resized.save(output_path)


def main() -> None:
    args = parse_args()
    maybe_clean_outputs(args.clean)
    preprocess_tabular_csv()
    extract_images_per_part()
    target_size = compute_average_image_size()
    resize_images(target_size)
    process_images(str(RESIZED_IMAGES_DIR), str(ENHANCED_IMAGES_DIR))

    print("Preprocessing complete.")
    print(f"Processed tabular CSV: {PROCESSED_TABULAR_PATH}")
    print(f"Images per part: {IMAGES_PER_PART_DIR}")
    print(f"Resized images: {RESIZED_IMAGES_DIR}")
    print(f"Enhanced images: {ENHANCED_IMAGES_DIR}")
    print(f"Average resize target: {target_size[0]}x{target_size[1]}")


if __name__ == "__main__":
    main()
