# src/data/image_loader.py

import logging
import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize

logger = logging.getLogger("src.data.image_loader")

def preprocess_termogram(data: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize and normalize a thermogram image to the `[0, 1]` range."""
    data_resized = resize(data, target_size, preserve_range=True, anti_aliasing=True)
    preprocessed_data = data_resized / 255.0
    return preprocessed_data

def _resolve_gender_value(row) -> str:
    """Resolve the file-name gender token expected by the dataset."""
    if "Gender_Original" in row and isinstance(row["Gender_Original"], str):
        return row["Gender_Original"]

    gender_value = row["Gender"]
    if isinstance(gender_value, (int, float, np.integer, np.floating)):
        return "M" if gender_value == 1 else "F"
    return str(gender_value)

def load_termogram_images(
    data_tabular,
    img_dir: str,
    target_size: tuple,
    return_valid_indices: bool = False,
) -> tuple:
    """Load paired left/right thermograms following the current dataset naming convention.

    When `return_valid_indices=True`, the function also returns the row indices from
    `data_tabular` whose paired images were successfully loaded. Existing callers can
    keep using the original two-value return signature.
    """
    if "Subject" not in data_tabular.columns or "Gender" not in data_tabular.columns:
        raise KeyError("`data_tabular` must contain `Subject` and `Gender` columns.")

    termograms_left, termograms_right = [], []
    valid_indices = []
    missing_subjects = []

    for idx, row in data_tabular.iterrows():
        subject_id = str(row["Subject"])
        gender = _resolve_gender_value(row)

        if subject_id.startswith("DM"):
            left_folder, right_folder = "DM Left", "DM Right"
        elif subject_id.startswith("CG"):
            left_folder, right_folder = "CG Left", "CG Right"
        else:
            logger.warning(f"Unknown group for subject {subject_id}, skipping.")
            continue

        img_path_left = os.path.join(img_dir, "Left", left_folder, f"{subject_id}_{gender}_L.png")
        img_path_right = os.path.join(img_dir, "Right", right_folder, f"{subject_id}_{gender}_R.png")

        if not (os.path.exists(img_path_left) and os.path.exists(img_path_right)):
            logger.warning(f"Image file not found for {subject_id}: {img_path_left} | {img_path_right}")
            missing_subjects.append(subject_id)
            continue

        try:
            img_left = imread(img_path_left)
            img_right = imread(img_path_right)
            img_left_normalized = preprocess_termogram(img_left, target_size)
            img_right_normalized = preprocess_termogram(img_right, target_size)
        except Exception as e:
            logger.error(f"Error loading or preprocessing image for subject {subject_id}: {e}")
            continue

        termograms_left.append(img_left_normalized)
        termograms_right.append(img_right_normalized)
        valid_indices.append(idx)

    logger.info(f"Total images loaded: {len(termograms_left)}. Missing: {len(missing_subjects)}")
    if missing_subjects:
        logger.warning(f"Missing subjects: {missing_subjects}")

    outputs = (np.array(termograms_left), np.array(termograms_right))
    if return_valid_indices:
        return outputs + (valid_indices,)
    return outputs

# Export functions
__all__ = ['preprocess_termogram', 'load_termogram_images']
