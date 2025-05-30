# src/data/image_loader.py

import os
import logging
from skimage.io import imread
from skimage.transform import resize
import numpy as np

logger = logging.getLogger("src.data.image_loader")

def preprocess_termogram(data: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize dan normalisasi termogram ke ukuran dan range [0,1] untuk feeding CNN."""
    data_resized = resize(data, target_size, preserve_range=True, anti_aliasing=True)
    preprocessed_data = data_resized / 255.0
    return preprocessed_data

def load_termogram_images(data_tabular, img_dir: str, target_size: tuple) -> tuple:
    """
    Memuat citra termogram kiri dan kanan secara paired dari data_tabular sesuai subject+gender.
    """
    termograms_left, termograms_right = [], []
    missing_subjects = []

    for idx, row in data_tabular.iterrows():
        subject_id = row['Subject']
        
        # Gunakan gender asli untuk nama file
        if 'Gender_Original' in row:
            gender = row['Gender_Original']
        else:
            gender_numeric = row['Gender']
            if isinstance(gender_numeric, (int, float)):
                gender = 'M' if gender_numeric == 1 else 'F'
            else:
                gender = gender_numeric
        
        if subject_id.startswith('DM'):
            left_folder, right_folder = 'DM Left', 'DM Right'
        elif subject_id.startswith('CG'):
            left_folder, right_folder = 'CG Left', 'CG Right'
        else:
            logger.warning(f"Unknown group for subject {subject_id}, skipping.")
            continue

        img_path_left = os.path.join(img_dir, 'Left', left_folder, f'{subject_id}_{gender}_L.png')
        img_path_right = os.path.join(img_dir, 'Right', right_folder, f'{subject_id}_{gender}_R.png')

        if not (os.path.exists(img_path_left) and os.path.exists(img_path_right)):
            logger.warning(f"Image file not found for {subject_id}: {img_path_left} | {img_path_right}")
            missing_subjects.append(subject_id)
            continue

        try:
            img_left = imread(img_path_left)
            img_right = imread(img_path_right)
            # Preprocess images
            img_left_resized = resize(img_left, target_size, preserve_range=True, anti_aliasing=True)
            img_right_resized = resize(img_right, target_size, preserve_range=True, anti_aliasing=True)
            img_left_normalized = img_left_resized / 255.0
            img_right_normalized = img_right_resized / 255.0
        except Exception as e:
            logger.error(f"Error loading or preprocessing image for subject {subject_id}: {e}")
            continue

        termograms_left.append(img_left_normalized)
        termograms_right.append(img_right_normalized)

    logger.info(f"Total images loaded: {len(termograms_left)}. Missing: {len(missing_subjects)}")
    if missing_subjects:
        logger.warning(f"Missing subjects: {missing_subjects}")

    return np.array(termograms_left), np.array(termograms_right)

# Export functions
__all__ = ['preprocess_termogram', 'load_termogram_images']