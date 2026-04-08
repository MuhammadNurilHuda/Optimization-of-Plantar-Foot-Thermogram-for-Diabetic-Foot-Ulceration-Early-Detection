# src/utils/image_enhancement.py

import cv2
import numpy as np
import logging

logger = logging.getLogger('src.utils.image_enhancement')

def posterize_image(image, bits):
    """
    Menerapkan efek posterize pada citra.
    """
    try:
        factor = 2 ** (8 - bits)
        posterized_image = (image // factor) * factor
        return posterized_image
    except Exception as e:
        logger.error(f"Error in posterize_image: {e}")
        raise

def solarize_image(image, threshold):
    """
    Menerapkan efek solarize pada citra.
    """
    try:
        solarized_image = np.where(image < threshold, image, 255 - image)
        return solarized_image.astype(np.uint8)
    except Exception as e:
        logger.error(f"Error in solarize_image: {e}")
        raise

def clahe_image(image, clip_limit, tile_grid_size):
    """
    Menerapkan CLAHE pada citra.
    """
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_image
    except Exception as e:
        logger.error(f"Error in clahe_image: {e}")
        raise

def adjust_gamma_image(image, gamma):
    """
    Menerapkan gamma adjustment pada citra.
    """
    try:
        if gamma == 0:
            raise ValueError("gamma must be non-zero")

        normalized = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        abs_gamma = abs(float(gamma))
        inv_gamma = 1.0 / abs_gamma

        if gamma > 0:
            mapped = normalized ** inv_gamma
        else:
            # Support negative gamma values without undefined 0**negative
            # operations by applying the same transform to the inverted range.
            mapped = 1.0 - ((1.0 - normalized) ** inv_gamma)

        table = np.clip(mapped * 255.0, 0, 255).astype("uint8")
        adjusted_image = cv2.LUT(image, table)
        return adjusted_image
    except Exception as e:
        logger.error(f"Error in adjust_gamma_image: {e}")
        raise
