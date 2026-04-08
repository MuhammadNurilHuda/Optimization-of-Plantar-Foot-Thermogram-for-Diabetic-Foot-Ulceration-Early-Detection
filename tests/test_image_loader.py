# tests/test_image_loader.py

import unittest
import os
import shutil
import pandas as pd
import numpy as np
from skimage.io import imsave

from src.data.image_loader import load_termogram_images

class TestImageLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Buat folder temp
        cls.test_img_dir = "tests/temp_enhancement"
        os.makedirs(cls.test_img_dir, exist_ok=True)
        os.makedirs(os.path.join(cls.test_img_dir, "Left", "CG Left"), exist_ok=True)
        os.makedirs(os.path.join(cls.test_img_dir, "Right", "CG Right"), exist_ok=True)
        # Dummy tabular, 2 subjek
        cls.df = pd.DataFrame({
            "Subject": ["CG001", "CG002"],
            "Gender": ["M", "F"]
        })
        # Buat dummy image 32x32 (uint8) utk tiap side pada setiap subjek
        dummy_img = np.ones((32, 32), dtype=np.uint8) * 100
        imsave(os.path.join(cls.test_img_dir, "Left/CG Left/CG001_M_L.png"), dummy_img)
        imsave(os.path.join(cls.test_img_dir, "Left/CG Left/CG002_F_L.png"), dummy_img)
        imsave(os.path.join(cls.test_img_dir, "Right/CG Right/CG001_M_R.png"), dummy_img)
        imsave(os.path.join(cls.test_img_dir, "Right/CG Right/CG002_F_R.png"), dummy_img)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_img_dir)

    def test_load_termogram_images(self):
        target_size = (16, 16)
        X_left, X_right = load_termogram_images(self.df, self.test_img_dir, target_size)
        self.assertEqual(X_left.shape, (2, 16, 16))
        self.assertEqual(X_right.shape, (2, 16, 16))
        # Check range & type
        self.assertTrue(np.all((X_left >= 0) & (X_left <= 1)))
        self.assertTrue(np.all((X_right >= 0) & (X_right <= 1)))
        self.assertAlmostEqual(np.max(X_left), 100/255.0, delta=0.01)
        self.assertAlmostEqual(np.max(X_right), 100/255.0, delta=0.01)

    def test_load_termogram_images_returns_valid_indices(self):
        target_size = (16, 16)
        df = pd.DataFrame({
            "Subject": ["CG001", "CG404", "CG002"],
            "Gender": ["M", "F", "F"]
        })

        X_left, X_right, valid_indices = load_termogram_images(
            df,
            self.test_img_dir,
            target_size,
            return_valid_indices=True,
        )

        self.assertEqual(X_left.shape[0], 2)
        self.assertEqual(X_right.shape[0], 2)
        self.assertEqual(valid_indices, [0, 2])

if __name__ == "__main__":
    unittest.main()
