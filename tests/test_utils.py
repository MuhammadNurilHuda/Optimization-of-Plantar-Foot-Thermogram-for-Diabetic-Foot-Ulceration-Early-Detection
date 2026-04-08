import unittest
import warnings

import numpy as np

from src.utils.image_enhancement import adjust_gamma_image


class TestImageEnhancementUtils(unittest.TestCase):
    def test_adjust_gamma_image_supports_negative_gamma_without_runtime_warning(self):
        image = np.array(
            [
                [0, 64, 128],
                [192, 224, 255],
            ],
            dtype=np.uint8,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error", RuntimeWarning)
            output = adjust_gamma_image(image, -1.5)

        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(len(caught), 0)


if __name__ == "__main__":
    unittest.main()
