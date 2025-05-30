# tests/test_tabular_preprocess.py

import unittest
import pandas as pd
import numpy as np
from src.data.tabular_preprocess import convert_gender_to_numeric, create_labels, preprocess_tabular

class TestTabularPreprocess(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Subject': ['DM001', 'CG002'],
            'Gender': ['M', 'F'],
            'Age': [60, 55],
            'Height': [170, 160]
        })

    def test_convert_gender(self):
        df2 = convert_gender_to_numeric(self.df.copy())
        self.assertSetEqual(set(df2['Gender']), {1, 0})

    def test_create_labels(self):
        df3 = create_labels(self.df.copy())
        self.assertListEqual(df3['label'].tolist(), [1, 0])

    def test_preprocess_tabular_fit(self):
        arr = self.df[['Age', 'Height']].values
        scaled, scaler = preprocess_tabular(arr)
        # Mean ~0, std ~1
        self.assertAlmostEqual(np.mean(scaled), 0, places=4)
        self.assertAlmostEqual(np.std(scaled), 1, places=4)
        self.assertIsNotNone(scaler)

    def test_preprocess_tabular_transform(self):
        arr = self.df[['Age', 'Height']].values
        scaled, scaler = preprocess_tabular(arr)
        scaled2, _ = preprocess_tabular(arr, scaler=scaler)
        np.testing.assert_array_almost_equal(scaled, scaled2)

if __name__ == '__main__':
    unittest.main()