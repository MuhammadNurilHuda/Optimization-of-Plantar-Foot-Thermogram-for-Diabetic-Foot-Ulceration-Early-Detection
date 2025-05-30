# tests\test_data_split.py

import unittest
import numpy as np
from src.data.data_split import split_data

class TestDataSplit(unittest.TestCase):
    def setUp(self):
        n = 10
        self.left = np.random.rand(n, 16, 16)
        self.right = np.random.rand(n, 16, 16)
        self.tabular = np.random.rand(n, 4)
        self.y = np.array([0, 1] * 5)

    def test_split_shape(self):
        res = split_data(self.left, self.right, self.tabular, self.y, test_size=0.3, random_state=42)
        (xtrain_l, xtest_l, xtrain_r, xtest_r, xtrain_tab, xtest_tab, y_tr, y_te) = res
        n_train = int(10 * (1-0.3))
        n_test = 10 - n_train
        self.assertEqual(xtrain_l.shape[0], n_train)
        self.assertEqual(xtest_l.shape[0], n_test)
        self.assertEqual(xtrain_r.shape[0], n_train)
        self.assertEqual(xtrain_tab.shape[0], n_train)
        self.assertEqual(y_tr.shape[0], n_train)
        self.assertEqual(y_te.shape[0], n_test)

if __name__ == '__main__':
    unittest.main()