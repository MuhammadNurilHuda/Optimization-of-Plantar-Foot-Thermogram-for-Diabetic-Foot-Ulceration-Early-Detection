# tests/test_training.py

import unittest
import numpy as np
import os
import shutil
from src.training import train_single_experiment
from src.models.model1 import create_model as create_model1

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.test_model_path = "tests/temp_model.keras"  
        self.test_scaler_path = "tests/temp_scaler.joblib"
        self.test_history_path = "tests/temp_history.csv"
        
    def tearDown(self):
        # Cleanup
        for path in [self.test_model_path, self.test_scaler_path, self.test_history_path]:
            if os.path.exists(path):
                os.remove(path)
                
    def test_train_dummy(self):
        # Create dummy data
        n_samples = 10
        X_train_left = np.random.rand(n_samples, 224, 224, 3)
        X_train_right = np.random.rand(n_samples, 224, 224, 3)
        X_train_tabular = np.random.rand(n_samples, 13)
        y_train = np.random.randint(0, 2, n_samples)
        
        X_val_left = np.random.rand(4, 224, 224, 3)
        X_val_right = np.random.rand(4, 224, 224, 3)
        X_val_tabular = np.random.rand(4, 13)
        y_val = np.random.randint(0, 2, 4)
        
        model, history, train_time = train_single_experiment(
            X_train_left, X_train_right, X_train_tabular, y_train,
            X_val_left, X_val_right, X_val_tabular, y_val,
            create_model1, "test_model",
            self.test_model_path,
            self.test_scaler_path,
            self.test_history_path,
            None,  # no tensorboard
            batch_size=4,
            epochs=2,
            learning_rate=0.001,
            random_seed=42
        )
        
        # Verifikasi
        self.assertTrue(os.path.isfile(self.test_model_path))
        self.assertTrue(os.path.isfile(self.test_history_path))
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        self.assertGreater(train_time, 0)

if __name__ == '__main__':
    unittest.main()