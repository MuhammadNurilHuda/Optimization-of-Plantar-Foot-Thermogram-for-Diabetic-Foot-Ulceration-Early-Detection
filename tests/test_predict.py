# tests\test_predict.py

import unittest
import os
import shutil
import numpy as np
import pandas as pd
import joblib
from skimage.io import imsave
import tensorflow as tf
from src.predict import predict_new_data
from src.models.model1 import create_model

class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "tests/temp_pred"
        cls.img_dir = os.path.join(cls.test_dir, "images")
        cls.model_path = os.path.join(cls.test_dir, "test_model.keras")
        cls.scaler_path = os.path.join(cls.test_dir, "test_scaler.joblib")
        cls.output_path = os.path.join(cls.test_dir, "predictions.csv")
        
        # Buat direktori
        os.makedirs(os.path.join(cls.img_dir, "Left/CG Left"), exist_ok=True)
        os.makedirs(os.path.join(cls.img_dir, "Right/CG Right"), exist_ok=True)
        os.makedirs(os.path.join(cls.img_dir, "Left/DM Left"), exist_ok=True)
        os.makedirs(os.path.join(cls.img_dir, "Right/DM Right"), exist_ok=True)
        
        # Buat dummy images dengan naming yang benar
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # CG100 dengan gender M (numeric: 1)
        imsave(os.path.join(cls.img_dir, "Left/CG Left/CG100_M_L.png"), arr)
        imsave(os.path.join(cls.img_dir, "Right/CG Right/CG100_M_R.png"), arr)
        
        # DM101 dengan gender F (numeric: 0)
        imsave(os.path.join(cls.img_dir, "Left/DM Left/DM101_F_L.png"), arr)
        imsave(os.path.join(cls.img_dir, "Right/DM Right/DM101_F_R.png"), arr)
        
        # Buat dummy model
        model = create_model((224, 224, 3), 13)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.save(cls.model_path)
        
        # Buat dummy scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        dummy_data = np.random.randn(10, 13)
        scaler.fit(dummy_data)
        joblib.dump(scaler, cls.scaler_path)
    
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_predict_new_data(self):
        test_data = pd.DataFrame({
            'Subject': ['CG100', 'DM101'],
            'Gender': ['M', 'F'],
            'General_right': [20.0, 25.0],
            'LCA_right': [21.0, 26.0],
            'LPA_right': [22.0, 27.0],
            'MCA_right': [23.0, 28.0],
            'MPA_right': [24.0, 29.0],
            'TCI_right': [25.0, 30.0],
            'General_left': [20.0, 25.0],
            'LCA_left': [21.0, 26.0],
            'LPA_left': [22.0, 27.0],
            'MCA_left': [23.0, 28.0],
            'MPA_left': [24.0, 29.0],
            'TCI_left': [25.0, 30.0]
        })
        
        features = ['Gender', 'General_right', 'LCA_right', 'LPA_right', 
                   'MCA_right', 'MPA_right', 'TCI_right', 'General_left', 
                   'LCA_left', 'LPA_left', 'MCA_left', 'MPA_left', 'TCI_left']
        
        result = predict_new_data(
            data_tabular=test_data,
            image_dir=self.img_dir,
            target_size=(224, 224),
            features_tabular=features,
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            save_output_path=self.output_path
        )
        
        # Verifikasi
        self.assertEqual(len(result), 2)
        self.assertIn('Subject', result.columns)
        self.assertIn('pred_prob', result.columns)
        self.assertIn('pred_label', result.columns)
        self.assertTrue(os.path.exists(self.output_path))

if __name__ == '__main__':
    unittest.main()