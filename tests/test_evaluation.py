# tests\test_evaluation.py

import unittest
import numpy as np
from src.models import create_model1
from src.evaluation import evaluate_model

class TestModelEvaluation(unittest.TestCase):
    def test_evaluate_dummy(self):
        # Dummy data & model train sedikit
        batch = 4
        X_l = np.random.rand(batch, 16, 16, 1)
        X_r = np.random.rand(batch, 16, 16, 1)
        X_t = np.random.rand(batch, 3)
        y = np.array([0,1,0,1])
        model = create_model1((16,16,1), 3)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit([X_l, X_r, X_t], y, epochs=1, batch_size=2, verbose=0)
        
        metrics = evaluate_model(model, X_l, X_r, X_t, y)
        self.assertIn('accuracy', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)

if __name__ == '__main__':
    unittest.main()