# tests\test_metrics.py

import unittest, os, sys
from unittest.mock import MagicMock, patch
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Menambahkan direktori proyek utama ke sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.utils import evaluate_model, plot_history


class TestMetrics(unittest.TestCase):

    def setUp(self):
        """
        Metode yang dijalankan sebelum setiap test.
        Kita akan membuat mock model dan data yang dibutuhkan.
        """
        # Mock model Keras
        self.model = MagicMock()
        # Mock perilaku model.evaluate
        # Misal, kembalikan (loss, accuracy) = (0.1234, 0.5678)
        self.model.evaluate.side_effect = [
            (0.1234, 0.5678),  # Hasil evaluasi test set
            (0.1111, 0.6000)   # Hasil evaluasi train set
        ]
        # Mock perilaku model.predict
        # Kembalikan probabilitas untuk data test
        # Misalnya, jika ada 5 sample di test, kita kembalikan 5 probabilitas
        self.model.predict.return_value = np.array([[0.2], [0.7], [0.4], [0.9], [0.55]])

        # Buat data tiruan
        # X_test_left dsb. kita asumsikan punya 5 data sample
        # demi kesederhanaan kita tidak peduli shape detail, hanya len => 5
        self.X_test_left = np.zeros((5, 10, 10, 3))
        self.X_test_right = np.zeros((5, 10, 10, 3))
        self.X_test_tabular = np.zeros((5, 4))
        self.y_test = np.array([0, 1, 0, 1, 1])  # Label biner
        
        # X_train dummy
        self.X_train_left = np.zeros((5, 10, 10, 3))
        self.X_train_right = np.zeros((5, 10, 10, 3))
        self.X_train_tabular = np.zeros((5, 4))
        self.y_train = np.array([1, 0, 0, 1, 0])
        
        # Buat dummy history jika diperlukan oleh plot_history
        self.dummy_history = MagicMock()
        self.dummy_history.history = {
            'accuracy': [0.5, 0.6, 0.7],
            'val_accuracy': [0.45, 0.58, 0.65],
            'loss': [1.0, 0.9, 0.8],
            'val_loss': [1.1, 0.95, 0.85]
        }

    @patch("matplotlib.pyplot.show")
    def test_evaluate_model(self, mock_plt_show):
        """
        Uji fungsi evaluate_model dengan data tiruan dan model mock.
        """
        # Panggil evaluate_model
        evaluate_model(
            model=self.model,
            X_train_left=self.X_train_left,
            X_train_right=self.X_train_right,
            X_train_tabular=self.X_train_tabular,
            y_train=self.y_train,
            X_test_left=self.X_test_left,
            X_test_right=self.X_test_right,
            X_test_tabular=self.X_test_tabular,
            y_test=self.y_test,
            model_name="ModelTest",
            history=self.dummy_history
        )
        # Pastikan model.evaluate dipanggil 2 kali (untuk test dan train)
        self.assertEqual(self.model.evaluate.call_count, 2)
        # Pastikan model.predict juga dipanggil 1 kali
        self.model.predict.assert_called_once()
        # Karena plt.show di-patch, tidak akan tampil plot

    @patch("matplotlib.pyplot.show")
    def test_plot_history(self, mock_plt_show):
        """
        Uji fungsi plot_history secara terpisah.
        """
        # Panggil plot_history dengan history dummy
        plot_history(self.dummy_history, model_name="PlotHistoryTest")
        # plt.show sudah di-patch sehingga tidak membuka jendela plot
        # Tidak ada assert detail, karena kita hanya cek bahwa fungsi jalan tanpa error

    @patch("matplotlib.pyplot.show")
    def test_plot_history_no_data(self, mock_plt_show):
        """
        Uji plot_history ketika history kosong atau tidak temukan key 'accuracy'/'loss'.
        """
        empty_history = MagicMock()
        empty_history.history = {}
        # Tidak seharusnya error walau data history tidak lengkap
        plot_history(empty_history, model_name="NoDataTest")

if __name__ == '__main__':
    unittest.main()