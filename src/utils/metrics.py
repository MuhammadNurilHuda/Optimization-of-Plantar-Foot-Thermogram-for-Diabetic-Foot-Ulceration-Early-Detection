# src\utils\metrics.py

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
roc_curve,
roc_auc_score
)

logger = logging.getLogger('src.utils.metrics')

def evaluate_model(
    model,
    X_train_left, X_train_right, X_train_tabular,
    y_train,
    X_test_left, X_test_right, X_test_tabular,
    y_test,
    model_name="UnnamedModel",
    history=None):
    """
    Mengevaluasi model pada data training & test, menampilkan metrik dan grafik.
    """
    try:
        # Evaluasi test set
        test_loss, test_acc = model.evaluate(
            [X_test_left, X_test_right, X_test_tabular],
            y_test,
            verbose=0)
        logger.info(f"{model_name} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

        # Evaluasi training set
        train_loss, train_acc = model.evaluate(
            [X_train_left, X_train_right, X_train_tabular],
            y_train,
            verbose=0)
        logger.info(f"{model_name} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        # Mengukur waktu inferensi
        start_time = time.time()
        y_pred_prob = model.predict([X_test_left, X_test_right, X_test_tabular])
        total_inference_time = time.time() - start_time
        n_samples = len(y_test)
        if n_samples > 0:
            logger.info(f"{model_name} | Inference {n_samples} sampel: {total_inference_time:.4f} s")
            logger.info(f"{model_name} | Inference per sampel: {(total_inference_time/n_samples):.6f} s")
        else:
            logger.warning(f"{model_name} | Tidak ada sampel test (n=0).")

        # Konversi probabilitas -> label biner
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Hitung metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logger.info(f"{model_name} | Accuracy : {acc:.4f}")
        logger.info(f"{model_name} | Precision: {prec:.4f}")
        logger.info(f"{model_name} | Recall   : {rec:.4f}")
        logger.info(f"{model_name} | F1-score : {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"{model_name} | Confusion Matrix:\n{cm}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title(f"{model_name} | Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.show()

        # ROC & AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        logger.info(f"{model_name} | ROC AUC : {roc_auc:.4f}")

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC (AUC={roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} | ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

        if history is not None:
            plot_history(history, model_name)

    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        raise
        
def plot_history(history, model_name="ModelHistory"):
    """
    Menampilkan grafik akurasi & loss dari history training.
    """
    try:
        acc = history.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', [])
        loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])

        if not acc or not loss:
            logger.warning(f"{model_name} | Data akurasi/loss tidak ditemukan.")
            return

        epochs = range(len(acc))

        plt.figure(figsize=(12, 5))

        # Plot akurasi
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label="Train Accuracy")
        if val_acc:
            plt.plot(epochs, val_acc, label="Val Accuracy")
        plt.legend(loc="lower right")
        plt.title(f"{model_name} | Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label="Train Loss")
        if val_loss:
            plt.plot(epochs, val_loss, label="Val Loss")
        plt.legend(loc="upper right")
        plt.title(f"{model_name} | Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error in plot_history: {e}")
        raise