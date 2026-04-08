# src\evaluation.py

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger("src.evaluation")

def evaluate_model(
    model,
    X_test_left, X_test_right, X_test_tabular, y_test,
    save_eval_path: str = None,
    measure_inference_time: bool = True
) -> dict:
    """
    Evaluasi model di test set. Return dict hasil, log output, serta (opsional) simpan ke CSV.
    """
    logger.info("Evaluasi model pada test set...")
    y_pred_prob = model.predict([X_test_left, X_test_right, X_test_tabular])
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
        logger.warning("ROC AUC tidak dapat dihitung karena hanya ada satu kelas pada y_test.")
    logger.info(f"Hasil evaluasi: {metrics}")

    if measure_inference_time:
        import time
        if len(y_test) == 0:
            raise ValueError("y_test tidak boleh kosong saat evaluasi.")
        idx = np.random.choice(len(y_test))
        sample = ([X_test_left[[idx]], X_test_right[[idx]], X_test_tabular[[idx]]])
        start = time.time()
        _ = model.predict(sample)
        elapsed = time.time() - start
        metrics["inference_time_one_sample_sec"] = elapsed
        logger.info(f"Inference time for one sample: {elapsed:.6f} sec")

    # Simpan ke CSV jika diinginkan
    if save_eval_path is not None:
        eval_df = pd.DataFrame([metrics])
        eval_df.to_csv(save_eval_path, index=False)
        logger.info(f"Hasil evaluasi disimpan di {save_eval_path}")

    return metrics
