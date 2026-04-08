# src\predict.py

import os
import pandas as pd
import joblib
import logging
import tensorflow as tf
from src.data.image_loader import load_termogram_images
from src.data.tabular_preprocess import convert_gender_to_numeric, preprocess_tabular

logger = logging.getLogger("src.predict")

def predict_new_data(
    data_tabular: pd.DataFrame,
    image_dir: str,
    target_size: tuple,
    features_tabular: list,
    model_path: str,
    scaler_path: str,
    save_output_path: str
):
    """Proses data baru (image & tabular) dan prediksi hasilnya, output ke CSV."""
    # Load model dan scaler
    logger.info("Memuat model...")
    
    # Support both .h5 and .keras formats
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    elif os.path.exists(model_path.replace('.h5', '.keras')):
        model = tf.keras.models.load_model(model_path.replace('.h5', '.keras'))
    elif os.path.exists(model_path.replace('.keras', '.h5')):
        model = tf.keras.models.load_model(model_path.replace('.keras', '.h5'))
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
        
    logger.info("Memuat scaler tabular...")
    scaler = joblib.load(scaler_path)
    
    # Preprocess tabular
    logger.info("Preprocessing tabular data")
    data_tabular = convert_gender_to_numeric(data_tabular)
    X_tabular, _ = preprocess_tabular(data_tabular[features_tabular].values, scaler)
    
    # Load + preprocess images
    logger.info("Memuat termogram (Left/Right)...")
    X_left, X_right, valid_indices = load_termogram_images(
        data_tabular,
        image_dir,
        target_size,
        return_valid_indices=True,
    )
    
    # Validasi jumlah sampel
    if len(X_left) == 0 or len(X_right) == 0:
        logger.error("Tidak ada gambar yang berhasil dimuat!")
        raise ValueError("Tidak ada gambar yang berhasil dimuat!")

    data_tabular = data_tabular.iloc[valid_indices].reset_index(drop=True)
    X_tabular = X_tabular[valid_indices]
    
    # Prediksi
    logger.info("Predicting data baru")
    y_pred_prob = model.predict([X_left, X_right, X_tabular])
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Simpan hasil
    result_df = pd.DataFrame({
        "Subject": data_tabular["Subject"].values, 
        "pred_prob": y_pred_prob.flatten(), 
        "pred_label": y_pred
    })
    result_df.to_csv(save_output_path, index=False)
    logger.info(f"Prediksi tersimpan di {save_output_path}")
    return result_df
