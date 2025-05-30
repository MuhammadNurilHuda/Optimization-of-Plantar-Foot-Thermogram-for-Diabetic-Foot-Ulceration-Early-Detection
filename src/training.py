# src\training.py

import os
import time
import logging
import logging.config
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import joblib

def train_single_experiment(
    X_train_left, X_train_right, X_train_tabular, y_train,
    X_val_left, X_val_right, X_val_tabular, y_val,
    model_builder, model_name: str,
    save_model_path: str,
    save_scaler_path: str = None,
    save_history_path: str = None,
    tensorboard_log_dir: str = None,
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    random_seed: int = 42
):
    logger = logging.getLogger("src.training")
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    logger.info(f"Mulai training: {model_name}")

    input_shape_image = X_train_left.shape[1:]
    input_shape_tabular = X_train_tabular.shape[1]
    model = model_builder(input_shape_image, input_shape_tabular)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = []
    if tensorboard_log_dir:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir))
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True))

    start_time = time.time()
    history = model.fit(
        [X_train_left, X_train_right, X_train_tabular],
        y_train,
        validation_data=([X_val_left, X_val_right, X_val_tabular], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training selesai dalam {training_time:.2f} detik.")

    # Save model - FIX: Remove .replace() karena path sudah benar
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    try:
        # Pastikan menggunakan format yang konsisten
        if save_model_path.endswith('.h5'):
            save_model_path = save_model_path.replace('.h5', '.keras')
        K.clear_session()
        model.save(save_model_path)
        logger.info(f"Model disimpan di {save_model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        # Alternative save method
        try:
            K.clear_session()
            model.save_weights(save_model_path.replace('.keras', '_weights.h5'))
            logger.info(f"Model weights disimpan di {save_model_path.replace('.keras', '_weights.h5')}")
        except Exception as e2:
            logger.error(f"Error saving weights: {e2}")

    # Save training history
    if save_history_path is not None:
        os.makedirs(os.path.dirname(save_history_path), exist_ok=True)
        pd.DataFrame(history.history).to_csv(save_history_path, index=False)
        logger.info(f"Training history disimpan di {save_history_path}")

    return model, history, training_time
