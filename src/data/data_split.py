# src\data\data_split.py

import logging
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger("src.data.data_split")

def split_data(
    X_left: np.ndarray, X_right: np.ndarray, X_tabular: np.ndarray, y: np.ndarray,
    test_size: float = 0.2, random_state: int = 42
):
    """
    Membagi data ke training dan test set, dengan logging shape.
    """
    try:
        (X_train_left, X_test_left,
         X_train_right, X_test_right,
         X_train_tabular, X_test_tabular,
         y_train, y_test) = train_test_split(
             X_left, X_right, X_tabular, y,
             test_size=test_size,
             random_state=random_state
        )
        logger.info(f"Train shape L: {X_train_left.shape}, R: {X_train_right.shape}, tabular: {X_train_tabular.shape}, y: {y_train.shape}")
        logger.info(f"Test shape  L: {X_test_left.shape},  R: {X_test_right.shape},  tabular: {X_test_tabular.shape},  y: {y_test.shape}")
        return (X_train_left, X_test_left,
                X_train_right, X_test_right,
                X_train_tabular, X_test_tabular,
                y_train, y_test)
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise