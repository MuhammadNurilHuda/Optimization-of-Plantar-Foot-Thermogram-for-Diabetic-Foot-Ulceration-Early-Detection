# src/data/tabular_preprocess.py

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("src.data.tabular_preprocess")

def convert_gender_to_numeric(data_tabular: pd.DataFrame) -> pd.DataFrame:
    if 'Gender' not in data_tabular.columns:
        logger.error("Kolom 'Gender' tidak ditemukan.")
        raise KeyError("Kolom 'Gender' tidak ditemukan.")
    data_tabular = data_tabular.copy()
    
    # Simpan gender original untuk keperluan load image
    data_tabular['Gender_Original'] = data_tabular['Gender']
    data_tabular['Gender'] = data_tabular['Gender'].map({'M': 1, 'F': 0})
    if data_tabular['Gender'].isna().any():
        invalid_values = sorted(data_tabular.loc[data_tabular['Gender'].isna(), 'Gender_Original'].astype(str).unique())
        logger.error(f"Nilai gender tidak valid: {invalid_values}")
        raise ValueError(f"Unsupported gender values: {invalid_values}")
    
    logger.info("Gender dikonversi menjadi numerik, original disimpan di 'Gender_Original'.")
    return data_tabular

def create_labels(data_tabular: pd.DataFrame, target_col: str = 'Subject') -> pd.DataFrame:
    if target_col not in data_tabular.columns:
        logger.error(f"Kolom '{target_col}' tidak ditemukan.")
        raise KeyError(f"Kolom '{target_col}' tidak ditemukan.")
    data_tabular = data_tabular.copy()
    data_tabular['label'] = data_tabular[target_col].apply(lambda x: 1 if str(x).startswith('DM') else 0)
    logger.info("Label 'label' berhasil dibuat (1 untuk DM, 0 untuk CG).")
    return data_tabular

def preprocess_tabular(data: np.ndarray, scaler: StandardScaler = None) -> (np.ndarray, StandardScaler):
    """
    Melakukan training scaling jika scaler==None, atau transform jika scaler diberikan.
    Return tuple: (scaled, scaler)
    """
    if data.ndim != 2:
        raise ValueError("Tabular data must be a 2D array.")
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)
    logger.info("Tabular data berhasil dinormalisasi.")
    return scaled, scaler
