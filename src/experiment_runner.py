# src/experiment_runner.py

import os
import yaml
import pandas as pd
import logging
import logging.config
import numpy as np

from src.models import create_model1, create_model2, create_model3, create_model4
from src.data.image_loader import load_termogram_images
from src.data.tabular_preprocess import preprocess_tabular
from src.data.data_split import split_data
from src.training import train_single_experiment
from src.evaluation import evaluate_model
from sklearn.preprocessing import StandardScaler

MODEL_FACTORY = {
    'model1': create_model1,
    'model2': create_model2,
    'model3': create_model3,
    'model4': create_model4,
}

def run_full_grid_experiment(config_path: str):
    logging.config.fileConfig('configs/logging.conf')
    logger = logging.getLogger("src.experiment_runner")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_data_dir = config['data']['base_dir']
    tabular_path = config['data']['tabular_path']
    features = config['data']['tabular_features']
    target_size = tuple(config['data']['input_shape'])

    grid_results = []
    count = 0

    for enhancement, param_list in config['data']['enhancements'].items():
        for param in param_list:
            img_dir = os.path.join(base_data_dir, enhancement, param)
            logger.info(f"Memproses {enhancement} | param: {param}")
            
            # Muat tabular data (sudah di-preprocess)
            data_tabular = pd.read_csv(tabular_path)
            
            # Data sudah memiliki label dan gender sudah numeric
            logger.info("Menggunakan data yang sudah di-preprocess")
            
            # Ambil label
            y = data_tabular['label'].values
            
            # Untuk image loader, kita perlu gender original (M/F)
            # Konversi balik dari numeric ke string untuk nama file
            data_tabular['Gender_Original'] = data_tabular['Gender'].map({1: 'M', 0: 'F'})
            
            # Ambil features untuk tabular (Gender sudah numeric)
            X_tabular_raw = data_tabular[features].values
            
            # Normalize tabular features
            scaler = StandardScaler()
            X_tabular = scaler.fit_transform(X_tabular_raw)
            
            # Muat termogram & preprocess
            X_left, X_right = load_termogram_images(data_tabular, img_dir, target_size)
            
            # Validasi: jumlah sample pada citra & tabular match?
            min_n = min(len(X_left), len(X_tabular))
            if min_n < 10:
                logger.warning(f"Jumlah sample sangat sedikit di {enhancement}-{param}, skip...")
                continue
            X_left, X_right, X_tabular, y = X_left[:min_n], X_right[:min_n], X_tabular[:min_n], y[:min_n]
            
            # Split train/test
            (X_train_left, X_test_left,
             X_train_right, X_test_right,
             X_train_tabular, X_test_tabular,
             y_train, y_test) = split_data(X_left, X_right, X_tabular, y, 
                                           test_size=config['split']['test_size'], 
                                           random_state=config['seed'])
            
            # Loop model
            for model_name in config['model']['architectures']:
                count += 1
                logger.info(f"GRID #{count}: {enhancement}|{param}|{model_name}")
                model_dir = os.path.join(config['model']['save_dir'], enhancement, param)
                os.makedirs(model_dir, exist_ok=True)
                tb_dir = os.path.join(config['logging']['tensorboard_dir'], enhancement, param, model_name)

                model_path = os.path.join(model_dir, f"{model_name}.keras")
                scaler_path = os.path.join(model_dir, f"{model_name}_scaler.joblib")
                hist_path = os.path.join(config['logging']['history_dir'], f"{enhancement}_{param}_{model_name}_history.csv")
                eval_csv = os.path.join(config['logging']['evaluation_dir'], f"{enhancement}_{param}_{model_name}_eval.csv")

                model, history, train_time = train_single_experiment(
                    X_train_left, X_train_right, X_train_tabular, y_train,
                    X_test_left, X_test_right, X_test_tabular, y_test,
                    MODEL_FACTORY[model_name], model_name,
                    model_path, scaler_path, hist_path, tb_dir,
                    batch_size=config['train']['batch_size'],
                    epochs=config['train']['epochs'],
                    learning_rate=config['train']['learning_rate'],
                    random_seed=config['seed']
                )
                
                # Simpan scaler
                import joblib
                joblib.dump(scaler, scaler_path)
                
                # Evaluasi
                metrics = evaluate_model(model, X_test_left, X_test_right, X_test_tabular, y_test, save_eval_path=eval_csv)
                result_row = {
                    "enhancement": enhancement,
                    "param": param,
                    "model": model_name,
                    **metrics,
                    "training_time_sec": train_time
                }
                logger.info(f"Result: {result_row}")
                grid_results.append(result_row)
                
                # Simpan hasil grid experiment
                pd.DataFrame(grid_results).to_csv(config['logging']['final_grid_summary'], index=False)

    logger.info("Seluruh eksperimen grid search selesai!")
    pd.DataFrame(grid_results).to_csv(config['logging']['final_grid_summary'], index=False)
    print("Grid search experiment finished. Result:", config['logging']['final_grid_summary'])