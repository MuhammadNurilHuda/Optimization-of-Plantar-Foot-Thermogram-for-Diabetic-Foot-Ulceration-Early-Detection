# Optimasi Termogram Telapak Kaki Untuk Deteksi Dini Ulkus Kaki Diabetik: Pendekatan Peningkatan Citra

## Deskripsi Proyek

Proyek ini mengembangkan pipeline Machine Learning yang komprehensif untuk deteksi dini ulkus kaki diabetes (Diabetic Foot Ulcer/DFU) menggunakan citra termogram dan data suhu telapak kaki. Sistem ini menggabungkan Convolutional Neural Network (CNN) untuk memproses citra termogram bilateral (kiri dan kanan) dengan data klinis tabular, menggunakan arsitektur multi-input deep learning.

Dataset yang digunakan merupakan data publik yang diterbitkan oleh Hernandez-Contreras et al. (DOI:10.1109/ACCESS.2019.2951356).

### Fitur Utama Pipeline

- **Grid Search Otomatis**: Eksperimen sistematis dengan 4 metode enhancement × multiple parameters × 4 arsitektur model
- **Dual-Input CNN**: Memproses citra termogram kaki kiri dan kanan secara bersamaan
- **Multi-Modal Learning**: Integrasi data citra dengan 13 fitur suhu tabular
- **Production-Ready**: Modular, scalable, dengan logging dan testing komprehensif
- **Real-time Monitoring**: Track progress eksperimen dan visualisasi hasil

## Struktur Proyek

```
project_root/
├── configs/
│   ├── config.yaml          # Konfigurasi utama eksperimen
│   ├── test_config.yaml     # Konfigurasi untuk test run
│   └── logging.conf         # Konfigurasi logging
├── data/
│   ├── external/            # Data mentah original
│   │   └── Plantar Thermogram Data Analysis.csv
│   └── processed/           # Data yang sudah diproses
│       ├── [Preprocessed]Plantar Thermogram Data Analysis.csv
│       └── image enhancement/
│           ├── CLAHE/
│           │   ├── 2.0_(8, 8)/
│           │   ├── 3.0_(6, 12)/
│           │   └── ...
│           ├── Gamma/
│           ├── Posterize/
│           └── Solarize/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_split.py           # Train-test splitting
│   │   ├── image_loader.py         # Termogram image loader
│   │   └── tabular_preprocess.py   # Tabular data preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model1/                 # Arsitektur model 1
│   │   │   └── model1.py
│   │   ├── model2/                 # Arsitektur model 2
│   │   ├── model3/                 # Arsitektur model 3
│   │   └── model4/                 # Arsitektur model 4
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_enhancement.py    # CLAHE, Gamma, Posterize, Solarize
│   │   └── metrics.py              # Evaluasi dan visualisasi
│   ├── evaluation.py               # Model evaluation
│   ├── experiment_runner.py        # Grid search runner
│   ├── predict.py                  # Inference untuk data baru
│   └── training.py                 # Training pipeline
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_image_enhancement.py
│   ├── test_models.py
│   ├── test_predict.py
│   ├── test_tabular_preprocess.py
│   └── test_training.py
├── logs/
│   ├── tfboard/                    # TensorBoard logs
│   ├── history_csv/                # Training history
│   ├── evaluasi_csv/               # Evaluation results
│   └── grid_experiment_summary.csv # Grid search summary
├── models/                         # Saved models
├── main.py                         # Main entry point
├── monitor_progress.py             # Real-time monitoring
├── analyze_results.py              # Post-experiment analysis
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Architectures

Proyek ini mengimplementasikan 4 arsitektur CNN-Tabular yang berbeda:

- **Model 1**: Basic CNN + Dense layers
- **Model 2**: Deeper CNN with batch normalization
- **Model 3**: CNN with spatial dropout
- **Model 4**: Advanced architecture with regularization

Setiap model menerima 3 input:

1. Citra termogram kaki kiri (224×224×3)
2. Citra termogram kaki kanan (224×224×3)
3. Data tabular 13 fitur suhu

## Tech Stack

### **Deep Learning Framework**

- **TensorFlow 2.10+** - Primary deep learning framework
- **Keras API** - High-level neural network API
- **CUDA 11.2+** - GPU acceleration support

### **Data Science & ML Libraries**

- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning utilities (train-test split, metrics, preprocessing)
- **Scikit-image** - Image processing and enhancement
- **OpenCV** - Advanced computer vision operations

### **Visualization & Monitoring**

- **Matplotlib** - Static plotting and visualizations
- **Seaborn** - Statistical data visualization
- **TensorBoard** - Real-time training monitoring and visualization

### **Development Tools**

- **Python 3.9+** - Primary programming language
- **Jupyter Notebook** - Exploratory data analysis and prototyping
- **Git** - Version control
- **PyYAML** - Configuration management
- **Joblib** - Model serialization

### **Testing & Quality Assurance**

- **Pytest** - Unit testing framework
- **unittest** - Python standard testing library
- **Logging** - Python logging module for debugging

### **Architecture Patterns**

- **Factory Pattern** - Model creation and management
- **Modular Design** - Separation of concerns (data, models, utils)
- **Pipeline Pattern** - End-to-end ML workflow
- **Grid Search** - Systematic hyperparameter optimization

### **MLOps Features**

- **Experiment Tracking** - Automated logging of all experiments
- **Model Versioning** - Systematic model saving with metadata
- **Reproducibility** - Seed management and configuration tracking
- **Performance Monitoring** - Real-time progress tracking

### **Key Technical Achievements**

- ✅ **Multi-Modal Deep Learning**: Integration of image and tabular data
- ✅ **Bilateral Image Processing**: Simultaneous left-right thermogram analysis
- ✅ **Automated Pipeline**: 60+ experiments with minimal manual intervention
- ✅ **Production-Ready Code**: Modular, tested, and documented
- ✅ **Scalable Architecture**: Easy to add new models or enhancement methods

### **Hardware Requirements**

- **Minimum**: 8GB RAM, NVIDIA GPU with 4GB VRAM
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Storage**: 20GB free space for full experiment

### **Deployment Ready**

- Model serialization in Keras format
- Standardized preprocessing pipeline
- Single-image inference capability
- Error handling and logging

## Image Enhancement Methods

### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)

- Parameters: clip_limit × tile_grid_size
- Configurations: 2.0*(8,8), 3.0*(6,12), 3.0*(8,8), 3.0*(16,16)

### 2. Gamma Adjustment

- Parameters: gamma value
- Configurations: -1.5, 0.5, 1.5, 2, 5

### 3. Posterize

- Parameters: bits
- Configurations: 1, 2, 3

### 4. Solarize

- Parameters: threshold
- Configurations: 64, 128, 192

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/username/diabetic-foot-ulcer-detection.git
cd diabetic-foot-ulcer-detection
```

### 2. Setup Environment

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verifikasi Instalasi

```bash
# Jalankan semua unit tests
python -m pytest tests/ -v
```

## Cara Penggunaan

### 1. Quick Test Run (4 eksperimen)

```bash
# Generate test configuration
python test_run.py

# Jalankan test experiment
python main.py --config configs/test_config.yaml
```

### 2. Full Grid Search (60 eksperimen)

```bash
# Terminal 1: Jalankan experiment
python main.py

# Terminal 2: Monitor progress
python monitor_progress.py

# Terminal 3: TensorBoard (optional)
python launch_tensorboard.py
```

### 3. Analisis Hasil

```bash
# Setelah experiment selesai
python analyze_results.py
```

### 4. Prediksi Data Baru

```bash
python predict_single.py \
    --tabular data_baru.csv \
    --images path/to/new/images \
    --model models/CLAHE/2.0_8,8/model1.keras \
    --scaler models/CLAHE/2.0_8,8/model1_scaler.joblib \
    --output predictions.csv
```

## Monitoring & Visualisasi

### TensorBoard

```bash
tensorboard --logdir logs/tfboard
```

Akses di: http://localhost:6006

### Real-time Progress Monitor

Monitor menampilkan:

- Jumlah eksperimen selesai
- Best accuracy & configuration
- Performance per enhancement method
- Latest experiments

## Hasil Eksperimen

Grid search akan menghasilkan:

- `logs/grid_experiment_summary.csv`: Rangkuman semua eksperimen
- `logs/experiment_analysis.png`: Visualisasi performa
- Model terbaik tersimpan di `models/`

### Metrik Evaluasi

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Inference time

## Tips & Best Practices

### 1. **Memory Management**

- Close aplikasi berat (browser, IDE) saat menjalankan full experiment
- Monitor RAM usage dengan Task Manager/htop
- Jika memory terbatas, kurangi `batch_size` di config.yaml
- Gunakan `del` untuk hapus variable besar yang tidak terpakai

### 2. **Experiment Management**

- **Selalu mulai dengan test run** untuk validasi pipeline
- Backup hasil secara berkala:
  ```bash
  cp -r logs/ logs_backup_$(date +%Y%m%d_%H%M%S)/
  ```
- Simpan config yang digunakan bersama hasil experiment
- Dokumentasikan perubahan hyperparameter

### 3. **Troubleshooting**

- **Jika experiment terputus**: Check `grid_experiment_summary.csv` untuk melihat progress terakhir
- **GPU out of memory**: Kurangi batch_size atau image resolution
- **Slow training**: Pastikan menggunakan GPU dengan `tf.config.list_physical_devices('GPU')`
- **Import errors**: Pastikan menjalankan dari root directory project

### 4. **Performance Optimization**

- Gunakan SSD untuk menyimpan dataset (faster I/O)
- Pre-load images ke memory jika RAM mencukupi
- Enable mixed precision training untuk GPU modern:
  ```python
  tf.keras.mixed_precision.set_global_policy('mixed_float16')
  ```

### 5. **Reproducibility**

- Selalu set random seed di config
- Catat versi library yang digunakan: `pip freeze > requirements_exact.txt`
- Gunakan Git untuk version control
- Tag commit untuk experiment penting

### 6. **Data Handling**

- Verify data integrity sebelum experiment:
  ```bash
  python check_data.py
  ```
- Pastikan tidak ada missing images
- Check data distribution untuk class imbalance
- Backup raw data di lokasi terpisah

### 7. **Model Selection**

- Mulai dengan model sederhana (model1) untuk baseline
- Compare training vs validation metrics untuk detect overfitting
- Save best model berdasarkan validation metric, bukan training

### 8. **Production Deployment**

- Test model dengan single prediction dulu
- Validate input preprocessing sama dengan training
- Monitor inference time untuk real-time requirements
- Implement error handling untuk edge cases

### 9. **Collaboration**

- Gunakan format naming yang konsisten
- Document experiment assumptions dan decisions
- Share results dalam format yang mudah dibaca (CSV, plots)
- Gunakan relative paths, bukan absolute

### 10. **Resource Planning**

- **Estimasi waktu**: ~2-5 jam untuk 60 experiments
- **Disk space**: ~5-10 GB untuk models dan logs
- **Best time to run**: Malam hari atau weekend untuk full experiment
- **Cloud alternative**: Gunakan Google Colab/Kaggle jika resource lokal terbatas
