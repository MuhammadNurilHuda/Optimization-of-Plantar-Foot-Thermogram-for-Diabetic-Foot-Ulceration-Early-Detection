# Optimization of Plantar Foot Thermogram for Diabetic Foot Ulceration Early Detection: An Image Enhancement Approach


## Overview

This repository contains the research codebase for a deep learning pipeline that supports early detection of diabetic foot ulceration (DFU) from plantar thermogram images and tabular plantar temperature measurements.

The project combines:

- Bilateral plantar thermogram images (left and right foot)
- Structured plantar temperature features
- Multiple image enhancement strategies
- Multi-input deep learning models for binary classification

The study goal is to improve the quality of thermal pattern representation before training, while preserving clinically relevant thermal information for DFU-related risk discrimination.

## Research Context

Diabetes mellitus can lead to severe complications, including diabetic foot ulcers. In this work, thermographic imaging is used as a non-invasive modality for early risk detection. The experimental pipeline evaluates whether image enhancement improves the downstream classification performance of a multimodal model that integrates:

- convolutional neural networks (CNNs) for left and right plantar thermograms, and
- dense neural layers for tabular plantar temperature features.

Based on the reported study results associated with this repository:

- solarize was the most effective enhancement strategy across the tested methods,
- the best models reached `97.06%` accuracy, and
- the best reported AUC was `1.000`.

These findings support the use of enhancement-aware thermogram preprocessing for improving both predictive performance and computational efficiency in DFU screening experiments.

## Dataset

The repository is built around the plantar thermogram dataset referenced in the project abstract and local data files. The codebase currently expects:

- raw thermogram assets under [data/raw](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/data/raw)
- an analysis table under [data/external/Plantar Thermogram Data Analysis.csv](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/data/external/Plantar%20Thermogram%20Data%20Analysis.csv)

The local raw dataset structure includes control (`CG...`) and diabetes (`DM...`) subjects, with left and right plantar images named using the convention:

```text
<Subject>_<Gender>_L.png
<Subject>_<Gender>_R.png
```

where `Gender` is encoded as `M` or `F` in file names.

## Method Summary

### Inputs

- Left plantar thermogram image
- Right plantar thermogram image
- 13 tabular features:
  - `Gender`
  - `General_right`, `LCA_right`, `LPA_right`, `MCA_right`, `MPA_right`, `TCI_right`
  - `General_left`, `LCA_left`, `LPA_left`, `MCA_left`, `MPA_left`, `TCI_left`

### Classification setup

- Binary label creation is subject based:
  - `DM* -> 1`
  - `CG* -> 0`
- Images are resized and normalized to `[0, 1]`
- Tabular features are standardized with `StandardScaler`

### Image enhancement methods implemented

- `CLAHE`
- `Gamma`
- `Posterize`
- `Solarize`

The currently configured parameter grid in [configs/config.yaml](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/configs/config.yaml) is:

- `CLAHE`: `2.0_(8, 8)`, `3.0_(6, 12)`, `3.0_(8, 8)`, `3.0_(16, 16)`
- `Gamma`: `-1.5`, `0.5`, `1.5`, `2`, `5`
- `Posterize`: `1`, `2`, `3`
- `Solarize`: `64`, `128`, `192`

### Model family

The repository defines four multi-input neural network variants in [src/models](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/models):

- `model1`: shallow bilateral CNN + tabular branch
- `model2`: deeper bilateral CNN + deeper tabular branch
- `model3`: deepest bilateral CNN among the current variants
- `model4`: asymmetric left/right image branches with deeper tabular branch

All four models output a single sigmoid score for binary classification.

## Repository Structure

```text
.
├── configs/                  Experiment and logging configuration
├── data/                     Raw and external data tracked in the repository
├── notebooks/                Research notebooks and exploratory experiments
├── scripts/                  Shell helpers for train / predict / evaluate
├── src/
│   ├── apply_image_enhancement.py
│   ├── evaluation.py
│   ├── experiment_runner.py
│   ├── predict.py
│   ├── training.py
│   ├── data/
│   ├── models/
│   └── utils/
├── tests/                    Unit tests for core pipeline components
├── main.py                   Main experiment entry point
├── test_run.py               Generates a reduced experiment config
└── README.md
```

## Core Pipeline

### 1. Image enhancement

[src/apply_image_enhancement.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/apply_image_enhancement.py) applies the implemented enhancement methods to image directories and saves enhanced outputs by method and parameter setting.

### 2. Data loading and preprocessing

- [src/data/image_loader.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/data/image_loader.py) loads paired left/right images using dataset naming conventions
- [src/data/tabular_preprocess.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/data/tabular_preprocess.py) converts gender, creates labels, and scales tabular data
- [src/data/data_split.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/data/data_split.py) splits multimodal arrays into train/test partitions

### 3. Training

[src/training.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/training.py) compiles a chosen architecture, trains it with early stopping, and stores model/history artifacts.

### 4. Evaluation

[src/evaluation.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/evaluation.py) computes accuracy, precision, recall, F1, ROC AUC, and single-sample inference time.

### 5. Grid experiment execution

[src/experiment_runner.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/experiment_runner.py) iterates through enhancement settings and model architectures, then saves per-run and aggregated summaries.

### 6. Inference

[src/predict.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/src/predict.py) loads a trained model and tabular scaler, preprocesses new cases, and exports predictions to CSV.

## Environment Setup

### Requirements

The project depends on:

- Python 3.9+
- TensorFlow
- scikit-learn
- pandas
- numpy
- OpenCV
- scikit-image
- matplotlib
- seaborn
- PyYAML
- joblib
- pytest

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Running the Project

### End-to-end order from raw data

If you are starting from the repository's raw dataset, use the following order.

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run dataset preprocessing once.
4. Run a reduced experiment to verify the pipeline.
5. Run the full experiment grid.

### 1. Create and activate the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Preprocess the dataset

```bash
python preprocess_dataset.py
```

This step creates the required `data/processed` artifacts automatically. You do not need to create the directories manually.

The preprocessing script generates:

- `data/processed/[Preprocessed]Plantar Thermogram Data Analysis.csv`
- `data/processed/images_per_part/...`
- `data/processed/resized_images/...`
- `data/processed/image enhancement/...`

What this step does:

- reads the raw tabular CSV from `data/external/Plantar Thermogram Data Analysis.csv`
- creates the binary `label` column from the subject identifier
- converts `Gender` from `M/F` to `1/0`
- standardizes the plantar temperature features
- copies left/right thermogram images from `data/raw` into the paired folder structure expected by the pipeline
- computes the average image size from the copied dataset
- resizes images into `data/processed/resized_images`
- applies all configured enhancement methods into `data/processed/image enhancement`

If you want to rebuild the processed outputs from scratch, use:

```bash
python preprocess_dataset.py --clean
```

### 4. Generate a reduced test configuration

```bash
python test_run.py
```

This generates [configs/test_config.yaml](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/configs/test_config.yaml), which limits the run to:

- 2 enhancement families
- 1 parameter setting per selected enhancement
- 2 model architectures

for a total of 4 experiments.

### 5. Run the reduced experiment first

```bash
python main.py --config configs/test_config.yaml
```

This is the recommended sanity check before launching the full grid.

### 6. Run the full configured experiment grid

```bash
python main.py --config configs/config.yaml
```

## Testing

Run the current automated tests with:

```bash
python3 -m pytest tests test_run.py -q
```

If `pytest` is unavailable in your environment, install project dependencies first.

## Current Codebase Notes

This repository is a research codebase rather than a packaged clinical application. A few practical expectations are worth making explicit:

- logging is file based and configured through [configs/logging.conf](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/configs/logging.conf),
- the preprocessing workflow is now exposed through [preprocess_dataset.py](/home/nurilhuda3333/projects/Optimization-of-Plantar-Foot-Thermogram-for-Diabetic-Foot-Ulceration-Early-Detection-An-Image-Enhan/preprocess_dataset.py),
- the provided tests focus on core pipeline behavior, not clinical validation.

## Citation

If you use this repository in academic work, cite:

- the original [plantar thermogram dataset](https://ieee-dataport.org/open-access/plantar-thermogram-database-study-diabetic-foot-complications) publication used by the project, and
- the corresponding [paper](https://doi.org/10.59190/stc.v5i2.273) associated with this implementation.
