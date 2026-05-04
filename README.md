# EEEM073 – AI and Sustainability
## Predicting Wildfire Burned Area in Central Africa Using Deep Learning

**Student:** Mir Ishaque Ali (6931803)  
**Module:** EEEM073 – AI and Sustainability, University of Surrey  
**Deadline:** Tuesday 05th May 2026, 16:00 GMT

---

## Project Overview

This project develops an AI pipeline to predict monthly wildfire burned area across Central Africa one month ahead, supporting SDG 13 (Climate Action) and SDG 15 (Life on Land). Three models are compared: a Random Forest baseline, an LSTM (temporal), and a Swin Transformer (spatial). Model compression via pruning and quantisation is applied to evaluate the sustainability of the AI models themselves.

---

## Repository Structure

```
├── 1_Data_Loading_and_Preprocessing.ipynb   # Load ESA Fire_cci NetCDF data, clean, save processed arrays
├── 2_Exploratory_Data_Analysis.ipynb        # EDA, statistics, spatial/temporal visualisations
├── 3_AI_Modelling.ipynb                     # Train Random Forest, LSTM, Swin Transformer
├── 4_Evaluation_and_XAI.ipynb               # Evaluate models, SHAP/GradientExplainer XAI
├── 5_Model_Compression.ipynb                # Pruning, quantisation, carbon footprint analysis
├── README.md                                # This file
└── data/                                    # Dataset files (see Dataset section below)
```

The notebooks are designed to be run **sequentially**. Each notebook saves its outputs (processed data, trained models, results) to disk so the next notebook can load them without re-running earlier steps.

---

## Dataset

**ESA Fire_cci Burned Area v5.1 — MODIS Grid Product**

### Included in this submission

The `data/` folder contains **72 NetCDF files** covering the period **January 2017 – December 2022** (6 years × 12 months), which is the subset used in this project. This corresponds to the training, validation, and test splits used across all notebooks.

### Full dataset

The complete ESA Fire_cci dataset spans **2001–2022**. Only the 2017–2022 subset was used in this project due to computational constraints. The full dataset is freely available from the CEDA Data Catalogue:

> https://catalogue.ceda.ac.uk/uuid/3628cb2fdba443588155e15dee8e5352

To use the full 2001–2022 data, download all NetCDF files from the link above, place them in the `data/` folder, and adjust the date range filter in Notebook 1 accordingly.

### Spatial coverage

Central Africa region: approximately **15°S–15°N, 5°E–45°E**, at 0.25° spatial resolution (~120 × 140 grid cells).

---

## Dependencies

### Python Version

Python **3.9** or higher is recommended. The code was developed and tested on Python 3.10.

### Install all dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas xarray zarr netcdf4 "dask[complete]" matplotlib seaborn scipy scikit-learn cartopy torch shap einops
```

### Notes on specific libraries

- **PyTorch (`torch`):** CPU-only installation is sufficient to reproduce all results. No GPU is required. For CPU-only install: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **Cartopy:** May require additional system dependencies. On Ubuntu/Debian: `sudo apt-get install libgeos-dev libproj-dev`. On Windows, use conda: `conda install -c conda-forge cartopy`
- **SHAP:** Used for TreeExplainer (Random Forest) and GradientExplainer (LSTM) in Notebook 4.
- **timm:** Not required — the Swin Transformer in this project is implemented from scratch using pure PyTorch (`torch.nn`) without any external vision library.

---

## Environment Setup (Recommended)

```bash
# Create environment
python -m venv eeem073_env

# Activate (Linux/Mac)
source eeem073_env/bin/activate

# Activate (Windows)
eeem073_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Alternatively, using conda:

```bash
conda create -n eeem073 python=3.10
conda activate eeem073
conda install -c conda-forge numpy pandas xarray zarr netcdf4 dask matplotlib seaborn scipy scikit-learn cartopy shap einops
pip install torch --index-url https://download.pytorch.org/whl/cpu
jupyter notebook
```

---

## How to Run

Run the notebooks **in order**:

1. **`1_Data_Loading_and_Preprocessing.ipynb`**  
   Loads the ESA NetCDF files from `data/`, extracts burned area and vegetation class features, handles missing values, normalises data, and saves processed arrays to `data/processed/`.

2. **`2_Exploratory_Data_Analysis.ipynb`**  
   Loads processed data. Produces descriptive statistics, spatial maps, seasonal decomposition, correlation matrix, and vegetation class distributions.

3. **`3_AI_Modelling.ipynb`**  
   Trains Random Forest (with RandomizedSearchCV), LSTM (with early stopping), and Swin Transformer. Saves trained models to `models/`.

4. **`4_Evaluation_and_XAI.ipynb`**  
   Loads trained models. Evaluates on 2022 test year (R², MAE, RMSE). Produces spatial residual maps, temporal accuracy plots, SHAP feature importance (RF), and GradientExplainer attribution (LSTM).

5. **`5_Model_Compression.ipynb`**  
   Applies weight pruning and dynamic quantisation to LSTM and Swin Transformer. Compares compressed vs baseline on accuracy, file size, inference speed, and estimated carbon footprint (gCO₂).

Each notebook prints its key results and saves figures to `figures/`.

---

## Reproducing Results

- All random seeds are fixed (`numpy.random.seed(42)`, `torch.manual_seed(42)`) for reproducibility.
- Training was performed on CPU; results should be identical across platforms.
- Expected runtime: Notebooks 1–2 ≈ 5–10 min each; Notebook 3 ≈ 30–60 min (LSTM/Swin training on CPU); Notebooks 4–5 ≈ 10–20 min each.

---

## Acknowledgements

Code structure and example implementations adapted from EEEM073 lab materials (University of Surrey, 2025–2026). External library usage is acknowledged inline in each notebook. SHAP library: Lundberg & Lee (2017). PyTorch: Paszke et al. (2019).
