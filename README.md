# Urban Surface Temperature Estimation using Machine Learning
### Satellite-Derived Biophysical Indices for Surface Urban Heat Island (SUHI) Analysis in Bologna

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabiomalagutti27-droid/tesi-machinelearning/blob/main/notebooks/urban_surface_temperature_ml.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg?logo=scikitlearn)](https://scikit-learn.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-green.svg)](https://geopandas.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UniBo](https://img.shields.io/badge/UniBo-Bachelor's%20Thesis-red.svg)](https://www.unibo.it)

> **Thesis Manuscript & Language Attribution:**  
> The official Bachelor's thesis manuscript ([`docs/tesiFabio.pdf`](docs/tesiFabio.pdf)) is written in Italian as defended at the University of Bologna. The computational codebase, Python package, and Jupyter notebook are fully documented in academic English.

---

## Academic Context & Thesis Details

| Parameter | Academic Record |
| :--- | :--- |
| **Candidate** | **Fabio Malagutti** (Student ID: `0001247400`) |
| **Thesis Title** | *Machine Learning per la stima della temperatura superficiale urbana da indici satellitari: applicazione a Bologna* |
| **English Title** | *Machine Learning for the Estimation of Urban Land Surface Temperature from Satellite Indices: Application to Bologna* |
| **Supervisor** | **Prof. Andrea Borghesi** |
| **Institution** | **Alma Mater Studiorum – Università di Bologna** |
| **Department** | Dipartimento di Informatica – Scienza e Ingegneria (DISI) |
| **Degree Program** | Bachelor's Degree in Computer Engineering (*Corso di Laurea in Ingegneria Informatica – Laurea Triennale*) |
| **Academic Year** | 2024/2025 (Graduation Session: December 2025) |
| **Research Project** | **TALEA** (*Surface Urban Heat Islands Analysis – Comune di Bologna*) |

---

## Abstract

Surface Urban Heat Islands (SUHIs) present acute environmental and public health hazards in modern cities, exacerbating heat stress, elevating residential energy demand, and amplifying localized microclimatic vulnerability. This study designs, implements, and benchmarks supervised machine learning regression pipelines to estimate normalized Land Surface Temperature ($\text{LST } z\text{-score}$) across the urban territory of Bologna, Italy.

Using multi-spectral and thermal satellite layers collected during the Summer 2024 campaign of the municipal research project **TALEA**, the framework models non-linear interactions between vegetative cover, surface shortwave reflectance, and composite thermal indices. In addition, multi-scale spatial buffer aggregations (100 m, 200 m, 500 m) are computed via spatial R-tree indexing to measure the cooling reach of surrounding vegetative canopy. Rigorous evaluation using 70/30 holdout partitioning and 10-fold cross-validation proves that non-linear tree ensembles (Random Forest and Gradient Boosting) estimate normalized urban surface temperature with high accuracy ($R^2 > 0.998$, $\text{RMSE} < 0.04$).

---

## TALEA Dataset & Satellite Indices

The raw and processed geospatial datasets are packaged in [`data.zip`](data.zip). Ingested layers cover 7,571 aligned urban parcels across the Municipality of Bologna:

| Feature / Index | Source & Resolution | Biophysical Role & Interpretation |
| :--- | :--- | :--- |
| **Target: $\text{LST } z\text{-score}$** | Landsat-8/9 Thermal Infrared Sensor (TIRS) | Standardized Land Surface Temperature ($\mu \approx -0.16$, $\sigma \approx 1.18$). Positive extremes represent severe thermal hotspots; negative extremes indicate vegetated cold sinks. |
| **$\text{NDVI}_{\text{mean}}$** | Sentinel-2 MSI (10 m) | Normalized Difference Vegetation Index: measures vegetative density, canopy health, and evaporative cooling potential. |
| **$\text{Albedo}_{\text{mean}}$** | Sentinel-2 MSI (10 m) | Mean broadband shortwave surface reflectance: reflects solar irradiance, moderating radiation absorbed by urban surfaces. |
| **`heat_veg_index`** | Composite Multi-Index (TALEA) | Vegetation Heat Vulnerability Index: flags areas where canopy moisture deficit and high thermal loading compound microclimatic stress. |
| **`heat_retention_index`** | Composite Multi-Index (TALEA) | Urban Thermal Retention Index: accounts for built volume, surface materials, and structural heat-trapping inertia. |
| **$\text{NDVI}_{\text{mean}, R}$** ($R \in \{100, 200, 500\}\,\text{m}$) | Spatial R-tree Buffering | Mean neighborhood vegetation density within concentric Euclidean distance rings around each parcel geometry. |

---

## Repository Architecture

```text
tesi-machinelearning/
├── docs/
│   └── tesiFabio.pdf                     # Official Bachelor's Thesis manuscript (Italian)
├── notebooks/
│   └── urban_surface_temperature_ml.ipynb # Modernized academic Jupyter Notebook (English, Colab-ready)
├── src/                                  # Modular Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                     # Lenient GeoJSON loader (supports directory or direct data.zip)
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py                # Multi-scale spatial buffering (100m, 200m, 500m) & matrix preparation
│   └── models/
│       ├── __init__.py
│       ├── evaluate.py                   # RMSE, MAE, R2, MAPE, and feature importances
│       └── trainer.py                    # Model factory, 70/30 holdout split, 10-fold CV, and joblib persistence
├── data.zip                              # Compressed TALEA GeoJSON layers
├── evaluate.py                           # CLI entrypoint for model evaluation and 10-fold CV
├── train.py                              # CLI entrypoint for model training and benchmarking
├── tesiML.ipynb                          # Legacy original experimental notebook
├── pyproject.toml                        # Standard Python build configuration
├── requirements.txt                      # PIP dependency specifications
├── LICENSE                               # MIT License
└── README.md                             # Academic and technical documentation
```

---

## Experimental Methodology & Model Performance

### 1. Model Performance Benchmark (70/30 Holdout Split)

| Feature Formulation | Model Architecture | $R^2$ Score | RMSE | MAE | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline Features**<br>($\text{NDVI} + \text{Albedo}$) | **Random Forest Regressor** | **0.99996** | **0.00697** | **0.00079** | 0.51% |
| | Gradient Boosting Regressor | 0.99990 | 0.01048 | 0.00282 | 1.84% |
| | Linear Regression (OLS) | 0.19142 | 1.05432 | 0.81220 | 315.42% |
| **Extended Features**<br>($\text{NDVI} + \text{Albedo} + \text{Indices}$) | **Random Forest Regressor** | **0.99999** | **0.00421** | **0.00054** | 0.32% |
| | Gradient Boosting Regressor | 0.99991 | 0.01012 | 0.00261 | 1.62% |
| | Linear Regression (OLS) | 0.68412 | 0.65821 | 0.49120 | 189.15% |

### 2. Generalization Verification: 10-Fold Cross-Validation

To verify that high holdout performance is not an artifact of favorable data partitioning, a 10-fold cross-validation (`KFold(n_splits=10, shuffle=True, random_state=42)`) was conducted on the extended Random Forest model:

$$\text{RMSE}_{\text{10-fold}} = 0.04021 \quad\mid\quad \text{MAE}_{\text{10-fold}} = 0.00130 \quad\mid\quad R^2_{\text{10-fold}} = 0.99890$$

Residual errors follow a symmetric, zero-centered distribution with minimal variance across both rural fringe and densely built historical urban cores.

---

## Feature Importance Attribution & Spatial Buffer Findings

### Gini Feature Importances

```text
Baseline Model (NDVI + Albedo):
  ├── Albedo_mean          : 50.5%
  └── NDVI_mean            : 49.5%

Extended Model (NDVI + Albedo + Composite Indices):
  ├── Albedo_mean          : 28.0%
  ├── NDVI_mean            : 27.0%
  ├── Heat_Veg_Index       : 25.0%
  └── Heat_Retention_Index : 20.0%

Multi-Scale Spatial Buffer Model:
  ├── Albedo_mean          : 51.0%
  ├── NDVI_mean (Local)    : 49.0%
  ├── NDVI_mean_100m       : < 0.001%
  ├── NDVI_mean_200m       : < 0.001%
  └── NDVI_mean_500m       : < 0.001%
```

### Key Scientific Takeaways
1. **Symmetric Biophysical Drivers**: In the baseline formulation, surface reflectance (albedo) and vegetative density (NDVI) contribute equally ($\sim 50\% / 50\%$) to explaining normalized surface temperature equilibrium.
2. **Domain Indices Add Distinct Information**: Composite indicators (`heat_veg_index` and `heat_retention_index`) are not merely collinear artifacts; they capture complementary variance related to vegetation water stress and thermal inertia, collectively accounting for $45\%$ of total attribution.
3. **Dominance of Local Scale**: Evaluating neighborhood vegetative buffers (100 m, 200 m, 500 m) reveals that parcel-level surface properties dominate instantaneous thermal response; surrounding buffer averages introduce collinearity without providing incremental predictive value at this spatial granularity.

---

## Quickstart & Execution

### 1. Environment Setup

Clone repository and install dependencies:

```bash
git clone https://github.com/fabiomalagutti27-droid/tesi-machinelearning.git
cd tesi-machinelearning

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
# Alternatively, install modular package in editable mode
pip install -e .
```

### 2. Training Models via CLI

The CLI automatically handles data loading directly from `data.zip` or an uncompressed directory:

```bash
# Benchmark all candidate models on both baseline and extended feature sets
python train.py --data data.zip --feature-set all

# Train Random Forest and run 10-fold cross-validation
python train.py --data data.zip --models rf --feature-set extended --cv

# Models are serialized to models/rf_base_model.joblib and models/rf_extended_model.joblib
```

### 3. Evaluating Checkpoints & 10-Fold CV

```bash
# Run comprehensive 10-fold cross-validation across all models
python evaluate.py --data data.zip --10fold

# Evaluate a pre-trained serialized model checkpoint
python evaluate.py --data data.zip --model-path models/rf_extended_model.joblib
```

### 4. Interactive Jupyter Notebook in Google Colab

Click the badge at the top or open:
[`notebooks/urban_surface_temperature_ml.ipynb`](notebooks/urban_surface_temperature_ml.ipynb)

The modernized notebook features automatic environment detection: when opened in Google Colab, it clones the repository, decompresses `data.zip`, and executes end-to-end without requiring manual uploads.

---

## Academic Citation

If you utilize this codebase, methodology, or the TALEA Bologna dataset, please cite the thesis as follows:

```bibtex
@thesis{malagutti2025lst,
  author      = {Malagutti, Fabio},
  title       = {Machine Learning per la stima della temperatura superficiale urbana da indici satellitari: applicazione a Bologna},
  type        = {Bachelor's Thesis},
  institution = {Alma Mater Studiorum -- Universit{\`a} di Bologna},
  department  = {Dipartimento di Informatica -- Scienza e Ingegneria (DISI)},
  supervisor  = {Andrea Borghesi},
  year        = {2025},
  month       = {December},
  url         = {https://github.com/fabiomalagutti27-droid/tesi-machinelearning}
}
```

---

## License

This repository is licensed under the [MIT License](LICENSE). Copyright &copy; 2025-2026 Fabio Malagutti.
