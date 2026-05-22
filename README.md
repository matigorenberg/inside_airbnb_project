# High-Demand Airbnb Listing Identification in Buenos Aires using Classification Models

This repository contains the code and resources for the Final Integrative Project of the Postgraduate Diploma in Data Science and Advanced Analytics (UTN). The goal is to classify the success potential of Airbnb listings in Buenos Aires (CABA) using Machine Learning techniques, following the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) methodology.

## Business Context

Airbnb hosts often lack data-driven guidance on what actually drives listing success. Decisions around pricing, presentation, and operational management are frequently based on intuition rather than evidence. This project addresses that gap by building a classification model that identifies high-demand listings based on operational and qualitative features — providing actionable insights for hosts, property managers, and short-term rental market analysts. A key finding challenges conventional wisdom: how a listing is managed and perceived matters more than where it is located.

## Key Results

- **Selected Model:** LightGBM with hyperparameter optimization and training set normalization.
- **Discriminative Capacity (ROC-AUC):** 0.90, excellent ability to differentiate between high- and low-demand listings.
- **Capture Efficiency (Recall):** 86%, successfully identifying the majority of successful market opportunities.
- **Precision vs. Random Baseline:** 52%, achieving a Lift of 2.3x over the market base probability.
- **Main Finding:** Operational management and perceived quality have greater predictive impact than geographic location and property infrastructure in the Buenos Aires Airbnb market.

## Project Structure
```bash
├── data/                           # Data folder
│   └── listings.zip                # Raw data (included for reproducibility)
├── inside_airbnb_ml_project.ipynb  # Main notebook with the full data lifecycle
├── README.md                       # Project documentation
└── .gitignore                      # Excluded files (virtual environments and temp files)
```
## Contents

- `data/`: Contains the raw dataset `listings.zip` (see notes below).
- `inside_airbnb_ml_project.ipynb`: Jupyter Notebook with the full analysis pipeline: data loading, cleaning, feature engineering, modeling (linear models vs. ensemble comparison including boosting), metric evaluation, and model interpretability via SHAP values.
- `README.md`: This file — project description, requirements, and execution instructions.

## Requirements

- Python 3.11 or higher.
- Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `shap`, `folium`.

## Running the Notebook

### Option A: Google Colab (Recommended)

1. Upload `inside_airbnb_ml_project.ipynb` to your Google Drive.
2. Load `listings.zip` in the "Files" panel (folder icon on the left) inside a folder named `data/`, or adjust the loading path in the notebook.
3. Run cells in order. Opening cells will automatically install any missing libraries (e.g., `shap`, `lightgbm`) if not already present in the environment.

### Option B: Local (Jupyter Notebook or JupyterLab)

1. Clone or download this repository.
2. Create a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate
```

3. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap folium
```

4. Launch Jupyter Notebook or JupyterLab and open the `.ipynb` file.
5. Make sure the path to `listings.zip` matches the data loading section of the notebook (preferably inside a folder named `data/`).
6. Run cells in order.

The notebook covers the following steps:

- Library imports and visual environment setup.
- Dataset loading.
- Initial data cleaning and feature engineering.
- Exploratory Data Analysis (EDA).
- Data preprocessing.
- Model training with stratified cross-validation (Logistic Regression, Random Forest, LightGBM, XGBoost).
- Hyperparameter optimization via GridSearchCV, prioritizing Recall maximization.
- In-depth interpretability analysis of the winning model (LightGBM) using SHAP values.

## Data & Reproducibility

The dataset corresponds to the Inside Airbnb extraction for Buenos Aires (CABA) with a cutoff date of January 30, 2025. The notebook reads the `.zip` file directly without decompressing it, as `pandas` supports this natively.

> **Important note on data:** Inside Airbnb periodically updates its file structure (newer versions may introduce significant changes in column names and data types). The original `listings.zip` used for training and validation is included in this repository to ensure full reproducibility of the presented results.

**Using newer data:** If you'd like to use a more recent dataset:
- Visit [Inside Airbnb](http://insideairbnb.com/get-the-data/) and download the `listings.csv.gz` file for Buenos Aires (Detailed Listings data).
- Decompress if needed.
- Follow the data loading instructions in the notebook.
- **Important:** Review and update column mappings and cleaning functions (e.g., `initial_data_cleaning`) to account for any structural changes in the source data.

## References

- Data source: [Inside Airbnb](http://insideairbnb.com/)
- Prediction algorithm: [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/)
- Interpretability framework: [SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations)
