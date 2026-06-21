# High-Demand Airbnb Listing Identification in Buenos Aires using Classification Models

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://inside-airbnb-project.streamlit.app/)

This repository contains the code and resources for the Final Integrative Project of the Postgraduate Diploma in Data Science and Advanced Analytics (UTN). The goal is to classify the success potential of Airbnb listings in Buenos Aires (CABA) using Machine Learning techniques, following the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) methodology.

The project is delivered in two complementary formats: a Jupyter Notebook covering the full research and modelling lifecycle, and a Streamlit web application that allows users to interact with the trained model and obtain predictions with SHAP-based explanations.

## Business Context

Airbnb hosts often lack data-driven guidance on what actually drives listing success. Decisions around pricing, presentation, and operational management are frequently based on intuition rather than evidence. This project addresses that gap by building a classification model that identifies high-demand listings based on operational and qualitative features, providing actionable insights for hosts, property managers, and short-term rental market analysts. A key finding challenges conventional wisdom: how a listing is managed and perceived matters more than where it is located.

## Key Results

- **Selected Model:** LightGBM with hyperparameter optimisation and training set normalisation.
- **Discriminative Capacity (ROC-AUC):** 0.896, reflecting excellent ability to differentiate between high- and low-demand listings.
- **Capture Efficiency (Recall):** 86%, successfully identifying the majority of successful market opportunities.
- **Precision vs. Random Baseline:** 52%, achieving a Lift of 2.3x over the market base probability.
- **Main Finding:** Operational management and perceived quality have greater predictive impact than geographic location and property infrastructure in the Buenos Aires Airbnb market.

## Project Structure

```
├── data/
│   └── listings.zip                    # Raw dataset (included for reproducibility)
├── model/                              # Generated artifacts (produced by train_model.py)
│   ├── pipeline.pkl                    # Fitted sklearn Pipeline (preprocessor + LightGBM)
│   └── config.json                     # Q75 threshold, optimal decision threshold, feature names
├── utils/
│   ├── __init__.py
│   └── features.py                     # Feature engineering functions and custom transformer
├── app.py                              # Streamlit web application
├── train_model.py                      # Script to train the model and save artifacts
├── inside_airbnb_ml_project.ipynb      # Research notebook with the full data lifecycle
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

> **Note on model artifacts:** The `model/` folder is included in this repository for convenience, allowing the Streamlit app to run immediately after cloning. To regenerate the artifacts from scratch, run `python train_model.py`.

## Requirements

Python 3.11 or higher, with the following libraries:

```
pandas, numpy, scikit-learn, lightgbm, xgboost, shap, streamlit, joblib, matplotlib, seaborn, folium
```

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Option A: Running the Streamlit App

This is the recommended path for interacting with the trained model. The app takes property characteristics as input and returns a success probability along with a SHAP-based explanation of the most influential factors.

### Step 1: Train the model and save artifacts

```bash
python train_model.py
```

This script loads `data/listings.zip`, runs the full preprocessing and training pipeline (approximately 30 seconds), and saves the fitted pipeline and configuration to the `model/` folder.

### Step 2: Launch the app

```bash
python -m streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Option B: Running the Research Notebook

The notebook covers the complete data science lifecycle and is intended for exploratory and methodological review.

### Option B.1: Google Colab

1. Upload `inside_airbnb_ml_project.ipynb` to your Google Drive.
2. Load `listings.zip` in the "Files" panel (folder icon on the left) inside a folder named `data/`, or adjust the loading path in the notebook.
3. Run cells in order. The notebook will automatically install any missing libraries (e.g., `shap`, `lightgbm`) if not already present in the Colab environment.

### Option B.2: Local (Jupyter Notebook or JupyterLab)

1. Clone or download this repository.
2. Create a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate      # macOS and Linux
env\Scripts\activate         # Windows
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook or JupyterLab and open the `.ipynb` file.
5. Ensure the path to `listings.zip` matches the data loading section of the notebook (preferably inside a folder named `data/`).
6. Run cells in order.

The notebook covers the following steps:

- Library imports and visual environment setup.
- Dataset loading.
- Initial data cleaning and feature engineering.
- Exploratory Data Analysis (EDA).
- Data preprocessing with a scikit-learn Pipeline (to prevent data leakage).
- Model training with stratified cross-validation (Logistic Regression, Random Forest, LightGBM, XGBoost).
- Hyperparameter optimisation via GridSearchCV, prioritising Recall maximisation.
- In-depth interpretability analysis of the winning model (LightGBM) using SHAP values.
- Hypothesis validation comparing an asset-only model (H1) against the full integral model (H2).

---

## Data and Reproducibility

The dataset corresponds to the Inside Airbnb extraction for Buenos Aires (CABA) with a cutoff date of January 30, 2025. Both the notebook and the training script read the `.zip` file directly without decompressing it, as `pandas` supports this natively.

> **Important note on data:** Inside Airbnb periodically updates its file structure (newer versions may introduce changes in column names and data types). The original `listings.zip` used for training and validation is included in this repository to ensure full reproducibility of the presented results.

**Using newer data:** If you would like to use a more recent dataset:

- Visit [Inside Airbnb](http://insideairbnb.com/get-the-data/) and download the `listings.csv.gz` file for Buenos Aires (Detailed Listings data).
- Decompress if needed.
- Follow the data loading instructions in the notebook or `train_model.py`.
- Review and update column mappings and cleaning functions (in `utils/features.py`) to account for any structural changes in the source data.

---

## References

- Data source: [Inside Airbnb](http://insideairbnb.com/)
- Prediction algorithm: [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/)
- Interpretability framework: [SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations)
- Web application framework: [Streamlit](https://streamlit.io/)
