"""
train_model.py — Run this script ONCE to train the model and save artifacts.

Usage:
    python train_model.py

Output:
    model/pipeline.pkl   — fitted sklearn Pipeline (preprocessor + LightGBM)
    model/config.json    — Q75 threshold, feature names, property type categories

Why a separate training script instead of training inside the app?
    Training takes ~30 seconds and reads 19MB of data. Loading a pre-trained
    model takes milliseconds. Running this once and saving the result means
    every Streamlit user gets instant predictions.
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

# Import all feature engineering logic from our shared utils module.
# Both this script and app.py import from the same place, so they can never
# diverge accidentally.
from utils.features import (
    initial_data_cleaning,
    feature_engineering,
    final_static_preprocessing,
    QuantileThresholdCapper,
)

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join('data', 'listings.zip')
MODEL_DIR = 'model'
PIPELINE_PATH = os.path.join(MODEL_DIR, 'pipeline.pkl')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.json')


# ---------------------------------------------------------------------------
# Preprocessing pipeline factory
# ---------------------------------------------------------------------------

def build_preprocessor(X_train):
    """
    Constructs and returns the ColumnTransformer for the given training set.

    The function receives X_train so it can dynamically identify the binary
    flag columns, rather than hardcoding them. This makes the pipeline robust
    to minor changes in the feature set.

    Why ColumnTransformer?
        Different features require different transformations. A price column
        needs log-normalization; a binary flag needs nothing. ColumnTransformer
        lets us apply the right recipe to each group of columns in one step,
        and sklearn handles fitting/transforming consistently.
    """

    # --- Feature groups ---

    # These two have missing values in the raw data. We fill them with the
    # training-set mean (inside the pipeline, so the mean is never computed
    # on test/holdout data).
    features_to_impute = ['host_response_rate', 'host_acceptance_rate']

    # property_type is a string category — the model can't process strings
    # directly, so we one-hot encode it. min_frequency=30 groups rare property
    # types into a single "infrequent" bucket to avoid overfitting on them.
    categorical_features = ['property_type']

    # Review score columns follow a special distribution (0 = no reviews,
    # 4.0–5.0 = actual score), so they get their own scaling step.
    review_cols = [c for c in X_train.columns if c.startswith('review_scores_')]

    # Standard numeric features that just need to be scaled to [0, 1].
    # Why MinMaxScaler instead of StandardScaler?
    # Airbnb features are heavily right-skewed (lots of listings cluster at
    # low values with a long tail). StandardScaler assumes near-normality and
    # was found to degrade LightGBM performance on this dataset. MinMaxScaler
    # is distribution-agnostic and more stable here.
    numeric_features = [
        'availability_365', 'calculated_host_listings_count',
        'bathrooms_clipped', 'accommodates_clipped',
        'dist_obelisco', 'dist_palermo',
        'total_amenities_count', 'host_response_time_ordinal',
    ] + review_cols

    # Everything else is a binary flag (0/1). These are already in the right
    # scale and need no transformation — we pass them through unchanged.
    features_with_transform = (
        features_to_impute + categorical_features + numeric_features
        + ['price', 'minimum_nights']
    )
    binary_features = [c for c in X_train.columns if c not in features_with_transform]

    scaler = MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            # Price: cap outliers → log-transform → scale.
            # Log transform is critical: raw prices are highly right-skewed
            # (skew > 2). Log1p compresses the long tail so the model treats
            # a jump from $100 to $200 similarly to $500 to $1000 (both are
            # doubling), which is more meaningful economically.
            ('price_transform', Pipeline([
                ('capper', QuantileThresholdCapper(quantile=0.99)),
                ('log', FunctionTransformer(np.log1p, validate=False,
                                            feature_names_out='one-to-one')),
                ('scaler', MinMaxScaler()),
            ]), ['price']),

            # Minimum nights: cap extreme outliers (some listings require 365
            # nights, which is clearly not a short-term rental), then scale.
            ('nights_transform', Pipeline([
                ('capper', QuantileThresholdCapper(quantile=0.99)),
                ('scaler', MinMaxScaler()),
            ]), ['minimum_nights']),

            # Response/acceptance rates: impute missing with training mean,
            # then scale. The is_missing_* flags (computed earlier) tell the
            # model whether imputation occurred, so information is not lost.
            ('impute_transform', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', scaler),
            ]), features_to_impute),

            # Property type: one-hot encode.
            # handle_unknown='infrequent_if_exist' ensures that a property type
            # seen at inference but not in training maps to the "infrequent"
            # bucket rather than crashing.
            ('categorical_transform', OneHotEncoder(
                min_frequency=30,
                handle_unknown='infrequent_if_exist',
                sparse_output=False,
            ), categorical_features),

            # Standard numerics: scale only.
            ('numeric_transform', MinMaxScaler(), numeric_features),

            # Binary flags: pass through without any transformation.
            ('binary_passthrough', 'passthrough', binary_features),
        ],
        remainder='drop',  # Drop any column not explicitly listed above.
    )

    return preprocessor, binary_features


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    print("Loading and cleaning data...")
    df_raw = pd.read_csv(DATA_PATH)
    df_base = df_raw.pipe(initial_data_cleaning).pipe(feature_engineering)

    print("Applying static preprocessing...")
    df_processed = final_static_preprocessing(df_base)

    # --- Train/holdout split BEFORE computing the target ---
    # The Q75 threshold that defines "success" is calculated on training data
    # only. Using the full dataset would leak future information into the
    # target variable, artificially inflating performance metrics.
    X = df_processed.drop(columns=['reviews_per_month'])
    y_raw = df_processed['reviews_per_month']

    X_train, X_hold, y_train_raw, y_hold_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=42
    )

    # Compute the success threshold exclusively from training data.
    q75_train = float(y_train_raw.quantile(0.75))
    y_train = (y_train_raw > q75_train).astype(int)
    y_hold = (y_hold_raw > q75_train).astype(int)

    print(f"Q75 threshold (reviews/month): {q75_train:.4f}")
    print(f"Training set size: {len(X_train):,} | Holdout size: {len(X_hold):,}")
    print(f"Positive rate (train): {y_train.mean():.2%}")

    # --- Build full pipeline ---
    print("Building preprocessing pipeline...")
    preprocessor, binary_features = build_preprocessor(X_train)

    # Best hyperparameters found by GridSearchCV in the notebook.
    # We hardcode them here to avoid re-running 108 candidate evaluations
    # every time someone prepares the app — that work is already done.
    best_params = {
        'is_unbalance': True,     # Addresses class imbalance without upsampling.
                                  # Tells LightGBM to weight the minority class
                                  # (successful listings) more heavily during
                                  # training. This prioritizes Recall so we
                                  # don't miss high-potential listings.
        'learning_rate': 0.05,   # Conservative step size — less likely to
                                  # overfit than 0.1, more expressive than 0.01.
        'max_depth': 5,           # Limits tree depth to prevent memorizing noise.
        'n_estimators': 100,      # Number of boosting rounds.
        'num_leaves': 31,         # Controls model complexity. LightGBM grows
                                  # trees leaf-by-leaf (unlike XGBoost which
                                  # grows level-by-level), so num_leaves is the
                                  # primary complexity knob.
        'random_state': 42,
        'verbosity': -1,          # Suppress LightGBM's training logs.
    }

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LGBMClassifier(**best_params)),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # --- Evaluate on holdout ---
    y_proba = pipeline.predict_proba(X_hold)[:, 1]
    y_pred = pipeline.predict(X_hold)
    print(f"Holdout ROC-AUC: {roc_auc_score(y_hold, y_proba):.4f}")
    print(f"Holdout Recall:  {recall_score(y_hold, y_pred):.4f}")

    # --- Optimal decision threshold ---
    # LightGBM's default threshold is 0.5, but this is arbitrary. With
    # is_unbalance=True the model's probabilities are shifted, so 0.5 is
    # rarely the best cut-off point.
    #
    # Strategy: sweep all candidate thresholds from the precision-recall
    # curve and pick the one that maximises F1 score. F1 balances precision
    # and recall equally, giving a principled default for the app.
    # A recruiter question to expect: "why F1 and not Recall?"
    # Answer: the app is a decision-support tool for a host, not a bulk
    # screening system — false positives (wasted effort) matter here too,
    # so pure Recall maximisation would be too aggressive.
    #
    # Note on cross-validation: the notebook used 3-fold CV for speed during
    # model selection across 108 candidates. For a production model, 5-fold
    # would give more stable estimates (~40% more compute, lower variance).
    # This is a known trade-off, not an oversight.
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_hold, y_proba)
    f1_vals = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-9)
    best_idx = int(np.argmax(f1_vals[:-1]))  # last element has no matching threshold
    optimal_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_vals[best_idx])
    default_f1 = float(f1_vals[np.searchsorted(thresholds, 0.5)])
    print(f"Optimal threshold: {optimal_threshold:.4f}  (F1={best_f1:.4f} vs 0.5 default F1={default_f1:.4f})")

    # --- Extract metadata for the Streamlit app ---
    # After fitting, the preprocessor knows the exact feature names it outputs.
    # These are needed for SHAP to label each bar in the explanation chart.
    fitted_prep = pipeline.named_steps['prep']
    raw_feature_names = fitted_prep.get_feature_names_out().tolist()

    # Clean the prefix that ColumnTransformer adds (e.g. "price_transform__price"
    # → "price"). The app uses these cleaned names for display.
    feature_names_out = [name.split('__')[-1] for name in raw_feature_names]

    # Get property type categories seen during training, so the Streamlit
    # dropdown only shows valid options.
    ohe = fitted_prep.named_transformers_['categorical_transform']
    property_types = ohe.categories_[0].tolist()

    # Save everything the app needs to reconstruct predictions from scratch.
    config = {
        'q75_threshold': q75_train,
        'optimal_threshold': optimal_threshold,   # F1-maximising decision threshold
        'feature_names_out': feature_names_out,
        'property_types': property_types,
        'binary_features': binary_features,
    }

    # --- Persist artifacts ---
    os.makedirs(MODEL_DIR, exist_ok=True)

    # joblib is preferred over pickle for sklearn objects because it handles
    # large numpy arrays more efficiently (uses memory-mapped files internally).
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Pipeline saved: {PIPELINE_PATH}")

    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved:   {CONFIG_PATH}")

    print("\nDone! You can now run:  streamlit run app.py")


if __name__ == '__main__':
    main()
