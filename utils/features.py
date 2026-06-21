"""
Feature engineering utilities for the Inside Airbnb ML project.

All functions here are extracted from the training notebook so that both
train_model.py and the Streamlit app share the exact same logic.
Keeping them in one place prevents the classic mistake of applying
slightly different preprocessing at training time vs. inference time.
"""

import ast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns the great-circle distance in km between two (lat, lon) points.

    Why pure math instead of a library like GeoPy?
    A single formula doesn't justify an extra dependency. This is vectorized
    with numpy so it works on full DataFrame columns just as fast.
    """
    R = 6371  # Earth's radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Landmark coordinates used as reference points for the distance features.
# The Obelisco is the geographic/tourist center of CABA; Palermo is the
# highest-density short-term rental neighborhood, so proximity to it is
# a meaningful market signal.
OBELISCO_LAT, OBELISCO_LON = -34.6037, -58.3816
PALERMO_LAT, PALERMO_LON = -34.5830, -58.4289


# ---------------------------------------------------------------------------
# Step 1 — Initial data cleaning
# ---------------------------------------------------------------------------

def initial_data_cleaning(df_raw):
    """
    Drops irrelevant columns, filters to 'Entire home/apt', and parses
    raw string fields (percentages, prices, booleans) into proper types.

    This step does NOT use any statistics from the data (no mean, no median),
    so it is safe to run on the full dataset before the train/test split.
    """
    df = df_raw.copy()

    # Columns that are either identifiers, free text, or redundant with
    # other columns (e.g. multiple availability windows when we keep 365-day).
    cols_to_drop = [
        'id', 'listing_url', 'scrape_id', 'source', 'picture_url', 'host_id', 'host_url',
        'host_name', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped',
        'host_verifications', 'host_neighbourhood', 'neighbourhood_group_cleansed',
        'calendar_updated', 'license', 'description', 'neighborhood_overview',
        'host_about', 'host_location', 'host_listings_count', 'neighbourhood',
        'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
        'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
        'availability_30', 'availability_60', 'availability_90', 'availability_eoy',
        'number_of_reviews_ltm', 'number_of_reviews_l30d', 'number_of_reviews_ly',
        'estimated_occupancy_l365d', 'estimated_revenue_l365d',
        'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms',
        'host_total_listings_count', 'calculated_host_listings_count_entire_homes',
        'bathrooms_text', 'host_has_profile_pic', 'has_availability',
        'maximum_nights', 'beds', 'bedrooms',
    ]
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Keep only entire apartments/homes — private rooms and shared spaces have
    # a fundamentally different competitive dynamic.
    df = df[df['room_type'] == 'Entire home/apt'].copy()
    df.drop(columns=['room_type'], inplace=True)

    # Remove atypical property types (castles, buses, caves, etc.) that would
    # add noise without enough samples to be statistically meaningful.
    exclude_types = [
        'Tiny home', 'Castle', 'Cave', 'Bus', 'Camper/RV', 'Pension',
        'Room in aparthotel', 'Tower', 'Entire guest suite',
    ]
    df = df[~df['property_type'].isin(exclude_types)]
    df = df[~df['property_type'].str.lower().str.contains('room')]

    # Parse percentage strings like "95%" → 0.95 (float, 0–1 scale).
    # The model treats these as continuous probabilities, not whole numbers.
    df['host_response_rate'] = (
        df['host_response_rate'].str.replace('%', '', regex=False).astype(float) / 100
    )
    df['host_acceptance_rate'] = (
        df['host_acceptance_rate'].str.replace('%', '', regex=False).astype(float) / 100
    )

    # Remove the "$" and "," from price strings → numeric.
    df['price'] = df['price'].str.replace(r'[\$,]', '', regex=True).astype(float)

    # Airbnb stores boolean columns as 't'/'f' strings.
    bool_map = {'t': 1, 'f': 0}
    for col in ['host_is_superhost', 'host_identity_verified', 'instant_bookable']:
        df[col] = df[col].map(bool_map).fillna(0)

    # Drop rows where price or host_since are missing — both are essential.
    df.dropna(subset=['price', 'host_since'], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Step 2 — Feature engineering (no statistics, safe before split)
# ---------------------------------------------------------------------------

def feature_engineering(df):
    """
    Creates derived features from existing columns.

    Key rule: every transformation here is deterministic (no mean, no quantile),
    so applying it before the train/test split introduces zero data leakage.
    Statistical operations (scaling, imputation) are handled inside the
    sklearn Pipeline, which is fit only on training data.
    """
    df = df.copy()

    # --- Host longevity ---
    df['host_since'] = pd.to_datetime(df['host_since'])
    df['last_scraped'] = pd.to_datetime(df['last_scraped'])
    df['host_longevity_years'] = (
        (df['last_scraped'] - df['host_since']).dt.days / 365.25
    ).round(1)

    # --- Zombie listing flag ---
    # A listing that has been active for 5+ years, has zero reviews, and is
    # still "available" is almost certainly abandoned. Including this flag
    # prevents the model from confusing a ghost listing with a new one.
    df['is_zombie'] = (
        (df['host_longevity_years'] > 5)
        & (df['number_of_reviews'] == 0)
        & (df['availability_365'] > 100)
    ).astype(int)

    # Drop listings with zero availability AND zero reviews — these are
    # effectively invisible to guests and pollute the training distribution.
    df = df[~((df['availability_365'] == 0) & (df['number_of_reviews'] == 0))].copy()

    # --- Missing data flags ---
    # Rather than silently imputing missing values, we create binary flags so
    # the model can learn that "this host never provided a response rate" is
    # itself a signal (hosts who don't fill in their profile tend to underperform).
    df['is_missing_reviews'] = df['reviews_per_month'].isna().astype(int)
    df['is_missing_response_data'] = df['host_response_rate'].isna().astype(int)
    df['is_missing_acceptance_data'] = df['host_acceptance_rate'].isna().astype(int)

    # --- Review score cleaning ---
    # Airbnb review scores range from 4.0 to 5.0 in practice. Missing values
    # are filled with 0 as a sentinel, and real scores are clipped to ≥4.0
    # to remove encoding artifacts. This lets the model distinguish between
    # "no reviews yet" (0) and "worst possible review" (4.0).
    score_cols = [c for c in df.columns if c.startswith('review_scores_')]
    df[score_cols] = df[score_cols].fillna(0)
    for col in score_cols:
        mask = df[col] > 0
        df.loc[mask, col] = df.loc[mask, col].clip(lower=4.0)

    # --- Geographic distance features ---
    # Raw lat/lon coordinates are not useful directly — the model would need to
    # learn the full 2D spatial relationship. Distance to two key landmarks
    # (tourist center + top rental neighborhood) compresses that into two
    # meaningful, interpretable numeric features.
    df['dist_obelisco'] = haversine_distance(
        df['latitude'], df['longitude'], OBELISCO_LAT, OBELISCO_LON
    )
    df['dist_palermo'] = haversine_distance(
        df['latitude'], df['longitude'], PALERMO_LAT, PALERMO_LON
    )

    # --- Amenity features ---
    # Parse the raw JSON-like amenities string into a Python list.
    df['amenities_list'] = df['amenities'].apply(
        lambda x: [i.strip().lower() for i in ast.literal_eval(x)]
        if pd.notnull(x) else []
    )
    df['total_amenities_count'] = df['amenities_list'].apply(len)

    # Binary flags for the 8 amenities that showed the strongest predictive
    # signal in the EDA phase. Using individual flags (instead of only the
    # total count) lets the model weight specific amenities differently.
    # Strings must exactly match Airbnb's format after lowercasing.
    # Bug fixed: original notebook used underscores ('hair_dryer',
    # 'room-darkening_shades', 'dishes_and_silverware') which never matched
    # the actual data — those features were always 0 during training.
    SELECTED_AMENITIES = [
        'hot water', 'hot water kettle', 'room-darkening shades',
        'bed linens', 'hair dryer', 'bidet',
        'dishes and silverware', 'cooking basics',
    ]
    for amenity in SELECTED_AMENITIES:
        # Normalise to a valid column name: spaces and hyphens → underscores.
        col_name = 'amenity_' + amenity.replace(' ', '_').replace('-', '_')
        df[col_name] = df['amenities_list'].apply(lambda x: 1 if amenity in x else 0)

    return df


# ---------------------------------------------------------------------------
# Step 3 — Final static preprocessing (no statistics, safe before split)
# ---------------------------------------------------------------------------

def final_static_preprocessing(df):
    """
    Applies fixed clipping rules and ordinal encoding before the train/test
    split. No quantile or mean is computed here — those live inside the
    sklearn Pipeline to prevent leakage.
    """
    df = df.copy()

    # Clip bathrooms to [1, 4]: listings with 0 bathrooms are data errors;
    # more than 4 is so rare that it would create an unreliable category.
    df['bathrooms_clipped'] = df['bathrooms'].fillna(1).replace(0, 1).clip(upper=4.0)

    # Clip accommodates to [1, 8]: properties for >8 guests are a niche
    # segment that behaves differently from typical short-term rentals.
    df['accommodates_clipped'] = df['accommodates'].clip(upper=8.0)

    # Re-apply review score clipping (defensive, in case feature_engineering
    # was not run or the DataFrame was modified between steps).
    review_cols = [c for c in df.columns if c.startswith('review_scores_')]
    for col in review_cols:
        df[col] = df[col].fillna(0).clip(lower=4.0)

    # Encode host response time as an ordinal integer (faster = higher number).
    # Why ordinal instead of one-hot? The categories have a natural order
    # (within an hour > a few hours > within a day > a few days), so an
    # ordinal encoding preserves that relationship without adding extra columns.
    response_map = {
        'within an hour': 4,
        'within a few hours': 3,
        'within a day': 2,
        'a few days or more': 1,
        'Unknown': 0,
    }
    df['host_response_time_ordinal'] = (
        df['host_response_time'].fillna('Unknown').map(response_map)
    )

    # Drop raw columns that have been replaced by engineered versions,
    # or that are identifiers/text not useful for modeling.
    cols_to_drop = [
        'name', 'host_since', 'last_scraped', 'amenities', 'amenities_list',
        'latitude', 'longitude', 'host_response_time', 'bathrooms',
        'accommodates', 'first_review', 'last_review', 'number_of_reviews',
        'neighbourhood_cleansed', 'host_longevity_years',
    ]
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    return df


# ---------------------------------------------------------------------------
# Custom sklearn transformer
# ---------------------------------------------------------------------------

class QuantileThresholdCapper(BaseEstimator, TransformerMixin):
    """
    Caps each feature at its N-th percentile, learned from training data only.

    Why a custom transformer instead of a hardcoded clip value?
    - Hardcoding is fragile: the right cap depends on the data distribution,
      which may shift if the dataset is updated.
    - By inheriting from BaseEstimator + TransformerMixin, this class plugs
      seamlessly into sklearn's Pipeline. That means the cap thresholds are
      learned during pipeline.fit() on training data and automatically
      applied consistently at inference time — no manual bookkeeping.

    Why not sklearn's built-in QuantileTransformer?
    - That transformer *redistributes* values (changes the shape of the
      distribution). We only want to *clip* extreme outliers while preserving
      the original scale for interpretability.
    """

    def __init__(self, quantile=0.99):
        self.quantile = quantile
        self.thresholds_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()
        # Learn the cap threshold for each column from training data only.
        self.thresholds_ = X_df.quantile(self.quantile)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.thresholds_:
                X_df[col] = X_df[col].clip(upper=self.thresholds_[col])
        return X_df.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)
