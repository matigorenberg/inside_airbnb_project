"""
app.py — Streamlit app for predicting Airbnb listing success in CABA, Argentina.

Run with:  streamlit run app.py

The app loads a pre-trained LightGBM pipeline and lets a user enter property
characteristics to get a success probability and a SHAP-based explanation.
"""

import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.features import haversine_distance, QuantileThresholdCapper  # noqa: F401
# QuantileThresholdCapper must be imported so joblib can deserialize the pipeline —
# it's a custom class that joblib needs to find in scope when loading pipeline.pkl.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate centroids of common CABA neighbourhoods.
# Used to compute dist_obelisco and dist_palermo without asking
# users to enter raw coordinates.
NEIGHBORHOODS = {
    "Palermo":              (-34.5830, -58.4289),
    "Recoleta":             (-34.5882, -58.3940),
    "San Telmo":            (-34.6218, -58.3736),
    "Puerto Madero":        (-34.6132, -58.3623),
    "Belgrano":             (-34.5601, -58.4572),
    "Almagro":              (-34.6088, -58.4180),
    "Villa Crespo":         (-34.5967, -58.4397),
    "Caballito":            (-34.6197, -58.4414),
    "Núñez":                (-34.5436, -58.4619),
    "Flores":               (-34.6307, -58.4620),
    "Balvanera / Once":     (-34.6120, -58.3970),
    "Centro / Microcentro": (-34.6037, -58.3816),
    "Villa Urquiza":        (-34.5759, -58.4869),
    "Colegiales":           (-34.5710, -58.4481),
    "Boedo":                (-34.6304, -58.4156),
}

OBELISCO = (-34.6037, -58.3816)
PALERMO  = (-34.5830, -58.4289)

# Maps the ordinal integer back to a human-readable label for the selectbox.
RESPONSE_TIME_OPTIONS = {
    "Within an hour":      4,
    "Within a few hours":  3,
    "Within a day":        2,
    "A few days or more":  1,
    "Unknown / Not set":   0,
}

# Human-readable labels for SHAP chart axes.
# Raw column names from the pipeline are technical; these are recruiter-friendly.
FEATURE_LABELS = {
    "price":                            "Nightly price",
    "minimum_nights":                   "Minimum nights",
    "host_response_rate":               "Host response rate",
    "host_acceptance_rate":             "Host acceptance rate",
    "host_response_time_ordinal":       "Response time",
    "calculated_host_listings_count":   "Number of host listings",
    "instant_bookable":                 "Instant bookable",
    "host_is_superhost":                "Superhost status",
    "host_identity_verified":           "Identity verified",
    "availability_365":                 "Annual availability (days)",
    "review_scores_rating":             "Overall rating",
    "review_scores_accuracy":           "Accuracy score",
    "review_scores_checkin":            "Check-in score",
    "review_scores_location":           "Location score",
    "review_scores_cleanliness":        "Cleanliness score",
    "review_scores_communication":      "Communication score",
    "review_scores_value":              "Value-for-money score",
    "dist_obelisco":                    "Distance to city centre (km)",
    "dist_palermo":                     "Distance to Palermo (km)",
    "bathrooms_clipped":                "Bathrooms",
    "accommodates_clipped":             "Guest capacity",
    "total_amenities_count":            "Total amenities",
    "is_missing_reviews":               "Missing reviews (data flag)",
    "is_missing_response_data":         "Response rate unknown (flag)",
    "is_missing_acceptance_data":       "Acceptance rate unknown (flag)",
    "is_zombie":                        "Inactive listing (flag)",
    "amenity_hot_water":                "Amenity: Hot water",
    "amenity_hot_water_kettle":         "Amenity: Hot water kettle",
    "amenity_room_darkening_shades":    "Amenity: Room-darkening shades",
    "amenity_bed_linens":               "Amenity: Bed linens",
    "amenity_hair_dryer":               "Amenity: Hair dryer",
    "amenity_bidet":                    "Amenity: Bidet",
    "amenity_dishes_and_silverware":    "Amenity: Dishes & silverware",
    "amenity_cooking_basics":           "Amenity: Cooking basics",
    "property_type_Entire condo":       "Property: Entire condo",
    "property_type_Entire home":        "Property: Entire home",
    "property_type_Entire loft":        "Property: Entire loft",
    "property_type_Entire rental unit": "Property: Rental unit",
    "property_type_Entire serviced apartment": "Property: Serviced apartment",
    "property_type_Entire townhouse":   "Property: Townhouse",
    "property_type_Entire vacation home": "Property: Vacation home",
    "property_type_infrequent_sklearn": "Property: Other type",
}


# ---------------------------------------------------------------------------
# Artifact loading (cached — runs once per server session, not per user action)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """
    Loads the trained pipeline and config, then builds a SHAP TreeExplainer.

    Why @st.cache_resource?
    This decorator caches heavy objects (models, explainers) across all users
    and browser refreshes. Without it, the pipeline would reload on every
    widget interaction — making the app unusably slow.
    """
    pipeline = joblib.load("model/pipeline.pkl")
    with open("model/config.json") as f:
        config = json.load(f)

    # Build the SHAP explainer once and cache it.
    # TreeExplainer is tailored for tree-based models (LightGBM, XGBoost).
    # It computes exact Shapley values in O(TLD) time, where T=trees, L=leaves,
    # D=depth — much faster than the model-agnostic KernelExplainer.
    clf = pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)

    return pipeline, config, explainer


# ---------------------------------------------------------------------------
# Input → DataFrame
# ---------------------------------------------------------------------------

def build_input_df(inputs: dict) -> pd.DataFrame:
    """
    Converts the sidebar widget values into a one-row DataFrame with exactly
    the column names and dtypes the pipeline preprocessor expects.

    Derived fields computed here (not collected directly from widgets):
    - is_missing_* flags
    - dist_obelisco / dist_palermo from neighborhood selection
    - review scores set to 0 when the listing has no reviews
    - is_zombie always 0 (an actively submitted listing is not a zombie)
    """
    lat, lon = NEIGHBORHOODS[inputs["neighborhood"]]

    # Derive missing-data flags BEFORE passing rates to the DataFrame.
    # The pipeline will impute NaN rates with the training mean; the flags
    # tell the model that imputation occurred (preserving that signal).
    response_rate   = inputs["host_response_rate"]    # float or None
    acceptance_rate = inputs["host_acceptance_rate"]  # float or None

    is_missing_response   = 1 if response_rate   is None else 0
    is_missing_acceptance = 1 if acceptance_rate is None else 0

    row = {
        # --- Property profile ---
        "price":                          inputs["price"],
        "minimum_nights":                 inputs["minimum_nights"],
        "property_type":                  inputs["property_type"],
        "accommodates_clipped":           float(inputs["accommodates"]),
        "bathrooms_clipped":              float(inputs["bathrooms"]),
        "availability_365":               inputs["availability_365"],
        "total_amenities_count":          inputs["total_amenities_count"],

        # --- Location (derived) ---
        "dist_obelisco": haversine_distance(lat, lon, *OBELISCO),
        "dist_palermo":  haversine_distance(lat, lon, *PALERMO),

        # --- Host management ---
        "host_response_time_ordinal":     inputs["host_response_time_ordinal"],
        "host_response_rate":             response_rate,    # may be NaN → pipeline imputes
        "host_acceptance_rate":           acceptance_rate,  # may be NaN → pipeline imputes
        "calculated_host_listings_count": inputs["calculated_host_listings_count"],
        "instant_bookable":               int(inputs["instant_bookable"]),
        "host_is_superhost":              int(inputs["host_is_superhost"]),
        "host_identity_verified":         int(inputs["host_identity_verified"]),

        # --- Reviews ---
        "review_scores_rating":        inputs["review_scores_rating"],
        "review_scores_accuracy":      inputs["review_scores_accuracy"],
        "review_scores_checkin":       inputs["review_scores_checkin"],
        "review_scores_location":      inputs["review_scores_location"],
        "review_scores_cleanliness":   inputs["review_scores_cleanliness"],
        "review_scores_communication": inputs["review_scores_communication"],
        "review_scores_value":         inputs["review_scores_value"],

        # --- Derived flags ---
        # is_missing_reviews is always 0 in the app: users must provide review
        # scores, so there is no missing data to flag. The flag exists in the
        # model because ~20% of raw Airbnb listings have no reviews_per_month
        # value — a data-quality issue in the source, not an app concept.
        "is_missing_reviews":          0,
        "is_missing_response_data":    is_missing_response,
        "is_missing_acceptance_data":  is_missing_acceptance,
        "is_zombie":                   0,  # always 0: active listing by definition

        # --- Amenities ---
        "amenity_hot_water":               int(inputs["amenity_hot_water"]),
        "amenity_hot_water_kettle":        int(inputs["amenity_hot_water_kettle"]),
        "amenity_room_darkening_shades":   int(inputs["amenity_room_darkening_shades"]),
        "amenity_bed_linens":              int(inputs["amenity_bed_linens"]),
        "amenity_hair_dryer":              int(inputs["amenity_hair_dryer"]),
        "amenity_bidet":                   int(inputs["amenity_bidet"]),
        "amenity_dishes_and_silverware":   int(inputs["amenity_dishes_and_silverware"]),
        "amenity_cooking_basics":          int(inputs["amenity_cooking_basics"]),
    }

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# SHAP chart
# ---------------------------------------------------------------------------

def get_shap_chart(pipeline, explainer, config, X_df: pd.DataFrame):
    """
    Returns a matplotlib Figure showing the top 10 SHAP values for X_df.

    Steps:
    1. Transform input through the fitted preprocessor (same steps as training).
    2. Compute SHAP values with the cached TreeExplainer.
    3. Pair values with feature names, keep top 10 by |SHAP|, plot.

    Color convention: blue = pushes prediction toward success,
                      red  = pushes prediction toward standard/failure.
    """
    prep = pipeline.named_steps["prep"]
    X_transformed = prep.transform(X_df)
    feature_names = config["feature_names_out"]

    shap_values = explainer.shap_values(X_transformed)

    # LightGBM can return either a list [class0_array, class1_array] or a
    # single 2D array depending on the shap version. We always want class-1
    # (probability of being a high-demand listing).
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]

    shap_df = pd.DataFrame({"feature": feature_names, "shap_value": vals})
    shap_df["abs"] = shap_df["shap_value"].abs()
    shap_df = shap_df.nlargest(10, "abs").sort_values("shap_value")

    # Map raw names to human-readable labels for the chart.
    shap_df["label"] = shap_df["feature"].map(
        lambda x: FEATURE_LABELS.get(x, x)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in shap_df["shap_value"]]
    ax.barh(shap_df["label"], shap_df["shap_value"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (positive = increases success probability)", fontsize=10)
    ax.set_title("Top 10 factors influencing this prediction", fontsize=12, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Airbnb Success Predictor — CABA",
        page_icon="🏠",
        layout="wide",
    )

    pipeline, config, explainer = load_artifacts()
    optimal_threshold = config["optimal_threshold"]
    property_types    = config["property_types"]

    # -----------------------------------------------------------------------
    # Sidebar — inputs
    # -----------------------------------------------------------------------
    st.sidebar.title("Property Details")
    st.sidebar.caption(
        "Fill in the details below and click **Predict** to see "
        "whether this listing is likely to be high-demand."
    )

    inputs = {}

    # --- Section 1: Property Profile ---
    st.sidebar.subheader("🏠 Property Profile")

    inputs["property_type"] = st.sidebar.selectbox(
        "Property type", property_types, index=property_types.index("Entire rental unit")
    )
    inputs["neighborhood"] = st.sidebar.selectbox(
        "Neighbourhood", list(NEIGHBORHOODS.keys())
    )
    inputs["price"] = st.sidebar.number_input(
        "Nightly price (ARS $)", min_value=0, value=15000, step=1000
    )
    inputs["accommodates"] = st.sidebar.slider(
        "Guest capacity", min_value=1, max_value=8, value=2
    )
    inputs["bathrooms"] = st.sidebar.slider(
        "Bathrooms", min_value=1.0, max_value=4.0, value=1.0, step=0.5
    )
    inputs["minimum_nights"] = st.sidebar.number_input(
        "Minimum nights required", min_value=1, value=2
    )
    inputs["availability_365"] = st.sidebar.slider(
        "Days available per year", min_value=0, max_value=365, value=200
    )

    # --- Section 2: Amenities ---
    st.sidebar.subheader("✅ Amenities")

    inputs["total_amenities_count"] = st.sidebar.number_input(
        "Total number of amenities listed", min_value=0, value=20
    )
    st.sidebar.caption("Select which key amenities the property offers:")

    col1, col2 = st.sidebar.columns(2)
    inputs["amenity_hot_water"]             = col1.checkbox("Hot water",            value=True)
    inputs["amenity_hot_water_kettle"]      = col2.checkbox("Hot water kettle",     value=False)
    inputs["amenity_room_darkening_shades"] = col1.checkbox("Darkening shades",     value=False)
    inputs["amenity_bed_linens"]            = col2.checkbox("Bed linens",           value=True)
    inputs["amenity_hair_dryer"]            = col1.checkbox("Hair dryer",           value=True)
    inputs["amenity_bidet"]                 = col2.checkbox("Bidet",                value=False)
    inputs["amenity_dishes_and_silverware"] = col1.checkbox("Dishes & silverware",  value=True)
    inputs["amenity_cooking_basics"]        = col2.checkbox("Cooking basics",       value=True)

    # --- Section 3: Host Management ---
    st.sidebar.subheader("🤝 Host Management")

    rt_label = st.sidebar.selectbox("Response time", list(RESPONSE_TIME_OPTIONS.keys()))
    inputs["host_response_time_ordinal"] = RESPONSE_TIME_OPTIONS[rt_label]

    unknown_response = st.sidebar.checkbox("Response rate unknown", value=False)
    if unknown_response:
        st.sidebar.caption("_Pipeline will impute with training-set mean._")
        inputs["host_response_rate"] = None
    else:
        inputs["host_response_rate"] = (
            st.sidebar.slider("Response rate (%)", 0, 100, 95) / 100
        )

    unknown_acceptance = st.sidebar.checkbox("Acceptance rate unknown", value=False)
    if unknown_acceptance:
        st.sidebar.caption("_Pipeline will impute with training-set mean._")
        inputs["host_acceptance_rate"] = None
    else:
        inputs["host_acceptance_rate"] = (
            st.sidebar.slider("Acceptance rate (%)", 0, 100, 90) / 100
        )

    inputs["calculated_host_listings_count"] = st.sidebar.number_input(
        "Number of active listings by host", min_value=1, value=1
    )
    inputs["instant_bookable"]    = st.sidebar.toggle("Instant bookable",    value=False)
    inputs["host_is_superhost"]   = st.sidebar.toggle("Superhost",           value=False)
    inputs["host_identity_verified"] = st.sidebar.toggle("Identity verified", value=True)

    # --- Section 4: Guest Reviews ---
    st.sidebar.subheader("⭐ Guest Reviews")

    for col, label in [
        ("review_scores_rating",        "Overall rating"),
        ("review_scores_accuracy",      "Accuracy"),
        ("review_scores_checkin",       "Check-in"),
        ("review_scores_location",      "Location"),
        ("review_scores_cleanliness",   "Cleanliness"),
        ("review_scores_communication", "Communication"),
        ("review_scores_value",         "Value for money"),
    ]:
        inputs[col] = st.sidebar.slider(label, 4.0, 5.0, 4.8, step=0.1)

    # --- Predict button ---
    st.sidebar.divider()
    predict_clicked = st.sidebar.button("🔍 Predict", use_container_width=True, type="primary")

    # -----------------------------------------------------------------------
    # Main area — header
    # -----------------------------------------------------------------------
    st.title("🏠 Airbnb Listing Success Predictor")
    st.markdown(
        """
        This tool predicts whether an Airbnb listing in **Buenos Aires (CABA)**
        will rank in the **top 25% by reviews per month** — a proven proxy for
        booking demand.

        Enter your listing's current details in the sidebar to get a **success
        probability** and a breakdown of which factors are helping or hurting
        your ranking.

        The model is a **LightGBM classifier** trained on ~23,000 listings from
        [Inside Airbnb](http://insideairbnb.com/) (January 2025), achieving
        a **ROC-AUC of 0.896** and **Recall of 86%** on unseen data.
        """
    )
    st.divider()

    # -----------------------------------------------------------------------
    # Prediction & results
    # -----------------------------------------------------------------------
    if predict_clicked:
        X_df       = build_input_df(inputs)
        y_proba    = float(pipeline.predict_proba(X_df)[0, 1])
        is_success = y_proba >= optimal_threshold

        # Metrics row
        col_prob, col_verdict = st.columns([1, 2])

        with col_prob:
            st.metric(
                label="Success Probability",
                value=f"{y_proba:.1%}",
                help=(
                    "Probability that this listing falls in the top 25% "
                    "by reviews/month in the CABA market."
                ),
            )

        with col_verdict:
            if is_success:
                st.success(
                    "**HIGH DEMAND** — This listing is predicted to be among the "
                    "top-performing properties in CABA by booking activity.",
                    icon="✅",
                )
            else:
                st.warning(
                    "**STANDARD** — This listing is predicted to be below the "
                    "high-demand threshold. See the factors below for guidance on "
                    "where to focus.",
                    icon="⚠️",
                )

        st.caption(
            f"Decision threshold: **{optimal_threshold:.2f}** "
            f"(F1-optimised on the holdout set — not the arbitrary 0.5 default)."
        )

        st.divider()

        # SHAP explanation
        st.subheader("What's driving this prediction?")
        st.markdown(
            "The chart below shows the **10 features that influenced this result most**. "
            "🔵 **Blue bars** push the prediction toward *success*; "
            "🔴 **red bars** push it toward *standard/failure*."
        )

        fig = get_shap_chart(pipeline, explainer, config, X_df)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "SHAP (SHapley Additive exPlanations) values measure each feature's "
            "marginal contribution to this individual prediction. "
            "They are computed with a **TreeExplainer**, which is exact (not approximate) "
            "for tree-based models like LightGBM."
        )

    else:
        st.info(
            "👈 Fill in the property details in the sidebar and click **Predict** "
            "to see the result.",
            icon="ℹ️",
        )


if __name__ == "__main__":
    main()
