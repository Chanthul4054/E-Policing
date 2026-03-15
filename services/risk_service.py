import joblib
import json
import pandas as pd
import numpy as np
import os
import shap
from extensions import db
from sqlalchemy import text
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR,"models")

# load scaler
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# load the models
CRIME_MODELS = {}
for crime in ["burglary" ,"theft" ,"vehicle" ,"robbery" ,"drugs", "stabbing"]:
    CRIME_MODELS[crime] = joblib.load(
        os.path.join(MODEL_DIR, f"rf_{crime}.pkl")
    )

# load feature order
with open("models/feature_list.json") as f:
    FEATURE_ORDER = json.load(f)

FEATURE_LABELS = {
    "avg_victim_age":"Average Victim Age",
    "holiday_ratio":"Holiday Crime Ratio",
    "rainy_ratio":"Rainy Weather Frequency",
    "unemployment_rate": "Unemployment Rate",
    "avg_income":"Average Household Income",
    "building_density":"Building Density",
    "land_area_density":"Land Area Density",
    "road_density":"Road Network Density",
    "log_population":"Population Density",
    "distance_km":"Distance to the Closest Police Station"
}

GLOBAL_SHAP_CACHE = {}

def get_global_shap_results(crime_type):
    global GLOBAL_SHAP_CACHE

    if crime_type in GLOBAL_SHAP_CACHE:
        return GLOBAL_SHAP_CACHE[crime_type]

    model = CRIME_MODELS[crime_type]

    # Build dataset for all GN divisions
    global_df = build_global_feature_dataset()
    X_global = global_df[FEATURE_ORDER]
    X_scaled = scaler.transform(X_global)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    global_plot = generate_global_shap_waterfall_plot(
        X_scaled,
        shap_values
    )

    # Handle SHAP output format
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    top_features = (
        pd.DataFrame({
            "feature": FEATURE_ORDER,
            "value": np.abs(shap_values_pos).mean(axis=0)
        })
        .sort_values("value", ascending=False)
        .head(3)["feature"]
        .tolist()
    )

    explanation = global_explanation(
        crime_type,
        top_features
    )

    GLOBAL_SHAP_CACHE[crime_type] = {
        "plot": global_plot,
        "text": explanation
    }

    return GLOBAL_SHAP_CACHE[crime_type]

def generate_local_shap_waterfall_plot(model, X_scaled, feature_names, explainer, shap_values):
    """Generate local SHAP waterfall plot and return as base64 image"""

    base_values=explainer.expected_value

    if isinstance(base_values, (list, np.ndarray)):
        base_values = base_values[1]

    base_values = float(np.array(base_values).flatten()[0])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    # select the single observation
    shap_values = shap_values[0]

    explanation = shap.Explanation(
        values=shap_values,
        base_values= base_values,
        data=pd.DataFrame(X_scaled, columns=feature_names).iloc[0],
        feature_names=[FEATURE_LABELS.get(f,f) for f in feature_names]
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()

    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
    
def generate_global_shap_waterfall_plot(X_background, shap_values):

    X_background = pd.DataFrame(X_background, columns=FEATURE_ORDER )

    # rename columns for readability
    X_background = X_background.rename(columns=FEATURE_LABELS)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_background,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close()

    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")

def local_explanation(crime_type, gn_division, top_features, risk_score):
    """Generates interpretation for a specific GN Division"""
    drivers = ", ".join(
        FEATURE_LABELS.get(f, f) for f in top_features["feature"].values[:2]
    )
    mitigator = FEATURE_LABELS.get(top_features["feature"].values[-1], top_features["feature"].values[-1])
    
    return(
        f"The elevated {crime_type} risk in {gn_division} is primarily driven by"
        f" {drivers}, which outweigh mitigating factors such as {mitigator}."
        f"Predicted risk score = {risk_score:.3f}."
    )

def global_explanation(crime_type, top_features):
    drivers = ", ".join(
        FEATURE_LABELS.get(f, f) for f in top_features[:3]
    )

    return(
        f"Across all GN divisions, {crime_type} risk is mainly influenced by"
        f" {drivers}. These factors consistently show the strongest contribution "
        f"to the model predictions."
    )
    
def run_risk_factor_pipeline(crime_type, features):
    """
    Runs ML inference and generates:
    - risk score
    - local SHAP explanation
    - global SHAP explanation (from cached per-crime results)
    """

    if crime_type not in CRIME_MODELS:
        raise ValueError("Invalid crime type")

    if not features:
        raise ValueError("Features are missing")

    # Convert single GN feature dictionary into dataframe
    X = pd.DataFrame([features])

    # Ensure required features exist and preserve model feature order
    try:
        X = X[FEATURE_ORDER]
    except KeyError as e:
        raise ValueError(f"Missing required feature(s): {e}")

    # Scale input features
    X_scaled = scaler.transform(X)

    # Load model for requested crime type
    model = CRIME_MODELS[crime_type]

    # Predict risk score (probability of crime = 1)
    risk = float(model.predict_proba(X_scaled)[0][1])

    # Local SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    local_waterfall_plot = generate_local_shap_waterfall_plot(
        model=model,
        X_scaled=X_scaled,
        feature_names=FEATURE_ORDER,
        explainer=explainer,
        shap_values=shap_values
    )

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    # Extract top 3 local driving factors
    shap_row = shap_values_pos[0]

    feature_importance = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "shap_value": shap_row
    })

    top_drivers = (
        feature_importance
        .reindex(feature_importance["shap_value"].abs().sort_values(ascending=False).index)
        .head(3)
        .reset_index(drop=True)
    )

    # Local interpretation text
    local_text = local_explanation(
        crime_type=crime_type,
        gn_division=features.get("gn_name", "Selected GN"),
        top_features=top_drivers,
        risk_score=risk
    )

    # Get global SHAP results for this crime type
    global_results = get_global_shap_results(crime_type)

    return {
        "gn_division": features.get("gn_name", "Selected GN"),
        "crime_type": crime_type,
        "risk_score": round(risk, 3),
        "top_features": top_drivers["feature"].tolist(),
        "shap_values": top_drivers["shap_value"].tolist(),
        "global_waterfall_plot": global_results["plot"],
        "local_waterfall_plot": local_waterfall_plot,
        "local_interpretation": local_text,
        "global_interpretation": global_results["text"]
    }

# fetch the GN list 
def get_all_gns():
    query = text("""
        SELECT DISTINCT gn_division
        FROM crime_data
        ORDER BY gn_division    
    """)

    with db.engine.connect() as conn:
        gn_name_list = conn.execute(query).fetchall()

    return [row[0] for row in gn_name_list]




# fetch gn level environmental, socio-economic, demographic,... data
def fetch_gn_features(gn_division_name):
    # 1)Fetch structural features
    query_struct = text("""
        SELECT
            "GN_population",
            "Avg_Household_Income",
            "Unemployment_Rate",
            "Building_Density",
            "Road_Density",
            "distance_to_station_km"
        FROM gn_division_info
        WHERE "admin4Name_en" = :gn
        LIMIT 1
    """)

    with db.engine.connect() as conn:
        struct = conn.execute(query_struct, {"gn":gn_division_name}).mappings().first()

    if not struct:
        raise ValueError("GN Division not found")
    
    # 2)Fetch contexual averages 
    query_contexual = text("""
        SELECT
            AVG(victim_age) AS avg_victim_age,
            AVG(CASE WHEN sex = 'f' THEN 1 ELSE 0 END) AS female_ratio,
            AVG(CASE WHEN is_holiday = '1' THEN 1 ELSE 0 END) AS holiday_ratio,
            AVG(CASE WHEN LOWER(weather) = 'rainy' THEN 1 ELSE 0 END) AS rainy_ratio
        FROM crime_data
        WHERE gn_division = :gn
    """
    )
    with db.engine.connect() as conn:
        context = conn.execute(query_contexual, {"gn":gn_division_name}).mappings().first()

    
    # if not context:
    #     raise ValueError("GN Division not found")

    # Handling missing values safely
    avg_victim_age = context["avg_victim_age"] or 30
    # female_victim_ratio = context["female_ratio"] or 0
    holiday_ratio = context["holiday_ratio"] or 0
    rainy_ratio = context["rainy_ratio"] or 0
    
    # 3) Urban ratios
    # temporary placeholders ################
    land_area_density = 0.5


    # 4) Transform to model schema
    gn_population = struct["GN_population"] or 1
    log_population = np.log1p(gn_population)

    distance_km = struct["distance_to_station_km"] or 0

    # distance_km = struct(static placeholder for now)
    current_week = datetime.now().isocalendar().week
    week_sin = np.sin(2 * np.pi * current_week / 52)
    week_cos = np.cos(2 * np.pi * current_week / 52)

    year = datetime.now().year


    features = {
        "avg_victim_age": float(avg_victim_age),
        "holiday_ratio": float(holiday_ratio),
        "rainy_ratio": float(rainy_ratio),
        "unemployment_rate": float(struct["Unemployment_Rate"] or 0),
        "avg_income": float(struct["Avg_Household_Income"] or 0),
        "building_density": float(struct["Building_Density"] or 0),
        "land_area_density": float(land_area_density),
        "road_density": float(struct["Road_Density"] or 0),
        "log_population": float(log_population),
        "distance_km": float(distance_km),
        "year": float(year)
    }
    features["gn_name"] = gn_division_name

    return features

def build_global_feature_dataset():
    """
    Builds feature dataset for all the GN divisions
    """

    gn_list = get_all_gns()

    rows = []

    for gn in gn_list:
        features = fetch_gn_features(gn)

        # remove helper field
        features.pop("gn_name", None)

        rows.append(features)

    df = pd.DataFrame(rows)

    return df




# . Remove hardcoded crime type
# Add a route that accepts the teammate’s JSON




# receive Component 2 output list

# build a filtered GN list from it

# show only those GN divisions in dropdown

# also use the related crime_type from Component 2 for the selected GN