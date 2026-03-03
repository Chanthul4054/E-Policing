import joblib
import json
import pandas as pd
import numpy as np
import os
import shap
from extensions import db
from sqlalchemy import text
from datetime import datetime

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


def run_risk_factor_pipeline(crime_type, features):
    """
    Runs ML inference and generate SHAP explanation
    """

    if crime_type not in CRIME_MODELS:
        raise ValueError("Invalid crime type")

    # convert to dataframe 
    X = pd.DataFrame([features])
    X = X[FEATURE_ORDER]

    # scale features
    X_scaled = scaler.transform(X)

    # PREDICT
    model = CRIME_MODELS[crime_type]
    # predict_proba - returns the probabilities of crime and no crime
    # float -  converts Numpy float in to python float(JSON safe)
    # this is the risk score shown on the dashboard
    risk = float(model.predict_proba(X_scaled)[0][1])

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Handle different SHAP output shapes
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif len(shap_values.shape) ==3:
        shap_values_pos = shap_values[:,:,1]
    else:
        shap_values_pos = shap_values

    shap_row = shap_values_pos[0]



    # get top 3 drivers
    shap_row = shap_values_pos[0]
    feature_importance = pd.DataFrame({
        "feature":FEATURE_ORDER,
        "shap_value":shap_row
    })

    top_drivers = (
        feature_importance
        .reindex(feature_importance["shap_value"].abs().sort_values(ascending=False).index)
        .head(3)
    )

    return {
        "risk_score":round(risk, 3),
        "top_features":top_drivers["feature"].tolist(),
        "shap_values":top_drivers["shap_value"].tolist()
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

    return features


        