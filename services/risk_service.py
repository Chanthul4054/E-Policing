import joblib
import json
import pandas as pd
import numpy as np
import os
import shap
from extensions import db
from sqlalchemy import text
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "risk_factor_models")
if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

CRIME_TYPES = ["burglary", "theft", "vehicle", "robbery", "drugs", "stabbing"]

CRIME_MODELS = {
    crime: joblib.load(os.path.join(MODEL_DIR, f"rf_{crime}.pkl"))
    for crime in CRIME_TYPES
}

SCALERS = {
    crime: joblib.load(os.path.join(MODEL_DIR, f"scaler_{crime}.pkl"))
    for crime in CRIME_TYPES
}

with open(os.path.join(MODEL_DIR, "feature_lists.json"), "r") as f:
    FEATURE_LISTS = json.load(f)

with open(os.path.join(MODEL_DIR, "thresholds.json"), "r") as f:
    THRESHOLDS = json.load(f)


FEATURE_LABELS = {
    "unemployment_rate": "Unemployment",
    "avg_income": "Household Income",

    "building_density": "Building Density",
    "land_area_density": "Land Use Intensity",
    "road_density": "Road Density",
    "log_population": "Population",
    "distance_km": "Distance to Police Station",

    "week_of_year": "Time of Year",

    "avg_victim_age_lag1": "Recent Victim Age",
    "avg_victim_age_lag2": "Recent Victim Age",
    "avg_victim_age_roll4": "Recent Victim Age",

    "female_victim_ratio_lag1": "Recent Female Victims",
    "female_victim_ratio_lag2": "Recent Female Victims",
    "female_victim_ratio_roll4": "Recent Female Victims",

    "urban_ratio_lag1": "Urban Activity",
    "urban_ratio_lag2": "Urban Activity",
    "urban_ratio_roll4": "Urban Activity",

    "holiday_ratio_lag1": "Recent Holidays",
    "holiday_ratio_lag2": "Recent Holidays",
    "holiday_ratio_roll4": "Recent Holidays",

    "rainy_ratio_lag1": "Recent Rainy Weather",
    "rainy_ratio_lag2": "Recent Rainy Weather",
    "rainy_ratio_roll4": "Recent Rainy Weather",
}
DISPLAY_EXCLUDE_FEATURES = {
    "year",
    "week_of_year",
    "week_sin",
    "week_cos",

    "unemployment_rate",
    "avg_income",

    "total_crimes_lag1",
    "total_crimes_lag2",
    "total_crimes_lag4",
    "total_crimes_roll4",
    "total_crimes_roll8",
    "total_crimes_trend_4v8",

    "burglary_count_lag1",
    "burglary_count_lag2",
    "burglary_count_lag4",
    "burglary_count_roll4",
    "burglary_count_roll8",
    "burglary_count_trend_4v8",

    "theft_count_lag1",
    "theft_count_lag2",
    "theft_count_lag4",
    "theft_count_roll4",
    "theft_count_roll8",
    "theft_count_trend_4v8",

    "vehicle_count_lag1",
    "vehicle_count_lag2",
    "vehicle_count_lag4",
    "vehicle_count_roll4",
    "vehicle_count_roll8",
    "vehicle_count_trend_4v8",

    "robbery_count_lag1",
    "robbery_count_lag2",
    "robbery_count_lag4",
    "robbery_count_roll4",
    "robbery_count_roll8",
    "robbery_count_trend_4v8",

    "drugs_count_lag1",
    "drugs_count_lag2",
    "drugs_count_lag4",
    "drugs_count_roll4",
    "drugs_count_roll8",
    "drugs_count_trend_4v8",

    "stabbing_count_lag1",
    "stabbing_count_lag2",
    "stabbing_count_lag4",
    "stabbing_count_roll4",
    "stabbing_count_roll8",
    "stabbing_count_trend_4v8",

    "avg_victim_age_lag1",
    "avg_victim_age_lag2",
    "avg_victim_age_roll4",

    "female_victim_ratio_lag1",
    "female_victim_ratio_lag2",
    "female_victim_ratio_roll4"
}

FEATURE_DESCRIPTIONS = {
    "unemployment_rate": "The percentage of people without jobs in the area.",
    "avg_income": "The average income of households in the area.",
    "building_density": "How closely buildings are packed together.",
    "land_area_density": "How much land is actively used compared to unused land.",
    "road_density": "How many roads are present in the area.",
    "log_population": "The number of people living in the area.",
    "distance_km": "How far the area is from the nearest police station.",

    "avg_victim_age_lag1": "Average age of victims in recent weeks.",
    "avg_victim_age_lag2": "Average age of victims earlier.",
    "avg_victim_age_roll4": "Average victim age over the last few weeks.",

    "female_victim_ratio_lag1": "Proportion of female victims recently.",
    "female_victim_ratio_lag2": "Proportion of female victims earlier.",
    "female_victim_ratio_roll4": "Proportion of female victims recently.",

    "urban_ratio_lag1": "How urban (developed) the area is.",
    "urban_ratio_lag2": "How urban (developed) the area is.",
    "urban_ratio_roll4": "How urban (developed) the area is.",

    "holiday_ratio_lag1": "Presence of holidays in recent weeks.",
    "holiday_ratio_lag2": "Presence of holidays earlier.",
    "holiday_ratio_roll4": "Holiday trend over recent weeks.",

    "rainy_ratio_lag1": "How much rain occurred recently.",
    "rainy_ratio_lag2": "Rainfall earlier.",
    "rainy_ratio_roll4": "Rainfall trend over recent weeks."
}
CRIME_LABELS = {
    "burglary": "Burglary",
    "theft": "Theft",
    "vehicle": "Vehicle Theft",
    "robbery": "Robbery",
    "drugs": "Drug-related Crime",
    "stabbing": "Stabbing"
}

CRIME_TYPE_ALIASES = {
    "vehicle_theft": "vehicle",
    "vehicle theft": "vehicle",
    "vehicle": "vehicle",
    "theft": "theft",
    "burglary": "burglary",
    "robbery": "robbery",
    "drugs": "drugs",
    "stabbing": "stabbing"
}

GLOBAL_SHAP_CACHE = {}

def build_global_feature_dataset(crime_type):
    gn_list = get_all_gns()
    feature_order = FEATURE_LISTS[crime_type]
    rows = []

    for gn in gn_list:
        try:
            row = fetch_latest_feature_row(gn)
            filtered_row = {f: row.get(f, 0.0) for f in feature_order}
            rows.append(filtered_row)
        except Exception:
            continue

    return pd.DataFrame(rows)

def get_global_shap_results(crime_type):
    global GLOBAL_SHAP_CACHE

    if crime_type in GLOBAL_SHAP_CACHE:
        return GLOBAL_SHAP_CACHE[crime_type]

    model = CRIME_MODELS[crime_type]
    scaler = SCALERS[crime_type]
    feature_order = FEATURE_LISTS[crime_type]

    global_df = build_global_feature_dataset(crime_type)

    X_global = global_df[feature_order]
    X_scaled = scaler.transform(X_global)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    global_plot = generate_global_shap_bar_plot(
        X_scaled,
        shap_values,
        feature_order
    )

    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif len(np.array(shap_values).shape) == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    global_feature_df = pd.DataFrame({
        "feature": feature_order,
        "shap_value": np.mean(shap_values_pos, axis=0),
        "avg_feature_value": np.mean(X_scaled, axis=0)
    })

    filtered_global = filter_display_features(global_feature_df, top_n=5)

    top_features = filtered_global["feature"].tolist()

    global_top_features = [
        {
            "label": make_dynamic_global_label(row["feature"], row["avg_feature_value"]),
            "description": FEATURE_DESCRIPTIONS.get(row["feature"], "No description available.")
        }
        for _, row in filtered_global.iterrows()
    ]

    GLOBAL_SHAP_CACHE[crime_type] = {
        "plot": global_plot,
        "text": global_explanation(crime_type, top_features, global_feature_df),
        "top_features": global_top_features
    }

    return GLOBAL_SHAP_CACHE[crime_type]

def generate_global_shap_bar_plot(X_background, shap_values, feature_order):
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(np.array(shap_values).shape) == 3:
        shap_values = shap_values[:, :, 1]

    mean_shap = np.mean(shap_values, axis=0)
    mean_feature_values = np.mean(X_background, axis=0)

    chart_df = pd.DataFrame({
        "feature": feature_order,
        "shap_value": mean_shap,
        "avg_feature_value": mean_feature_values
    })

    chart_df = filter_display_features(chart_df, top_n=5).copy()

    chart_df["label"] = chart_df.apply(
        lambda row: make_dynamic_global_label(row["feature"], row["avg_feature_value"]),
        axis=1
    )

    chart_df = chart_df.sort_values("shap_value")

    print(chart_df[["feature", "avg_feature_value", "shap_value"]])

    colors = ["#e56b6f" if v > 0 else "#57cc99" for v in chart_df["shap_value"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.barh(chart_df["label"], chart_df["shap_value"], color=colors, height=0.55)
    ax.axvline(0, color="#94a3b8", linewidth=1)

    ax.set_title("Top 5 factors affecting crime risk across the division", fontsize=12, pad=10)
    ax.set_xlabel("Left = reduces risk   |   Right = increases risk", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xticks([])

    max_abs = max(abs(chart_df["shap_value"].min()), abs(chart_df["shap_value"].max()), 0.001)
    ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220, facecolor="white")
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_local_plot(feature_importance, gn_name, crime_type):
    chart_df = filter_display_features(feature_importance, top_n=3)
    chart_df["label"] = chart_df.apply(
        lambda row: make_dynamic_feature_label(row["feature"], row["feature_value"]),
        axis=1
    )
    chart_df = chart_df.sort_values("shap_value")

    colors = ["#e56b6f" if v > 0 else "#57cc99" for v in chart_df["shap_value"]]

    plt.figure(figsize=(8.2, 3.8))
    plt.barh(chart_df["label"], chart_df["shap_value"], color=colors, height=0.55)
    plt.axvline(0, color="#94a3b8", linewidth=1.1)

    crime_name = CRIME_LABELS.get(crime_type, crime_type.title())
    plt.title(f"Main factors affecting {crime_name.lower()} risk in {gn_name}", fontsize=12)
    plt.xlabel("Left = reduces risk    |    Right = increases risk", fontsize=10)
    plt.ylabel("")
    plt.xticks([])
    plt.tick_params(axis='y', labelsize=11)
    max_abs = max(abs(chart_df["shap_value"].min()), abs(chart_df["shap_value"].max()), 0.001)
    plt.xlim(-max_abs * 1.15, max_abs * 1.15)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    plt.close()

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def interpret_risk_score(risk_score):
    if risk_score < 0.2:
        return "Low Risk", "Routine monitoring is sufficient"
    elif risk_score < 0.5:
        return "Moderate Risk", "Increased awareness is recommended"
    elif risk_score < 0.75:
        return "High Risk", "Consider increasing patrol presence"
    else:
        return "Very High Risk", "Immediate attention and action required"
    

def format_feature_list(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

EPS = 1e-6

def local_explanation(crime_type, gn_division, top_features, risk_score):
    crime_name = CRIME_LABELS.get(crime_type, crime_type.title())

    increasing_features = [
        make_dynamic_feature_label(row["feature"], row["feature_value"]).lower()
        for _, row in top_features.iterrows()
        if row["shap_value"] > EPS
    ]

    reducing_features = [
        make_dynamic_feature_label(row["feature"], row["feature_value"]).lower()
        for _, row in top_features.iterrows()
        if row["shap_value"] < -EPS
    ]

    risk_label, action = interpret_risk_score(risk_score)

    parts = []

    if increasing_features:
        parts.append(
            f"factors increasing {crime_name.lower()} risk include {format_feature_list(increasing_features)}"
        )

    if reducing_features:
        parts.append(
            f"factors reducing {crime_name.lower()} risk include {format_feature_list(reducing_features)}"
        )

    if not parts:
        parts.append(
            f"no strong increasing or reducing factors are clearly shown for {crime_name.lower()} risk"
        )

    return (
        f"In {gn_division}, " + ". ".join(parts) + ". "
        f"This area currently shows a {risk_label.lower()} ({risk_score:.0%}). {action}."
    )

def global_explanation(crime_type, top_features, global_feature_df):
    crime_name = CRIME_LABELS.get(crime_type, crime_type.title())

    top_rows = (
        global_feature_df
        .set_index("feature")
        .loc[top_features]
        .reset_index()
    )

    increasing_features = [
        make_dynamic_global_label(row["feature"], row["avg_feature_value"])
        for _, row in top_rows.iterrows()
        if row["shap_value"] > EPS
    ]

    reducing_features = [
        make_dynamic_global_label(row["feature"], row["avg_feature_value"])
        for _, row in top_rows.iterrows()
        if row["shap_value"] < -EPS
    ]

    parts = []

    if increasing_features:
        parts.append(
            f"the main patterns increasing {crime_name.lower()} risk are {format_feature_list(increasing_features)}"
        )

    if reducing_features:
        parts.append(
            f"the main patterns reducing {crime_name.lower()} risk are {format_feature_list(reducing_features)}"
        )

    if not parts:
        parts.append(
            f"no strong increasing or reducing patterns are clearly shown for {crime_name.lower()} risk"
        )

    return (
        f"Across all Grama Niladhari divisions, " + ". ".join(parts) + "."
    )
 
def run_risk_factor_pipeline(crime_type, features):
    normalized_crime_type = CRIME_TYPE_ALIASES.get(crime_type.lower().strip())

    if normalized_crime_type not in CRIME_MODELS:
        raise ValueError("Invalid crime type")

    crime_label = CRIME_LABELS.get(
        normalized_crime_type,
        normalized_crime_type.replace("_", " ").title()
    )

    model = CRIME_MODELS[normalized_crime_type]
    scaler = SCALERS[normalized_crime_type]
    feature_order = FEATURE_LISTS[normalized_crime_type]
    threshold = float(THRESHOLDS.get(normalized_crime_type, 0.5))

    X = pd.DataFrame([{f: features.get(f, 0.0) for f in feature_order}])

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_order
    )

    risk = float(model.predict_proba(X_scaled)[0][1])
    predicted_positive = risk >= threshold

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif len(np.array(shap_values).shape) == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    shap_row = shap_values_pos[0]

    feature_importance = pd.DataFrame({
        "feature": feature_order,
        "shap_value": shap_row,
        "feature_value": X_scaled.iloc[0].values
    })

    top_drivers = (
        feature_importance
        .reindex(feature_importance["shap_value"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )

    top_drivers = filter_display_features(top_drivers, top_n=3)

    local_plot = generate_local_plot(
        feature_importance=top_drivers,
        gn_name=features.get("gn_name", "Selected GN"),
        crime_type=normalized_crime_type
    )

    global_results = get_global_shap_results(normalized_crime_type)

    return {
        "gn_division": features.get("gn_name", "Selected GN"),
        "crime_type": normalized_crime_type,
        "crime_label": crime_label,
        "feature_week": str(features.get("week", "")),
        "risk_score": round(risk, 3),
        "risk_percentage": round(risk * 100, 1),
        "threshold": round(threshold, 3),
        "predicted_positive": predicted_positive,
        "risk_level": get_risk_level(risk, threshold),
        "top_features": [
            {
                "label": make_dynamic_feature_label(row["feature"], row["feature_value"]),
                "description": FEATURE_DESCRIPTIONS.get(row["feature"], "No description available.")
            }
            for _, row in top_drivers.iterrows()
        ],
        "global_top_features": global_results["top_features"],
        "shap_values": top_drivers["shap_value"].tolist(),
        "global_waterfall_plot": global_results["plot"],
        "local_waterfall_plot": local_plot,
        "local_interpretation": local_explanation(
            crime_type=normalized_crime_type,
            gn_division=features.get("gn_name", "Selected GN"),
            top_features=top_drivers,
            risk_score=risk
        ),
        "global_interpretation": global_results["text"]
    }

# fetch the GN list 
def get_all_gns():
    query = text("""
        SELECT DISTINCT gn_division
        FROM risk_feature_store
        ORDER BY gn_division
    """)

    with db.engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    return [row[0] for row in rows]

def get_risk_level(risk_score, threshold):
    if risk_score < threshold * 0.75:
        return "Low"
    elif risk_score < threshold:
        return "Moderate"
    else:
        return "High"

# to read data from the risk_feature_store
def fetch_latest_feature_row(gn_division_name):
    query = text("""
        SELECT *
        FROM risk_feature_store
        WHERE gn_division = :gn
        ORDER BY week DESC
        LIMIT 1
    """)

    with db.engine.connect() as conn:
        row = conn.execute(query, {"gn": gn_division_name}).mappings().first()

    if not row:
        raise ValueError("No feature data found for selected GN division")

    return dict(row)

def filter_display_features(feature_df, top_n=5):
    filtered = feature_df.loc[
        ~feature_df["feature"].isin(DISPLAY_EXCLUDE_FEATURES)
    ].copy()

    filtered = filtered.reindex(
        filtered["shap_value"].abs().sort_values(ascending=False).index
    ).head(top_n)

    return filtered.reset_index(drop=True)

def make_dynamic_feature_label(feature, feature_value):
    base = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

    if feature == "distance_km":
        return "Greater Distance to Police Station" if feature_value > 0 else "Shorter Distance to Police Station"

    if feature in {"avg_income"}:
        return f"Higher {base}" if feature_value > 0 else f"Lower {base}"

    if feature in {
        "unemployment_rate", "building_density", "land_area_density",
        "road_density", "log_population", "avg_victim_age_lag1",
        "avg_victim_age_lag2", "avg_victim_age_roll4",
        "female_victim_ratio_lag1", "female_victim_ratio_lag2", "female_victim_ratio_roll4",
        "urban_ratio_lag1", "urban_ratio_lag2", "urban_ratio_roll4",
        "holiday_ratio_lag1", "holiday_ratio_lag2", "holiday_ratio_roll4",
        "rainy_ratio_lag1", "rainy_ratio_lag2", "rainy_ratio_roll4"
    }:
        return f"Higher {base}" if feature_value > 0 else f"Lower {base}"

    return base

def make_dynamic_global_label(feature, avg_feature_value):
    base = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

    if feature == "distance_km":
        return "Greater Distance to Police Station" if avg_feature_value > 0 else "Shorter Distance to Police Station"

    return f"Higher {base}" if avg_feature_value > 0 else f"Lower {base}"