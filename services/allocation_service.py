from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text, bindparam

from extensions import db

logger = logging.getLogger(__name__)

# Constants

_BASE_DIR = Path(__file__).resolve().parent.parent
_MODEL_DIR = _BASE_DIR / "models"

MODEL_PATH = _MODEL_DIR / "resource_allocation_xgboost_model.pkl"
FEATURES_PATH = _MODEL_DIR / "resource_model_features.pkl"

CRIME_TYPE_MAP = {
    "drugs": 0, "robbery": 1, "theft": 2,
    "vehicle_theft": 3, "burglary": 4, "stabbing": 5,
}

_DEFAULT_MODEL_FEATURES: List[str] = [
    "risk_score_next_week",
    "gn_population",
    "gn_distance_m",
    "Avg_Household_Income",
    "Unemployment_Rate",
    "Building_Density",
    "Road_Density",
    "Land_Area_Density",
    "historical_crime_count",
    "crime_type_diversity",
    "holiday_crime_ratio",
    "night_crime_ratio",
]

_OUTPUT_COLUMNS: List[str] = [
    "gn_pcode",
    "gn_division",
    "crime_type",
    "risk_score_next_week",
    "predicted_demand_score",
    "officers_allocated",
    "allocation_rank",
]

# Exceptions

class HotspotInputError(Exception):
    """Raised when the hotspot module output is invalid or incomplete."""


class ModelLoadError(Exception):
    """Raised when the XGBoost model or its feature list cannot be loaded."""


class FeatureError(Exception):
    """Raised when required features are missing and cannot be recovered."""


# Model loading

def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the pre-trained XGBoost regression model from disk."""
    if not model_path.exists():
        raise ModelLoadError(
            f"Model file not found at '{model_path}'. "
        )
    try:
        model = joblib.load(model_path)
        logger.info("XGBoost resource-allocation model loaded from %s", model_path)
        return model
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load model from '{model_path}': {exc}"
        ) from exc


def load_model_features(features_path: Path = FEATURES_PATH) -> List[str]:
    
    if not features_path.exists():
        logger.warning(
            "Feature list file not found at '%s'. Using default feature list.",
            features_path,
        )
        return list(_DEFAULT_MODEL_FEATURES)
    try:
        features = joblib.load(features_path)
        logger.info("Model features loaded: %s", features)
        return list(features)
    except Exception as exc:
        logger.warning(
            "Could not load feature list from '%s': %s. Using defaults.",
            features_path,
            exc,
        )
        return list(_DEFAULT_MODEL_FEATURES)


# Hotspot JSON → DataFrame

def parse_hotspot_output(hotspot_output: Dict[str, Any]) -> pd.DataFrame:
    
    if not isinstance(hotspot_output, dict):
        raise HotspotInputError(
            "hotspot_output must be a dictionary. "
            f"Received type: {type(hotspot_output).__name__}"
        )

    status = hotspot_output.get("status", "").strip().lower()
    if status != "success":
        raise HotspotInputError(
            f"Hotspot module returned non-success status: '{status}'. "
        )

    predictions = hotspot_output.get("predictions")
    if not predictions or not isinstance(predictions, list):
        raise HotspotInputError(
            "Hotspot output is missing the 'predictions' list or it is empty."
        )

    crime_type = hotspot_output.get("crime_type", "unknown")

    records: List[Dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        gn_name = pred.get("gn_name") or pred.get("pcode_id")
        if not gn_name:
            raise HotspotInputError(
                f"Prediction at index {idx} is missing 'gn_name'. "
                f"Entry: {pred}"
            )
        risk_score = pred.get("risk_score")
        if risk_score is None:
            raise HotspotInputError(
                f"Prediction for '{gn_name}' is missing 'risk_score'."
            )
        records.append(
            {
                "gn_pcode": str(gn_name).strip(),
                "crime_type": str(crime_type).strip(),
                "risk_score_next_week": float(
                    np.clip(pd.to_numeric(risk_score, errors="coerce") or 0.0, 0.0, 1.0)
                ),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise HotspotInputError("Parsed hotspot predictions produced an empty DataFrame.")

    logger.info("Parsed %d hotspot predictions for crime_type='%s'.", len(df), crime_type)
    return df


# Feature preparation

def prepare_features(
    hotspot_df: pd.DataFrame,
    gn_feature_df: pd.DataFrame,
    model_features: List[str],
) -> pd.DataFrame:
    
    if gn_feature_df is None or gn_feature_df.empty:
        raise FeatureError(
            "GN feature DataFrame is empty or None. "
        )

    if "gn_pcode" not in gn_feature_df.columns:
        raise FeatureError(
            "GN feature DataFrame is missing the required 'gn_pcode' column. "
            f"Available columns: {list(gn_feature_df.columns)}"
        )

    gn_feature_clean = gn_feature_df.copy()
    gn_feature_clean["gn_pcode"] = gn_feature_clean["gn_pcode"].astype(str).str.strip()

    merged = hotspot_df.merge(gn_feature_clean, on="gn_pcode", how="left")

    if merged.empty:
        logger.warning(
            "Merge produced zero rows. Verify that gn_pcode values match "
            "between the hotspot output and the GN feature DataFrame."
        )

    for col in model_features:
        if col not in merged.columns:
            logger.warning("Missing feature column '%s' — filling with 0.", col)
            merged[col] = 0

    for col in model_features:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    return merged


# Prediction

def predict_demand(
    df: pd.DataFrame,
    model: Any,
    model_features: List[str],
) -> pd.DataFrame:
    
    X = df[model_features].values
    dmat = xgb.DMatrix(X, feature_names=model_features)

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    raw_scores = booster.predict(dmat)

    df = df.copy()
    df["predicted_demand_score"] = np.clip(raw_scores, 0.0, 1.0)
    logger.info(
        "Demand scores — min=%.4f, max=%.4f, mean=%.4f",
        df["predicted_demand_score"].min(),
        df["predicted_demand_score"].max(),
        df["predicted_demand_score"].mean(),
    )
    return df


# Officer allocation
def allocate_officers(df: pd.DataFrame, total_officers: int) -> pd.DataFrame:
    
    df = df.copy()
    total_demand = df["predicted_demand_score"].sum()

    if total_demand <= 0 or total_officers <= 0:
        df["officers_allocated"] = 0
        df["allocation_rank"] = df["predicted_demand_score"].rank(
            method="dense", ascending=False
        ).astype(int)
        return df

    df["_raw_alloc"] = (df["predicted_demand_score"] / total_demand) * total_officers
    df["officers_allocated"] = df["_raw_alloc"].apply(math.floor)
    df["_remainder"] = df["_raw_alloc"] - df["officers_allocated"]

    remaining = total_officers - int(df["officers_allocated"].sum())
    if remaining > 0:
        top_indices = df["_remainder"].nlargest(remaining).index
        df.loc[top_indices, "officers_allocated"] += 1

    df["officers_allocated"] = df["officers_allocated"].astype(int)
    df["allocation_rank"] = (
        df["predicted_demand_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    df.drop(columns=["_raw_alloc", "_remainder"], inplace=True, errors="ignore")

    logger.info(
        "Allocated %d / %d officers across %d GN divisions.",
        df["officers_allocated"].sum(),
        total_officers,
        len(df),
    )
    return df


# Result formatting

def format_results(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    if "gn_division" not in df.columns:
        df["gn_division"] = "N/A"

    available = [c for c in _OUTPUT_COLUMNS if c in df.columns]
    result = df[available].copy()
    result = result.sort_values("allocation_rank", ascending=True).reset_index(drop=True)
    return result


# Core entry point

def run_resource_allocation(
    hotspot_output: Dict[str, Any],
    gn_feature_df: pd.DataFrame,
    total_officers: int = 120,
    model_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
) -> pd.DataFrame:
    _model_path = Path(model_path) if model_path else MODEL_PATH
    _features_path = Path(features_path) if features_path else FEATURES_PATH

    hotspot_df = parse_hotspot_output(hotspot_output)
    model = load_model(_model_path)
    model_features = load_model_features(_features_path)
    merged_df = prepare_features(hotspot_df, gn_feature_df, model_features)

    if merged_df.empty:
        logger.warning("No rows after merging — returning empty result.")
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    scored_df = predict_demand(merged_df, model, model_features)
    allocated_df = allocate_officers(scored_df, total_officers)
    result_df = format_results(allocated_df)

    logger.info(
        "Resource allocation complete — %d GN divisions, %d officers deployed.",
        len(result_df),
        result_df["officers_allocated"].sum(),
    )
    return result_df


def allocation_to_dict(result_df: pd.DataFrame) -> List[Dict[str, Any]]:
    return result_df.to_dict(orient="records")


# DB helpers


def _load_allowed_gn_codes() -> set:
    path = os.path.join(_BASE_DIR, "static", "gn_div_info", "gn_name_mapping.json")
    with open(path, "r") as f:
        return set(json.load(f).values())


def _fetch_gn_features(gn_pcodes: list) -> pd.DataFrame:
    """Fetch GN-level features from the database for the given pcodes."""
    if not gn_pcodes:
        return pd.DataFrame()

    q = text("""
        SELECT "admin4Pcode"          AS gn_pcode,
               "admin4Name_en"        AS gn_division,
               "GN_population"        AS gn_population,
               "distance_to_station_km" * 1000 AS gn_distance_m,
               "Avg_Household_Income",
               "Unemployment_Rate",
               "Building_Density",
               "Road_Density",
               "closest_police_station"
        FROM gn_division_info
        WHERE "admin4Pcode" IN :g
    """).bindparams(bindparam("g", expanding=True))

    with db.engine.connect() as conn:
        rows = conn.execute(q, {"g": gn_pcodes}).mappings().all()

    return pd.DataFrame(rows)


# Flask pipeline wrapper


def run_allocation_pipeline(total_officers, max_gns_to_cover, min_per_gn, crime_type="drugs"):
    
    from routes.hotspot_routes import generate_risk_scores

    allowed = _load_allowed_gn_codes()
    raw_predictions = generate_risk_scores(crime_type)
    predictions = [p for p in raw_predictions if p["gn_name"] in allowed]

    if not predictions:
        return crime_type, pd.DataFrame()

    hotspot_output = {
        "crime_type": crime_type,
        "status": "success",
        "predictions": predictions,
    }

    gn_pcodes = [p["gn_name"] for p in predictions]
    gn_feature_df = _fetch_gn_features(gn_pcodes)

    if len(predictions) > max_gns_to_cover:
        sorted_preds = sorted(predictions, key=lambda p: p["risk_score"], reverse=True)
        hotspot_output["predictions"] = sorted_preds[:max_gns_to_cover]

    result_df = run_resource_allocation(
        hotspot_output=hotspot_output,
        gn_feature_df=gn_feature_df,
        total_officers=total_officers,
    )

    # Attach closest_police_station for dashboard display
    if not gn_feature_df.empty and "closest_police_station" in gn_feature_df.columns:
        station_map = gn_feature_df.set_index("gn_pcode")["closest_police_station"]
        result_df["closest_police_station"] = (
            result_df["gn_pcode"].map(station_map).fillna("N/A")
        )
    else:
        result_df["closest_police_station"] = "N/A"

    # Compute normalised demand for diminishing-curves chart
    total_demand = result_df["predicted_demand_score"].sum()
    if total_demand > 0:
        result_df["demand_norm"] = result_df["predicted_demand_score"] / total_demand
    else:
        result_df["demand_norm"] = 0

    return crime_type, result_df


# Diminishing returns charts


def diminishing_curves_from_df(df, k, max_total, step=5):
    d = df[df["officers_allocated"] > 0]
    if d.empty:
        return [], [], []

    demand = d["demand_norm"].to_numpy()
    totals, total_benefits, marginal = [], [], []
    prev = 0.0

    for T in range(0, max_total + 1, step):
        officers = (demand * T).astype(int)
        b = np.sum(demand * (1 - np.exp(-k * officers)))
        totals.append(T)
        total_benefits.append(float(b))
        marginal.append(float(b - prev))
        prev = b

    return totals, total_benefits, marginal