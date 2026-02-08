import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import text, bindparam
from extensions import db

FEATURES = ["risk_score", "distance_to_station_km", "GN_population", "crime_type_enc", "risk_rank"]
MODEL_PATH = "models/resource_allocation_xgboost.json"

CRIME_TYPE_MAP = {
    "drugs": 0, "robbery": 1, "theft": 2,
    "vehical theft": 3, "buglary": 4, "stabbing": 5,
}

def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

def run_allocation_pipeline(total_officers, max_gns_to_cover, min_per_gn):
    # hotspot_output comes from your hotspot module
    hotspot_output = {
        "crime_type": "drugs",
        "predictions": [
            {"gn_name": "LK2130145", "risk_score": 0.69},
            {"gn_name": "LK2130120", "risk_score": 0.68},
        ]
    }

    crime_type = hotspot_output["crime_type"]
    df = pd.DataFrame(hotspot_output["predictions"])
    df["crime_type"] = crime_type
    df["crime_type_enc"] = CRIME_TYPE_MAP[crime_type]
    df["risk_rank"] = df["risk_score"].rank(ascending=False)

    # Fetch GN features
    gn_names = df["gn_name"].tolist()
    q = text("""
        SELECT "admin4Pcode" AS gn_name,
               "GN_population",
               "distance_to_station_km",
               "closest_police_station"
        FROM gn_division_info
        WHERE "admin4Pcode" IN :g
    """).bindparams(bindparam("g", expanding=True))

    with db.engine.connect() as conn:
        extra = conn.execute(q, {"g": gn_names}).mappings().all()

    df = df.merge(pd.DataFrame(extra), on="gn_name", how="left")

    X = df[FEATURES]
    dmat = xgb.DMatrix(X, feature_names=FEATURES)

    model = load_model()
    df["allocation_score"] = model.get_booster().predict(dmat)

    # allocation logic (reuse yours)
    df["pred_alloc"] = df["allocation_score"].clip(lower=0)
    df["pred_alloc_norm"] = df["pred_alloc"] / df["pred_alloc"].sum()
    df["assigned_officers"] = (df["pred_alloc_norm"] * total_officers).astype(int)

    return crime_type, df


def diminishing_curves_from_df(df, k, max_total, step=5):
    d = df[df["assigned_officers"] > 0]
    if d.empty:
        return [], [], []

    demand = d["pred_alloc_norm"].to_numpy()
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
