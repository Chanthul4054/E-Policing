from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, bindparam
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import xgboost as xgb

load_dotenv()

db = SQLAlchemy()
app = Flask(__name__)

db_uri = os.getenv("DB_URI")
if not db_uri:
    raise RuntimeError("DB_URI is not set in .env")

app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# ---------------- Resource Allocation pipeline ----------------
FEATURES = ["risk_score", "distance_to_station_km", "GN_population", "crime_type_enc", "risk_rank"]
MODEL_PATH = "models/resource_allocation_xgboost.json"

CRIME_TYPE_MAP = {
    "drugs": 0,
    "robbery": 1,
    "theft": 2,
    "vehical theft": 3,
    "buglary": 4,
    "stabbing": 5,
}

def load_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

def fetch_gn_features(gn_names: list[str]) -> pd.DataFrame:
    if not gn_names:
        return pd.DataFrame(columns=["gn_name", "GN_population", "distance_to_station_km", "closest_police_station"])

    q = text("""
        SELECT
            "admin4Pcode" AS "gn_name",
            "GN_population" AS "GN_population",
            "distance_to_station_km" AS "distance_to_station_km",
            "closest_police_station" AS "closest_police_station"
        FROM gn_division_info
        WHERE "admin4Pcode" IN :gn_names
    """).bindparams(bindparam("gn_names", expanding=True))

    with db.engine.connect() as conn:
        rows = conn.execute(q, {"gn_names": gn_names}).mappings().all()

    return pd.DataFrame(rows)

def build_df_from_hotspot_output(hotspot_output: dict) -> pd.DataFrame:
    crime_type = hotspot_output.get("crime_type", "drugs")
    preds = hotspot_output.get("predictions", [])
    if not preds:
        return pd.DataFrame()

    df = pd.DataFrame(preds)
    df["crime_type"] = crime_type

    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df.loc[~df["risk_score"].between(0, 1), "risk_score"] = np.nan
    df["risk_score"] = df["risk_score"].fillna(df["risk_score"].median())

    df["crime_type_enc"] = df["crime_type"].map(CRIME_TYPE_MAP).fillna(0).astype(int)
    df["risk_rank"] = df["risk_score"].rank(ascending=False, method="first")

    return df[["gn_name", "risk_score", "crime_type", "crime_type_enc", "risk_rank"]].copy()

def allocate_officers(df: pd.DataFrame, total_officers=500, max_gns_to_cover=80, min_per_gn=1) -> pd.DataFrame:
    df = df.copy()
    df["pred_alloc"] = pd.to_numeric(df["allocation_score"], errors="coerce").fillna(0.0).clip(lower=0.0)

    df = df.sort_values("pred_alloc", ascending=False).reset_index(drop=True)
    deploy = df.head(max_gns_to_cover).copy()
    non_deploy = df.iloc[max_gns_to_cover:].copy()

    s = float(deploy["pred_alloc"].sum())
    deploy["pred_alloc_norm"] = (deploy["pred_alloc"] / s) if (s > 0 and np.isfinite(s)) else (1.0 / len(deploy) if len(deploy) else 0.0)

    deploy["assigned_officers"] = np.floor(deploy["pred_alloc_norm"] * total_officers).astype(int)
    deploy.loc[deploy["assigned_officers"] < min_per_gn, "assigned_officers"] = min_per_gn

    diff = int(total_officers - deploy["assigned_officers"].sum())
    deploy = deploy.sort_values("pred_alloc_norm", ascending=False).reset_index(drop=True)

    i = 0
    while diff != 0 and i < 100000 and len(deploy) > 0:
        idx = i % len(deploy)
        if diff > 0:
            deploy.loc[idx, "assigned_officers"] += 1
            diff -= 1
        else:
            if deploy.loc[idx, "assigned_officers"] > min_per_gn:
                deploy.loc[idx, "assigned_officers"] -= 1
                diff += 1
        i += 1

    non_deploy["pred_alloc_norm"] = 0.0
    non_deploy["assigned_officers"] = 0

    return pd.concat([deploy, non_deploy], ignore_index=True)

def run_allocation_pipeline(total_officers, max_gns_to_cover, min_per_gn):
    """
    Runs your full pipeline and returns (crime_type, df_out)
    """
    # Replace this with real hotspot output
    hotspot_output = {
        "crime_type": "drugs",
        "predictions": [
            {"gn_encoded": 31, "gn_name": "LK2130145", "risk_score": 0.6906274641},
            {"gn_encoded": 26, "gn_name": "LK2130120", "risk_score": 0.6855819647},
            {"gn_encoded": 29, "gn_name": "LK2130135", "risk_score": 0.6193563941},
        ],
        "status": "success"
    }

    crime_type = hotspot_output.get("crime_type", "unknown")

    df_hot = build_df_from_hotspot_output(hotspot_output)
    if df_hot.empty:
        return crime_type, pd.DataFrame()

    df_db = fetch_gn_features(df_hot["gn_name"].tolist())
    df = df_hot.merge(df_db, on="gn_name", how="left")

    df["GN_population"] = pd.to_numeric(df["GN_population"], errors="coerce")
    df["distance_to_station_km"] = pd.to_numeric(df["distance_to_station_km"], errors="coerce")

    pop_med = float(df["GN_population"].median()) if df["GN_population"].notna().any() else 0.0
    dist_med = float(df["distance_to_station_km"].median()) if df["distance_to_station_km"].notna().any() else 0.0

    df.loc[df["GN_population"].isna() | (df["GN_population"] <= 0), "GN_population"] = pop_med
    df.loc[df["distance_to_station_km"].isna() | (df["distance_to_station_km"] < 0), "distance_to_station_km"] = dist_med
    df["closest_police_station"] = df["closest_police_station"].fillna("Unknown")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features for model: {missing}. Have: {df.columns.tolist()}")

    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["risk_score"] = df["risk_score"].fillna(df["risk_score"].median())
    df["distance_to_station_km"] = df["distance_to_station_km"].fillna(df["distance_to_station_km"].median())
    df["GN_population"] = df["GN_population"].fillna(df["GN_population"].median())
    df["crime_type_enc"] = df["crime_type_enc"].fillna(0)
    df["risk_rank"] = df["risk_rank"].fillna(df["risk_rank"].median())

    X = df.loc[:, FEATURES].copy()
    dmat = xgb.DMatrix(X, feature_names=FEATURES)

    model = load_model()
    df["allocation_score"] = model.get_booster().predict(dmat)
    df["allocation_score"] = pd.to_numeric(df["allocation_score"], errors="coerce").fillna(0.0)

    df = allocate_officers(
        df,
        total_officers=total_officers,
        max_gns_to_cover=max_gns_to_cover,
        min_per_gn=min_per_gn
    )

    return crime_type, df

def diminishing_curves_from_df(df, k=0.25, max_total_officers=500, step=5):
    d = df[df["assigned_officers"] > 0].copy()
    if d.empty:
        return [], [], []

    d = d.sort_values("pred_alloc_norm", ascending=False).reset_index(drop=True)
    demand = d["pred_alloc_norm"].to_numpy()

    totals = []
    total_benefits = []
    marginal = []
    prev = 0.0

    for T in range(0, int(max_total_officers) + 1, int(step)):
        officers = (demand * T).astype(int)
        b = float(np.sum(demand * (1.0 - np.exp(-float(k) * officers))))
        totals.append(T)
        total_benefits.append(b)
        marginal.append(b - prev)
        prev = b

    return totals, total_benefits, marginal

# ---------------- Routes ----------------

@app.route("/")
def index():
    total_officers = int(request.args.get("officers", 500))
    max_gns_to_cover = int(request.args.get("topk", 80))
    min_per_gn = int(request.args.get("min_per_gn", 1))
    k = float(request.args.get("k", 0.25))
    chart_max = int(request.args.get("chart_max", total_officers))

    crime_type, df = run_allocation_pipeline(total_officers, max_gns_to_cover, min_per_gn)

    cols = [
        "gn_name",
        "crime_type",
        "risk_score",
        "distance_to_station_km",
        "GN_population",
        "closest_police_station",
        "risk_rank",
        "allocation_score",
        "pred_alloc",
        "pred_alloc_norm",
        "assigned_officers",
    ]

    rows = []
    if not df.empty:
        df = df.sort_values(["assigned_officers", "allocation_score"], ascending=[False, False])
        rows = df[cols].to_dict(orient="records")

    return render_template(
        "index.html",
        cols=cols,
        rows=rows,
        crime_type=crime_type,
        total_officers=total_officers,
        max_gns_to_cover=max_gns_to_cover,
        min_per_gn=min_per_gn,
        k=k,
        chart_max=chart_max,
    )

@app.route("/api/diminishing")
def api_diminishing():
    total_officers = int(request.args.get("officers", 500))
    max_gns_to_cover = int(request.args.get("topk", 80))
    min_per_gn = int(request.args.get("min_per_gn", 1))
    k = float(request.args.get("k", 0.25))
    chart_max = int(request.args.get("chart_max", total_officers))
    step = int(request.args.get("step", 5))

    _, df = run_allocation_pipeline(total_officers, max_gns_to_cover, min_per_gn)
    totals, total_benefits, marginal = diminishing_curves_from_df(df, k=k, max_total_officers=chart_max, step=step)

    return jsonify({
        "totals": totals,
        "total_benefits": total_benefits,
        "marginal_benefits": marginal
    })

if __name__ == "__main__":
    app.run(debug=True)