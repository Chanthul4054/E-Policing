from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from dotenv import load_dotenv
import os
import pandas as pd
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

MODEL_PATH = os.path.join("models", "models/resource_allocation_xgboost.json")

# The model expects these exact feature names 
FEATURES = [
    "risk_score",
    "distance_to_station_km",
    "GN_population",
    "crime_type_enc",
    "risk_rank",
]

def load_hotspot_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

def fetch_feature_rows(limit=200):
    """
    IMPORTANT: change `hotspot_features` to the real table/view that contains
    these columns: risk_score, distance_to_station_km, GN_population,
    crime_type_enc, risk_rank

    Also include an id/location column so you can identify the row in the output.
    """
    sql = text(f"""
        SELECT
            id,
            {", ".join(FEATURES)}
        FROM hotspot_features
        LIMIT :limit
    """)

    with db.engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()

    # rows is list[dict]
    return pd.DataFrame(rows)

@app.route("/")
def index():
    df = fetch_feature_rows(limit=200)

    if df.empty:
        return render_template("index.html", cols=[], rows=[])

    model = load_hotspot_model()

    # Predict allocation score
    X = df[FEATURES]
    df["allocation_score"] = model.predict(X)

    # Optional: sort by highest score first
    df = df.sort_values("allocation_score", ascending=False)

    cols = df.columns.tolist()
    rows = df.to_dict(orient="records")
    return render_template("index.html", cols=cols, rows=rows)

if __name__ == "__main__":
    app.run(debug=True)
