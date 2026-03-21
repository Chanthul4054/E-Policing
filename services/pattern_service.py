import json
import requests
import pandas as pd
from functools import lru_cache
from flask import request as flask_request

# ── Load dataset once at startup ────────────────────────────────
try:
    _df = pd.read_csv("data/CrimeData_Final.csv")
    _df["_year"] = pd.to_datetime(_df["date"], format="%m/%d/%Y", errors="coerce").dt.year
except Exception as e:
    _df = None
    print("Warning: Could not load crime dataset:", e)


# ── Fetch live risk scores from hotspot API ──────────────────────
def fetch_risk_scores(crime_type):
    try:
        # Forward the session cookies so login_required passes
        cookies = flask_request.cookies

        response = requests.get(
            "http://localhost:5000/hotspot/predict",
            params={"type": crime_type},
            cookies=cookies,
            timeout=10
        )
        data = response.json()
        if data.get("status") == "success":
            return data["predictions"]
        else:
            print("Hotspot API error:", data)
            return []
    except Exception as e:
        print("Failed to fetch risk scores:", e)
        return []


# ── Pattern strength calculator ──────────────────────────────────
def pattern_strength(top_patterns):
    if not top_patterns:
        return "none"
    best       = max(top_patterns, key=lambda r: r.get("lift", 1.0) * r.get("confidence", 0.0))
    lift       = best.get("lift",       1.0)
    confidence = best.get("confidence", 0.0)
    if lift > 2.0 and confidence >= 0.7:
        return "high"
    elif lift >= 1.5 or confidence >= 0.5:
        return "medium"
    elif lift >= 1.2:
        return "low"
    else:
        return "none"


# ── Main pattern detection ───────────────────────────────────────
def get_detected_patterns(crime_type, location_type=None, time_filter=None):

    with open("models/final_rule_database.json", "r") as f:
        rules = json.load(f)

    # ✅ Get live predictions from hotspot API with session cookies
    live_predictions = fetch_risk_scores(crime_type)

    if not live_predictions:
        print(f"Warning: No predictions returned for {crime_type}")
        return {"status": "success", "crime_type": crime_type, "predictions": []}

    top10 = sorted(live_predictions, key=lambda x: x['risk_score'], reverse=True)[:10]

    results = []

    for item in top10:
        gn_code    = item['pcode_id']
        risk_score = item['risk_score']

        matched_rules = [
            rule for rule in rules
            if rule['gn_pcode']    == gn_code
            and rule['crime_type'] == crime_type
            and (
                location_type is None
                or location_type == 'all'
                or rule.get('location_type', '').lower() == location_type.lower()
            )
            and (
                time_filter is None
                or time_filter == 'all'
                or time_filter.lower() in rule.get('pattern_text', '').lower().replace('time late night', 'time night')
            )
        ]

        heatmap_rules = [
            rule for rule in rules
            if rule['gn_pcode']    == gn_code
            and rule['crime_type'] == crime_type
        ]

        matched_rules = sorted(matched_rules, key=lambda x: x['confidence'], reverse=True)
        heatmap_rules = sorted(heatmap_rules, key=lambda x: x['confidence'], reverse=True)

        top5 = matched_rules[:5]

        results.append({
            "gn_pcode":         gn_code,
            "gn_division":      top5[0]['gn_division'] if top5 else item.get('display_name', gn_code),
            "crime_type":       crime_type,
            "risk_score":       risk_score,
            "top_patterns":     top5,
            "all_patterns":     heatmap_rules,
            "pattern_strength": pattern_strength(top5)
        })

    return {
        "status":      "success",
        "crime_type":  crime_type,
        "predictions": results
    }


# ── Crime trend by year ──────────────────────────────────────────
@lru_cache(maxsize=20)
def get_crime_trend(crime_type):
    YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
    if _df is None:
        return {"labels": [str(y) for y in YEARS], "data": [0] * len(YEARS)}
    filtered      = _df[_df["crime"].str.lower() == crime_type.lower()].copy()
    yearly_counts = filtered.groupby("_year").size().to_dict() if not filtered.empty else {}
    data          = [int(yearly_counts.get(y, 0)) for y in YEARS]
    return {"labels": [str(y) for y in YEARS], "data": data}