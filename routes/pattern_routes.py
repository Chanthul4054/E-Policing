import os
from flask import Blueprint, render_template, request, jsonify, send_file, session
from flask_login import login_required
from services.pattern_service import (
    get_detected_patterns,
    get_crime_trend)

pattern_bp = Blueprint("pattern", __name__, url_prefix="/pattern")

@pattern_bp.route("/")
@login_required
def index():
    crime_type = session.get('selected_crime_type', 'burglary')
    return render_template("pattern.html", selected_crime_type=crime_type)

@pattern_bp.route("/get-risk-data")
@login_required
def get_risk_data():
    crime_type = session.get('selected_crime_type', 'burglary')
    location_type = request.args.get("location_type", "all")
    time_filter = request.args.get("time_filter", "all")

    result = get_detected_patterns(crime_type, location_type, time_filter)

    if result.get("status") == "success":
        compact_results = []
        seen = set()

        for item in result.get("predictions", []):
            gn = item.get("gn_division")
            crime = item.get("crime_type")

            if gn and crime and (gn, crime) not in seen:
                compact_results.append({
                    "gn_division": gn,
                    "crime_type": crime
                })
                seen.add((gn, crime))

        session["pattern_results"] = compact_results
        print("Stored pattern_results directly in browser session:", session.get("pattern_results"))

    return jsonify(result)

@pattern_bp.route("/get-trend")
@login_required
def get_trend():
    crime_type = session.get('selected_crime_type', 'burglary')
    return jsonify(get_crime_trend(crime_type))

@pattern_bp.route("/map-data")
@login_required
def map_data():
    file_path = os.path.join(os.getcwd(), "data/sri-lanka-map.geojson")
    return send_file(file_path, mimetype="application/json")