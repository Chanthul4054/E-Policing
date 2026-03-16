import os
from flask import Blueprint, render_template, request, jsonify, send_file, session
from flask_login import login_required
from services.pattern_service import get_detected_patterns, get_crime_trend

pattern_bp = Blueprint("pattern", __name__, url_prefix="/pattern")

@pattern_bp.route("/")
@login_required
def index():
    crime_type = session.get('selected_crime_type', 'burglary')
    return render_template("pattern.html", selected_crime_type=crime_type)

@pattern_bp.route("/get-risk-data")
@login_required
def get_risk_data():
    crime_type    = session.get('selected_crime_type', 'burglary')
    location_type = request.args.get("location_type", "all")
    time_filter   = request.args.get("time_filter",   "all")
    return jsonify(get_detected_patterns(crime_type, location_type, time_filter))

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