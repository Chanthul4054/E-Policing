from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from services.risk_service import run_risk_factor_pipeline, get_all_gns, fetch_gn_features

risk_bp = Blueprint("risk", __name__)

@risk_bp.route("/", methods=["GET","POST"])
@login_required
def index():
    gn_name_list = get_all_gns()
    result = None
    if request.method == "POST":
        gn_division = request.form.get("gn_division")
        
        features = fetch_gn_features(gn_division)
        crime_type = "burglary"
        result = run_risk_factor_pipeline(crime_type, features)

        if not crime_type:
            return render_template(
                "risk.html",
                gn_name_list=gn_name_list,
                result=None,
                error = "Crime type not set"
            )
        if not gn_division:
            return render_template(
                "risk.html",
                gn_name_list=gn_name_list,
                result=None,
                error = "GN Division not set"
            )

    return render_template(
        "risk.html",
        gn_name_list=gn_name_list,
        result=result
    )



@risk_bp.route("/api/explain", methods=["POST"])
@login_required
def explain():
    payload = request.get_json()

    crime_type = payload.get("crime_type")
    features = payload.get("features")
    
    try:
        result =run_risk_factor_pipeline(crime_type, features)
    except ValueError as e:
        return jsonify({"error":str(e)}),400
    
    return jsonify({
        "crime_type": crime_type,
        "risk_score":result["risk_score"],
        "top_features":result["top_features"]
    })

