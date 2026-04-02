from flask import Blueprint, render_template, request, jsonify, session
from flask_login import login_required
from services.risk_service import run_risk_factor_pipeline, fetch_latest_feature_row

risk_bp = Blueprint("risk", __name__)

@risk_bp.route("/", methods=["GET", "POST"])
@login_required
def index():
    pattern_results = session.get("pattern_results", [])

    gn_options = []
    seen = set()

    for item in pattern_results:
        gn = item.get("gn_division")
        crime = item.get("crime_type")

        if gn and crime and (gn, crime) not in seen:
            gn_options.append({
                "gn_division": gn,
                "crime_type": crime,
                "selection_key": f"{gn}|{crime}"
            })
            seen.add((gn, crime))

    result = None
    selected_key = None

    if request.method == "POST":
        selected_key = request.form.get("gn_selection")

        if not selected_key:
            return render_template(
                "risk.html",
                gn_options=gn_options,
                selected_key=None,
                result=None,
                error="No GN division selection was received."
            )

        try:
            selected_gn, crime_type = selected_key.split("|", 1)
        except ValueError:
            return render_template(
                "risk.html",
                gn_options=gn_options,
                selected_key=None,
                result=None,
                error="Invalid selection format received."
            )

        selected_item = next(
            (
                item for item in pattern_results
                if item.get("gn_division") == selected_gn
                and item.get("crime_type") == crime_type
            ),
            None
        )

        if not selected_item:
            return render_template(
                "risk.html",
                gn_options=gn_options,
                selected_key=None,
                result=None,
                error="Selected GN division and crime type are not available from Component 2 output."
            )

        try:
            features = fetch_latest_feature_row(selected_gn)
            features["gn_name"] = selected_gn
            result = run_risk_factor_pipeline(crime_type, features)
        except ValueError as e:
            return render_template(
                "risk.html",
                gn_options=gn_options,
                selected_key=selected_key,
                result=None,
                error=str(e)
            )
        except Exception as e:
            print("RISK PIPELINE ERROR:", repr(e))
            return render_template(
                "risk.html",
                gn_options=gn_options,
                selected_key=selected_key,
                result=None,
                error=f"An unexpected error occurred while generating the risk analysis: {str(e)}"
            )
    print("pattern_results in session at /risk/:", session.get("pattern_results"))

    return render_template(
    "risk.html",
    gn_options=gn_options,
    selected_key=selected_key,
    result=result,
    error=None
    )
    
@risk_bp.route("/api/load-pattern-results", methods=["POST"])
@login_required
def load_pattern_results():
    payload = request.get_json()
    print("Received payload:", payload)

    if not isinstance(payload, list) or not payload:
        print("Payload invalid or empty")
        return jsonify({"error": "Invalid or empty payload"}), 400

    compact_results = []
    seen = set()

    for item in payload:
        gn = item.get("gn_division")
        crime = item.get("crime_type")

        if gn and crime and (gn, crime) not in seen:
            compact_results.append({
                "gn_division": gn,
                "crime_type": crime
            })
            seen.add((gn, crime))

    session["pattern_results"] = compact_results
    print("Stored in session:", session.get("pattern_results"))

    gn_options = [
        {
            "gn_division": item["gn_division"],
            "crime_type": item["crime_type"],
            "selection_key": f'{item["gn_division"]}|{item["crime_type"]}'
        }
        for item in compact_results
    ]

    return jsonify({
        "message": "Pattern results loaded successfully",
        "gn_options": gn_options
    })

# @risk_bp.route("/clear-session")
# @login_required
# def clear_session():
#     session.clear()
#     return "Session cleared!"