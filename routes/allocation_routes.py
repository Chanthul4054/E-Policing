from flask import Blueprint, render_template, request, jsonify, session
from flask_login import login_required

from services.allocation_service import (
    run_allocation_pipeline,
    run_resource_allocation,
    allocation_to_dict,
    diminishing_curves_from_df,
    HotspotInputError,
    ModelLoadError,
    FeatureError,
    CRIME_TYPE_MAP,
    _fetch_gn_features,
)

allocation_bp = Blueprint("allocation", __name__)


# Dashboard routes


@allocation_bp.route("/")
@login_required
def index():
    total_officers = int(request.args.get("officers", 500))
    topk = int(request.args.get("topk", 80))
    min_per_gn = int(request.args.get("min_per_gn", 2))
    k = float(request.args.get("k", 0.25))
    chart_max = int(request.args.get("chart_max", 250))

    crime_type_req = session.get("selected_crime_type", "drugs")
    if crime_type_req not in CRIME_TYPE_MAP:
        crime_type_req = "drugs"

    crime_type, df = run_allocation_pipeline(
        total_officers=total_officers,
        max_gns_to_cover=topk,
        min_per_gn=min_per_gn,
        crime_type=crime_type_req,
    )

    cols = [
        "gn_division",
        "closest_police_station",
        "officers_allocated",
    ]

    rows = []
    total_assigned = 0
    coverage = 0

    if not df.empty:
        df = df.sort_values(
            ["officers_allocated", "predicted_demand_score"],
            ascending=[False, False],
        )
        rows = df[cols].to_dict(orient="records")
        total_assigned = int(df["officers_allocated"].sum())
        coverage = len(df)

    return render_template(
        "allocation.html",
        cols=cols,
        rows=rows,
        crime_type=crime_type,
        total_officers=total_officers,
        max_gns_to_cover=topk,
        min_per_gn=min_per_gn,
        k=k,
        chart_max=chart_max,
        total_assigned=total_assigned,
        coverage=coverage,
    )


@allocation_bp.route("/api/diminishing")
@login_required
def api_diminishing():
    total_officers = int(request.args.get("officers", 200))
    topk = int(request.args.get("topk", 80))
    min_per_gn = int(request.args.get("min_per_gn", 2))
    k = float(request.args.get("k", 0.25))
    chart_max = int(request.args.get("chart_max", 250))

    crime_type_req = session.get("selected_crime_type", "drugs")
    if crime_type_req not in CRIME_TYPE_MAP:
        crime_type_req = "drugs"

    _, df = run_allocation_pipeline(total_officers, topk, min_per_gn, crime_type=crime_type_req)
    totals, total_benefits, marginal = diminishing_curves_from_df(df, k, chart_max)

    return jsonify({
        "totals": totals,
        "total_benefits": total_benefits,
        "marginal_benefits": marginal,
    })


@allocation_bp.route("/api/allocate-resources", methods=["POST"])
@login_required
def api_allocate_resources():
    
    payload = request.get_json(force=True)
    hotspot_dict = payload.get("hotspot_output", payload)
    total_officers = int(payload.get("total_officers", 120))
    min_per_gn = int(payload.get("min_per_gn", 0))

    # Fetch GN features from the database for all predicted pcodes
    predictions = hotspot_dict.get("predictions", [])
    gn_pcodes = [
        p.get("gn_name") or p.get("pcode_id", "")
        for p in predictions
    ]
    gn_df = _fetch_gn_features(gn_pcodes)

    try:
        result_df = run_resource_allocation(
            hotspot_output=hotspot_dict,
            gn_feature_df=gn_df,
            total_officers=total_officers,
            min_per_gn=min_per_gn
        )
    except HotspotInputError as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 422
    except ModelLoadError as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 500
    except FeatureError as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 422

    records = allocation_to_dict(result_df)
    return jsonify({
        "status": "success",
        "total_officers_deployed": int(result_df["officers_allocated"].sum()),
        "gn_divisions_covered": len(result_df),
        "allocations": records,
    })
