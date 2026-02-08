from flask import Blueprint, render_template, request, jsonify
from services.allocation_service import run_allocation_pipeline
from services.allocation_service import diminishing_curves_from_df

allocation_bp = Blueprint("allocation", __name__)

@allocation_bp.route("/")
def index():
    total_officers = int(request.args.get("officers", 500))
    topk = int(request.args.get("topk", 80))
    min_per_gn = int(request.args.get("min_per_gn", 1))
    k = float(request.args.get("k", 0.25))
    chart_max = int(request.args.get("chart_max", total_officers))

    crime_type, df = run_allocation_pipeline(
        total_officers=total_officers,
        max_gns_to_cover=topk,
        min_per_gn=min_per_gn
    )

    cols = [
        "gn_name", "crime_type", "risk_score",
        "distance_to_station_km", "GN_population",
        "closest_police_station", "risk_rank",
        "allocation_score", "pred_alloc",
        "pred_alloc_norm", "assigned_officers"
    ]

    rows = []
    if not df.empty:
        df = df.sort_values(["assigned_officers", "allocation_score"], ascending=[False, False])
        rows = df[cols].to_dict(orient="records")

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
    )

@allocation_bp.route("/api/diminishing")
def api_diminishing():
    total_officers = int(request.args.get("officers", 500))
    topk = int(request.args.get("topk", 80))
    min_per_gn = int(request.args.get("min_per_gn", 1))
    k = float(request.args.get("k", 0.25))
    chart_max = int(request.args.get("chart_max", total_officers))

    _, df = run_allocation_pipeline(total_officers, topk, min_per_gn)
    totals, total_benefits, marginal = diminishing_curves_from_df(df, k, chart_max)

    return jsonify({
        "totals": totals,
        "total_benefits": total_benefits,
        "marginal_benefits": marginal
    })
