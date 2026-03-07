from flask import Blueprint, render_template, request, jsonify
from services.allocation_service import run_allocation_pipeline, CRIME_TYPE_MAP
from services.allocation_service import diminishing_curves_from_df
from flask_login import login_required

allocation_bp = Blueprint("allocation", __name__)

@allocation_bp.route("/")
@login_required
def index():
    total_officers = int(request.args.get("officers", 500))
    topk = int(request.args.get("topk", 80))
    min_per_gn = int(request.args.get("min_per_gn", 2))
    k = float(request.args.get("k", 0.25))
    chart_max = int(request.args.get("chart_max", 250))
    from flask import session
    crime_type_req = session.get("selected_crime_type", "drugs")
    if crime_type_req not in CRIME_TYPE_MAP:
        crime_type_req = "drugs"

    crime_type, df = run_allocation_pipeline(
        total_officers=total_officers,
        max_gns_to_cover=topk,
        min_per_gn=min_per_gn,
        crime_type=crime_type_req
    )


    cols = [
        "gn_name",
        "closest_police_station",
        "pred_alloc_norm", 
        "assigned_officers"
    ]

    rows = []
    total_assigned = 0
    coverage = 0
    
    if not df.empty:
        df = df.sort_values(["assigned_officers", "allocation_score"], ascending=[False, False])
        rows = df[cols].to_dict(orient="records")
        total_assigned = int(df["assigned_officers"].sum())
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
        coverage=coverage
    )

@allocation_bp.route("/api/diminishing")
@login_required
def api_diminishing():
    total_officers = int(request.args.get("officers", 200))
    topk = int(request.args.get("topk", 80))
    min_per_gn = int(request.args.get("min_per_gn", 2))
    k = float(request.args.get("k", 0.25))
    chart_max = int(request.args.get("chart_max", 250))
    from flask import session
    crime_type_req = session.get("selected_crime_type", "drugs")
    if crime_type_req not in CRIME_TYPE_MAP:
        crime_type_req = "drugs"

    _, df = run_allocation_pipeline(total_officers, topk, min_per_gn, crime_type=crime_type_req)
    totals, total_benefits, marginal = diminishing_curves_from_df(df, k, chart_max)

    return jsonify({
        "totals": totals,
        "total_benefits": total_benefits,
        "marginal_benefits": marginal
    })
