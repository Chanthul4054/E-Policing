from flask import Blueprint, render_template, request
from flask_login import login_required
from extensions import db
from sqlalchemy import text

records_bp = Blueprint("records", __name__)

@records_bp.route("/")
@login_required
def index():
    page = request.args.get("page", 1, type=int)
    per_page = 50
    offset = (page - 1) * per_page
    
    crime = request.args.get('crime', '').strip()
    year = request.args.get('year', '').strip()
    month = request.args.get('month', '').strip()
    
    conditions = []
    params = {"limit": per_page, "offset": offset}
    
    if crime:
        conditions.append("LOWER(crime) = lower(:crime)")
        params["crime"] = crime
    if year:
        conditions.append("date LIKE :year")
        params["year"] = f"%/{year}"
    if month:
        try:
            m_int = int(month)
            conditions.append("(date LIKE :m_2d OR date LIKE :m_1d)")
            params["m_2d"] = f"{m_int:02d}/%"
            params["m_1d"] = f"{m_int}/%"
        except ValueError:
            pass
            
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    
    # Get total count
    count_query = text(f"SELECT COUNT(*) FROM crime_data {where_clause}")
    with db.engine.connect() as conn:
        total = conn.execute(count_query, params).scalar()
        
    total_pages = (total + per_page - 1) // per_page if total > 0 else 1
    
    # Fetch data
    query = text(f"""
        SELECT crime_id, crime, location, date, time, victim_age, sex, weather, gn_division 
        FROM crime_data 
        {where_clause}
        ORDER BY crime_id DESC
        LIMIT :limit OFFSET :offset
    """)
    
    with db.engine.connect() as conn:
        records = conn.execute(query, params).mappings().fetchall()
        
    return render_template(
        "records.html", 
        records=records, 
        page=page, 
        total_pages=total_pages,
        total_records=total,
        crime_filter=crime,
        year_filter=year,
        month_filter=month
    )
