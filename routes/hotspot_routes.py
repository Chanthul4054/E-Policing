from flask import Blueprint, render_template
from flask_login import login_required

hotspot_bp = Blueprint("hotspot", __name__)

@hotspot_bp.route("/")
@login_required
def index():
    
    return render_template("hotspot.html")
