from flask import Blueprint, render_template
from flask_login import login_required

risk_bp = Blueprint("risk", __name__)

@risk_bp.route("/")
@login_required
def index():
    return render_template("risk.html")
