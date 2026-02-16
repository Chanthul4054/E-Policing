from flask import Blueprint, render_template
from flask_login import login_required

pattern_bp = Blueprint("pattern", __name__)

@pattern_bp.route("/")
@login_required
def index():
    return render_template("pattern.html")
