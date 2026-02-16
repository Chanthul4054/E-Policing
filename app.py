from flask import Flask, render_template, session
from dotenv import load_dotenv
import os

from extensions import db
from routes.auth_routes import auth_bp, init_login
from routes.allocation_routes import allocation_bp
from routes.hotspot_routes import hotspot_bp
from routes.pattern_routes import pattern_bp
from routes.risk_routes import risk_bp

load_dotenv()

def create_app():
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
    app.config["SESSION_PERMANENT"] = False

    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise RuntimeError("DB_URI is not set in .env")

    app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # Blueprints
    init_login(app)
    app.register_blueprint(auth_bp)
    app.register_blueprint(allocation_bp)
    app.register_blueprint(hotspot_bp, url_prefix="/hotspot")
    app.register_blueprint(pattern_bp, url_prefix="/pattern")
    app.register_blueprint(risk_bp, url_prefix="/risk")

    return app

app = create_app()

@app.before_request
def make_session_non_permanent():
    session.permanent = False  

@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403

if __name__ == "__main__":
    app.run(debug=True)
