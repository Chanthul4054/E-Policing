from flask import Flask
from dotenv import load_dotenv
import os

from extensions import db
from routes.allocation_routes import allocation_bp

load_dotenv()

app = Flask(__name__)

db_uri = os.getenv("DB_URI")
if not db_uri:
    raise RuntimeError("DB_URI is not set")

app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# Register your module
app.register_blueprint(allocation_bp)

if __name__ == "__main__":
    app.run(debug=True)
