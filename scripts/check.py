import json
from app import create_app
from extensions import db
from sqlalchemy import text

app = create_app()
with app.app_context():
    with db.engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM crime_data LIMIT 1")).mappings().first()
        with open("check_output.json", "w") as f:
            json.dump(dict(row), f, default=str)
