from app import create_app
from extensions import db
from sqlalchemy import text

app = create_app()
with app.app_context():
    with db.engine.connect() as conn:
        q = text("SELECT date FROM crime_data LIMIT 5")
        res = conn.execute(q).fetchall()
        print([r[0] for r in res])
