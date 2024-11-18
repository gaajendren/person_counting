from datetime import datetime
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine, inspect
from personCount.Database.init import db
from personCount.counter import app


class PersonExit(db.Model):
    __tablename__ = 'person_exit'
    id = db.Column(db.Integer, primary_key=True)
    track_id = db.Column(db.VARCHAR(255), nullable=False)
    img = db.Column(db.VARCHAR(255), nullable=False)
    embedding = db.Column(db.JSON, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)



with app.app_context():
    try:
        inspector = inspect(db.engine)

        if not inspector.has_table('person_exit'):
            db.create_all()
            print("Occupancy table created.")
        else:
            print("Occupancy table already exists.")
    except OperationalError as e:
        print(f"Error creating tables: {e}")
