from datetime import datetime
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine, inspect
from personCount.Database.init import db
from personCount.counter import app


class PersonEnter(db.Model):
    __tablename__ = 'person_enter'
    id = db.Column(db.Integer, primary_key=True)
    track_id = db.Column(db.VARCHAR(255), nullable=False)
    person_id_exit = db.Column(db.Integer, db.ForeignKey('person_exit.id'), nullable=True)
    embedding = db.Column(db.JSON, nullable=False)
    img = db.Column(db.VARCHAR(255), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)



with app.app_context():
    try:
        inspector = inspect(db.engine)

        if not inspector.has_table('person_enter'):
            db.create_all()
            print("Occupancy table created.")
        else:
            print("Occupancy table already exists.")
    except OperationalError as e:
        print(f"Error creating tables: {e}")
