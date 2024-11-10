from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine, inspect
from personCount.Database.init import db
from personCount.counter import app


class Occupancy(db.Model):
    __tablename__ = 'occupancy'
    id = db.Column(db.Integer, primary_key=True)
    Date = db.Column(db.Date, nullable=False)
    Count = db.Column(db.JSON, nullable=False)


with app.app_context():
    try:
        inspector = inspect(db.engine)

        if not inspector.has_table('occupancy'):
            db.create_all()
            print("Occupancy table created.")
        else:
            print("Occupancy table already exists.")
    except OperationalError as e:
        print(f"Error creating tables: {e}")
