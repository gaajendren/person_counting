from sqlalchemy.exc import OperationalError
from personCount.Database.init import db
from personCount.counter import app


class Occupancy(db.Model):
    __tablename__ = 'occupancy'
    id = db.Column(db.Integer, primary_key=True)
    Date = db.Column(db.Date, nullable=False)
    Count = db.Column(db.JSON, nullable=False)


with app.app_context():
    try:
        if not db.engine.dialect.has_table(db.engine, 'occupancy'):
            db.create_all()
    except OperationalError as e:
        print(f"Error creating tables: {e}")
