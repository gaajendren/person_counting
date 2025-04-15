from personCount.Database.init import db



class Setting(db.Model):
    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    roi = db.Column(db.String(255), nullable=False)
    exit_roi = db.Column(db.String(255), nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    is_manual = db.Column(db.Integer, nullable=False)

