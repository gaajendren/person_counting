from personCount.Database.occupancy import Occupancy
from datetime import date, datetime
from personCount.Database.init import db
from sqlalchemy.exc import SQLAlchemyError


def add_record(count):
    try:

        occupancy_record = Occupancy.query.filter_by(Date=date.today()).first()



        if occupancy_record:

            old_count = list(occupancy_record.Count)
            new_count = {'Time': datetime.now().time().strftime("%H:%M:%S"), 'Count': count}
            old_count.append(new_count)

            occupancy_record.Count = old_count
            db.session.add(occupancy_record)
            db.session.commit()

        else:

            new_row = Occupancy(Date=date.today(),Count=[{'Time': datetime.now().time().strftime("%H:%M:%S"), 'Count': count}])
            db.session.add(new_row)
            db.session.commit()

    except SQLAlchemyError as e:

        db.session.rollback()
        print(f"Database error occurred: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
