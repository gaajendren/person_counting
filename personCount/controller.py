import json

from torch.nn.functional import embedding

from personCount.Database.occupancy import Occupancy
from personCount.Database.person_img import PersonEnter
from personCount.Database.exit_person import PersonExit
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



def upload_face_enter(track_id, best_face):

    try:
        print(f"Uploading PersonEnter with track_id: {track_id}")
        embedding_str = json.dumps(best_face[1])
        new_data = PersonEnter(track_id=track_id,embedding=embedding_str, img=best_face[0], timestamp=datetime.now())
        db.session.add(new_data)
        db.session.commit()
        print('sucess-update')
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Database error occurred: {e}")

    except Exception as e:
        print(f'An unexpected error occurred: {e}')


def update_face_enter(enter_person_id, exit_id):

    try:
        print(f"Updating record for enter_person_id: {enter_person_id}, with exit_id: {exit_id}")
        entry = PersonEnter.query.filter_by(track_id=int(enter_person_id)).first()
        if entry:
            print(f"Found record for track_id = {enter_person_id}: {entry}")
            entry.person_id_exit = int(exit_id)
            db.session.commit()
            print(f"Successfully updated person_id_exit for track_id = {enter_person_id}")
        else:
            print("PersonEnter record not found.")

    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Database error occurred: {e}")
    except Exception as e:
        print(f'An unexpected error occurred: {e}')



def upload_face_exit(track_id, best_face, enter_person_id ):

    try:

        print(f"Uploading PersonExit with track_id: {track_id}")
        embedding_str = json.dumps( best_face[1])
        new_data = PersonExit(track_id=track_id , embedding= embedding_str ,img=best_face[0] ,timestamp=datetime.now())
        db.session.add(new_data)
        db.session.commit()

        if enter_person_id is not None:
            print(f"Updating PersonExit with track_id: {track_id}, enter_person_id: {enter_person_id}")
            exit_person = PersonExit.query.filter_by(track_id= int(track_id)).first()
            if exit_person:
                print(f"Created PersonExit ID: {exit_person.id}")
                if enter_person_id is not None:
                    print(f"Linking enter_person_id {enter_person_id} to exit_person_id {exit_person.id}")
                    update_face_enter(enter_person_id, exit_person.id)
                else:
                    print("enter_person_id is None, skipping update.")
            else:
                print(f"No matching PersonExit found for track_id {track_id}")

    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Database error occurred: {e}")

    except Exception as e:
        print(f'An unexpected error occurred: {e}')