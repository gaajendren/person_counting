import os
import queue
import shutil
import threading
from datetime import time
from queue import Queue

from apscheduler.schedulers.background import BackgroundScheduler
from deepface.DeepFace import represent
from flask import Flask, Response
from flask_cors import CORS
from flask_socketio import SocketIO
from mtcnn import MTCNN
from ultralytics import YOLO

from personCount.Database.init import *
from personCount.config import *
from personCount.controller import *
from personCount.deepSort import *
from personCount.fn import *

detector = MTCNN()
app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")

CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000", "supports_credentials": True}})

init_db(app)

scheduler = BackgroundScheduler()

modal = YOLO('/trained_yolo_model/yolov8s.pt')
track_enter = DeepSort()
track_exit = DeepSort()

person_enter= {}
person_exit= {}

person_count = 0

enter_uncheck_track_id = []
exit_uncheck_track_id = []

image_count =0

face_queue = Queue()

save_face = Queue()

latest_enter_frame = None
latest_exit_frame = None
counting_active = False

error = None

START_TIME = time(4, 30)
END_TIME = time(3, 20)

# for frame skip
enter_frame_count =0
exit_frame_count =0

def is_within_counting_time():
   now = datetime.now().time()
   if START_TIME > END_TIME:
      return now >= START_TIME or now <= END_TIME
   else:
      return START_TIME <= now <= END_TIME


def background_image_process(way):
   global counting_active, enter_frame_count, exit_frame_count, error

   if way == 'enter' :
      screen = cv2.VideoCapture('C:/Users/gaaje/PycharmProjects/person_counting/personCount/test_video/test6.mp4')
   else:
      screen = cv2.VideoCapture('C:/Users/gaaje/PycharmProjects/person_counting/personCount/test_video/test6.mp4')

   if not screen.isOpened():
      error = 'Error: Could not open video source.'
      raise RuntimeError("Error: Could not open video source.")

   while True:
      success, frame = screen.read()

      if not success:
         error = 'Error: Could not read video source.'
         raise RuntimeError("Error: Could not read video source.")

      if way == 'enter':
         skip, enter_frame_count = frame_limit_skip(enter_frame_count, 30, 3)
      else:
         skip, exit_frame_count = frame_limit_skip(exit_frame_count, 30, 3)

      if skip:
         continue
      if is_within_counting_time():
         if way== 'enter':
            person_counting(frame, way, person_enter, track_enter)
         else:
            person_counting(frame, way, person_exit , track_exit)
      else:
         counting_active = False
         continue


def person_counting(frame, way, person , track):
   global person_enter,person_exit, latest_enter_frame, latest_exit_frame, counting_active, socketio, person_count, image_count

   resized_frame = aspect_ratio_resize(frame)
   height, width, _ = resized_frame.shape

   results = modal(resized_frame, stream=True, classes=[0])

   for r in results:
      detections = []
      boxes = r.boxes

      for box in boxes:
         x1,y1,x2,y2 = box.xyxy[0]
         x1, y1, x2, y2 = int(x1), int(y1) , int(x2), int(y2)

         if cv2.pointPolygonTest(np.array(rio),(x2,y2), False) < 0:
            continue

         detections.append([x1, y1, x2, y2, float(box.conf[0])])
         # cv2.putText(resized_frame, str(round(float(box.conf[0]), 2) ) ,(x1 + 30,y1 ), cv2.FONT_HERSHEY_PLAIN, 2 , (255,255,255), thickness=2)
         cv2.circle(resized_frame, (x2,y2), 3, (255,0,0), thickness=-1)
         cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

      track.update(resized_frame, detections)

      for tracker in track.tracks:
         bbox = tracker.bbox
         x1, y1, x2, y2 = bbox
         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
         track_id = tracker.track_id


         # start ...................................................................

         if track_id not in person:
             person[track_id] = {"entered": False, "face": [], 'checked': False}
             print(f'1-{track_id}')
         if line[0][0] < x2 < line[1][0] and y2 > line[0][1]:
             print(f'2-{track_id}')
             print(person[track_id]['face'])
             print(person[track_id]["checked"])
             if not person[track_id]["checked"]:
                 print(f'3-{track_id}')
                 if check_left_face_process(track_id, person, way) and person[track_id]['face']:
                    print(f'4-{track_id}')
                    save_face.put((track_id, person[track_id], person))
             print(f'5-{track_id}')
             person[track_id]["entered"] = True

         else:
             print(f'6-{track_id}')
             if not person[track_id]["entered"] and person[track_id]["checked"] == False:
                 print(f'7-{track_id}')
                 # face start
                 face_image = resized_frame[y1:y2, x1:x2]
                 face_queue.put((track_id, face_image, way))

         if way == 'enter':
             print(f'8-{track_id}')
             if track_id not in enter_uncheck_track_id:
                 print(f'9-{track_id}')
                 update_check(person,way)
         else:
             print(f'10-{track_id}')
             if track_id not in exit_uncheck_track_id:
                 print(f'11-{track_id}')
                 update_check(person,way)

         # end............................

         # face_image = resized_frame[y1:y2, x1:x2]
         cv2.putText(resized_frame, str(track_id), (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2 , (255,255,255), thickness=2)


   cv2.polylines(resized_frame, [np.array(rio)], True,(0,255,255), thickness=1)
   cv2.line(resized_frame, line[0], line[1], (0, 255, 0), thickness=2)


   if way == 'enter':
      latest_enter_frame= resized_frame
   else:
      latest_exit_frame = resized_frame

   counting_active = True

   person_count = text_count(person_enter, person_exit ,width, resized_frame)

   socketio.emit('person_count', {'count': person_count, })

def update_check(person, way):
    global enter_uncheck_track_id, exit_uncheck_track_id

    if way == 'enter':
        uncheck_trackId = enter_uncheck_track_id
    else:
        uncheck_trackId = exit_uncheck_track_id

    for uncheck_id in uncheck_trackId:
      print(f'18-{uncheck_id}')
      if  person[uncheck_id]['face'] and check_left_face_process(uncheck_id, person, way):
          print(f'19-{uncheck_id}')
          save_face.put((uncheck_id, person[uncheck_id], person))
          print('unchecked_yes')

def check_left_face_process(track_id ,  person,way):
    global face_queue, enter_uncheck_track_id, exit_uncheck_track_id
    print(f'20-{track_id}')
    if way == 'enter':
        uncheck_trackId = enter_uncheck_track_id
    else:
        uncheck_trackId = exit_uncheck_track_id

    complete_check = True
    print(f'21-{track_id}')
    with face_queue.mutex:
        for item in face_queue.queue:
            if item[0]  == track_id:
                complete_check = False
                print(f'13-{track_id}')
                if track_id not in uncheck_trackId:
                    uncheck_trackId.append(track_id)
                    print(f'14-{track_id}')
                print(f'yes{item}')
            break
    print(f'unche{uncheck_trackId}')
    if complete_check:
        print(f'15-{track_id}')
        if track_id in uncheck_trackId:
            uncheck_trackId.remove(track_id)
            print(f'16-{track_id}')
        person[track_id]["checked"] = True
        print(f'17-{track_id}')
    return complete_check




def face_detection():
    global image_count, person_exit , person_exit
    while True:
        try:
            face_data = face_queue.get(block=True)
            track_id, face_image,way = face_data
            print(f'181-{track_id}')
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            results = detector.detect_faces(face_image_rgb)

            for result in results:
                confidence = result['confidence']
                if confidence >= 0.8:
                    # Get the bounding box

                    fx, fy, fwidth, fheight = result['box']
                    fx2, fy2 = fx + fwidth, fy + fheight

                    # Crop the face from the frame
                    cropped_face = face_image[fy:fy2, fx:fx2]

                    try:

                        embedding = represent(cropped_face, model_name='Facenet', enforce_detection=False)[0]["embedding"]

                        image_count += 1
                        file_name = f'{track_id}{image_count}{way}.jpg'
                        cv2.imwrite(  f'C:/Users/gaaje/PycharmProjects/person_counting/personCount/face_img/{file_name}',face_image)
                        face_array = [f'{file_name}', embedding, confidence]

                        if way == 'enter':
                            print(f'121-{face_array}')
                            person_enter[track_id]['face'].append(face_array)

                        else:
                            person_exit[track_id]['face'].append(face_array)

                    except Exception as e:
                        print(f"Error encoding face: {e}")

                    face_queue.task_done()

        except queue.Empty:
            continue


def filter_store_face():

    while True:
        try:
             track_id, face_data, person = save_face.get(block=True)
             print(f'check-error{track_id}')
             print(f'sucesss-{track_id}')

             best_face = max(face_data['face'], key=lambda x: x[2])

             best_face_file_path = f'C:/Users/gaaje/PycharmProjects/person_counting/personCount/face_img/{best_face[0]}'
             new_path = f'C:/Users/gaaje/PycharmProjects/person_counting/personCount/person_img/{best_face[0]}'

             shutil.move(best_face_file_path, new_path)

             for face in person[track_id]['face']:

                 face_image = face[0]
                 file_path = f'C:/Users/gaaje/PycharmProjects/person_counting/personCount/face_img/{face_image}'

                 if face_image != best_face[0]:
                     print(f"Attempting to delete: {file_path}")
                     if os.path.exists(file_path):
                         os.remove(file_path)
                         print(f"Deleted: {file_path}")
                     else:
                         print(f"{file_path} does not exist.")

             person[track_id]['face'] = []

             person[track_id]['face'].append(best_face)

             if person == person_enter:
                 with app.app_context():
                    upload_face_enter(track_id,best_face)

             else:
                 all_embeddings = []

                 for person_id, person_data in person_enter.items():
                     if 'face' in person_data and person_data['face']:
                        all_embeddings.append((person_id, person_data['face'][0][1]))


                 person_id_enter = batch_compare(np.array(best_face[1]), np.array(all_embeddings, dtype=object))

                 if person_id_enter is not None and person_id_enter != 0:
                     with app.app_context():
                         upload_face_exit(track_id, best_face, int(person_id_enter))


             save_face.task_done()

        except queue.Empty:
            continue

def batch_compare(current_embedding, all_embeddings):
    min_distance = float('inf')
    closes_person = 0

    for track_id, embedding in all_embeddings:
        distance = np.linalg.norm(embedding - current_embedding)
        print(f'{track_id}-{distance}')

        if distance < min_distance:
            min_distance = distance
            closes_person = track_id

    print(f'best_compare-{closes_person}')
    return closes_person





# def normalize(embedding):
#     return embedding / np.linalg.norm(embedding)


def encode_frame(frame):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return buffer.tobytes()

def generate_video(way):
    global latest_enter_frame, latest_exit_frame

    while counting_active:

        frame = latest_enter_frame if way == 'enter' else latest_exit_frame

        if frame is None:
            continue

        frame_bytes = encode_frame(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def scheduled_task():

   with app.app_context():
      add_record(person_count)



scheduler.add_job(scheduled_task, trigger='cron', hour='8-23,0-4', minute='*/1')
if not scheduler.running:
    scheduler.start()

   

@app.route('/current_status', methods=['GET'])
def status():
     if counting_active is False:
        return f"Counting only start {START_TIME} until {END_TIME}"
     if error is not None:
        return f"{error}"
     return f'success'


@app.route('/video_feed')
def video_feed_enter():

   if counting_active is False:

      return f'currently offline'

   return Response(generate_video('enter'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_exit')
def video_feed_exit():

   if counting_active is False:

      return f'currently offline'

   return Response(generate_video('exit'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    threading.Thread(target=background_image_process, args=('enter',) ,daemon=True).start()
    threading.Thread(target=background_image_process, args=('exit',) ,daemon=True).start()
    threading.Thread(target=face_detection, daemon=True).start()
    threading.Thread(target=filter_store_face, daemon=True).start()

    socketio.run(app, host='127.0.0.1', port=5000)
