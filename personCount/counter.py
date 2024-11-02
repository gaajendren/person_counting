import cv2
import numpy as np
from ultralytics import YOLO
from fn import *
from config import *
from deepSort import *
from flask import Flask, Response
from flask_socketio import SocketIO
from flask_cors import CORS
from datetime import datetime, time
from Database.occupancy import *
from Database.init import *
from flask_sqlalchemy import SQLAlchemy
import threading
import cvzone

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")

CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000", "supports_credentials": True}})

init_db(app)

modal = YOLO('/personCount/trained_yolo_model/yolov8s.pt')
track = DeepSort()

person_enter= {}

person_exit= {}

latest_enter_frame = None
latest_exit_frame = None
counting_active = False

error = None

START_TIME = time(4, 30)
END_TIME = time(3, 30)

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
      screen = cv2.VideoCapture('/personCount/test_video/test6.mp4')
   else:
      screen = cv2.VideoCapture('/personCount/test_video/test6.mp4')

   if not screen.isOpened():
      error = 'Error: Could not open video source.'
      raise RuntimeError("Error: Could not open video source.")

   while True:
      success, frame = screen.read()

      if not success:
         error = 'Error: Could not read video source.'
         raise RuntimeError("Error: Could not read video source.")

      if way == 'enter':
         skip, enter_frame_count = frame_limit_skip(enter_frame_count, 30, 2)
      else:
         skip, exit_frame_count = frame_limit_skip(exit_frame_count, 30, 2)

      if skip:
         continue
      if is_within_counting_time():
         if way== 'enter':
            person_counting(frame, way, person_enter)
         else:
            person_counting(frame, way, person_exit)
      else:
         counting_active = False
         continue



def person_counting(frame, way, person):
   global socketio, person_enter,person_exit, latest_enter_frame, latest_exit_frame, counting_active

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

         if line[0][0] < x2 < line[1][0] and y2 > line[0][1]:

            if track_id not in person:
               person[track_id] = track_id

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


def encode_frame(frame):
   _, buffer = cv2.imencode('.jpg', frame)
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


   # cv2.setMouseCallback('i', mouse_coordinate)

   # for person_id, details in person.items():
   #    print(f"ID: {person_id}")
   #    print(f"  Entered: {details.get('entered', False)}")
   #    print(f"  Faces: {details.get('face', None)}\n")


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

    socketio.run(app, host='127.0.0.1', port=5000)