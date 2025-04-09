import base64
import math
import os
import queue
import shutil
import threading
from queue import Queue
import cv2
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from deepface.DeepFace import represent
from flask import Flask, Response, send_from_directory
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO
from tensorflow.python.util.numpy_compat import np_array
from ultralytics import YOLO
from personCount.Database.init import *
from personCount.config import *
from personCount.controller import *
from personCount.fn import *
from personCount.Yunet.yunet import YuNet
from personCount.sort.sort import *
from datetime import datetime
import time as timer
import onnxruntime as ort
from scipy.spatial.distance import cosine
from collections import defaultdict
from threading import Lock
import personCount.config as config


app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000", "supports_credentials": True}})

init_db(app)

scheduler = BackgroundScheduler()

modal = YOLO('/trained_yolo_model/yolov8s.pt')

track_enter = Sort(max_age=20)
track_exit = Sort(max_age=20)


face_recognizer = ort.InferenceSession(f"{absolute_path}/Onnx/mobilefacenet.onnx")

detector = YuNet(
    f"{absolute_path}/Yunet/face_detection_yunet_2023mar.onnx",
    (320, 320),
    confThreshold=0.8,
    nmsThreshold=0.3,
    topK=5000)


person_enter = {}
person_exit = {}

person_count = 0

person_count_enter = []
person_count_exit = []

image_count = 0

face_queue = Queue()
save_face = Queue()

latest_enter_frame = None
latest_exit_frame = None

counting_active = True

error = None

enter_frame_count = 0
exit_frame_count = 0

counter_lock = Lock()

active_face_tracks_enter = defaultdict(int)
active_face_tracks_exit = defaultdict(int)

enter_embeddings = np.empty((0, 128))
enter_ids = []

unchecked_track_id_enter = []
unchecked_track_id_exit = []

with app.app_context():
    initial_config()

def is_within_counting_time():

    now = datetime.now().time()
    if config.START_TIME > config.END_TIME:
        return now >= config.START_TIME or now <= config.END_TIME
    else:
        return config.START_TIME <= now <= config.END_TIME


def background_image_process(way):
    global counting_active, enter_frame_count, exit_frame_count, error

    if way == 'enter':
        screen = cv2.VideoCapture(absolute_path + camera_enter)
        screen.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    else:
        screen = cv2.VideoCapture(absolute_path + camera_exit)
        screen.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    screen.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not screen.isOpened():
        raise RuntimeError("Error: Could not open video source.")

    while True:
        success, frame = screen.read()

        if not success:
            error = 'Error: Could not read video source.'
            print(error)

            if way == 'enter':
                screen = cv2.VideoCapture(absolute_path + camera_enter , apiPreference=cv2.CAP_FFMPEG)
                screen.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                screen.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                screen = cv2.VideoCapture(absolute_path + camera_exit , apiPreference=cv2.CAP_FFMPEG)
                screen.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                screen.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        if way == 'enter':
            skip, enter_frame_count = frame_limit_skip(enter_frame_count)
        else:
            skip, exit_frame_count = frame_limit_skip(exit_frame_count)

        if skip:
            continue

        if is_within_counting_time():
            if way == 'enter':
                person_counting(frame, way, person_enter, track_enter)
            else:
                person_counting(frame, way, person_exit, track_exit)
        else:
            counting_active = False
            print('no')
            continue


def person_counting(frame, way, person, track):
    global person_enter, person_exit, latest_enter_frame, latest_exit_frame, counting_active, socketio, person_count, image_count, person_count_enter, person_count_exit,unchecked_track_id_exit,unchecked_track_id_enter

    resized_frame = aspect_ratio_resize(frame)
    ori_frame = resized_frame.copy()
    height, width, _ = resized_frame.shape

    r_io = config.rio if way == 'enter' else config.exit_rio

    try:
        results = modal(resized_frame, verbose=False, stream=True, classes=[0])
        detections = np.empty((0, 5))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100

                if cv2.pointPolygonTest(np.array(r_io), (x2, y2), False) < 0:
                    continue

                currentArray = np_array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

                cv2.circle(resized_frame, (x2, y2), 3, (255, 0, 0), thickness=-1)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

            result_tracker = track.update(detections)

            for tracker in result_tracker:
                x1, y1, x2, y2, track_id = tracker
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                # start ...................................................................

                if track_id not in person:
                    in_count_list = track_id in (person_count_enter if way == 'enter' else person_count_exit)

                    if not in_count_list:
                        print(f"New person detected: {track_id}")
                        person[track_id] = {
                            "entered": False,
                            "face": [],
                            'checked': False,
                            'img': []
                        }
                    else:
                        continue


                if line[0][0] < x2 < line[1][0] and y2 > line[0][1] and person[track_id]["entered"] == False:

                    if way == 'enter':
                        if track_id not in person_count_enter:
                            person_count_enter.append(track_id)
                        unchecked_track_id_enter.append(track_id)
                    else:
                        if track_id not in person_count_exit:
                            person_count_exit.append(track_id)
                        unchecked_track_id_exit.append(track_id)

                    print(f"Person {track_id} crossed the line for '{way}'")

                    person[track_id]["entered"] = True

                else:

                    if not person[track_id]["entered"]:
                        # face start
                        person_image = resized_frame[y1:y2, x1:x2]

                        ori_person = ori_frame[y1:y2, x1:x2]

                        if len(person[track_id]['img']) == 0:
                            person[track_id]['img'] = ori_person

                        print(f"Adding face image for person in the queue {track_id}...")

                        with counter_lock:
                            (active_face_tracks_enter if way == 'enter' else active_face_tracks_exit)[track_id] += 1

                        face_queue.put((track_id, person_image, way))

                update_check(person,way)

                cv2.putText(resized_frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)



            cv2.polylines(resized_frame, [np.array(r_io)], True, (0, 255, 255), thickness=1)

        cv2.line(resized_frame, line[0], line[1], (0, 255, 0), thickness=2)

        if way == 'enter':
            latest_enter_frame = resized_frame
        else:
            latest_exit_frame = resized_frame

        person_count = text_count(person_count_enter, person_count_exit, width, resized_frame)

        socketio.emit('person_count', {'count': person_count, })

    except Exception as e:
        print(f"Error while loading model: {e}")


def update_check(person, way):
    global unchecked_track_id_enter,unchecked_track_id_exit

    print('it update_check')
    with counter_lock:
        unchecked_track_id = (unchecked_track_id_enter if way == 'enter' else unchecked_track_id_exit)

        if not unchecked_track_id:
            return

        for track_id in unchecked_track_id:
            if track_id not in (active_face_tracks_enter if way == 'enter' else active_face_tracks_exit):
                person[track_id]['checked'] = True
                unchecked_track_id.remove(track_id)
                save_face.put((track_id, person, way))



def get_mobilefacenet_embedding(face_img):

    try:

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (112, 112))
        face_img = (face_img.astype(np.float32) - 127.5) / 128.0

        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)

        embedding = face_recognizer.run(None, {'x': face_img})[0][0]
        return embedding
    except Exception as e:
        print(f"Embedding failed: {str(e)}")
        return None


def face_detection():
    global image_count, person_exit, person_exit, enter_embeddings
    while True:
        try:
            face_data = face_queue.get(block=True)
            track_id, face_image, way = face_data

            print(f"Processing face data for track_id {track_id} in {way} direction...")

            if (way == 'enter' and person_enter[track_id]["checked"]) or (way == 'exit' and person_exit[track_id]["checked"]):
                with counter_lock:
                   del (active_face_tracks_enter if way == 'enter' else active_face_tracks_exit)[track_id]
                   face_queue.task_done()
                continue

            h, w = face_image.shape[:2]
            detector.setInputSize((w, h))

            faces = detector.infer(face_image)

            for face in faces:
                confidence = face[-1]
                if confidence >= 0.9:
                    x1, y1, w, h = [int(v) for v in face[:4]]
                    cropped_face = face_image[y1:y1 + h, x1:x1 + w]

                    try:

                        embedding =  get_mobilefacenet_embedding(cropped_face)

                        image_count += 1
                        face_array = [embedding, confidence]

                        if way == 'enter':

                            with counter_lock:
                                enter_embeddings = np.vstack([enter_embeddings, normalize(embedding)])
                                enter_ids.append(track_id)

                        print(f"Saved face image for track_id {track_id} with confidence {confidence:.2f}")

                        if way == 'enter':
                            person_enter[track_id]['face'].append(face_array)  # exception on here
                        else:
                            person_exit[track_id]['face'].append(face_array)

                    except Exception as e:
                        print(f"Error encoding face: {e}")

            with counter_lock:
                if (active_face_tracks_enter[track_id] if way == 'enter' else active_face_tracks_exit[track_id]) == 1:
                    if way == 'enter':
                        del active_face_tracks_enter[track_id]
                    else:
                        del active_face_tracks_exit[track_id]
                else:
                    (active_face_tracks_enter if way == 'enter' else active_face_tracks_exit)[track_id] -= 1

            face_queue.task_done()

        except queue.Empty:
            print(f"empty")
            continue


def filter_store_face():
    global enter_embeddings,enter_ids
    while True:
        try:
            track_id, person, way = save_face.get(block=True)

            if track_id not in person:
                save_face.task_done()
                continue

            file_name = f'{track_id}{image_count}{way}.jpg'
            cv2.imwrite(f'{absolute_path }/person_img/{file_name}', person[track_id]['img'])

            # for face in person[track_id]['face']:
                # person[track_id]['face'] = []
                # person[track_id]['face'].append(best_face)

            best_face = max( person[track_id]['face'], key=lambda x: x[1])

            if person is person_enter:
                with app.app_context():
                    print(f"Uploading face data for entry: track_id {track_id}")
                    upload_face_enter(track_id, best_face, file_name)

            else:

                matched_person_id, confidence = batch_compare(normalize(best_face[0]))

                if matched_person_id != -1:
                    with app.app_context():
                        print(f"Uploading face data for exit: track_id {track_id}, matched with person_id {matched_person_id}")
                        upload_face_exit(track_id, best_face, int(matched_person_id), file_name)

                        with counter_lock:

                            mask = np.array(enter_ids) != matched_person_id
                            enter_embeddings = enter_embeddings[mask]
                            enter_ids = list(np.array(enter_ids)[mask])

                        del person_enter[int(matched_person_id)]
                        del person_exit[track_id]
                else:
                    upload_face_exit(track_id, best_face, enter_person_id=None, filename =file_name)


            save_face.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error : {e}")


def batch_compare(current_embedding, threshold=0.4):
    with counter_lock:

        similarities = np.dot(enter_embeddings, current_embedding)
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        # Threshold check
        if best_sim >= threshold:
            print(f"Best match: Person {best_idx} with similarity {best_sim:.4f}")
            return enter_ids[best_idx], best_sim
        return -1, 0.0


def normalize(embedding):
    return embedding / np.linalg.norm(embedding)


MAX_FPS = 10
FRAME_INTERVAL = 1.0 / MAX_FPS


def encode_frame(frame):

    try:

        _, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 60,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1
        ])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Encoding error: {e}")
        return None


def generate_video(way):

    global latest_enter_frame, latest_exit_frame

    last_frame_time = timer.time()

    while counting_active:
        try:
            # Get current frame
            frame = latest_enter_frame if way == 'enter' else latest_exit_frame
            if frame is None:
                timer.sleep(0.01)
                continue

            # Frame rate control
            current_time = timer.time()
            elapsed = current_time - last_frame_time
            if elapsed < FRAME_INTERVAL:
                timer.sleep(FRAME_INTERVAL - elapsed)
                continue
            last_frame_time = current_time

            # Encode and send
            frame_bytes = encode_frame(frame)
            if frame_bytes:
                socketio.emit(f'video_frame_{way}', {'frame': frame_bytes})

            timer.sleep(0.001)

        except Exception as e:
            print(f"Stream error ({way}): {e}")
            timer.sleep(1)  # Prevent tight error loop


def scheduled_task():
    with app.app_context():
        add_record(person_count)


scheduler.add_job(scheduled_task, trigger='cron', hour='8-23,0-4', minute='*/1')
if not scheduler.running:
    scheduler.start()

PERSON_IMG_FOLDER = r"D:/laragon/www/PersonCouting/personCount/person_img"

# Route to serve files from 'person_img' folder
@app.route('/person_img/<path:filename>')
def serve_person_img(filename):
    try:
        print('sendinggg....')
        return send_from_directory(PERSON_IMG_FOLDER, filename)
    except FileNotFoundError:
        print('faileddddd....')
        return "File not found", 404


@app.route('/frame_update', methods=['GET'])
@cross_origin(origins="http://library_occupancy.test")
def frame():

    enter_frame = encode_frame(latest_enter_frame)
    exit_frame = encode_frame(latest_exit_frame)

    return {'frame' : [enter_frame,exit_frame]}


@app.route('/current_status', methods=['GET'])
def status():
    if counting_active is False:
        return f"Counting only start {START_TIME} until {END_TIME}"
    if error is not None:
        return f"{error}"
    return f'success'


@app.route('/setting_update', methods=['POST'])
def handle_settings_update():
    with app.app_context():
        roi, exit_roi , startTime, endTime = update_flask_config()

    if roi and exit_roi and startTime and endTime:
        with app.app_context():
            update_config(roi, exit_roi, startTime, endTime )


@socketio.on('request_video_feed')
def handle_video_feed_request(data):
    way = data.get('way', 'enter')

    if not counting_active:
        return {'status': 'offline'}

    if not hasattr(handle_video_feed_request, 'active_threads'):
        handle_video_feed_request.active_threads = {'enter': None, 'exit': None}

    if handle_video_feed_request.active_threads[way] is None or not handle_video_feed_request.active_threads[way].is_alive():
        thread = threading.Thread(target=generate_video, args=(way,), daemon=True)
        thread.start()
        handle_video_feed_request.active_threads[way] = thread



if __name__ == '__main__':


    threading.Thread(target=background_image_process, args=('enter',), daemon=True).start()
    threading.Thread( target=background_image_process, args=('exit',), daemon=True).start()
    threading.Thread(target=face_detection, daemon=True).start()
    threading.Thread(target=filter_store_face, daemon=True).start()
    socketio.run(app, host='127.0.0.1', port=5000)
