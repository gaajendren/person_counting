import base64
import math
import queue
import threading
import time as timer
from collections import defaultdict
from queue import Queue
from threading import Lock
import cv2
import onnxruntime as ort
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, send_from_directory
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO
from tensorflow.python.util.numpy_compat import np_array
from ultralytics import YOLO
from ffmpegcv import VideoCaptureStreamRT
import personCount.config as config
from personCount.Database.init import *
from personCount.Yunet.yunet import YuNet
from personCount.config import *
from personCount.controller import *
from personCount.fn import *
from personCount.sort.sort import *

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000", "supports_credentials": True}})

init_db(app)

scheduler = BackgroundScheduler()

cv2.ocl.setUseOpenCL(True)

modal = YOLO("C:/Users/gaaje/PycharmProjects/person_counting/personCount/trained_yolo_model/yolov8s.onnx")


track_enter = Sort(max_age=20)
track_exit = Sort(max_age=20)

providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

face_recognizer = ort.InferenceSession(f"{absolute_path}/Onnx/mobilefacenet.onnx" , providers=providers)


detector = YuNet(
    f"{absolute_path}/Yunet/face_detection_yunet_2023mar.onnx",
    (320, 320),
    confThreshold=0.8,
    nmsThreshold=0.3,
    topK=5000,
    backendId=cv2.dnn.DNN_BACKEND_OPENCV,
    targetId=cv2.dnn.DNN_TARGET_OPENCL)


person_enter = {}
person_exit = {}

person_count = 0

person_count_enter = []
person_count_exit = []

image_count = 0

face_queue = Queue()
save_face = Queue()

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

latest_enter_frame = None
latest_exit_frame = None
frame_lock = Lock()

start = timer.perf_counter()

def is_within_counting_time():
    now = datetime.now().time()
    if config.START_TIME > config.END_TIME:
        return now >= config.START_TIME or now <= config.END_TIME
    else:
        return config.START_TIME <= now <= config.END_TIME



def background_image_process(way):
    global  enter_frame_count, exit_frame_count, error, start

    if way == 'enter':
        screen = cv2.VideoCapture(
            'rtspsrc location=rtsp://admin:555888Gaa$@192.168.1.103:554/Streaming/Channels/101 latency=0 ! '
            'rtph264depay ! h264parse ! '
            'd3d11h264dec  ! videoconvert ! '
            'video/x-raw,format=BGR ! '
            'appsink sync=false',
            cv2.CAP_GSTREAMER
        )
    else:
        screen = cv2.VideoCapture(camera_exit)

    if not screen.isOpened():
        raise RuntimeError("Error: Could not open video source.")

    while True:
        success, frame = screen.read()

        if not success:
            error = 'Error: Could not read video source.'
            end = timer.perf_counter()
            elapsed_time = end - start
            print(f"Elapsed time: {elapsed_time} seconds")
            print(error)

            if way == 'enter':
                screen = cv2.VideoCapture(
                     'rtspsrc location=rtsp://admin:555888Gaa$@192.168.1.103:554/Streaming/Channels/101 latency=0 ! '
                     'rtph264depay ! h264parse ! '
                     'd3d11h264dec ! videoconvert ! '
                     'video/x-raw,format=BGR ! appsink sync=false',
                    cv2.CAP_GSTREAMER
                )
            else:
                screen = cv2.VideoCapture( camera_exit)


            if not screen.isOpened():
                raise RuntimeError("Error: Could not open video source.")

            continue

        if way == 'enter':
            skip, enter_frame_count = frame_limit_skip(enter_frame_count)
        else:
            skip, exit_frame_count = frame_limit_skip(exit_frame_count)

        if skip:
            continue

        if is_within_counting_time() and config.is_auto:
            config.counting_active = True

            if way == 'enter':
                person_counting(frame, way, person_enter, track_enter)
            else:
                person_counting(frame, way, person_exit, track_exit)
        else:
            config.counting_active = False
            print('no')
            continue


def person_counting(frame, way, person, track):
    global person_enter, person_exit, latest_enter_frame, latest_exit_frame, socketio, person_count, image_count, person_count_enter, person_count_exit, unchecked_track_id_exit, unchecked_track_id_enter

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

                update_check(person, way)

                cv2.putText(resized_frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                            thickness=2)

            cv2.polylines(resized_frame, [np.array(r_io)], True, (0, 255, 255), thickness=1)

        cv2.line(resized_frame, line[0], line[1], (0, 255, 0), thickness=2)

        with frame_lock:
            if way == 'enter':
                latest_enter_frame = resized_frame
            else:
                latest_exit_frame = resized_frame

        person_count = text_count(person_count_enter, person_count_exit)

        socketio.emit('person_count', {'count': person_count, })

    except Exception as e:
        print(f"Error while loading model: {e}")


def update_check(person, way):
    global unchecked_track_id_enter, unchecked_track_id_exit

    with counter_lock:
        unchecked_track_id = (unchecked_track_id_enter if way == 'enter' else unchecked_track_id_exit)

        if not unchecked_track_id:
            return

        for track_id in unchecked_track_id:
            if track_id not in (active_face_tracks_enter if way == 'enter' else active_face_tracks_exit):
                person[track_id]['checked'] = True
                unchecked_track_id.remove(track_id)
                save_face.put((track_id, person, way))


def get_mobileFacenet_embedding(face_img):
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

            if (way == 'enter' and person_enter[track_id]["checked"]) or (
                    way == 'exit' and person_exit[track_id]["checked"]):
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

                        embedding = get_mobileFacenet_embedding(cropped_face)

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
    global enter_embeddings, enter_ids
    while True:
        try:
            track_id, person, way = save_face.get(block=True)

            if track_id not in person:
                save_face.task_done()
                continue

            file_name = f'{track_id}{image_count}{way}.jpg'
            cv2.imwrite(f'{absolute_path}/person_img/{file_name}', person[track_id]['img'])

            best_face = max(person[track_id]['face'], key=lambda x: x[1])

            if person is person_enter:
                with app.app_context():
                    print(f"Uploading face data for entry: track_id {track_id}")
                    upload_face_enter(track_id, best_face, file_name)

            else:

                matched_person_id, confidence = batch_compare(normalize(best_face[0]))

                if matched_person_id != -1:
                    with app.app_context():
                        print(
                            f"Uploading face data for exit: track_id {track_id}, matched with person_id {matched_person_id}")
                        upload_face_exit(track_id, best_face, int(matched_person_id), file_name)

                        with counter_lock:
                            mask = np.array(enter_ids) != matched_person_id
                            enter_embeddings = enter_embeddings[mask]
                            enter_ids = list(np.array(enter_ids)[mask])

                        del person_enter[int(matched_person_id)]
                        del person_exit[track_id]
                else:
                    upload_face_exit(track_id, best_face, enter_person_id=None, filename=file_name)

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


def encode_frame(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Encoding error: {e}")
        return None


def generate_video(way):
    global latest_enter_frame, latest_exit_frame

    target_fps = 30
    frame_interval = 1.0 / target_fps

    while config.counting_active:
        try:
            start_time = time.time()

            with frame_lock:
                frame = latest_enter_frame if way == 'enter' else latest_exit_frame

            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit(f'video_frame_{way}', {'frame': frame_b64})

            # Maintain stable FPS
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_interval - elapsed))

        except Exception as e:
            print(f"Stream error ({way}): {str(e)}")
            time.sleep(0.5)


def scheduled_task():
    with app.app_context():
        add_record(person_count)


scheduler.add_job(scheduled_task, trigger='cron', hour='8-23,0-4', minute='*/15')
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

    return {'frame': [enter_frame, exit_frame]}


@app.route('/current_status', methods=['GET'])
def status():
    if config.counting_active is False and config.is_auto is True:
        return f"Counting only start {START_TIME} until {END_TIME}"
    if error is not None:
        return f"{error}"
    return f'success'


@app.route('/setting_update', methods=['POST'])
def handle_settings_update():
    with app.app_context():
        roi, exit_roi, startTime, endTime, is_manual = update_flask_config()

    if roi and exit_roi and startTime and endTime and is_manual:
        with app.app_context():
            update_config(roi, exit_roi, startTime, endTime, is_manual)


@app.route('/manual_start', methods=['POST'])
def handle_auto_start(data):
    with app.app_context():
        if data == 'start':
           config.is_auto = True
        elif data == 'end':
           config.is_auto = False


@socketio.on('request_video_feeds')
def handle_video_feeds(data):
    ways = data.get('ways', ['enter', 'exit'])

    if not config.counting_active:
        return {'status': 'offline'}

    # Start thread for each requested stream
    for way in ways:
        if not hasattr(handle_video_feeds, 'threads'):
            handle_video_feeds.threads = {}

        if way not in handle_video_feeds.threads or not handle_video_feeds.threads[way].is_alive():
            thread = threading.Thread(target=generate_video, args=(way,), daemon=True)
            thread.start()
            handle_video_feeds.threads[way] = thread



if __name__ == '__main__':
    threading.Thread(target=background_image_process, args=('enter',), daemon=True).start()
    threading.Thread(target=background_image_process, args=('exit',), daemon=True).start()
    threading.Thread(target=face_detection, daemon=True).start()
    threading.Thread(target=filter_store_face, daemon=True).start()
    socketio.run(app, host='127.0.0.1', port=5000)
