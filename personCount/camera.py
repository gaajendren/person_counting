import cv2
import threading
import queue
from personCount.fn import *
from personCount.sort.sort import *
from datetime import time, datetime
import time as timer

error = None

START_TIME = time(4, 30)
END_TIME = time(3, 20)


class Camera():

    def __init__(self, rtsp_url, way):

        self.rtsp_url = rtsp_url
        self.way = way
        self.person = {}
        self.track = Sort(max_age = 20)
        self.frame_count = 0
        self.frame_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.t = threading.Thread(target=self.read_rtsp_stream)
        self.t.daemon = True
        self.t.start()


    def read_rtsp_stream(self):
        global counting_active, enter_frame_count, exit_frame_count, error


        screen = cv2.VideoCapture(self.rtsp_url)

        print(f"Starting video reading for way {self.way}")

        if not screen.isOpened():
            error = 'Error: Could not open video source.'

        frame_counter = 0
        start_time = timer.time()

        while True:
            screen.grab()

            success, frame = screen.retrieve()

            if not success:
                error = 'Error: Could not read video source.'
                continue


            skip, enter_frame_count = frame_limit_skip(self.frame_count)

            if skip:
                continue

            if self.is_within_counting_time():
                    self.person_counting(frame)
            else:
                continue

            frame_counter += 1  # Increment the frame counter

            # Check if 10 frames have been processed
            if frame_counter >= 50:
                end_time = timer.time()  # Record the end time
                elapsed_time = end_time - start_time  # Calculate the elapsed time
                print(f"Time taken to process 10 frames: {elapsed_time:.2f} seconds {enter_frame_count}")
                break


    def is_within_counting_time(self):
        now = datetime.now().time()
        if START_TIME > END_TIME:
            return now >= START_TIME or now <= END_TIME
        else:
            return START_TIME <= now <= END_TIME


    def person_counting(frame, self):
        global  counting_active, socketio, person_count

        resized_frame = aspect_ratio_resize(frame)
        ori_frame = resized_frame.copy()
        height, width, _ = resized_frame.shape

        update_check(person)

        print(f"Detecting people in frame for '{way}'...")

        results = modal(resized_frame, verbose=False, stream=True, classes=[0])
        detections = np.empty((0, 5))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100

                if cv2.pointPolygonTest(np.array(rio), (x2, y2), False) < 0:
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
                    print(f"New person detected: {track_id}")
                    person[track_id] = {"entered": False, "face": [], 'checked': False}

                if line[0][0] < x2 < line[1][0] and y2 > line[0][1]:

                    if way == 'enter':
                        if track_id not in person_count_enter:
                            person_count_enter.append(track_id)
                    else:
                        if track_id not in person_count_exit:
                            person_count_exit.append(track_id)

                    print(f"Person {track_id} crossed the line for '{way}'")

                    if not person[track_id]["checked"]:

                        if check_left_face_process(track_id, person, way) and person[track_id]['face']:
                            print(f"Saving face data for person {track_id}...")
                            save_face.put((track_id, person[track_id], person))

                    person[track_id]["entered"] = True

                else:

                    if not person[track_id]["entered"] and person[track_id]["checked"] == False:
                        # face start
                        face_image = resized_frame[y1:y2, x1:x2]
                        ori_person = ori_frame[y1:y2, x1:x2]
                        print(f"Adding face image for person in the queue {track_id}...")
                        face_queue.put((track_id, face_image, way, ori_person))

                # end............................

                # face_image = resized_frame[y1:y2, x1:x2]
                cv2.putText(resized_frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                            thickness=2)

        cv2.polylines(resized_frame, [np.array(rio)], True, (0, 255, 255), thickness=1)
        cv2.line(resized_frame, line[0], line[1], (0, 255, 0), thickness=2)

        if way == 'enter':
            latest_enter_frame = resized_frame

        else:
            latest_exit_frame = resized_frame

        counting_active = True

        person_count = text_count(person_count_enter, person_count_exit, width, resized_frame)

        socketio.emit('person_count', {'count': person_count, })