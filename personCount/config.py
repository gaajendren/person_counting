from datetime import time, datetime
from personCount.Database.setting import Setting
import json
from threading import Lock

config_lock = Lock()

counting_active = True
is_auto = True
START_TIME = time()
END_TIME = time()
rio = []
exit_rio = []

def initial_config():
    global START_TIME,END_TIME, rio , exit_rio ,counting_active,is_auto

    setting = Setting.query.first()

    START_TIME = setting.start_time
    END_TIME = setting.end_time


    initial_points = json.loads(setting.roi)
    initial_roi = [(int(point["x"]) * 2, int(point["y"]) * 2) for point in initial_points]
    initial_roi[2], initial_roi[3] = initial_roi[3], initial_roi[2]

    rio = initial_roi

    initial_points_exit = json.loads(setting.exit_roi)
    initial_roi_exit = [(int(point["x"]) * 2, int(point["y"]) * 2) for point in initial_points_exit]
    initial_roi_exit[2], initial_roi_exit[3] = initial_roi_exit[3], initial_roi_exit[2]

    exit_rio = initial_roi_exit

    if setting.is_manual == 0:
        counting_active = False
        is_auto = False
    else:
        counting_active = True
        is_auto = True




line = [(146*2,132*2),(442,132*2)]



absolute_path = "C:/Users/gaaje/PycharmProjects/person_counting/personCount"

camera_enter = "rtsp://admin:555888Gaa$@192.168.1.103:554/Streaming/Channels/101"
camera_exit = "rtsp://admin:555888Gaa$@192.168.1.103:554/Streaming/Channels/101"

test_video_enter = 'C:\\Users\\gaaje\\PycharmProjects\\person_counting\\personCount\\test_video\\enter.mp4'
test_video_exit = 'C:\\Users\\gaaje\\PycharmProjects\\person_counting\\personCount\\test_video\\exit.mp4'

def update_config():
    global rio , exit_rio, START_TIME,END_TIME,counting_active,is_auto
    print('success ongoing update')

    setting = Setting.query.first()

    START_TIME = setting.start_time
    END_TIME = setting.end_time

    points = json.loads(setting.roi)
    roi = [(int(point["x"]) * 2, int(point["y"]) * 2) for point in points]
    roi[2], roi[3] = roi[3], roi[2]

    rio = roi

    points_exit = json.loads(setting.exit_roi)
    roi_exit = [(int(point["x"]) * 2, int(point["y"]) * 2) for point in points_exit]
    roi_exit[2], roi_exit[3] = roi_exit[3], roi_exit[2]

    exit_rio = roi_exit

    with config_lock:
        if setting.is_manual == 0:
            counting_active = False
            is_auto = False
        else:
            counting_active = True
            is_auto = True
