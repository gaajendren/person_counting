from datetime import time, datetime
from personCount.Database.setting import Setting
import json


START_TIME = time()
END_TIME = time()
rio = []
exit_rio = []

def initial_config():
    global START_TIME,END_TIME, rio , exit_rio

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



line = [(290,355),(462,355)]



absolute_path = "D:/laragon/www/PersonCouting/personCount"

camera_enter = "/test_video/enter.mp4"
camera_exit = "/test_video/exit.mp4"


def update_config(roi, exit_roi, start_time , end_time ):
    global rio , exit_rio, START_TIME,END_TIME

    points = json.loads(roi)
    roi = [(int(point["x"]) * 2, int(point["y"]) * 2) for point in points]
    roi[2], roi[3] = roi[3], roi[2]

    rio = roi

    points_exit = json.loads(exit_roi)
    roi_exit = [(int(point["x"]) * 2, int(point["y"]) * 2) for point in points_exit]
    roi_exit[2], roi_exit[3] = roi_exit[3], roi_exit[2]

    exit_rio = roi_exit

    END_TIME = start_time
    START_TIME = end_time