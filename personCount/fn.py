import cv2
import cvzone

def aspect_ratio_resize(frame, width = 640):
   if frame is None:
      print("Error: Frame is None, skipping resize.")
      return None

   height = 320
   return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# c_frame_rate is the frame rate in the video and n_fps is new fps need to achieve

def frame_limit_skip(count):

   r = 2  #here it say how many skip

   count += 1
   if count % r != 0:
      return True, count

   return False, count


def mouse_coordinate(event, x, y, flags, param):
   if event == cv2.EVENT_MOUSEMOVE:
      position = [x, y]
      print(position)

def text_count(person_enter, person_exit):

   count = len(person_enter) - len(person_exit)

   if count <0:
      count =0

   return count




