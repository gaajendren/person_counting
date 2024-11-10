import cv2
import cvzone

def aspect_ratio_resize(frame, width = 640):
   h,w = frame.shape[:2]
   ratio = width / float(w)
   height = int(h * ratio)
   return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# c_frame_rate is the frame rate in the video and n_fps is new fps need to achieve

def frame_limit_skip(count, c_frame_rate, n_fps):

   r = c_frame_rate / n_fps

   count += 1
   if count % r != 0:
      return True, count

   return False, count


def mouse_coordinate(event, x, y, flags, param):
   if event == cv2.EVENT_MOUSEMOVE:
      position = [x, y]
      print(position)

def text_count(person_enter, person_exit, width, frame):
   count = len(person_enter) - len(person_exit)
   if count < 0:
      count = 0
   text = f'People Count: {count}'
   text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
   text_x = width - text_size[0] + 40
   text_y = text_size[1]
   cvzone.putTextRect(frame, text, (text_x, text_y), scale=1.5, thickness=2, colorT= (0, 255, 0), offset=5, border=1 )
   return count








