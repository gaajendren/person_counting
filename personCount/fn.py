import cv2
import cvzone

def aspect_ratio_resize(frame, width = 640):
   if frame is None:
      print("Error: Frame is None, skipping resize.")
      return None
   h,w = frame.shape[:2]
   ratio = width / float(w)
   height = int(h * ratio)
   return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# c_frame_rate is the frame rate in the video and n_fps is new fps need to achieve

def frame_limit_skip(count):

   r = 8

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

   if count <0:
      count =0

   # text = f'People Count: {count}'
   # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
   # text_x = width - text_size[0] + 40
   # text_y = text_size[1]
   # cvzone.putTextRect(frame, text, (text_x, text_y), scale=1.5, thickness=2, colorT= (0, 255, 0), offset=5, border=1 )

   return count


import cv2
import numpy as np
from collections import Counter


def get_dominant_color(image, k=3):
   """
   Extracts the dominant color from an image (ROI of clothing).
   :param image: Cropped person image (clothing area)
   :param k: Number of dominant colors to consider
   :return: Most common color name (e.g., 'Red', 'Blue', 'Black')
   """
   # Resize for faster processing
   image = cv2.resize(image, (50, 50))

   # Convert to RGB
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   # Reshape image data
   pixels = image.reshape(-1, 3)

   # K-Means clustering to find main colors
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=k, n_init=10)
   kmeans.fit(pixels)
   colors = kmeans.cluster_centers_.astype(int)

   # Count occurrences of each color
   labels = kmeans.labels_
   color_counts = Counter(labels)

   # Find the most common color
   dominant_color = colors[color_counts.most_common(1)[0][0]]

   # Convert to color name
   return get_color_name(dominant_color)


def get_color_name(rgb):
   """
   Maps RGB values to basic color names.
   :param rgb: Tuple (R, G, B)
   :return: Closest color name
   """
   colors = {
      "Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255),
      "Yellow": (255, 255, 0), "Orange": (255, 165, 0), "Purple": (128, 0, 128),
      "Black": (0, 0, 0), "White": (255, 255, 255), "Gray": (128, 128, 128)
   }

   # Find closest color
   min_dist = float('inf')
   best_match = "Unknown"

   for name, color_rgb in colors.items():
      dist = np.linalg.norm(np.array(rgb) - np.array(color_rgb))
      if dist < min_dist:
         min_dist = dist
         best_match = name

   return best_match





