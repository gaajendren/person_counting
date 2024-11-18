from personCount.deep_sort.deep_sort.tracker import *
from personCount.deep_sort.deep_sort.detection import Detection
from personCount.deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from personCount.deep_sort.tools import generate_detections
import numpy as np



class DeepSort:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):

        metrix = NearestNeighborDistanceMetric('cosine', 0.4, 100)
        self.tracker = Tracker(metrix)
        self.encoder = generate_detections.create_box_encoder('C:/Users/gaaje/PycharmProjects/person_counting/personCount/modal-data/mars-small128.pb', batch_size=1)



    def update(self, frame , detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return


        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        det = []
        for bbox_id, bbox in enumerate(bboxes):
            det.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(det)
        self.update_tracks()


    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            id = track.track_id

            tracks.append(Track(id, bbox))

        self.tracks = tracks



class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox

