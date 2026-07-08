import os, warnings
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = '0'

import numpy as np
from ultralytics import YOLO
from .base_tracker import BaseTracker

class StrongSortTracker(BaseTracker):
    def __init__(self, model_path, config=None):
        super().__init__(config)
        self.model = YOLO(model_path) if isinstance(model_path, str) else model_path
        self._tracker = None
        self._last_ids = []

    def _get_boxmot(self):
        if self._tracker is None:
            try:
                from boxmot import StrongSort
                self._tracker = StrongSort()
            except Exception as e:
                print(f"  StrongSort init failed: {e}")
                raise
        return self._tracker

    def track(self, source, **kwargs):
        conf = kwargs.pop("conf", 0.25)
        iou = kwargs.pop("iou", 0.45)
        results = self.model(source, conf=conf, iou=iou, verbose=False)
        self._last_ids = []
        for r in results:
            if r.boxes is None:
                continue
            dets = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            if len(dets) == 0:
                continue
            try:
                tracker = self._get_boxmot()
                img = r.orig_img
                tracks = tracker.update(dets, scores, cls_ids, img)
                if tracks is not None and len(tracks) > 0:
                    for t in tracks:
                        x1, y1, x2, y2, tid, conf, cls = t[:7]
                        self._last_ids.append(int(tid))
            except:
                pass
        return results

    def get_tracked_objects(self):
        return [{"track_id": tid} for tid in set(self._last_ids)]
