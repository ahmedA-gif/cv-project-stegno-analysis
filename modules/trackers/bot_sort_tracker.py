from ultralytics import YOLO
from .base_tracker import BaseTracker

class BotSortTracker(BaseTracker):
    def __init__(self, model_path, config=None):
        super().__init__(config)
        self.model = YOLO(model_path) if isinstance(model_path, str) else model_path
        self._last_results = None

    def track(self, source, **kwargs):
        conf = kwargs.pop("conf", 0.25)
        iou = kwargs.pop("iou", 0.45)
        results = self.model.track(
            source,
            tracker="botsort.yaml",
            persist=True,
            conf=conf,
            iou=iou,
            **kwargs
        )
        self._last_results = results
        return results

    def get_tracked_objects(self):
        if self._last_results is None:
            return []
        objects = []
        for r in self._last_results:
            if r.boxes is None or r.boxes.id is None:
                continue
            for box, tid, conf, cls in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy()
            ):
                objects.append({
                    "track_id": int(tid),
                    "box": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "label": r.names[int(cls)]
                })
        return objects
