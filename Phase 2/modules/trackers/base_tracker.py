from abc import ABC, abstractmethod

class BaseTracker(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def track(self, source, **kwargs):
        pass

    @abstractmethod
    def get_tracked_objects(self):
        pass

    def reset(self):
        pass
