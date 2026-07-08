from .byte_tracker import ByteTracker
from .bot_sort_tracker import BotSortTracker
from .deep_sort_tracker import DeepSortTracker
from .oc_sort_tracker import OCSortTracker
from .strong_sort_tracker import StrongSortTracker

_TRACKER_REGISTRY = {
    "bytetrack": ByteTracker,
    "botsort": BotSortTracker,
    "deepsort": DeepSortTracker,
    "ocsort": OCSortTracker,
    "strongsort": StrongSortTracker,
}

SUPPORTED_TRACKERS = list(_TRACKER_REGISTRY.keys())

def get_tracker(name, model, config=None):
    name = name.lower().strip()
    if name not in _TRACKER_REGISTRY:
        raise ValueError(f"Unknown tracker '{name}'. Available: {SUPPORTED_TRACKERS}")
    return _TRACKER_REGISTRY[name](model, config)

def list_trackers():
    return SUPPORTED_TRACKERS

def register_tracker(name, cls):
    _TRACKER_REGISTRY[name] = cls
    global SUPPORTED_TRACKERS
    SUPPORTED_TRACKERS = list(_TRACKER_REGISTRY.keys())
