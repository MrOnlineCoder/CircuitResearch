"""YOLO model loading and inference (device detector, crossing detector, orientation classifiers)."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ORIENTATION_TASKS, Settings


@dataclass
class Box:
    label: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


def _torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Models:
    """Lazily loads the trained weights from a models/ directory."""

    def __init__(self, models_dir, settings: Settings):
        self.dir = Path(models_dir)
        self.s = settings
        self.device = _torch_device()

    @lru_cache(maxsize=None)
    def _load(self, name: str):
        from ultralytics import YOLO

        path = self.dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing model {path}. Train it (see TRAINING.md) and copy it here.")
        return YOLO(str(path))

    def has(self, name: str) -> bool:
        return (self.dir / f"{name}.pt").exists()

    def devices(self, image: np.ndarray) -> list[Box]:
        """All objects from the device detector (components, gnd/vdd, ports, hop arcs)."""
        r = self._load("devices").predict(image, imgsz=self.s.imgsz_devices, conf=self.s.device_conf,
                                          agnostic_nms=True, device=self.device, verbose=False)[0]
        names = r.names
        return [Box(names[int(c)], float(p), *map(float, xyxy))
                for xyxy, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]

    def crossings(self, image: np.ndarray) -> list[Box]:
        """Crossings/junctions from the OBB model, as axis-aligned boxes."""
        r = self._load("jumpers").predict(image, imgsz=self.s.imgsz_jumpers, conf=self.s.jumper_conf,
                                          device=self.device, verbose=False)[0]
        names = r.names
        out = []
        for poly, c, p in zip(r.obb.xyxyxyxy.tolist(), r.obb.cls.tolist(), r.obb.conf.tolist()):
            xs, ys = [pt[0] for pt in poly], [pt[1] for pt in poly]
            out.append(Box(names[int(c)], float(p), min(xs), min(ys), max(xs), max(ys)))
        return out

    def classify(self, task: str, crop: Image.Image):
        """(label, confidence) from an orientation classifier, or None if not available."""
        assert task in ORIENTATION_TASKS, task
        if not self.has(task):
            return None
        r = self._load(task).predict(crop.convert("RGB"), device=self.device, verbose=False)[0]
        top = int(r.probs.top1)
        return r.names[top], float(r.probs.top1conf)
