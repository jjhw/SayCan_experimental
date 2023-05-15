from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Iterable
import numpy as np

@dataclass
class Item:
    name: str
    mask: np.ndarray
    score: float
    pos: np.ndarray
    area: int
    additional_names: Iterable[str]

    def __post_init__(self):
        self.mask = self.mask.astype(np.uint8)
        self.pos = self.pos.astype(np.int32)
        self.area = int(self.area)
        self.additional_names = list(_add_name.lower() for _add_name in self.additional_names)

@dataclass
class Circle(Item):
    radius: float

@dataclass
class Box(Item):
    bbox: np.ndarray
    bbox_area: int
    # phrase: str
    angle: float
    width: int
    height: int
    rotated_rect: np.ndarray
    rotated_rect_area: int