"""terminal-soup: Base simulation interface."""
from __future__ import annotations

import numpy as np


class Simulation:
    name: str = "Base Simulation"

    def __init__(self):
        self.w = 0
        self.h = 0
        self.rng = np.random.default_rng()

    def resize(self, w: int, h: int):
        if (self.w, self.h) != (w, h):
            self.w = w
            self.h = h
            self.reset()

    def reset(self):
        raise NotImplementedError

    def step(self, dt: float) -> np.ndarray:
        """Returns (H, W) uint8 indexed intensity array (0..255)."""
        raise NotImplementedError

    def seed(self, x: int, y: int, radius: int = 5):
        """Drops an interactive seed / disturbance at (x, y)."""
        pass

    def mutate(self):
        """Randomly alters simulation parameters / chaotic lava lamp evolution."""
        pass

    def get_info(self) -> str:
        """Returns brief text for HUD."""
        return self.name
