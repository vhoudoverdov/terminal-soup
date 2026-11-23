"""termarium: 256-color palettes with cycling & thermal mapping."""
from __future__ import annotations

import numpy as np

_I = np.arange(256)


def _cyclic_stops(colors):
    n = len(colors)
    xs = np.arange(n + 1) / n
    cols = np.array(list(colors) + [colors[0]], dtype=np.float64)
    t = _I / 256.0
    return np.stack([np.interp(t, xs, cols[:, c]) for c in range(3)], axis=1)


def _cosine(a, b, c, d):
    t = (_I / 256.0)[:, None]
    a, b, c, d = (np.array(v, dtype=np.float64) for v in (a, b, c, d))
    return 255.0 * np.clip(a + b * np.cos(2 * np.pi * (c * t + d)), 0.0, 1.0)


PALETTES = [
    ("Psychedelic Bio", _cyclic_stops([(10, 0, 40), (0, 240, 180), (255, 0, 128), (255, 220, 20), (50, 0, 200)])),
    ("Deep Coral Reef", _cyclic_stops([(0, 15, 30), (0, 120, 180), (255, 100, 80), (255, 230, 140), (140, 20, 160)])),
    ("Electric Lava Lamp", _cyclic_stops([(20, 0, 10), (180, 0, 60), (255, 90, 0), (255, 240, 60), (255, 0, 100)])),
    ("Acid Matrix", _cyclic_stops([(0, 10, 0), (0, 160, 40), (140, 255, 20), (0, 255, 220), (220, 255, 180)])),
    ("Thermal Cosine", _cosine((.5, .5, .5), (.5, .5, .5), (1, 1, 1), (0, .33, .67))),
    ("Abyssal Chem", _cyclic_stops([(5, 5, 20), (20, 60, 140), (80, 220, 255), (240, 255, 255), (160, 80, 240)])),
]

_RAMP = (np.clip(_I / 64.0, 0.0, 1.0) ** 1.2)[:, None]
_WHITE = (np.clip((_I - 210) / 45.0, 0.0, 1.0) ** 2 * 0.4)[:, None]


class PaletteManager:
    def __init__(self, index: int = 0, speed: float = 12.0):
        self.index = index % len(PALETTES)
        self.prev_index = self.index
        self.blend = 1.0
        self.offset = 0.0
        self.speed = speed
        self.cycle_enabled = False

    @property
    def name(self) -> str:
        return PALETTES[self.index][0]

    def next(self, step: int = 1):
        self.prev_index = self.index
        self.index = (self.index + step) % len(PALETTES)
        self.blend = 0.0

    def update(self, dt: float):
        if self.cycle_enabled:
            self.offset += dt * self.speed
        self.blend = min(1.0, self.blend + dt / 0.6)

    def lut(self) -> np.ndarray:
        idx = (_I + int(self.offset)) % 256
        hue = PALETTES[self.index][1][idx]
        if self.blend < 1.0:
            hue = hue * self.blend + PALETTES[self.prev_index][1][idx] * (1.0 - self.blend)
        col = hue * _RAMP
        col = col + (255.0 - col) * _WHITE
        col[0] = 0.0
        return np.clip(col, 0, 255).astype(np.uint8)
