"""termarium: Continuous Lenia Simulation.

Fluid, continuous generalization of Conway's Game of Life.
Generates self-organizing, gliding, dividing organic solitons and micro-organisms.
"""
from __future__ import annotations

import numpy as np
from .base import Simulation


class Lenia(Simulation):
    name = "Continuous Lenia (Fluid Life)"

    def __init__(self):
        super().__init__()
        self.A = np.empty((0, 0), dtype=np.float32)
        self.R = 10  # Kernel radius
        self.mu = 0.135
        self.sigma = 0.015
        self.dt = 0.15
        self.K = None
        self.K_fft = None

    def _build_kernel(self):
        r = self.R
        y, x = np.ogrid[-r:r+1, -r:r+1]
        dist = np.hypot(x, y) / r
        # Ring kernel with peak at 0.5
        kernel = np.exp(-((dist - 0.5)**2) / 0.08) * (dist <= 1.0)
        s = kernel.sum()
        if s > 0:
            kernel /= s
        self.K = kernel.astype(np.float32)

    def reset(self):
        self._build_kernel()
        self.A = np.zeros((self.h, self.w), dtype=np.float32)
        # Drop a few random living amoebic seeds
        for _ in range(5):
            cx = self.rng.integers(self.w // 4, 3 * self.w // 4)
            cy = self.rng.integers(self.h // 4, 3 * self.h // 4)
            self.seed(cx, cy, radius=self.R + 2)

    def seed(self, x: int, y: int, radius: int = 12):
        if self.w == 0 or self.h == 0:
            return
        x0, x1 = max(0, x - radius), min(self.w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.h, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(xx - x, yy - y) / max(1, radius)
        blob = np.exp(-dist**2 / 0.4) * (dist <= 1.0) * self.rng.uniform(0.6, 1.0, dist.shape)
        self.A[y0:y1, x0:x1] = np.clip(self.A[y0:y1, x0:x1] + blob, 0.0, 1.0)

    def mutate(self):
        # Mutate growth parameters to shift organism species
        self.mu = float(np.clip(self.mu + self.rng.uniform(-0.01, 0.01), 0.11, 0.17))
        self.sigma = float(np.clip(self.sigma + self.rng.uniform(-0.003, 0.003), 0.010, 0.025))

    def step(self, dt: float) -> np.ndarray:
        if self.w == 0 or self.h == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        # 2D Circular Convolution using FFT for fast O(N log N) field aggregation
        A = self.A
        Kh, Kw = self.K.shape
        padded_K = np.zeros_like(A)
        # Place kernel centered with periodic wrap
        cy, cx = Kh // 2, Kw // 2
        padded_K[:Kh - cy, :Kw - cx] = self.K[cy:, cx:]
        padded_K[:Kh - cy, -cx:] = self.K[cy:, :cx]
        padded_K[-cy:, :Kw - cx] = self.K[:cy, cx:]
        padded_K[-cy:, -cx:] = self.K[:cy, :cx]

        U = np.fft.ifft2(np.fft.fft2(A) * np.fft.fft2(padded_K)).real

        # Continuous Growth mapping G(u) in [-1, 1]
        G = 2.0 * np.exp(-((U - self.mu)**2) / (2.0 * (self.sigma**2))) - 1.0

        # Field update
        self.A = np.clip(A + self.dt * G, 0.0, 1.0)

        intensity = (self.A * 255.0).astype(np.uint8)
        return intensity

    def get_info(self) -> str:
        return f"Lenia [mu={self.mu:.3f} sigma={self.sigma:.3f}]"
