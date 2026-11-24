"""terminal-soup: Gray-Scott Reaction-Diffusion Simulation.

Models chemical morphogens U and V reacting and diffusing on a 2D toroidal grid
using a 9-point isotropic Patra-Karttunen Laplacian stencil and membrane-relief shading.
"""
from __future__ import annotations

import numpy as np
from .base import Simulation

# Iconic (F, k, substeps) parameter regimes:
REGIMES = [
    ("Coral Reef / Maze",   0.0545, 0.0620, 6),
    ("Mitosis / Bacteria",  0.0367, 0.0649, 6),
    ("Solitons / Waves",    0.0140, 0.0540, 6),
    ("Cheetah Spots",       0.0300, 0.0620, 6),
    ("Fingerprint Spirals", 0.0180, 0.0510, 6),
    ("Wandering Worms",     0.0580, 0.0650, 6),
]


class GrayScott(Simulation):
    name = "Gray-Scott Reaction-Diffusion"

    def __init__(self):
        super().__init__()
        self.regime_idx = 0
        self.F = REGIMES[0][1]
        self.k = REGIMES[0][2]
        self.substeps = REGIMES[0][3]
        self.Du = 0.16
        self.Dv = 0.08
        self.U = np.empty((0, 0), dtype=np.float32)
        self.V = np.empty((0, 0), dtype=np.float32)

    def reset(self):
        self.U = np.ones((self.h, self.w), dtype=np.float32)
        self.V = np.zeros((self.h, self.w), dtype=np.float32)

        # Seed with several organic clusters
        count = max(6, (self.w * self.h) // 2500)
        for _ in range(count):
            rx = int(self.rng.integers(self.w // 5, 4 * self.w // 5))
            ry = int(self.rng.integers(self.h // 5, 4 * self.h // 5))
            self.seed(rx, ry, radius=int(self.rng.integers(4, 9)))

    def seed(self, x: int, y: int, radius: int = 6):
        if self.w == 0 or self.h == 0:
            return
        x0, x1 = max(0, x - radius), min(self.w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.h, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - x) ** 2 + (yy - y) ** 2
        mask = dist2 <= radius ** 2
        n_pts = int(np.sum(mask))
        if n_pts > 0:
            self.U[y0:y1, x0:x1][mask] = 0.50 + self.rng.uniform(-0.06, 0.06, n_pts).astype(np.float32)
            self.V[y0:y1, x0:x1][mask] = 0.25 + self.rng.uniform(-0.06, 0.06, n_pts).astype(np.float32)

    def next_regime(self):
        self.regime_idx = (self.regime_idx + 1) % len(REGIMES)
        self.F = REGIMES[self.regime_idx][1]
        self.k = REGIMES[self.regime_idx][2]
        self.substeps = REGIMES[self.regime_idx][3]

    def mutate(self):
        self.F = float(np.clip(self.F + self.rng.uniform(-0.002, 0.002), 0.012, 0.062))
        self.k = float(np.clip(self.k + self.rng.uniform(-0.002, 0.002), 0.048, 0.068))

    @staticmethod
    def _laplacian_9pt(X: np.ndarray) -> np.ndarray:
        """9-point rotationally isotropic discrete Laplacian (cardinal 2/3, diagonal 1/6, center -10/3)."""
        n = np.roll(X, -1, axis=0)
        s = np.roll(X, 1, axis=0)
        e = np.roll(X, -1, axis=1)
        w = np.roll(X, 1, axis=1)
        ne = np.roll(n, -1, axis=1)
        nw = np.roll(n, 1, axis=1)
        se = np.roll(s, -1, axis=1)
        sw = np.roll(s, 1, axis=1)
        return (2.0 / 3.0) * (n + s + e + w) + (1.0 / 6.0) * (ne + nw + se + sw) - (10.0 / 3.0) * X

    def step(self, dt: float) -> np.ndarray:
        if self.w == 0 or self.h == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        U, V = self.U, self.V
        F, k = self.F, self.k
        Du, Dv = self.Du, self.Dv

        for _ in range(self.substeps):
            lapU = self._laplacian_9pt(U)
            lapV = self._laplacian_9pt(V)
            UV2 = U * V * V
            U += Du * lapU - UV2 + F * (1.0 - U)
            V += Dv * lapV + UV2 - (F + k) * V
            np.clip(U, 0.0, 1.0, out=U)
            np.clip(V, 0.0, 1.0, out=V)

        # Auto-nucleate if the dish ever goes completely extinct
        if float(V.max()) < 0.02:
            self.seed(self.w // 2, self.h // 2, radius=7)

        # Smooth membrane-relief shading: combine morphogen ratio V/(U+V) with edge gradient |grad V|
        ratio = V / (U + V + 1e-5)
        core = np.clip(ratio * 1.65, 0.0, 1.0) ** 0.85
        gx = np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)
        gy = np.roll(V, -1, axis=0) - np.roll(V, 1, axis=0)
        rim = np.hypot(gx, gy) * 1.8
        shaded = np.clip(core * 0.82 + rim * 0.35, 0.0, 1.0)
        return (shaded * 255.0).astype(np.uint8)

    def get_info(self) -> str:
        regime_name = REGIMES[self.regime_idx][0]
        return f"Gray-Scott [{regime_name}] F={self.F:.4f} k={self.k:.4f}"
