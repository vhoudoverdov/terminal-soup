"""termarium: Gray-Scott Reaction-Diffusion Simulation.

Models chemical morphogens U and V reacting and diffusing on a 2D toroidal grid.
Produces spots, labyrinths, dividing cells, and traveling solitary waves.
"""
from __future__ import annotations

import numpy as np
from .base import Simulation

# Iconic (F, k) parameter regimes:
REGIMES = [
    ("Coral Reef / Maze", 0.0545, 0.0620),
    ("Mitosis / Bacteria", 0.0367, 0.0649),
    ("Solitons / Waves", 0.0140, 0.0540),
    ("Cheetah Spots", 0.0300, 0.0620),
    ("Fingerprint Spirals", 0.0180, 0.0510),
    ("Wandering Worms", 0.0580, 0.0650),
]


class GrayScott(Simulation):
    name = "Gray-Scott Reaction-Diffusion"

    def __init__(self):
        super().__init__()
        self.regime_idx = 0
        self.F = REGIMES[0][1]
        self.k = REGIMES[0][2]
        self.Du = 0.16
        self.Dv = 0.08
        self.U = np.empty((0, 0), dtype=np.float32)
        self.V = np.empty((0, 0), dtype=np.float32)

    def reset(self):
        self.U = np.ones((self.h, self.w), dtype=np.float32)
        self.V = np.zeros((self.h, self.w), dtype=np.float32)

        # Seed with several clusters in center
        for _ in range(6):
            rx = self.rng.integers(self.w // 4, 3 * self.w // 4)
            ry = self.rng.integers(self.h // 4, 3 * self.h // 4)
            self.seed(rx, ry, radius=self.rng.integers(4, 9))

    def seed(self, x: int, y: int, radius: int = 6):
        if self.w == 0 or self.h == 0:
            return
        x0, x1 = max(0, x - radius), min(self.w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.h, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x)**2 + (yy - y)**2 <= radius**2
        self.U[y0:y1, x0:x1][mask] = 0.50 + self.rng.uniform(-0.05, 0.05, np.sum(mask))
        self.V[y0:y1, x0:x1][mask] = 0.25 + self.rng.uniform(-0.05, 0.05, np.sum(mask))

    def next_regime(self):
        self.regime_idx = (self.regime_idx + 1) % len(REGIMES)
        self.F = REGIMES[self.regime_idx][1]
        self.k = REGIMES[self.regime_idx][2]

    def mutate(self):
        # Subtle drift in (F, k) space for living lava lamp effect
        self.F = np.clip(self.F + self.rng.uniform(-0.002, 0.002), 0.010, 0.065)
        self.k = np.clip(self.k + self.rng.uniform(-0.002, 0.002), 0.045, 0.070)

    def step(self, dt: float) -> np.ndarray:
        if self.w == 0 or self.h == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        # Multiple sub-steps for PDE stability
        U, V = self.U, self.V
        F, k = self.F, self.k
        Du, Dv = self.Du, self.Dv

        for _ in range(4):
            # 5-point discrete Laplacian with periodic boundaries
            lapU = (np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) +
                    np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1) - 4.0 * U)
            lapV = (np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
                    np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4.0 * V)

            UV2 = U * V * V
            U += (Du * lapU - UV2 + F * (1.0 - U))
            V += (Dv * lapV + UV2 - (F + k) * V)
            np.clip(U, 0.0, 1.0, out=U)
            np.clip(V, 0.0, 1.0, out=V)

        # V morphogen concentration mapped to 0..255
        intensity = np.clip(V * 3.2 * 255.0, 0, 255).astype(np.uint8)
        return intensity

    def get_info(self) -> str:
        regime_name = REGIMES[self.regime_idx][0]
        return f"Gray-Scott [{regime_name}] F={self.F:.4f} k={self.k:.4f}"
