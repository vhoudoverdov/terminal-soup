"""terminal-soup: Continuous Lenia Simulation.

Fluid, continuous generalization of Conway's Game of Life (Bert Wang-Chak Chan).
Features pre-cached rFFT multi-ring kernels, asymmetric glider seeding, and
population homeostasis.
"""
from __future__ import annotations

import numpy as np
from .base import Simulation

# Curated multi-ring Lenia species: (name, radius_R, ring_weights_beta, mu, sigma, dt)
LENIA_SPECIES = [
    ("Orbium Gliders",         12, (1.0,),             0.150, 0.017, 0.18),
    ("Hydrogeminium Organelle", 15, (1.0, 0.65, 0.35), 0.135, 0.016, 0.16),
    ("Gemium Dividing Cells",  13, (1.0, 0.50),        0.142, 0.018, 0.18),
    ("Scutium Pulsers",        13, (0.55, 1.0),        0.158, 0.020, 0.16),
]


class Lenia(Simulation):
    name = "Continuous Lenia (Fluid Life)"

    def __init__(self):
        super().__init__()
        self.A = np.empty((0, 0), dtype=np.float32)
        self.regime_idx = 0
        self._apply_species(0)
        self.K = None
        self.K_rfft = None

    def _apply_species(self, idx: int):
        self.regime_idx = idx % len(LENIA_SPECIES)
        _, self.R, self.betas, self.mu, self.sigma, self.dt = LENIA_SPECIES[self.regime_idx]

    def _build_kernel(self):
        if self.w == 0 or self.h == 0:
            return
        r = min(self.R, min(self.w, self.h) // 3)
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        dist = np.hypot(x, y) / max(1, r)

        # Multi-ring Gaussian shell kernel K(r) with concentric rings weighted by beta_i
        n_rings = len(self.betas)
        kernel = np.zeros_like(dist, dtype=np.float32)
        for i, beta in enumerate(self.betas):
            r0 = i / n_rings
            r1 = (i + 1) / n_rings
            mid = 0.5 * (r0 + r1)
            width = 0.22 / n_rings
            ring = beta * np.exp(-((dist - mid) ** 2) / (2.0 * width ** 2)) * ((dist >= r0) & (dist <= r1))
            kernel += ring.astype(np.float32)

        s = float(kernel.sum())
        if s > 0:
            kernel /= s
        self.K = kernel

        # Precompute and cache the periodic padded rFFT of the kernel once per resize/species
        Kh, Kw = kernel.shape
        padded_K = np.zeros((self.h, self.w), dtype=np.float32)
        cy, cx = Kh // 2, Kw // 2
        padded_K[:Kh - cy, :Kw - cx] = kernel[cy:, cx:]
        padded_K[:Kh - cy, -cx:] = kernel[cy:, :cx]
        padded_K[-cy:, :Kw - cx] = kernel[:cy, cx:]
        padded_K[-cy:, -cx:] = kernel[:cy, :cx]
        self.K_rfft = np.fft.rfft2(padded_K)

    def reset(self):
        self._build_kernel()
        self.A = np.zeros((self.h, self.w), dtype=np.float32)
        count = max(5, (self.w * self.h) // 2200)
        for _ in range(count):
            cx = int(self.rng.integers(self.w // 6, 5 * self.w // 6))
            cy = int(self.rng.integers(self.h // 6, 5 * self.h // 6))
            self.seed(cx, cy, radius=self.R + 1)

    def seed(self, x: int, y: int, radius: int = 12):
        """Spawn an asymmetric crescent soliton organism with a random heading angle."""
        if self.w == 0 or self.h == 0:
            return
        x0, x1 = max(0, x - radius), min(self.w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.h, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dy, dx = yy - y, xx - x
        dist = np.hypot(dx, dy) / max(1.0, float(radius))
        angle = np.arctan2(dy, dx)
        heading = float(self.rng.uniform(0, 2.0 * np.pi))

        # Front-to-back dipole asymmetry + hollow core so the soliton self-propels
        ring_shell = np.exp(-((dist - 0.45) ** 2) / 0.12) + 0.6 * np.exp(-(dist ** 2) / 0.08)
        asymmetry = 0.72 + 0.35 * np.cos(angle - heading)
        blob = np.clip(ring_shell * asymmetry * (dist <= 1.0), 0.0, 1.0).astype(np.float32)
        self.A[y0:y1, x0:x1] = np.clip(self.A[y0:y1, x0:x1] + blob, 0.0, 1.0)

    def next_regime(self):
        self._apply_species(self.regime_idx + 1)
        self._build_kernel()

    def mutate(self):
        self.mu = float(np.clip(self.mu + self.rng.uniform(-0.006, 0.006), 0.125, 0.168))
        self.sigma = float(np.clip(self.sigma + self.rng.uniform(-0.002, 0.002), 0.013, 0.023))

    def step(self, dt: float) -> np.ndarray:
        if self.w == 0 or self.h == 0 or self.K_rfft is None:
            return np.zeros((1, 1), dtype=np.uint8)

        A = self.A
        # Fast O(N log N) circular convolution via cached real FFT kernel
        U = np.fft.irfft2(np.fft.rfft2(A) * self.K_rfft, s=A.shape)

        # Bell-curve growth function G(U) in [-1, 1]
        G = 2.0 * np.exp(-((U - self.mu) ** 2) / (2.0 * (self.sigma ** 2))) - 1.0
        A = np.clip(A + self.dt * G, 0.0, 1.0).astype(np.float32)

        # Population homeostasis: prevent extinction or screen saturation
        mean_mass = float(A.mean())
        if mean_mass < 0.015:
            rx = int(self.rng.integers(self.w // 5, 4 * self.w // 5))
            ry = int(self.rng.integers(self.h // 5, 4 * self.h // 5))
            self.A = A
            self.seed(rx, ry, radius=self.R + 1)
            A = self.A
        elif mean_mass > 0.55:
            A *= 0.94

        self.A = A
        # Slight gamma lift so delicate outer membranes and nuclei are richly visible
        shaded = np.clip(A ** 0.85, 0.0, 1.0)
        return (shaded * 255.0).astype(np.uint8)

    def get_info(self) -> str:
        s_name = LENIA_SPECIES[self.regime_idx][0]
        return f"Lenia [{s_name}] mu={self.mu:.3f} sigma={self.sigma:.3f}"
