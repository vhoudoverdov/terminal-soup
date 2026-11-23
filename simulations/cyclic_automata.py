"""terminal-soup: Cyclic Cellular Automata (CCA).

Generates self-organizing Archimedean spirals, crystal defects,
and rock-paper-scissors demoscene waves (Fisch, Gravner & Griffeath).
"""
from __future__ import annotations

import numpy as np
from .base import Simulation

# Curated (name, range_r, num_states, threshold, moore_neighborhood, step_rate_hz)
# Chosen specifically inside the "turbulent / spiral" phase-transition window so
# the automaton never suffers early fixation (freezing).
CCA_REGIMES = [
    ("Archimedean Spirals", 2, 14, 3, True, 20.0),
    ("Turbulent Vortex",    1, 8,  2, True, 22.0),
    ("Diamond Target Waves", 1, 14, 1, False, 20.0),
    ("Wide Ribbon Spirals", 3, 16, 5, True, 18.0),
    ("Boiling Membranes",   2, 10, 4, True, 20.0),
    ("Hyper-Crystal",       1, 5,  3, True, 24.0),
]


class CyclicAutomata(Simulation):
    name = "Cyclic Cellular Automata"

    def __init__(self):
        super().__init__()
        self.regime_idx = 0
        self._apply_regime(0)
        self.grid = np.empty((0, 0), dtype=np.int16)
        self.prev_grid = np.empty((0, 0), dtype=np.int16)
        self.accum = 0.0
        self.smooth = np.empty((0, 0), dtype=np.float32)

    def _apply_regime(self, idx: int):
        self.regime_idx = idx % len(CCA_REGIMES)
        _, self.radius, self.num_states, self.threshold, self.moore, self.rate = CCA_REGIMES[self.regime_idx]
        self._offsets = [
            (dy, dx)
            for dy in range(-self.radius, self.radius + 1)
            for dx in range(-self.radius, self.radius + 1)
            if not (dy == 0 and dx == 0)
            and (self.moore or (abs(dy) + abs(dx) <= self.radius))
        ]

    def reset(self):
        self.grid = self.rng.integers(0, self.num_states, size=(self.h, self.w), dtype=np.int16)
        # Nucleate several topological spiral defects ("demons") so waves self-sustain forever
        for _ in range(5):
            cx = int(self.rng.integers(self.w // 5, 4 * self.w // 5))
            cy = int(self.rng.integers(self.h // 5, 4 * self.h // 5))
            self._stamp_spiral_defect(cx, cy, radius=max(5, self.radius * 3))
        self.prev_grid = self.grid.copy()
        self.smooth = self.grid.astype(np.float32) / max(1, self.num_states)
        self.accum = 0.0

    def _stamp_spiral_defect(self, cx: int, cy: int, radius: int = 7):
        """Stamp a winding-number-1 pinwheel of states 0..N-1 around (cx, cy).

        In CCA dynamics, a closed loop of states 0 -> 1 -> ... -> N-1 forms a
        topologically protected rotor ('demon') that continuously emits spiral waves
        and prevents the lattice from ever freezing.
        """
        if self.w == 0 or self.h == 0:
            return
        x0, x1 = max(0, cx - radius), min(self.w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(self.h, cy + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dy, dx = yy - cy, xx - cx
        dist = np.hypot(dx, dy)
        mask = dist <= radius
        chirality = self.rng.choice([-1.0, 1.0])
        angle = (np.arctan2(dy, dx) * chirality) % (2.0 * np.pi)
        # Add radial twist so the seed already has a spiral arm
        phase = (angle / (2.0 * np.pi) + dist / max(2.0, radius * 1.5)) % 1.0
        states = (phase * self.num_states).astype(np.int16) % self.num_states
        self.grid[y0:y1, x0:x1][mask] = states[mask]

    def seed(self, x: int, y: int, radius: int = 7):
        self._stamp_spiral_defect(x, y, radius=max(radius, self.radius * 3))

    def next_regime(self):
        self._apply_regime(self.regime_idx + 1)
        self.reset()

    def mutate(self):
        # Cycle to a new non-freezing regime or perturb state count within viable bounds
        self.next_regime()

    def _ca_tick(self):
        G = self.grid
        self.prev_grid = G
        target = (G + 1) % self.num_states

        neighbors = np.zeros_like(G, dtype=np.int16)
        for dy, dx in self._offsets:
            shifted = np.roll(np.roll(G, dy, axis=0), dx, axis=1)
            neighbors += (shifted == target)

        advance = neighbors >= self.threshold
        self.grid = np.where(advance, target, G)

        # Anti-fixation guard: if fewer than 0.5% of cells advanced, inject spiral rotors
        if np.count_nonzero(advance) < (self.w * self.h * 0.005):
            for _ in range(3):
                rx = int(self.rng.integers(4, max(5, self.w - 4)))
                ry = int(self.rng.integers(4, max(5, self.h - 4)))
                self._stamp_spiral_defect(rx, ry, radius=max(5, self.radius * 3))

    def step(self, dt: float) -> np.ndarray:
        if self.w == 0 or self.h == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        self.accum += dt * self.rate
        ticks = min(int(self.accum), 3)
        if ticks > 0:
            self.accum -= int(self.accum)
            for _ in range(ticks):
                self._ca_tick()

        # Sub-frame cyclic interpolation on the unit circle so waves glide smoothly at 60 FPS
        frac = float(np.clip(self.accum, 0.0, 1.0))
        n = float(self.num_states)
        delta = ((self.grid - self.prev_grid) % self.num_states).astype(np.float32)
        interp_state = (self.prev_grid.astype(np.float32) + frac * delta) % n
        intensity = np.clip((interp_state / max(1.0, n - 1.0)) * 254.0 + 1.0, 1, 255).astype(np.uint8)
        return intensity

    def get_info(self) -> str:
        r_name = CCA_REGIMES[self.regime_idx][0]
        return f"Cyclic CA [{r_name}] R={self.radius} N={self.num_states} T={self.threshold}"
