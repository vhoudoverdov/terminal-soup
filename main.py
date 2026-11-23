"""termarium: Cellular Automata & Reaction-Diffusion Terrarium.

Interactive psychedelic pattern simulator in your terminal.
"""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np

from palettes import PaletteManager
from renderer import Presenter, Terminal
from simulations import SIMULATIONS

HELP = [
    "termarium  Terrarium Controls",
    "",
    "1-3 / \u2190 \u2192   switch simulation mode",
    "Mouse       click / drag to seed chemical / life",
    "Space       randomize / re-seed lattice",
    "M           mutate parameters (lava lamp shift)",
    "P           next color palette",
    "C           toggle continuous palette flow",
    "R           next parameter regime (Gray-Scott)",
    "D           color detail (256/128/64 levels)",
    "H           toggle bottom HUD",
    "?           toggle this help window",
    "Q / Esc     quit",
]


class TerrariumApp:
    def __init__(self, args):
        self.args = args
        self.sims = [cls() for cls in SIMULATIONS]
        self.cur_sim_idx = (args.sim - 1) % len(self.sims)
        self.pal = PaletteManager(args.palette - 1)
        self.presenter = Presenter(shift=args.detail)
        self.hud = not args.no_hud
        self.help = False
        self.fps = 0.0
        self.auto_mutate = args.auto
        self.last_mutate = time.perf_counter()
        self.toast = ""
        self.toast_until = 0.0

    @property
    def sim(self):
        return self.sims[self.cur_sim_idx]

    def say(self, msg: str, secs: float = 3.0):
        self.toast = msg
        self.toast_until = time.perf_counter() + secs

    def select_sim(self, idx: int):
        self.cur_sim_idx = idx % len(self.sims)
        self.say(f"Active: {self.sim.name}")

    def handle_keys(self, keys):
        for k in keys:
            if k in ("q", "Q", "\x1b", "\x03"):
                return False
            if k in "123" and len(k) == 1:
                self.select_sim(int(k) - 1)
            elif k == "RIGHT":
                self.select_sim(self.cur_sim_idx + 1)
            elif k == "LEFT":
                self.select_sim(self.cur_sim_idx - 1)
            elif k == " ":
                self.sim.reset()
                self.say("Terrarium re-seeded")
            elif k in ("m", "M"):
                self.sim.mutate()
                self.say(f"Mutated: {self.sim.get_info()}")
            elif k in ("r", "R"):
                if hasattr(self.sim, "next_regime"):
                    self.sim.next_regime()
                    self.say(f"Regime: {self.sim.get_info()}")
            elif k in ("p", "P"):
                self.pal.next()
                self.say(f"Palette: {self.pal.name}")
            elif k in ("c", "C"):
                self.pal.cycle_enabled = not self.pal.cycle_enabled
                self.say(f"Palette flow: {'ON' if self.pal.cycle_enabled else 'OFF'}")
            elif k in ("d", "D"):
                self.presenter.shift = (self.presenter.shift + 1) % 3
                self.say(f"Color detail: {256 >> self.presenter.shift} levels")
            elif k in ("h", "H"):
                self.hud = not self.hud
            elif k == "?":
                self.help = not self.help
        return True

    def handle_mouse(self, mouse_events):
        for m_type, px, py in mouse_events:
            if m_type in ("down", "drag"):
                self.sim.seed(px, py, radius=6)

    def overlay(self, cols, rows):
        s = []
        if self.hud:
            bar = " \u2502 "
            flow = " [FLOW]" if self.pal.cycle_enabled else ""
            auto = " [AUTO]" if self.auto_mutate else ""
            if time.perf_counter() < self.toast_until:
                info = self.toast
            else:
                info = self.sim.get_info()
            text = (f" termarium{bar}[{self.cur_sim_idx + 1}] {info}{bar}"
                    f"{self.pal.name}{flow}{bar}{self.fps:4.1f} fps{auto}{bar}? help")
            s.append(f"\x1b[{rows};1H\x1b[0m\x1b[38;2;230;230;240;48;2;12;12;28m"
                     + text[:cols].ljust(cols))

        if self.help:
            bw = max(len(x) for x in HELP) + 4
            x0 = max(1, (cols - bw) // 2 + 1)
            y0 = max(1, (rows - len(HELP) - 2) // 2)
            st = "\x1b[0m\x1b[38;2;140;255;200;48;2;10;10;32m"
            s.append(f"\x1b[{y0};{x0}H{st}\u250c" + "\u2500" * (bw - 2) + "\u2510")
            for i, line in enumerate(HELP):
                s.append(f"\x1b[{y0 + 1 + i};{x0}H{st}\u2502 " + line.ljust(bw - 4) + " \u2502")
            s.append(f"\x1b[{y0 + len(HELP) + 1};{x0}H{st}\u2514" + "\u2500" * (bw - 2) + "\u2518")
        return "".join(s)

    def run(self, term):
        target = 1.0 / self.args.fps
        start = last = time.perf_counter()
        cols, rows = term.size()
        frame_no = 0

        while True:
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now

            keys, mouse = term.parse_events()
            if not self.handle_keys(keys):
                break
            self.handle_mouse(mouse)

            if frame_no % 10 == 0:
                size = term.size()
                if size != (cols, rows):
                    cols, rows = size
                    term.write("\x1b[0m\x1b[2J")

            # Check lava lamp auto-mutation
            if self.auto_mutate and (now - self.last_mutate > 8.0):
                self.sim.mutate()
                self.last_mutate = now

            # Step simulation on sub-pixel grid: (cols, rows * 2)
            w, h = cols, rows * 2
            self.sim.resize(w, h)
            idx = self.sim.step(dt)

            self.pal.update(dt)
            lut = self.pal.lut()

            term.write(self.presenter.compose(idx, lut) + self.overlay(cols, rows))
            frame_no += 1

            if dt > 0:
                self.fps += (1.0 / dt - self.fps) * 0.05
            spare = target - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)

            if self.args.duration and (now - start > self.args.duration):
                break


def parse_args():
    p = argparse.ArgumentParser(prog="termarium", description="Cellular Automata & Reaction-Diffusion Terrarium")
    p.add_argument("--sim", type=int, default=1, choices=(1, 2, 3), help="Starting simulation: 1=Gray-Scott, 2=Cyclic, 3=Lenia")
    p.add_argument("--palette", type=int, default=1, help="Starting palette 1-6")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--detail", type=int, default=1, choices=(0, 1, 2), help="Color detail (0=256, 1=128, 2=64)")
    p.add_argument("--auto", action="store_true", help="Enable continuous lava lamp parameter drift")
    p.add_argument("--no-hud", action="store_true", help="Hide HUD bar")
    p.add_argument("--duration", type=float, default=0, help="Exit after N seconds")
    return p.parse_args()


def main():
    args = parse_args()
    app = TerrariumApp(args)
    try:
        with Terminal() as term:
            app.run(term)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
