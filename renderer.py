"""terminal-soup: Optimized ANSI half-block terminal presenter & mouse/keyboard handler."""
from __future__ import annotations

import os
import shutil
import sys
import numpy as np

HALF = "\u2580"


class Presenter:
    """Renders 2D indexed uint8 buffer (H x W) into a single ANSI true-color string."""
    def __init__(self, shift: int = 1):
        self.shift = shift

    def compose(self, idx: np.ndarray, lut: np.ndarray, row0: int = 1) -> str:
        if self.shift:
            idx = (idx >> self.shift) << self.shift
        h = idx.shape[0] // 2 * 2
        top, bot = idx[0:h:2], idx[1:h:2]
        cols = lut.tolist()
        fg = np.array([f"\x1b[38;2;{r};{g};{b}m" for r, g, b in cols], dtype=object)
        bg = np.array([f"\x1b[48;2;{r};{g};{b}m" for r, g, b in cols], dtype=object)

        fch = np.ones(top.shape, bool)
        bch = np.ones(top.shape, bool)
        fch[:, 1:] = top[:, 1:] != top[:, :-1]
        bch[:, 1:] = bot[:, 1:] != bot[:, :-1]

        cells = np.where(fch, fg[top], "") + np.where(bch, bg[bot], "") + HALF
        out = []
        for r in range(cells.shape[0]):
            out.append(f"\x1b[{r + row0};1H")
            out.append("".join(cells[r].tolist()))
        return "".join(out)


class Terminal:
    """Manages raw terminal mode, mouse tracking, alt-screen, and non-blocking input."""
    def __enter__(self):
        self.win = os.name == "nt"
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if self.win:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        else:
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

        # Alt-screen, hide cursor, clear, enable mouse tracking (SGR mode 1006 + 1000)
        self.write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[?1000h\x1b[?1006h")
        return self

    def __exit__(self, *exc):
        # Disable mouse tracking, show cursor, restore main screen
        self.write("\x1b[?1006l\x1b[?1000l\x1b[0m\x1b[2J\x1b[?25h\x1b[?1049l")
        if not self.win:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    @staticmethod
    def write(s: str):
        sys.stdout.write(s)
        sys.stdout.flush()

    @staticmethod
    def size():
        s = shutil.get_terminal_size((100, 30))
        return max(20, s.columns), max(8, s.lines)

    def parse_events(self):
        """Returns (keys, mouse_events) where mouse_event is (event_type, x, y_pixel)."""
        keys = []
        mouse = []
        raw = ""
        if self.win:
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    code = msvcrt.getwch()
                    keys.append({"K": "LEFT", "M": "RIGHT", "H": "UP", "P": "DOWN"}.get(code, ""))
                else:
                    raw += ch
        else:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                raw += os.read(self.fd, 256).decode(errors="ignore")

        # Parse SGR mouse sequences: \x1b[<btn;x;yM or \x1b[<btn;x;ym
        i = 0
        while i < len(raw):
            if raw.startswith("\x1b[<", i):
                end_m = raw.find("M", i)
                end_rel = raw.find("m", i)
                end = min(end_m if end_m != -1 else 99999, end_rel if end_rel != -1 else 99999)
                if end != 99999:
                    payload = raw[i + 3:end]
                    release = raw[end] == "m"
                    parts = payload.split(";")
                    if len(parts) == 3:
                        try:
                            b, cx, cy = int(parts[0]), int(parts[1]) - 1, int(parts[2]) - 1
                            # cy is terminal row; in half-block pixels, py = cy * 2
                            m_type = "up" if release else ("drag" if (b & 32) else "down")
                            mouse.append((m_type, cx, cy * 2))
                        except ValueError:
                            pass
                    i = end + 1
                    continue
            if raw.startswith("\x1b[", i) and i + 2 < len(raw):
                keys.append({"D": "LEFT", "C": "RIGHT", "A": "UP", "B": "DOWN"}.get(raw[i + 2], ""))
                i += 3
            else:
                keys.append(raw[i])
                i += 1
        return keys, mouse
