# Modeling Emergent Dynamical Systems and Artificial Life inside a Terminal

**termarium** is an interactive terminal terrarium that simulates **emergent self-organization**, **biological morphogenesis**, and **spatial evolutionary game theory** in real time, purely with Unicode half-blocks (`▀`) and 24-bit ANSI true-color escape sequences.

Created with Google Antigravity

---

## Overview: Local Rules, Global Emergence

Every simulation in **termarium** operates on the same foundational principle of [complex dynamical systems](https://en.wikipedia.org/wiki/Complex_system): **no cell has global knowledge of the screen.** Each point on the 2D grid only interacts with its immediate neighbors. Through nonlinear feedback—autocatalysis, cyclic dominance, or cooperative growth, millions of local interactions spontaneously organize into macroscopic coral reefs, dividing bacteria, rotating Archimedean spirals, and swimming lifeforms.

Together, the three engines span the mathematical spectrum from **discrete cellular automata** to **continuous partial differential equations (PDEs)**:

| Engine | Space & Time | Cell State | Mathematical Class | Game-Theoretic Analog |
| :--- | :--- | :--- | :--- | :--- |
| **1. Gray-Scott** | Continuous Limit ($\nabla^2, \Delta t$) | $(U, V) \in [0, 1]^2$ | [Reaction-Diffusion PDE](https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system) | Predator-Prey / Resource-Consumer |
| **2. Cyclic Automata** | Discrete Lattice & Ticks | $s \in \{0, \dots, N-1\}$ | [Cyclic Cellular Automaton](https://en.wikipedia.org/wiki/Cyclic_cellular_automaton) | Spatial Rock-Paper-Scissors |
| **3. Continuous Lenia** | Fine Grid ($\to \mathbb{R}^2, \Delta t$) | $A \in [0, 1]$ | Integro-Differential Convolution | Cooperation vs. Overcrowding |

---

## The Three Simulation Engines

### 1. Gray-Scott Reaction-Diffusion (`--sim 1`)
First proposed by Alan Turing in his landmark 1952 paper on [The Chemical Basis of Morphogenesis](https://en.wikipedia.org/wiki/The_Chemical_Basis_of_Morphogenesis) and popularized by John E. Pearson's 1993 classification ([*Complex Patterns in a Simple System*](https://arxiv.org/abs/patt-sol/9304003)), the Gray-Scott model simulates two virtual chemical morphogens $U$ (nutrient) and $V$ (autocatalyst) reacting via $U + 2V \rightarrow 3V$ and diffusing across a 2D surface:

$$\frac{\partial U}{\partial t} = D_u \nabla^2 U - UV^2 + F(1 - U), \qquad \frac{\partial V}{\partial t} = D_v \nabla^2 V + UV^2 - (F + k)V$$

By varying the feed rate $F$ and removal/kill rate $k$ (press **`R`** to cycle regimes or **`M`** to mutate), the same equation produces **Coral Reefs & Labyrinths**, **Mitosis (Dividing Cells)**, **Pulsing Solitons**, **Cheetah Spots**, and **Fingerprint Spirals**.

### 2. Cyclic Cellular Automata (`--sim 2`)
Introduced by David Griffeath and colleagues ([*Threshold-Range Scaling of Excitable Cellular Automata*](https://doi.org/10.1016/0167-2789(91)90220-4)), a [Cyclic Cellular Automaton](https://en.wikipedia.org/wiki/Cyclic_cellular_automaton) assigns each cell one of $N$ discrete states $\{0, 1, \dots, N-1\}$. At each step, a cell in state $s$ is consumed by state $(s + 1) \bmod N$ if at least $T$ neighbors within radius $R$ are already in state $(s + 1) \bmod N$.

In [evolutionary game theory](https://en.wikipedia.org/wiki/Evolutionary_game_theory), this models **non-transitive cyclic competition** (generalized Rock-Paper-Scissors). Around topologically protected winding-number-1 phase defects ("demons"), the system self-organizes into immortal, counter-rotating **Archimedean spiral waves** and turbulent crystal domains.

### 3. Continuous Lenia / SmoothLife (`--sim 3`)
Developed by Bert Wang-Chak Chan ([*Lenia: Biology of Artificial Life*, Complex Systems 2019](https://arxiv.org/abs/1812.05433); see also [Lenia on Wikipedia](https://en.wikipedia.org/wiki/Lenia) and Stephan Rafler's [*SmoothLife*](https://arxiv.org/abs/1111.1567)), Lenia generalizes [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) by making space, time, and cell states continuous:

$$A(t + \Delta t) = \operatorname{clip}\!\Big(A(t) + \Delta t \, G\big(K * A(t)\big),\; 0,\; 1\Big), \qquad G(u) = 2\exp\!\left(-\frac{(u - \mu)^2}{2\sigma^2}\right) - 1$$

To run in real time in Python, **termarium** evaluates the 2D circular kernel convolution $K * A$ in $O(N \log N)$ via Fast Fourier Transforms (`np.fft.fft2`), producing smooth, self-stabilizing soliton "micro-organisms" that glide, pulsate, and divide.

---

## Under the Hood: 60 FPS Terminal Software Rendering

1. **Square-Pixel Half-Block Canvas (`▀`)**: Terminal character cells have a ~`1:2` aspect ratio. By pairing the upper half-block character `▀` (`U+2580`) with a 24-bit foreground color (`\x1b[38;2;R;G;Bm`) for the top pixel and a 24-bit background color (`\x1b[48;2;R;G;Bm`) for the bottom pixel, a `120 × 40` terminal window becomes a **`120 × 80` square-pixel grid**.
2. **256-Color Cyclic Lookup Tables**: Each simulation outputs a `uint8` intensity field (`0..255`) mapped through a 256-entry RGB lookup table (supporting piecewise cyclic gradients and [Inigo Quilez cosine palettes](https://iquilezles.org/articles/palettes/)).
3. **Run-Length Escape Elision**: Vectorized horizontal diffing emits ANSI color sequences only when a pixel's foreground or background index differs from its left neighbor, flushing each frame in a single tear-free `sys.stdout.write()` call.
4. **SGR 1006 Mouse Tracking**: Enables `\x1b[?1000h\x1b[?1006h` so mouse clicks and drags map directly to sub-pixel coordinates $(x, 2y)$ to inject chemical or biological seeds live.

---

## Quick Start

**termarium** requires only Python 3.10+ and `numpy`:

```powershell
# Install NumPy if not already installed
pip install numpy

# Launch the interactive terrarium (defaults to Gray-Scott Reaction-Diffusion)
python main.py
```

### Launch Examples

```powershell
python main.py --sim 1             # Gray-Scott Reaction-Diffusion
python main.py --sim 2             # Cyclic Cellular Automata (Archimedean spirals)
python main.py --sim 3             # Continuous Lenia (Fluid Artificial Life)
python main.py --auto              # Autonomous "Lava Lamp" mode (slow parameter drift)
python main.py --sim 2 --palette 5 # Cyclic Automata with Thermal Cosine palette
```

> **Tip:** Use a modern true-color terminal (**Windows Terminal**, **iTerm2**, **Alacritty**, **Kitty**, or **WezTerm**). Zooming out your terminal font size (`Ctrl` + `-`) dramatically increases the simulation grid resolution!

---

## Interactive Controls

| Input | Action | Description |
| :--- | :--- | :--- |
| **Mouse Click / Drag** | **Drop Seed** | Inject chemical morphogens, spiral defects, or Lenia organisms at cursor |
| **`1` / `2` / `3`** | **Switch Engine** | `1`: Gray-Scott, `2`: Cyclic Automata, `3`: Continuous Lenia |
| **`←` / `→`** | **Prev / Next Engine** | Cycle backward or forward through the 3 simulation engines |
| **`R`** | **Cycle Regime** | Step through curated parameter regimes (Corals, Mitosis, Spirals, Vortices, etc.) |
| **`M`** | **Mutate Parameters** | Perturb $(F, k)$, neighborhood rules, or growth $(\mu, \sigma)$ on the fly |
| **`Space`** | **Re-seed Terrarium** | Clear and re-nucleate the grid with fresh random colonies |
| **`P`** | **Next Color Palette** | Crossfade between 6 palettes (*Psychedelic Bio*, *Deep Coral*, *Lava Lamp*, *Acid Matrix*, *Thermal Cosine*, *Abyssal Chem*) |
| **`C`** | **Toggle Palette Flow** | Continuously rotate palette hues across existing structures |
| **`D`** | **Color Detail Level** | Cycle `256` $\rightarrow$ `128` $\rightarrow$ `64` color levels (lower = faster on slow terminals) |
| **`H`** | **Toggle HUD** | Show or hide the bottom telemetry bar |
| **`?`** | **Toggle Help** | Show or hide the in-terminal hotkey overlay |
| **`Q` / `Esc`** | **Quit** | Restore terminal state and exit |

---

## Project Structure

```text
termarium/
├── main.py                      # Main 60 FPS loop, CLI parser, HUD & hotkey controller
├── renderer.py                  # Half-block (▀) ANSI compiler, RLE deduplicator, SGR 1006 mouse parser
├── palettes.py                  # 256-color cyclic & cosine palette LUT generator
└── simulations/
    ├── __init__.py              # Simulation registry
    ├── base.py                  # Abstract Simulation interface
    ├── gray_scott.py            # Gray-Scott 2D reaction-diffusion PDE solver & 6 Turing regimes
    ├── cyclic_automata.py       # Multi-state Cyclic CA with topological spiral defect nucleation
    └── lenia.py                 # FFT-accelerated Continuous Lenia (concentric ring convolution)
```
