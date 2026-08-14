# Stochastic Chemical Dynamics Testbed
A controlled numerical framework for testing ODE and SDE solvers on a stochastic chemical kinetics system.

---

## Overview

This repository contains a numerical framework for simulating and analyzing stochastic chemical reaction dynamics, focused on reproducibility, numerical control and method comparison.

The code implements:

- Exact stochastic simulation via Gillespie SSA
- Diffusion approximations via the Chemical Langevin Equation (CLE)
- Multiple deterministic and stochastic numerical integrators
- A fixed, reproducible randomness architecture
- An analysis and visualization suite


---

## Motivation

Stochastic chemical kinetics involves several layers of approximation:

1. Physical modeling
   - Chemical Master Equation (CME)
   - Diffusion approximations (CLE)

2. Numerical discretization
   - Deterministic ODE solvers for drift
   - Stochastic integrators for noise

3. Randomness handling
   - Pseudorandom number generation
   - Monte Carlo sampling

Each layer can influence results in subtle and sometimes severe ways. This framework isolates and controls those influences so that observed differences can be attributed to specific numerical or modeling choices

---

## Model

The current implementation focuses on the Schlögl autocatalytic reaction system, an open, well-mixed, nonequilibrium chemical network known for bistability and noise-induced switching. The framework supports additional reaction systems, but the Schlögl model serves as the primary test case.

---

## Features

### Stochastic Models
- SSA (CME)
  - Gillespie Direct Method
  - Exact, event-driven simulation
- CLE (SDE)
  - Itô interpretation
  - Multiplicative noise
  - Fixed timestep formulation

---

### Deterministic Integrators (Drift)
- Explicit Runge–Kutta (orders 1–7)
- Adams–Bashforth
- Adams–Moulton
- BDF methods
- Singly diagonally implicit Runge–Kutta (SDIRK)
- Fully implicit Runge–Kutta (Gauss, Radau IIA, Lobatto IIIA)


---

### Stochastic Integrators (Diffusion)
- Euler–Maruyama
- Milstein
- Tamed Euler
- Split-step / balanced Euler


---

### Operator Splitting
- Lie splitting
- Strang splitting

Splitting is handled explicitly at the simulation level.

---

### Randomness Control

A single global seed controls the entire experiment. Independent seeds are deterministically derived for SSA and CLE noise, and the two streams are fully decoupled. CLE noise is pre-generated and frozen, so re-running with identical inputs produces identical results. This enables fair solver comparisons and exact reproducibility.

---

## Analysis and Visualization

### CLE-only Analysis (fast, exploratory)
- Representative trajectories
- Ensemble mean and variance vs time
- Stationary distributions (linear and log scale)
- Quantile / box / violin plots
- Autocorrelation functions
- Switching and residence-time statistics
- Solver bias summaries
- Performance vs accuracy comparisons

### CLE vs SSA Analysis (validation)
- Stationary distribution overlays
- Quantile comparisons
- Mean / variance bias
- Probability mass in physical regions
- Switching rate and MFPT comparisons
- Time-resolved moment comparisons

CLE results are shown immediatel while SSA runs separately due to its computational cost.

---

## Design Philosophy

This project prioritizes correctness, reproducibility, transparency, and controlled experiments over raw speed, convenience, abstraction, and black-box simulation. SSA is slow by nature; CLE is fast but approximate. The framework makes these tradeoffs visible.

---

## Limitations

- Single-species reaction system (by design)
- Fixed timestep CLE (no adaptivity)
- SSA performance is limited by the inherent cost of exact simulation
- No enforcement of physical constraints beyond simple non-negativity


---

## Reproducibility

All results are reproducible given identical parameters, identical numerical method choices, and an identical global seed. This applies to SSA results, CLE trajectories, solver comparisons, and visualizations.

---

## Repository Structure

```text
.
├── cle_frontend.py       # Input handling and experiment orchestration
├── cle_backend.py        # Core simulation logic (CLE + SSA)
├── cle_visuals.py        # Analysis and plotting
├── theme.json            # Custom GUI theme
├── solvers/
│   ├── irk.py            # Collocation FIRK
│   ├── linear_multistep.py # Adams and BDF Multistep
│   ├── rk.py             # Explicit RK
│   ├── sdirk.py          # Singly Diagonally IRK
│   └── sde.py            # Stochastic integrators
└── generation/
    ├── bdf.py            # BDF Coefficient Generator
    ├── gauss_legendre.py # Gauss-Legendre Generator (fixed typo)
    ├── lobatto.py        # LobattoIIIA Generator
    ├── radau.py          # RadauIIA Generator
    └── multistep.py      # Adams Bashforth and Moulton Generator
```
