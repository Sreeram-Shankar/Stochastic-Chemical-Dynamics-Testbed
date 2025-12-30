# Stochastic Chemical Dynamics Testbed  
*A controlled numerical framework for SSA, CLE, and solver comparison*

---

## Overview

This repository contains a small but carefully designed numerical framework for simulating and analyzing **stochastic chemical reaction dynamics**, with a focus on **reproducibility, numerical control, and method comparison** rather than raw performance.

The code implements:

- Exact stochastic simulation via **Gillespie SSA**
- Diffusion approximations via the **Chemical Langevin Equation (CLE)**
- Multiple deterministic and stochastic numerical integrators
- A fixed, reproducible randomness architecture
- A comprehensive analysis and visualization suite

The primary goal of this project is **not** to provide the fastest simulator, but to provide a **transparent and trustworthy numerical laboratory** for studying how modeling and numerical choices influence observed stochastic behavior.

---

## Motivation

In stochastic chemical kinetics, multiple layers of approximation are involved:

1. **Physical modeling**
   - Chemical Master Equation (CME)
   - Diffusion approximations (CLE)

2. **Numerical discretization**
   - Deterministic ODE solvers for drift
   - Stochastic integrators for noise

3. **Randomness handling**
   - Pseudorandom number generation
   - Monte Carlo sampling

Each of these layers can influence results in subtle (and sometimes severe) ways.

This framework is designed to make those influences **explicit, isolated, and reproducible**, so that observed differences can be attributed to numerical or modeling choices rather than uncontrolled randomness.

---

## Model

The current implementation focuses on the **Schlögl autocatalytic reaction system**, an open, well-mixed, nonequilibrium chemical network known to exhibit bistability and noise-induced switching.

The framework is general enough to support additional reaction systems, but the Schlögl model serves as a compact and well-studied test case.

---

## Features

### Stochastic Models
- **SSA (CME)**
  - Gillespie Direct Method
  - Exact, event-driven simulation
- **CLE (SDE)**
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
- Fully implicit Runge–Kutta (Gauss, Radau IIA, Lobatto IIIC)

All deterministic integrators are implemented as **true step methods**, not restarted solvers.

---

### Stochastic Integrators (Diffusion)
- Euler–Maruyama
- Milstein
- Tamed Euler
- Split-step / balanced Euler

Stochastic integrators **do not generate randomness internally**; they consume externally provided noise.

---

### Operator Splitting
- Lie splitting
- Strang splitting

Splitting is handled explicitly at the simulation level, not hidden inside solvers.

---

### Randomness Control

A central design principle of this project is:

> **Randomness is an input, not a side effect.**

- A single **global seed** controls the entire experiment
- Independent seeds are deterministically derived for:
  - SSA
  - CLE noise
- SSA and CLE randomness are fully decoupled
- CLE noise is **pre-generated and frozen**
- Re-running with identical inputs produces **identical results**

This enables fair solver comparisons and exact reproducibility.

---

## Analysis and Visualization

The framework includes a comprehensive set of diagnostics.

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

CLE results are shown immediately; SSA runs separately due to its computational cost.

---

## Design Philosophy

This project intentionally prioritizes:

- **Correctness over speed**
- **Reproducibility over convenience**
- **Transparency over abstraction**
- **Controlled experiments over black-box simulation**

SSA is expected to be slow.  
CLE is expected to be fast but imperfect.  

The framework is built to make those tradeoffs visible rather than hidden.

---

## Limitations

- Single-species reaction system (by design)
- Fixed timestep CLE (no adaptivity)
- SSA performance is limited by the inherent cost of exact simulation
- No attempt is made to enforce physical constraints beyond simple non-negativity

These limitations are acknowledged explicitly and are considered acceptable for the scope of this project.

---

## Intended Use

This code is suitable for:

- Numerical analysis of stochastic chemical systems
- Studying solver-induced bias and approximation error
- Reproducible Monte Carlo experiments
- Educational or research-focused exploration

It is **not** intended as a production-grade chemical simulator.

---

## Reproducibility

All results are reproducible given:
- Identical parameters
- Identical numerical method choices
- Identical global seed

This applies to:
- SSA results
- CLE trajectories
- Solver comparisons
- Visualizations

---

## Repository Structure

.
├── cle_frontend.py     # Input handling and experiment orchestration
├── cle_backend.py      # Core simulation logic (CLE + SSA)
├── cle_visuals.py      # Analysis and plotting
├── solvers/
│   ├── irk.py               # Collocation FIRK
│   ├── linear_multistep.py  # Adams and BDF Multistep
│   ├── rk.py                # Explicit RK
│   ├── sdirk.py             # Singly Diagonally IRK
│   └── sde.py               # Stochastic integrators
├── generation/
│   ├── bdf.py             # BDF Coefficient Generator
│   ├── gauss_legednre.py  # Gauss-Legendre Generator
│   ├── lobatto.py         # LobattoIIIC Generator
│   ├── radau.py           # radauIIA Geneator
│   └── multistep.py        # Adams Bashforth and Moulton Generator

*Questions, comments, and constructive criticism are welcome.*
```
