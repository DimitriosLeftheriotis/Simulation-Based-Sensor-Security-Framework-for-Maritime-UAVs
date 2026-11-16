# Simulation testbed

Self-contained Python Monte Carlo testbed for the maritime-UAV sensor-security
framework. It reproduces every results table and results figures in the
manuscript. It is a **horizontal kinematic simulation**, not ArduPilot SITL:
the vehicle, sensors, attacks (A1–A7) and the four framework layers are modelled
in pure Python so the whole study runs from one dependency set.

## What it models

- Closed-loop guidance that drives the *estimated* position along a 500 m × 500 m
  patrol route, so an undetected spoof walks the *true* vehicle off route, while
  isolating a sensor makes the filter coast on inertial dead reckoning.
- GNSS (10 Hz) + IMU; six synthetic AIS vessels with radar/EO-IR observations.
- Framework: 4-state KF with NIS gate + CUSUM (calibrated empirically on
  attack-free runs), maritime common-mode vs. idiosyncratic cross-check with
  Hungarian association, trust layer + response state machine, MAVLink-2 signing
  and a Gilbert–Elliott link.
- Baselines B0 (no defence), B1 (innovation gating), B2 (implied-speed
  threshold), and the full framework F.

## Setup & Reproduce with `uv`

Using [`uv`](https://github.com/astral-sh/uv):

```sh
# Create virtual environment and install dependencies
uv venv
uv sync

# Run the simulation campaign (~11 min for N=100)
uv run python run_sim.py --N 100 --out results.json
# or directly via the installed CLI:
# uv run run-sim --N 100 --out results.json

# Generate all LaTeX tables (.txt) and figures (.pdf) in artifacts/ (<0.2s)
uv run python scripts/generate_all.py
# or individually:
# uv run python scripts/make_tables.py
# uv run python scripts/make_figs.py
```

Or using standard `pip`:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_sim.py --N 100 --out results.json
python scripts/generate_all.py
```

All reported numbers are produced by running this code; none are hand-set. Both
tables and figures consume `results.json` as the single source of truth,
guaranteeing complete mathematical consistency across paper artifacts. The
thresholds are calibrated on attack-free runs, so re-running reproduces the
paper's values up to Monte Carlo noise.

## Artifacts & Outputs

All generated paper artifacts are saved under `artifacts/`:
* `artifacts/tables/`: Individual LaTeX bodies (`tab9_detection.txt`, `tab10_mission.txt`, `tab11_ablation.txt`, `tab12_overhead.txt`, `tab13_robustness.txt`, `stats_significance.txt`) and combined `all_tables.txt`.
* `artifacts/figures/`: Publication-ready vector PDFs (`fig_roc.pdf`, `fig_traj.pdf`, `fig_robust.pdf`).

## Codebase Structure

The simulation is organised as a modular Python package `maritime_uav`:

| Path | Purpose |
|------|---------|
| `maritime_uav/config.py` | Physical constants, sensor noise, matrices, runtime calibration state |
| `maritime_uav/dynamics.py` | Reference trajectory generation, vessel tracks, Kalman filter (`KF`) |
| `maritime_uav/attacks.py` | Attack injection logic (A1–A7: jump, naive/consistent drift, jamming, IMU bias, AIS spoofing/suppression, link loss); A5/A6 hooks are called from the L3 scan loop |
| `maritime_uav/layers.py` | L2 (NIS/CUSUM), B2 detector, L3 maritime cross-check, L4 trust & isolation, fusion |
| `maritime_uav/simulation.py` | Monte Carlo trial core loop (`run_once`) |
| `maritime_uav/calibration.py` | Detection threshold calibration (`calibrate`) and Isolation Forest training |
| `maritime_uav/evaluation.py` | Batch Monte Carlo driver (`mc`), confidence intervals, hypothesis tests, benchmark |
| `maritime_uav/cli.py` | CLI argument parsing and experimental sequence orchestration |
| `run_sim.py` | Top-level runner script (`python run_sim.py ...`) |
| `scripts/generate_all.py` | Orchestrates generating all tables and figures in one step |
| `scripts/make_tables.py` | Emits LaTeX table `.txt` files directly from `results.json` |
| `scripts/make_figs.py` | Pure vector plotting script rendering figures from `results.json` |
| `artifacts/` | Directory storing generated `.txt` tables and `.pdf` figures |
