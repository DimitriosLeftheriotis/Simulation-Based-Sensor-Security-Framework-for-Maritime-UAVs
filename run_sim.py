#!/usr/bin/env python3
"""
Entry point for the maritime UAV sensor-security simulation framework.

Modular components in the `maritime_uav` package:
  - `maritime_uav.config`: parameters, noise levels, filter matrices, runtime config
  - `maritime_uav.dynamics`: reference trajectories, vessel motion, Kalman filter
  - `maritime_uav.attacks`: attack injection models (A1-A7)
  - `maritime_uav.layers`: L2 (NIS/CUSUM), L2 IMU monitor, baseline B2, L3 cross-check, L4 trust & isolation, fusion
  - `maritime_uav.link`: C2 uplink model, Layer 1 MAVLink 2 signing, A7 link attacks
  - `maritime_uav.simulation`: core single-run Monte Carlo trial (`run_once`)
  - `maritime_uav.calibration`: detector calibration and Isolation Forest training/eval
  - `maritime_uav.evaluation`: Monte Carlo batch runner, confidence intervals, hypothesis tests, benchmarks
  - `maritime_uav.cli`: main command-line interface orchestrator

Usage:
    python run_sim.py --N 100 --out results.json
"""
import sys
from maritime_uav import *
from maritime_uav.cli import main

if __name__ == '__main__':
    main()
