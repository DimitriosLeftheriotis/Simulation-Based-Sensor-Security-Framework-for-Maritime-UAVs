#!/usr/bin/env python3
"""Retired in v5.

This parallel driver duplicated the experiment logic of maritime_uav/cli.py
with the pre-v5 interfaces (tuple-valued if_eval, false-alarm rate per 260 s,
oracle association).  To avoid results that silently diverge from run_sim.py,
use the single documented driver:

    python run_sim.py --N 100 --out results.json
    python scripts/generate_all.py
"""
import sys
sys.exit(__doc__)
