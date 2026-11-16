"""
Sensor-security framework for maritime UAV operations.

Package re-exports for backward compatibility with ``import maritime_uav as S``.
"""
from .config import *          # noqa: F401,F403 – all constants
from .config import CFG
from .dynamics import make_reference, vessel_tracks, KF, route_distance
from .attacks import inject_attacks
from .layers import L2Detector, B2Detector, IMUMonitor, L3CrossCheck, L4Trust, fuse
from .link import C2Link, L1Verifier
from .simulation import run_once
from .calibration import calibrate, train_if, if_eval
from .evaluation import (ci95, mc, binom_ci, robustness,
                         calib_generalization, mcnemar_p, overhead_bench,
                         summarize, link_summary,
                         roc_bench, sample_trajectory)
from .cli import main
