"""
Simulation constants, Kalman-filter matrices, and mutable calibration state.

All parameters that were previously module-level globals in the monolithic
the original monolithic script live here.  Mutable calibration/threshold values are stored in
a module-level ``SimConfig`` dataclass instance (``CFG``) so they can be
updated by ``calibrate()`` without relying on ``global`` statements scattered
across modules.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2

# ---- Fixed simulation parameters (Table: simparams) ------------------------
DT = 0.1
SPEED = 8.0
RECT = 500.0
SIGMA_P = 0.8
SIGMA_V = 0.1
SIGMA_A = 0.3
SIGMA_IMU_A = 0.3
GNSS_TIMEOUT = 1.0
RECOVER_TOL = 5.0
GEOFENCE = 50.0
PFA = 1e-3
M_CONSEC = 3
CUSUM_DELTA = 4.0
CUSUM_H = 40.0
ALPHA_DOWN = 0.5
ALPHA_UP = 0.02
TAU_ISO = 0.3
TAU_REC = 0.6
SIGMA_MAX = 150.0
ISO_TIMEOUT = 60.0
T_EVAL = 40.0                # s, bounded post-onset response-evaluation window
RADAR_SCAN = 1.0
SIGMA_R = 20.0
SIGMA_BEAR = np.deg2rad(1.0)
SIGMA_PSI = np.deg2rad(1.0)  # AHRS heading uncertainty (radar->geo conversion)
PD_RADAR = 0.9
EOIR_RANGE = 3000.0
N_VESSELS = 6
SIGMA_AIS = 15.0             # AIS reported-position noise (m)
SIGMA_AIS_VEL = 0.3          # AIS reported-velocity (SOG/COG) noise (m/s)
AIS_REPORT_MIN = 4.0
AIS_REPORT_MAX = 12.0        # AIS report interval range (s)
DET_WINDOW = 120.0           # s: detection must occur within this window after onset
CM_THRESH = 40.0
CM_CONSEC = 5
GE_PGB = 0.05
GE_PBG = 0.4
GE_LOSS_BAD = 0.5
A6_DROP_P = 0.3              # A6: per-scan probability a radar-AIS pair is dropped
MAVLINK_SIGN_BYTES = 13
CTRL_GAIN = 0.8
A_MAX = 4.0                  # m/s^2 commanded-acceleration (turn) limit
B2_SPEED_LIMIT = 3.0 * SPEED

# ---- IMU monitor (L2, Table 6) and IMU attack A4 ------------------------------
SIGMA_MAG = np.deg2rad(2.0)      # magnetometer heading noise per 10 Hz sample
SIGMA_MAG_OFF = np.deg2rad(0.5)  # residual post-calibration magnetometer offset (per run)
IMU_TAU = 5.0                    # s, time constant of the EMA applied to both IMU residuals
IMU_CONSEC = 10                  # samples (1 s) a residual must stay above its threshold
SIGMA_CV = 1.0                   # m/s^2, process noise of the model-based fallback used
                                 # for prediction when the IMU is isolated
MODEL_BIAS_SD = 0.3              # m/s^2, per-run bias of the fallback's acceleration model
MODEL_NOISE_SD = 0.5             # m/s^2, white error of the fallback's acceleration model
                                 # (the kinematic vehicle has perfect actuation, so the
                                 # fallback is given an explicit model mismatch)
A4_BIAS = (0.2, 1.0)             # m/s^2, accelerometer bias (A4, bias variant)
A4_PSI_BIAS = (2.0, 6.0)         # deg, residual AHRS heading bias (A4, bias variant)
A4S_SCALE = (0.1, 0.5)           # accelerometer scale-factor error (A4, scale variant 'A4s')

# ---- C2 uplink and Layer 1 (MAVLink 2 signing), attack A7 on the C2 link -------
HB_PERIOD = 1.0                  # s, GCS HEARTBEAT period
GCS_FS_TIMEOUT = 5.0             # s, GCS-loss failsafe of the baseline autopilot (RTL)
LINK_LOSS_LIMIT = 30.0           # s, framework link-loss limit before CONTINGENCY
LINK_LOSS_GOOD = 0.01            # frame-loss probability, good state
LINK_LOSS_BAD = 0.95             # frame-loss probability, bad (jammed) state
LINK_LAT_GOOD = (0.02, 0.06)     # s, one-way latency range, good state
LINK_LAT_BAD = (0.10, 0.50)      # s, one-way latency range, bad state
A7J_PI_BAD = (0.2, 0.6)          # stationary fraction of time in the bad state (A7j)
A7J_BURST = (2.0, 20.0)          # s, mean bad-state burst length (A7j)
A7R_COPIES = 3                   # replayed copies of the recorded command (A7r)
A7I_PERIOD = 1.0                 # s, period of forged command frames (A7i)
A7I_OFFSET = (200.0, 600.0)      # m, distance of the attacker's target from the UAV (A7i)

# ---- Kalman-filter system matrices ------------------------------------------
F = np.array([[1, 0, DT, 0],
              [0, 1, 0, DT],
              [0, 0, 1,  0],
              [0, 0, 0,  1]], float)

B = np.array([[0.5 * DT**2, 0],
              [0, 0.5 * DT**2],
              [DT, 0],
              [0, DT]], float)

Q = SIGMA_A**2 * (B @ B.T) + 1e-6 * np.eye(4)
Q_CV = SIGMA_CV**2 * (B @ B.T) + 1e-6 * np.eye(4)   # IMU-isolated (model-based) prediction
R0 = np.diag([SIGMA_P**2, SIGMA_P**2, SIGMA_V**2, SIGMA_V**2])
GATE = chi2.ppf(1 - PFA, 4)


# ---- Mutable calibration / threshold state ----------------------------------
@dataclass
class SimConfig:
    """Run-time mutable thresholds set by ``calibrate()`` and sensitivity
    sweeps (``robustness()``, ``calib_generalization()``)."""
    emp_gate: float = GATE
    cusum_c0: float = 4 + CUSUM_DELTA
    cusum_h_cal: float = CUSUM_H
    cm_thresh: float = CM_THRESH
    spread_thresh: float = CM_THRESH
    # IMU-monitor thresholds; infinite (inactive) until calibrate() sets them,
    # so the calibration runs themselves are unaffected by the monitor.
    imu_acc_thresh: float = float('inf')
    imu_head_thresh: float = float('inf')
    # Parameters that sensitivity sweeps temporarily override:
    n_vessels: int = N_VESSELS
    pd_radar: float = PD_RADAR
    sigma_p: float = SIGMA_P


CFG = SimConfig()
