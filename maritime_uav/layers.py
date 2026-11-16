"""
Detection and trust layers:
  - L2: NIS gating + CUSUM detector
  - L2: IMU monitor (accelerometer vs GNSS-derived acceleration, AHRS heading
        vs magnetometer), Table 6
  - L3: Maritime cross-check (radar + AIS)
  - L4: Trust update, GNSS isolation, contingency
  - Fusion decision
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .attacks import inject_ais_attacks, a4_heading_bias
from .config import (
    DT, SIGMA_R, SIGMA_BEAR, SIGMA_PSI, SIGMA_AIS, SIGMA_AIS_VEL,
    EOIR_RANGE, AIS_REPORT_MIN, AIS_REPORT_MAX,
    M_CONSEC, RADAR_SCAN, CM_CONSEC,
    ALPHA_DOWN, ALPHA_UP, TAU_ISO, TAU_REC,
    SIGMA_MAX, ISO_TIMEOUT, GNSS_TIMEOUT,
    B2_SPEED_LIMIT, SPEED, CFG,
    SIGMA_MAG, IMU_TAU, IMU_CONSEC,
)


class L2Detector:
    """Innovation-based detector: NIS chi-square gate + CUSUM."""

    def __init__(self):
        self.consec_gate = 0
        self.cusum = 0.0
        self.peak_cusum = 0.0

    def update(self, d2, is_attack_phase):
        self.consec_gate = self.consec_gate + 1 if d2 > CFG.emp_gate else 0
        self.cusum = max(0.0, self.cusum + d2 - CFG.cusum_c0)
        if is_attack_phase:
            self.peak_cusum = max(self.peak_cusum, self.cusum)
        return self.consec_gate >= M_CONSEC or self.cusum > CFG.cusum_h_cal


class B2Detector:
    """Baseline B2: threshold on implied speed from position buffer."""

    def __init__(self):
        self.pos_buf = []

    def update(self, z_pos):
        self.pos_buf.append(z_pos.copy())
        if len(self.pos_buf) > int(1.0 / DT):
            self.pos_buf.pop(0)
            implied = np.linalg.norm(self.pos_buf[-1] - self.pos_buf[0]) / 1.0
            return implied > B2_SPEED_LIMIT
        return False


class IMUMonitor:
    """IMU checks of Table 6, run at the 10 Hz cycle rate.

    * Acceleration residual: accelerometer reading at step k-1 against the
      GNSS-derived acceleration (v_k - v_{k-1})/dt, evaluated whenever a valid
      GNSS velocity is available at both steps.  It uses the GNSS *velocity*,
      which position-spoofing leaves intact, so it remains usable while the
      GNSS position is rejected.
    * Heading residual: AHRS (gyro-integrated) heading against the
      magnetometer heading.  It is independent of GNSS.

    Each residual is smoothed by an exponential moving average with time
    constant IMU_TAU; a check fails when the norm of its EMA stays above a
    threshold calibrated on attack-free runs for IMU_CONSEC samples.
    """

    def __init__(self):
        self.alpha = DT / IMU_TAU
        self.ema_a = np.zeros(2)
        self.ema_h = 0.0
        self.consec_a = 0
        self.consec_h = 0
        self.prev_v = None
        self.prev_a = None
        self.stat_a = 0.0
        self.stat_h = 0.0

    def update(self, a_imu, v_gnss, gnss_ok, dpsi):
        """Return ``(acc_alarm, head_alarm)`` for this step."""
        if gnss_ok and self.prev_v is not None and self.prev_a is not None:
            r = self.prev_a - (v_gnss - self.prev_v) / DT
            self.ema_a = (1 - self.alpha) * self.ema_a + self.alpha * r
            self.stat_a = float(np.hypot(self.ema_a[0], self.ema_a[1]))
            self.consec_a = self.consec_a + 1 if self.stat_a > CFG.imu_acc_thresh else 0
        self.prev_v = v_gnss.copy() if gnss_ok else None
        self.prev_a = a_imu.copy()
        self.ema_h = (1 - self.alpha) * self.ema_h + self.alpha * dpsi
        self.stat_h = abs(self.ema_h)
        self.consec_h = self.consec_h + 1 if self.stat_h > CFG.imu_head_thresh else 0
        return self.consec_a >= IMU_CONSEC, self.consec_h >= IMU_CONSEC


class L3CrossCheck:
    """Maritime cross-check: radar-AIS association and common-mode residual."""

    def __init__(self, n_vessels, rng):
        self.consec_cm = 0
        self.consec_ais = 0
        self.peak_cm = 0.0
        # AIS realism: per-vessel report interval and stale position/velocity
        self.t_rep = rng.uniform(AIS_REPORT_MIN, AIS_REPORT_MAX, n_vessels)
        self.ais_last_t = np.zeros(n_vessels)

    def init_ais(self, vessels_k0, vves, rng, n_vessels):
        """Initialise last-reported AIS positions and velocities."""
        self.ais_last_pos = vessels_k0 + rng.normal(0, SIGMA_AIS, (n_vessels, 2))
        self.ais_last_vel = vves + rng.normal(0, SIGMA_AIS_VEL, (n_vessels, 2))

    def scan(self, k, t, truth_k, kf_x, vessels, vves, n_vessels,
             attacks, attacking, ais_off, psi_bias, rng,
             is_attack_phase, rec, rec_cm, rec_sp,
             mag_heading=False, mag_off=0.0):
        """Run one radar scan.  Returns ``(cm_alarm, ais_alarm, mar_ok,
        mar_common, mar_n, false_tracks)``.

        With ``mag_heading`` (IMU isolated by L4) the radar-to-geo conversion
        uses the magnetometer heading instead of the AHRS heading: the error
        then has the magnetometer's statistics and no IMU-induced bias.  The
        same single normal draw is consumed either way."""
        pd_radar = CFG.pd_radar
        psi_err = rng.normal(0, SIGMA_PSI)
        if mag_heading:
            psi_err = psi_err * (SIGMA_MAG / SIGMA_PSI) + mag_off
        else:
            psi_err += a4_heading_bias(attacks, attacking, psi_bias)
        cps, sps = np.cos(psi_err), np.sin(psi_err)
        ais_p, rad_rel = [], []
        false_tracks = 0
        for vi in range(n_vessels):
            rel = vessels[vi, k] - truth_k[:2]
            if np.linalg.norm(rel) > EOIR_RANGE or rng.random() > pd_radar:
                continue
            rng_m = np.linalg.norm(rel) + rng.normal(0, SIGMA_R)
            bear = np.arctan2(rel[1], rel[0]) + rng.normal(0, SIGMA_BEAR)
            rr = rng_m * np.array([np.cos(bear), np.sin(bear)])
            rr = np.array([cps * rr[0] - sps * rr[1],
                           sps * rr[0] + cps * rr[1]])
            # AIS: periodic, noisy position + velocity; dead-reckoned between
            # reports (real AIS carries SOG/COG), so staleness is bounded.
            if t - self.ais_last_t[vi] >= self.t_rep[vi]:
                self.ais_last_pos[vi] = vessels[vi, k] + rng.normal(0, SIGMA_AIS, 2)
                self.ais_last_vel[vi] = vves[vi] + rng.normal(0, SIGMA_AIS_VEL, 2)
                self.ais_last_t[vi] = t
            ap = self.ais_last_pos[vi] + self.ais_last_vel[vi] * (t - self.ais_last_t[vi])
            ap, dropped = inject_ais_attacks(ap, vi, attacks, attacking, ais_off, rng)
            if dropped:                         # A6: pair dropped
                false_tracks += 1
                continue
            ais_p.append(ap)
            rad_rel.append(rr)

        cm_alarm = ais_alarm = False
        mar_ok = False
        mar_common = None
        mar_n = 1

        if len(ais_p) >= 3:
            ais_p = np.array(ais_p)
            rad_rel = np.array(rad_rel)
            C = np.linalg.norm(ais_p[:, None, :] -
                               (truth_k[:2] + rad_rel)[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(C)
            resid = (ais_p[ri] - kf_x[:2]) - rad_rel[ci]
            common = np.median(resid, axis=0)
            spread = float(np.max(np.linalg.norm(resid - common, axis=1)))
            cmag = float(np.linalg.norm(common))
            if rec and not is_attack_phase:
                rec_cm.append(cmag)
                rec_sp.append(spread)
            if cmag > CFG.cm_thresh and spread < CFG.spread_thresh:
                self.consec_cm += 1
            else:
                self.consec_cm = 0
            cm_alarm = self.consec_cm >= CM_CONSEC
            self.consec_ais = (self.consec_ais + 1
                               if (spread > CFG.spread_thresh and cmag < CFG.cm_thresh)
                               else 0)
            ais_alarm = self.consec_ais >= CM_CONSEC
            if is_attack_phase:
                self.peak_cm = max(self.peak_cm, cmag)
            if spread < CFG.spread_thresh:       # geometry self-consistent
                mar_ok = True
                mar_common = common
                mar_n = len(ais_p)

        return cm_alarm, ais_alarm, mar_ok, mar_common, mar_n, false_tracks


class L4Trust:
    """Trust layer: trust score, GNSS isolation, and contingency logic."""

    def __init__(self):
        self.tau = 1.0
        self.isolated = False
        self.state = 'Nominal'
        self.iso_since = None
        self.aborted = False
        # IMU trust (same asymmetric update and hysteresis as GNSS)
        self.tau_imu = 1.0
        self.imu_isolated = False
        self.imu_iso_t = None

    def update_trust(self, l2_alarm, cm_alarm):
        if l2_alarm or cm_alarm:
            self.tau *= (1 - ALPHA_DOWN)
        else:
            self.tau += ALPHA_UP * (1 - self.tau)
        if self.tau < TAU_ISO:
            self.isolated = True
        elif self.tau > TAU_REC:
            self.isolated = False

    def update_imu_trust(self, imu_alarm, t):
        if imu_alarm:
            self.tau_imu *= (1 - ALPHA_DOWN)
        else:
            self.tau_imu += ALPHA_UP * (1 - self.tau_imu)
        if self.tau_imu < TAU_ISO:
            if not self.imu_isolated and self.imu_iso_t is None:
                self.imu_iso_t = t
            self.imu_isolated = True
        elif self.tau_imu > TAU_REC:
            self.imu_isolated = False

    def update_state(self, alarm, kf, t):
        if self.isolated:
            self.state = 'Isolated'
            self.iso_since = self.iso_since if self.iso_since else t
        elif alarm:
            self.state = 'Degraded'
        else:
            self.state = 'Nominal'
            self.iso_since = None
        if (kf.pos_std() > SIGMA_MAX or
                (self.iso_since and t - self.iso_since > ISO_TIMEOUT)):
            self.state = 'Contingency'
            self.aborted = True


def fuse(kf, z, config, gnss_valid, use_filter, l2_alarm, thr_alarm,
         isolated, tau, use_L4, consec_gate):
    """Apply the GNSS measurement to the filter according to the fusion
    policy of the active configuration."""
    if not (gnss_valid and use_filter):
        return
    if config == 'B0':
        pass
    elif config == 'B1':
        if consec_gate < M_CONSEC:
            kf.update(z, 1.0)
    elif config == 'B2':
        if not thr_alarm:
            kf.update(z, 1.0)
    else:  # 'F'
        if not isolated:
            Rs = 1.0 / max(tau, 1e-3) ** 2 if use_L4 else 1.0
            kf.update(z, Rs)
