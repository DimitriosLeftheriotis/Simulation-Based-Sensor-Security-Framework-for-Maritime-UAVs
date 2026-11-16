"""
Attack injection: modify GNSS measurements / IMU / AIS reports / radar
detections according to the active attack set.

GNSS / IMU attacks handled in ``inject_attacks``:
  A1   sudden-jump GNSS spoofing
  A2n  naive GNSS drift (position only)
  A2c  velocity-consistent GNSS drift (velocity offset ramped over 10 s)
  A3   GNSS jamming / dropout (no valid fix for the whole interval)
  A4   IMU bias injection: accelerometer bias plus the residual AHRS heading
       bias it induces (the heading part is applied where the heading is
       consumed, through ``a4_heading_bias``)
  A4s  IMU scale-factor injection on the accelerometer
  A7   Gilbert-Elliott erasure of the GNSS measurement stream (scenario S3)

Maritime attacks, applied inside the L3 radar scan:
  A5   AIS position spoofing: constant offset on the reported AIS position of
       one real vessel (vessel 0), through ``inject_ais_attacks``
  A6   radar blinding: each radar detection is suppressed with probability
       ``A6_DROP_P``, through ``radar_suppressed`` (the AIS report remains)

The C2-uplink realisations of A7 (jamming 'A7j', replay 'A7r', injection
'A7i') act on command frames and are handled in ``link.py``.

v5: random attack directions are unit vectors (``random_vector``); the
earlier code drew (cos a, sin b) with two independent angles, whose norm
varied between 0 and sqrt(2).
"""
import numpy as np
from .config import GE_PGB, GE_PBG, GE_LOSS_BAD, A6_DROP_P


def random_vector(rng, lo, hi):
    """Magnitude uniform in [lo, hi] along a uniformly random direction."""
    mag = rng.uniform(lo, hi)
    th = rng.uniform(0, 2 * np.pi)
    return mag * np.array([np.cos(th), np.sin(th)])


def inject_attacks(z, a_imu, attacks, t, t_on, k, k_jam_off,
                   jump, drift, imu_bias, rng, link_bad,
                   acc_true=None, a4s_scale=0.0):
    """Modify measurement *z* and IMU reading *a_imu* in place.
    Returns ``(gnss_valid, link_bad)``."""
    gnss_valid = True

    if 'A1' in attacks:
        z[:2] += jump
    if 'A2n' in attacks:                       # naive: position only
        z[:2] += drift * (t - t_on)
    if 'A2c' in attacks:                       # consistent: pos + ramped vel
        z[:2] += drift * (t - t_on)
        z[2:] += drift * min(1.0, (t - t_on) / 10.0)
    if 'A3' in attacks and k < k_jam_off:
        gnss_valid = False
    if 'A7' in attacks:                        # Gilbert-Elliott erasure
        link_bad = (rng.random() < GE_PGB) if not link_bad else (rng.random() > GE_PBG)
        if link_bad and rng.random() < GE_LOSS_BAD:
            gnss_valid = False

    if 'A4' in attacks:                        # accelerometer bias
        a_imu[:] = a_imu + imu_bias
    if 'A4s' in attacks and acc_true is not None:   # accelerometer scale factor
        a_imu[:] = a_imu + a4s_scale * acc_true

    return gnss_valid, link_bad


def inject_ais_attacks(ap, vi, attacks, attacking, ais_off):
    """A5: constant offset *ais_off* on the reported AIS position of vessel 0."""
    if attacking and 'A5' in attacks and vi == 0:
        return ap + ais_off
    return ap


def radar_suppressed(attacks, attacking, rng):
    """A6 (blinding): True if this radar detection is suppressed.  Draws from
    *rng* only when attacking with A6 active."""
    return bool(attacking and 'A6' in attacks and rng.random() < A6_DROP_P)


def a4_heading_bias(attacks, attacking, psi_bias):
    """Residual AHRS heading bias induced by A4 (rad); 0 when A4 is inactive."""
    return psi_bias if (attacking and 'A4' in attacks) else 0.0
