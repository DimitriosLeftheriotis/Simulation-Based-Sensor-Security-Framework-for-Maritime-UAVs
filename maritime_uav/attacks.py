"""
Attack injection: modify GNSS measurements / IMU / AIS reports according to
the active attack set.

GNSS / IMU attacks handled here:
  A1   sudden-jump GNSS spoofing
  A2n  naive GNSS drift (position only)
  A2c  velocity-consistent GNSS drift
  A3   GNSS jamming / dropout
  A4   IMU bias injection: accelerometer bias plus the residual AHRS heading
       bias it induces (the heading part is applied where the heading is
       consumed, i.e. in the radar-to-geo conversion of L3 and in the IMU
       monitor's heading residual)
  A4s  IMU scale-factor injection on the accelerometer
  A7   burst loss (Gilbert-Elliott) on the MAVLink stream that carries the
       injected GNSS (GPS_INPUT) in the testbed; used in scenario S3

AIS attacks (applied through ``inject_ais_attacks``, called from inside the
per-vessel loop of the L3 scan so that the order of the RNG draws is kept):
  A5   AIS position spoofing: constant offset on the reported AIS position of
       an existing vessel (vessel 0); no new vessel is created
  A6   AIS/track suppression: each detected radar-AIS pair is dropped with
       probability ``A6_DROP_P`` (counted in ``false_tracks``)

The A4 heading bias is applied where the heading is consumed, through
``a4_heading_bias``.

The C2-uplink realisations of A7 (jamming 'A7j', replay 'A7r', injection
'A7i') act on command frames and are handled in ``link.py``.
"""
import numpy as np
from .config import GE_PGB, GE_PBG, GE_LOSS_BAD, A6_DROP_P


def inject_attacks(z, a_imu, attacks, t, t_on, k, k_jam_off,
                   jump, drift, imu_bias, rng, link_bad,
                   acc_true=None, a4s_scale=0.0):
    """Modify measurement *z*, IMU reading *a_imu*, and return updated
    gnss_valid flag and link_bad state.

    Parameters are the pre-generated random attack magnitudes produced in
    ``run_once`` setup.  Returns ``(gnss_valid, link_bad)``.
    """
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
    if 'A7' in attacks:                        # Gilbert-Elliott link loss
        link_bad = (rng.random() < GE_PGB) if not link_bad else (rng.random() > GE_PBG)
        if link_bad and rng.random() < GE_LOSS_BAD:
            gnss_valid = False

    if 'A4' in attacks:                        # accelerometer bias
        a_imu[:] = a_imu + imu_bias
    if 'A4s' in attacks and acc_true is not None:   # accelerometer scale factor
        a_imu[:] = a_imu + a4s_scale * acc_true

    return gnss_valid, link_bad


def inject_ais_attacks(ap, vi, attacks, attacking, ais_off, rng):
    """AIS-layer attacks applied to one vessel's AIS report inside the L3 scan.

      A5  AIS position spoofing: constant offset *ais_off* on vessel 0
      A6  AIS/track suppression: the radar-AIS pair is dropped with
          probability ``A6_DROP_P`` (counted as a false track)

    Must be called at the same point of the per-vessel loop as before, so the
    order of the draws from *rng* is unchanged: ``rng.random()`` is consumed
    only when attacking with A6 active.  Returns ``(ap, dropped)``.
    """
    if not attacking:
        return ap, False
    if 'A5' in attacks and vi == 0:
        ap = ap + ais_off
    if 'A6' in attacks and rng.random() < A6_DROP_P:
        return ap, True
    return ap, False


def a4_heading_bias(attacks, attacking, psi_bias):
    """Residual AHRS heading bias induced by A4 (rad); 0 when A4 is inactive.
    Draws no random numbers."""
    return psi_bias if (attacking and 'A4' in attacks) else 0.0
