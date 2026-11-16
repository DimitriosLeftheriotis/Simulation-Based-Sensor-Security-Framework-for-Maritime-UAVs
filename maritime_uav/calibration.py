"""
Detector threshold calibration and learned (Isolation Forest) detector.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from .config import DT, PFA, CUSUM_DELTA, CM_CONSEC, M_CONSEC, IMU_CONSEC, DET_WINDOW, CFG


def calibrate(dur, K=40):
    """Set the NIS gate and CUSUM thresholds from attack-free runs to a target
    false-alarm rate, as described in the paper (the manoeuvre-induced
    innovations at the patrol corners make the nominal chi-square gate unsafe).

    Updates ``CFG`` in-place and returns a summary dict.
    """
    from .simulation import run_once  # deferred to break circular import

    runs = [run_once([], 'F', 50000 + i, dur, None, rec=True) for i in range(K)]
    d2all = np.concatenate([np.array(r['d2']) for r in runs if len(r['d2'])])
    CFG.emp_gate = float(np.quantile(d2all, 1 - PFA))
    CFG.cusum_c0 = float(d2all.mean() + CUSUM_DELTA)

    maxc = []
    for r in runs:
        g = mg = 0.0
        for d in r['d2']:
            g = max(0.0, g + d - CFG.cusum_c0)
            mg = max(mg, g)
        maxc.append(mg)
    CFG.cusum_h_cal = float(np.quantile(maxc, 0.95)) * 1.3 + 5.0

    # calibrate the maritime common-mode threshold against the actual
    # CM_CONSEC-consecutive-scan rule (per run), so realistic stale/noisy AIS
    # does not inflate the false-alarm rate.
    def sustained_q(key, q=0.97):
        s = []
        for r in runs:
            c = np.array(r[key])
            if len(c) >= CM_CONSEC:
                s.append(float(np.min(
                    np.lib.stride_tricks.sliding_window_view(c, CM_CONSEC),
                    axis=1).max()))
            elif len(c):
                s.append(float(c.max()))
        return float(np.quantile(s, q)) + 3.0 if s else None

    CFG.cm_thresh = sustained_q('cmrec') or CFG.cm_thresh
    CFG.spread_thresh = sustained_q('sprec') or CFG.spread_thresh

    # IMU monitor: same sustained-exceedance rule, applied to the smoothed
    # acceleration and heading residuals (IMU_CONSEC consecutive samples).
    # The monitor was inactive (infinite thresholds) during these runs, so
    # it did not influence the GNSS and maritime calibration above.
    def sustained_imu(key, q=0.97, margin=1.3):
        s = []
        for r in runs:
            c = np.array(r[key], float)
            c = c[~np.isnan(c)]
            if len(c) >= IMU_CONSEC:
                s.append(float(np.min(
                    np.lib.stride_tricks.sliding_window_view(c, IMU_CONSEC),
                    axis=1).max()))
        return float(np.quantile(s, q)) * margin if s else float('inf')

    CFG.imu_acc_thresh = sustained_imu('imu_a')
    CFG.imu_head_thresh = sustained_imu('imu_h')

    return dict(emp_gate=CFG.emp_gate, cusum_c0=CFG.cusum_c0,
                cusum_h=CFG.cusum_h_cal, cm_thresh=CFG.cm_thresh,
                spread_thresh=CFG.spread_thresh, d2_mean=float(d2all.mean()),
                imu_acc_thresh=CFG.imu_acc_thresh,
                imu_head_thresh_deg=float(np.rad2deg(CFG.imu_head_thresh)))


def train_if(dur, K=30):
    """Train the optional isolation-forest learned detector on attack-free
    feature vectors and set its alarm threshold to a low attack-free FPR."""
    from .simulation import run_once

    X = []
    for i in range(K):
        for f, att, _ in run_once([], 'F', 60000 + i, dur, rec=True)['feats']:
            if not att:
                X.append(f)
    X = np.array(X)
    model = IsolationForest(contamination=0.01, random_state=0,
                            n_estimators=120).fit(X)
    thr = float(np.quantile(model.score_samples(X), 0.005))
    return model, thr


def if_eval(model, thr, attacks, N, base, dur):
    """Evaluate the learned detector with the same rules as ``run_once``:
    alarm on M_CONSEC consecutive anomalous scores; detection = first alarm
    within DET_WINDOW after onset; false alarm = an alarm before onset.  Time
    comes from the step index of each feature sample (samples are missing
    while GNSS is jammed).  Alarm onsets over the whole run are counted for
    the attack-free false-alarm rate."""
    from .simulation import run_once

    rows = []
    for i in range(N):
        r = run_once(attacks, 'F', base + i, dur, 'noL3', rec=True)
        feats, t_on = r['feats'], r['t_on']
        X = np.array([f for f, _, _ in feats])
        ks = [kk for _, _, kk in feats]
        sc = model.score_samples(X)
        below = 0
        det_t = None
        fa = False
        onsets = 0
        for s, kk in zip(sc, ks):
            t = kk * DT
            below = below + 1 if s < thr else 0
            if below == M_CONSEC:
                onsets += 1
            if below >= M_CONSEC:
                if t < t_on:
                    fa = True
                elif t <= t_on + DET_WINDOW and det_t is None:
                    det_t = t
        rows.append(dict(detected=det_t is not None,
                         latency=(det_t - t_on) if det_t is not None else np.nan,
                         false_before=fa, alarm_onsets=onsets, t_on=t_on))
    return rows
