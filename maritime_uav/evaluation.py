"""
Statistical evaluation helpers, Monte Carlo driver, benchmarks, and
robustness / calibration-generalisation sweeps.
"""
import time, hashlib, tracemalloc
import numpy as np
from scipy.stats import t as tdist, wilcoxon, binomtest
from scipy.optimize import linear_sum_assignment
from .config import DT, SPEED, SIGMA_R, MAVLINK_SIGN_BYTES, CFG


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (None, None)
    m = float(x.mean())
    if len(x) < 2:
        return (m, 0.0)
    return (m, float(tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))))


def mc(attacks, config, N, base, ablation=None, dur=260.0):
    from .simulation import run_once
    return [run_once(attacks, config, base + i, dur, ablation) for i in range(N)]


def binom_ci(k, n):
    """95% binomial CI half-width (Wilson) for a detection/completion rate."""
    if n == 0:
        return 0.0
    p = k / n
    z = 1.96
    lo = (p + z * z / (2 * n) - z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)
    hi = (p + z * z / (2 * n) + z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)
    return float((hi - lo) / 2)


def robustness(N, dur):
    """R1: maritime detection of the consistent drift vs vessel count and radar
    quality, to show the cross-check is not an artefact of clean synthetic radar."""
    from .simulation import run_once

    base_v, base_pd = CFG.n_vessels, CFG.pd_radar
    out = {'vessels': {}, 'radar_pd': {}}
    for nv in [2, 3, 4, 6, 8]:
        CFG.n_vessels = nv
        r = [run_once(['A2c'], 'F', 80000 + i, dur) for i in range(N)]
        out['vessels'][str(nv)] = float(np.mean([x['detected'] for x in r]))
    CFG.n_vessels = base_v
    for pd in [0.5, 0.7, 0.9]:
        CFG.pd_radar = pd
        r = [run_once(['A2c'], 'F', 81000 + i, dur) for i in range(N)]
        out['radar_pd'][f'{pd:.1f}'] = float(np.mean([x['detected'] for x in r]))
    CFG.pd_radar = base_pd
    return out


def calib_generalization(N, dur):
    """A2: false-alarm rate when the deployment GNSS-noise level differs from the
    (in-distribution) level the thresholds were calibrated on."""
    from .simulation import run_once

    base = CFG.sigma_p
    out = {}
    for mult, lab in [(1.0, 'matched'), (1.5, 'x1.5'), (2.0, 'x2.0')]:
        CFG.sigma_p = base * mult
        r = [run_once([], 'F', 90000 + i, dur) for i in range(N)]
        out[lab] = float(np.mean([x['false_before'] for x in r]) / (dur / 3600.0))
    CFG.sigma_p = base
    return out


def mcnemar_p(a, b):
    """Exact McNemar test on paired boolean detection vectors a, b."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    nd = b01 + b10
    if nd == 0:
        return 1.0
    return float(binomtest(min(b01, b10), nd, 0.5).pvalue)


def overhead_bench():
    from .dynamics import KF
    from .layers import IMUMonitor
    from .link import Frame, L1Verifier, signature, _ts, GCS_SYS, GCS_COMP, MSG_HEARTBEAT
    reps = 20000
    kf = KF(np.zeros(4))
    z = np.array([1., 1., .1, .1])
    a = np.array([.1, .1])
    tracemalloc.start()
    t = time.perf_counter()
    for _ in range(reps):
        kf.predict(a)
        kf.innovation(z)
        kf.update(z, 1.0)
    l2 = (time.perf_counter() - t) / reps * 1e3
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ap = np.random.rand(6, 2) * 100
    rr = np.random.rand(6, 2) * 100
    t = time.perf_counter()
    for _ in range(reps):
        C = np.linalg.norm(ap[:, None, :] - rr[None, :, :], axis=2)
        linear_sum_assignment(C)
    l3 = (time.perf_counter() - t) / reps * 1e3
    t = time.perf_counter()
    for _ in range(reps):
        x = 0.5 * 0.9  # trust update + state machine (scalar)
    l4 = (time.perf_counter() - t) / reps * 1e3
    msg = b'x' * 50
    t = time.perf_counter()
    for _ in range(reps):
        hashlib.sha256(b'secret' + msg).digest()[:6]
    l1 = (time.perf_counter() - t) / reps * 1e3
    # L1 as executed in the loop: full verification of a signed frame
    # (signature recomputation, constant-time compare, replay window)
    key = bytes(32)
    ver = L1Verifier(key)
    frames = []
    for i in range(reps):
        fr = Frame(i, GCS_SYS, GCS_COMP, MSG_HEARTBEAT, bytes(9), 'legit')
        fr.ts = _ts(i * 1.0)
        fr.sig = signature(key, fr)
        frames.append(fr)
    t = time.perf_counter()
    for fr in frames:
        ver.check(fr)
    l1_verify = (time.perf_counter() - t) / reps * 1e3
    # L2 IMU monitor (acceleration and heading residuals, both tests)
    mon = IMUMonitor()
    av = np.array([.1, .1]); vv = np.array([8., 0.])
    t = time.perf_counter()
    for _ in range(reps):
        mon.update(av, vv, True, 0.01)
    l2_imu = (time.perf_counter() - t) / reps * 1e3
    return dict(L1=l1, L2=l2, L3=l3, L4=l4, total=l1 + l2 + l3 + l4,
                mem_mb=peak / 1e6, mav_bytes=MAVLINK_SIGN_BYTES,
                L1_verify=l1_verify, L2_imu=l2_imu,
                total_ext=l1_verify + l2 + l2_imu + l3 + l4)


def summarize(runs, keys=('rmse', 'maxerr', 'recov', 'xtrack_max')):
    """Mean and 95% CI half-width of the continuous metrics, plus the usual
    proportions, for a list of run dicts."""
    out = {}
    for kk in keys:
        m, h = ci95([x[kk] for x in runs])
        out[kk], out[kk + '_ci'] = m, h
    for kk in ('detected', 'completion', 'aborted', 'imu_detected', 'imu_isolated',
               'hijacked', 'corridor_ok', 'link_abort', 'false_before'):
        v = [bool(x[kk]) for x in runs]
        out[kk] = float(np.mean(v))
        out[kk + '_ci'] = binom_ci(int(np.sum(v)), len(v))
    lm, lh = ci95([x['latency'] for x in runs])
    out['latency'], out['latency_ci'] = lm, lh
    im, ih = ci95([x['imu_latency'] for x in runs])
    out['imu_latency'], out['imu_latency_ci'] = im, ih
    srcs = {}
    for x in runs:
        srcs[str(x['alarm_src'])] = srcs.get(str(x['alarm_src']), 0) + 1
    out['first_alarm_src'] = srcs
    return out


def link_summary(runs):
    """Metrics for the command-link experiments (attack A7 on the C2 uplink)."""
    s = summarize(runs)
    rej = {kk: float(np.mean([x['rejected'][kk] for x in runs]))
           for kk in ('unsigned', 'bad_sig', 'replay')}
    fr = [x['first_reject_t'] for x in runs]
    s.update(rejected_mean=rej,
             legit_rejected=float(np.mean([x['legit_rejected'] for x in runs])),
             restarts=float(np.mean([x['restarts'] for x in runs])),
             max_link_gap=ci95([x['max_link_gap'] for x in runs]),
             link_abort_t=ci95([np.nan if x['link_abort_t'] is None else x['link_abort_t']
                                for x in runs]))
    return s


def roc_bench(N, dur=260.0):
    """Peak detector scores under consistent drift (A2c) vs attack-free baseline
    under passive monitoring ('MON') for Figure 3 ROC/AUC analysis."""
    from .simulation import run_once
    aa = [run_once(['A2c'], 'MON', 10000 + i, dur) for i in range(N)]
    ff = [run_once([], 'MON', 40000 + i, dur) for i in range(N)]
    return {
        'innov_att': [float(r['peak_cusum']) for r in aa],
        'mar_att': [float(r['peak_cm']) for r in aa],
        'innov_fre': [float(r['peak_cusum']) for r in ff],
        'mar_fre': [float(r['peak_cm']) for r in ff]
    }


def sample_trajectory(dur=260.0, base_seed=2000, max_search=100):
    """Extract coordinates for a representative detected S1 trial under
    GNSS jump (A1) for Figure 4."""
    from .simulation import run_once
    from .config import GEOFENCE, DT, T_EVAL
    for sd in range(base_seed, base_seed + max_search):
        r = run_once(['A1'], 'F', sd, dur, trace=True)
        if r['detected'] and r['alarm_t'] and r['maxerr'] < GEOFENCE:
            return {
                'seed': sd,
                'k_on': int(r['k_on']),
                'alarm_t': float(r['alarm_t']) if r['alarm_t'] is not None else None,
                't_on': float(r['t_on']),
                'truth': np.round(r['truth'], 2).tolist(),
                'xhat': np.round(r['xhat'], 2).tolist(),
                'gnss': np.round(r['gnss'], 2).tolist(),
                'dt': float(DT),
                't_eval': float(T_EVAL),
                'geofence': float(GEOFENCE),
            }
    return None
