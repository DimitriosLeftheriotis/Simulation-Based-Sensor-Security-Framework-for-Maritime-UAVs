"""
Core simulation loop: ``run_once()`` runs a single Monte Carlo trial.

This is the thinned-down orchestrator that delegates to ``dynamics``,
``attacks``, ``layers`` and ``link``.

Random-number discipline: every quantity that existed before the IMU monitor
and the C2-link model were added is drawn from the main RNG in exactly the
original order.  The IMU-monitor noise, the A4 scale factor and the whole C2
link use separate RNG streams derived from the same seed, so experiments
that do not exercise these components reproduce the earlier results exactly.
"""
import numpy as np
from .config import (
    DT, SPEED, SIGMA_P, SIGMA_V, SIGMA_IMU_A, SIGMA_R, SIGMA_PSI,
    SIGMA_MAG, SIGMA_MAG_OFF, A4S_SCALE, Q_CV, MODEL_BIAS_SD, MODEL_NOISE_SD,
    T_EVAL, RADAR_SCAN, CTRL_GAIN, A_MAX,
    GNSS_TIMEOUT, RECOVER_TOL, GEOFENCE, DET_WINDOW, CFG,
)
from .dynamics import make_reference, vessel_tracks, KF, route_distance
from .attacks import inject_attacks, a4_heading_bias
from .layers import (L2Detector, B2Detector, IMUMonitor, L3CrossCheck,
                     L4Trust, fuse)
from .link import C2Link


def run_once(attacks, config, seed, duration=260.0, ablation=None,
             rec=False, trace=False):
    """Run one Monte Carlo trial and return a results dict."""
    rng = np.random.default_rng(seed)
    n = int(duration / DT)
    n_vessels = CFG.n_vessels

    d2rec = []; cmrec = []; sprec = []
    imu_a_rec = []; imu_h_rec = []
    feats = []

    ref = make_reference(n)
    ref_vel = np.vstack([(ref[1] - ref[0]) / DT, np.diff(ref, axis=0) / DT])
    vessels = vessel_tracks(n, n_vessels, rng)

    # Attack parameters (random per trial)
    t_on = rng.uniform(60, 120); k_on = int(t_on / DT)
    k_eval = min(n, k_on + int(T_EVAL / DT))
    jump = rng.uniform(50, 500) * np.array([np.cos(rng.uniform(0, 2 * np.pi)),
                                             np.sin(rng.uniform(0, 2 * np.pi))])
    drate = rng.uniform(0.2, 1.0)
    drift = drate * np.array([np.cos(rng.uniform(0, 2 * np.pi)),
                              np.sin(rng.uniform(0, 2 * np.pi))])
    jam_dur = rng.uniform(10, 60); k_jam_off = k_on + int(jam_dur / DT)
    imu_bias = rng.uniform(0.2, 1.0) * np.array([np.cos(rng.uniform(0, 2 * np.pi)),
                                                   np.sin(rng.uniform(0, 2 * np.pi))])
    ais_off = rng.uniform(80, 300) * np.array([np.cos(rng.uniform(0, 2 * np.pi)),
                                                np.sin(rng.uniform(0, 2 * np.pi))])
    psi_bias = np.deg2rad(rng.uniform(2.0, 6.0))

    # Independent stream for the IMU monitor (AHRS / magnetometer noise) and
    # for the A4 scale factor -- does not perturb the main RNG.
    rng_imu = np.random.default_rng([int(seed), 4])
    mag_off = rng_imu.normal(0, SIGMA_MAG_OFF)
    ahrs_noise = rng_imu.normal(0, SIGMA_PSI, n)
    mag_noise = rng_imu.normal(0, SIGMA_MAG, n)
    a4s_scale = rng_imu.uniform(*A4S_SCALE)
    # model-based fallback (IMU isolated): commanded acceleration with an
    # explicit per-run bias and white model error
    model_bias = rng_imu.normal(0, MODEL_BIAS_SD, 2)
    model_noise = rng_imu.normal(0, MODEL_NOISE_SD, (n, 2))

    # Vessel velocities and AIS init
    vves = (vessels[:, 1] - vessels[:, 0]) / DT

    # Configuration flags
    use_filter = config != 'B0'
    use_gate = config in ('B1', 'F', 'MON')
    use_L3 = config in ('F', 'MON') and ablation != 'noL3'
    use_L4 = config == 'F' and ablation != 'noL4'
    use_imu_mon = config in ('F', 'MON') and ablation != 'noIMU'
    use_L1 = config == 'F' and ablation != 'noL1'

    # State
    truth = np.zeros((n, 4)); truth[0, :2] = ref[0]; truth[0, 2:] = ref_vel[0]
    kf = KF(np.array([ref[0, 0], ref[0, 1], ref_vel[0, 0], ref_vel[0, 1]]))
    gnss_hist = np.zeros((n, 2)); xhat_hist = np.zeros((n, 2))
    errs = []

    # Detectors
    l2det = L2Detector()
    b2det = B2Detector()
    imumon = IMUMonitor()
    l3 = L3CrossCheck(n_vessels, rng)
    l3.init_ais(vessels[:, 0], vves, rng, n_vessels)
    l4 = L4Trust()
    link = C2Link(attacks, use_L1, seed, t_on, k_on, framework=(config == 'F'))

    alarm_t = None; false_before = False; last_fix_t = 0.0
    alarm_src = None; imu_alarm_t = None
    prev_v = ref_vel[0].copy()
    false_tracks = 0; link_bad = False

    for k in range(n):
        t = k * DT
        attacking = k >= k_on
        est = kf.x[:2].copy()

        # C2 uplink: delivery, L1 verification, command handling
        link_alarm = link.step(k, t, truth[k, :2])

        # Guidance law (AUTO: follow the stored mission from its start step;
        # GUIDED: fly to the commanded target)
        if link.mode == 'AUTO':
            kr = min(max(k - link.k0, 0), n - 1)
            v_des = ref_vel[kr] + CTRL_GAIN * (ref[kr] - est)
        else:
            dvec = link.target - est
            dist = np.linalg.norm(dvec)
            v_des = dvec / max(dist, 1e-6) * min(SPEED, CTRL_GAIN * dist)
        s = np.linalg.norm(v_des)
        if s > 1.5 * SPEED:
            v_des *= 1.5 * SPEED / s
        dv = v_des - prev_v
        amag = np.linalg.norm(dv) / DT
        if amag > A_MAX:
            dv *= A_MAX / amag
        v_cmd = prev_v + dv
        acc = (v_cmd - prev_v) / DT; prev_v = v_cmd.copy()
        if k + 1 < n:
            truth[k + 1, :2] = truth[k, :2] + v_cmd * DT
            truth[k + 1, 2:] = v_cmd

        # IMU
        a_imu = acc + rng.normal(0, SIGMA_IMU_A, 2)

        # GNSS measurement (before attack injection)
        gnss_valid = True
        z = np.empty(4)
        z[:2] = truth[k, :2] + rng.normal(0, CFG.sigma_p, 2)
        z[2:] = truth[k, 2:] + rng.normal(0, SIGMA_V, 2)

        # Attack injection
        if attacking:
            gnss_valid, link_bad = inject_attacks(
                z, a_imu, attacks, t, t_on, k, k_jam_off,
                jump, drift, imu_bias, rng, link_bad,
                acc_true=acc, a4s_scale=a4s_scale)

        # KF predict (IMU isolated: model-based prediction from the commanded
        # acceleration with model mismatch, inflated process noise)
        imu_iso = use_L4 and use_imu_mon and l4.imu_isolated
        if use_filter:
            if imu_iso:
                kf.predict(acc + model_bias + model_noise[k], Q_CV)
            else:
                kf.predict(a_imu)

        # --- Detection ---
        l2_alarm = thr_alarm = cm_alarm = ais_alarm = False
        acc_alarm = head_alarm = False
        mar_ok = False; mar_common = None; mar_n = 1

        if gnss_valid:
            last_fix_t = t
            if use_filter:
                d2 = kf.innovation(z)
                nu = z - kf.x
                feats.append(([d2, float(np.hypot(nu[0], nu[1])),
                               float(np.hypot(nu[2], nu[3])),
                               float(np.hypot(kf.x[2], kf.x[3]))], k >= k_on))
                if rec and k < k_on:
                    d2rec.append(d2)
                if use_gate:
                    l2_alarm = l2det.update(d2, k >= k_on)
                if config == 'B2':
                    thr_alarm = b2det.update(z[:2])
            else:
                kf.x[:2] = z[:2]; kf.x[2:] = z[2:]
        else:
            if t - last_fix_t > GNSS_TIMEOUT and config in ('B1', 'B2', 'F'):
                l2_alarm = True

        # L2 IMU monitor: accelerometer vs GNSS-derived acceleration whenever a
        # valid GNSS velocity is available (the position-spoofing attacks leave
        # the GNSS velocity intact, and the consistent drift changes it by at most
        # 0.1 m/s^2, below the calibrated threshold); AHRS heading vs
        # magnetometer at every step, independent of GNSS.
        if use_imu_mon:
            gnss_vel_ok = gnss_valid
            dpsi = (ahrs_noise[k] + a4_heading_bias(attacks, attacking, psi_bias)
                    - (mag_noise[k] + mag_off))
            acc_alarm, head_alarm = imumon.update(a_imu, z[2:], gnss_vel_ok, dpsi)
            if rec and k < k_on:
                imu_a_rec.append(imumon.stat_a if gnss_vel_ok else np.nan)
                imu_h_rec.append(imumon.stat_h)
        imu_alarm = acc_alarm or head_alarm

        # L3 maritime cross-check
        if use_L3 and k % int(RADAR_SCAN / DT) == 0 and gnss_valid:
            cm_alarm, ais_alarm, mar_ok, mar_common, mar_n, ft = l3.scan(
                k, t, truth[k], kf.x, vessels, vves, n_vessels,
                attacks, attacking, ais_off, psi_bias, rng,
                k >= k_on, rec and k < k_on, cmrec, sprec,
                mag_heading=imu_iso, mag_off=mag_off)
            false_tracks += ft

        # L4 trust + isolation
        if use_L4:
            l4.update_trust(l2_alarm, cm_alarm)
            if use_imu_mon:
                l4.update_imu_trust(imu_alarm, t)

        # maritime absolute-position fix
        if config == 'F' and use_L4 and l4.isolated and mar_ok:
            kf.update_pos(kf.x[:2] + mar_common, SIGMA_R**2 / mar_n)

        # Fusion decision
        fuse(kf, z, config, gnss_valid, use_filter, l2_alarm, thr_alarm,
             l4.isolated if use_L4 else False,
             l4.tau if use_L4 else 1.0,
             use_L4, l2det.consec_gate)

        # Alarm aggregation (navigation alarms drive the state machine; L1
        # rejections are reported as link-attack alarms)
        nav_alarm = (l2_alarm if config in ('B1',) else
                     thr_alarm if config == 'B2' else
                     (l2_alarm or cm_alarm or ais_alarm or imu_alarm) if config == 'F' else False)
        alarm = nav_alarm or (link_alarm and config == 'F')
        if alarm:
            if t_on <= t <= t_on + DET_WINDOW and alarm_t is None:
                alarm_t = t
                alarm_src = ('L2-GNSS' if l2_alarm else 'L2-IMU' if imu_alarm else
                             'L3-CM' if cm_alarm else 'L3-AIS' if ais_alarm else
                             'L1' if link_alarm else 'B2' if thr_alarm else 'other')
            if t < t_on:
                false_before = True
        if imu_alarm and imu_alarm_t is None and t >= t_on:
            imu_alarm_t = t

        # L4 state machine
        if use_L4:
            l4.update_state(nav_alarm, kf, t)

        xhat_hist[k] = kf.x[:2]; gnss_hist[k] = z[:2]
        if k_on <= k < k_eval:
            errs.append(np.linalg.norm(kf.x[:2] - truth[k, :2]))

    # --- Post-run metrics ---
    errs = np.array(errs) if errs else np.array([0.0])
    detected = alarm_t is not None
    latency = (alarm_t - t_on) if detected else np.nan
    rmse = float(np.sqrt(np.mean(errs**2)))
    maxerr = float(np.max(errs))
    recov = np.nan
    if detected:
        post = np.linalg.norm(xhat_hist[k_on:k_eval] - truth[k_on:k_eval, :2], axis=1)
        after = int((alarm_t - t_on) / DT)
        bel = np.where(post < RECOVER_TOL)[0]
        bel = bel[bel >= after]
        if len(bel):
            recov = bel[0] * DT
    completion = maxerr < GEOFENCE
    xtrack = float(np.max(route_distance(truth[k_on:k_eval, :2]))) if k_eval > k_on else 0.0

    out = dict(detected=detected, latency=latency, rmse=rmse, maxerr=maxerr,
               recov=recov, completion=completion, false_before=false_before,
               false_tracks=false_tracks,
               peak_cusum=l2det.peak_cusum, peak_cm=l3.peak_cm,
               aborted=l4.aborted,
               alarm_src=alarm_src,
               imu_detected=(imu_alarm_t is not None and imu_alarm_t <= t_on + DET_WINDOW),
               imu_latency=((imu_alarm_t - t_on) if (imu_alarm_t is not None
                            and imu_alarm_t <= t_on + DET_WINDOW) else np.nan),
               imu_isolated=l4.imu_iso_t is not None,
               xtrack_max=xtrack, corridor_ok=xtrack < GEOFENCE)
    out.update(link.summary())
    if rec:
        out['d2'] = d2rec; out['feats'] = feats
        out['cmrec'] = cmrec; out['sprec'] = sprec
        out['imu_a'] = imu_a_rec; out['imu_h'] = imu_h_rec
    if trace:
        out.update(truth=truth[:, :2].copy(), xhat=xhat_hist, gnss=gnss_hist,
                   t_on=t_on, alarm_t=alarm_t, k_on=k_on)
    return out
