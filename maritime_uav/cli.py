"""
Command-line interface and experiment orchestration for SIFIS simulation.
"""
import argparse, json, time, sys
import numpy as np
from scipy.stats import wilcoxon

from .config import CFG
from .calibration import calibrate, train_if, if_eval
from .evaluation import (ci95, mc, binom_ci, robustness,
                         calib_generalization, mcnemar_p, overhead_bench,
                         summarize, link_summary, far_summary, holm,
                         roc_bench, sample_trajectory)


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo simulation for maritime UAV sensor-security framework.")
    ap.add_argument('--N', type=int, default=100, help="Number of Monte Carlo trials per scenario.")
    ap.add_argument('--out', default='results.json', help="Output JSON results path.")
    ap.add_argument('--roc-N', type=int, default=None, help="Number of Monte Carlo trials for the ROC figure (defaults to N).")
    ap.add_argument('--duration', type=float, default=260.0, help="Simulation duration (seconds).")
    args = ap.parse_args()

    N = args.N
    dur = args.duration
    t0 = time.perf_counter()

    out = {'N': N, 'duration': dur}
    out['calibration'] = calibrate(dur)
    if_model, if_thr = train_if(dur)
    print(f"calibrated + trained IF  ({time.perf_counter()-t0:.0f}s)", file=sys.stderr)

    det_attacks = {'A1': ['A1'], 'A2n': ['A2n'], 'A2c': ['A2c'], 'A3': ['A3'], 'A5': ['A5'],
                   'A4': ['A4'], 'A4s': ['A4s']}
    det_cfg = {'B1': ('B1', None), 'B2': ('B2', None), 'L2': ('F', 'noL3'), 'F': ('F', None)}
    out['detection'] = {}
    det_runs = {}

    for an, ac in det_attacks.items():
        out['detection'][an] = {}
        det_runs[an] = {}
        for cn, (cfg, abl) in det_cfg.items():
            r = mc(ac, cfg, N, 1000, abl, dur)
            det_runs[an][cn] = r
            lm, lh = ci95([x['latency'] for x in r])
            dd = [x['detected'] for x in r]
            out['detection'][an][cn] = dict(
                dr=float(np.mean(dd)),
                dr_ci=binom_ci(int(np.sum(dd)), len(dd)),
                lat=lm,
                lat_ci=lh
            )
            if an in ('A4', 'A4s'):
                im, ih = ci95([x['imu_latency'] for x in r])
                out['detection'][an][cn].update(
                    imu_dr=float(np.mean([x['imu_detected'] for x in r])),
                    imu_lat=im, imu_lat_ci=ih,
                    first_alarm_src=summarize(r)['first_alarm_src'])

        # learned-detector (isolation forest) column
        ir = if_eval(if_model, if_thr, ac, N, 1000, dur)
        det_runs[an]['IF'] = ir
        dd = [x['detected'] for x in ir]
        lm, lh = ci95([x['latency'] for x in ir])
        out['detection'][an]['IF'] = dict(
            dr=float(np.mean(dd)),
            dr_ci=binom_ci(int(np.sum(dd)), len(dd)),
            lat=lm,
            lat_ci=lh
        )
        print(f"det {an} {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    # attack-free false alarms: alarm onsets per hour over the whole run
    out['far'] = {}
    for cn, (cfg, abl) in det_cfg.items():
        r = mc([], cfg, N, 7000, abl, dur)
        out['far'][cn] = far_summary(r, dur)
    ir = if_eval(if_model, if_thr, [], N, 7000, dur)
    out['far']['IF'] = far_summary(ir, dur)

    scen = {'S1': ['A1', 'A3'], 'S2': ['A5', 'A6'], 'S3': ['A2c', 'A7'], 'S4': ['A1', 'A4']}
    out['mission'] = {}
    mis_runs = {}

    def mission_row(r):
        s_ = summarize(r)
        comp = [x['completion'] for x in r]
        return dict(rmse=s_['rmse'], rmse_ci=s_['rmse_ci'],
                    maxerr=s_['maxerr'], maxerr_ci=s_['maxerr_ci'],
                    recov=s_['recov'], recov_ci=s_['recov_ci'],
                    completion=float(np.mean(comp) * 100),
                    completion_ci=binom_ci(int(np.sum(comp)), len(comp)) * 100,
                    corridor=s_['corridor_ok'] * 100, corridor_ci=s_['corridor_ok_ci'] * 100,
                    xtrack=s_['xtrack_max'],
                    safe_abort=s_['aborted'] * 100,
                    imu_dr=s_['imu_detected'], imu_lat=s_['imu_latency'])

    for sn, ac in scen.items():
        out['mission'][sn] = {}
        mis_runs[sn] = {}
        for cfg in ['B0', 'B1', 'B2', 'F']:
            r = mc(ac, cfg, N, 2000, None, dur)
            mis_runs[sn][cfg] = r
            out['mission'][sn][cfg] = mission_row(r)
        print(f"mis {sn} {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    # S4 with the IMU monitor removed (ablation): isolates its contribution
    r = mc(scen['S4'], 'F', N, 2000, 'noIMU', dur)
    mis_runs['S4']['F-noIMU'] = r
    out['mission']['S4']['F-noIMU'] = mission_row(r)

    # ---- IMU attack (A4) in isolation and during GNSS jamming --------------
    imu_scen = {'A4': ['A4'], 'A4s': ['A4s'], 'A3+A4': ['A3', 'A4']}
    imu_cfg = {'B0': ('B0', None), 'B1': ('B1', None), 'B2': ('B2', None),
               'F-noIMU': ('F', 'noIMU'), 'F': ('F', None)}
    out['imu'] = {}; imu_runs = {}
    for sn, ac in imu_scen.items():
        out['imu'][sn] = {}; imu_runs[sn] = {}
        for cn, (cfg, abl) in imu_cfg.items():
            r = mc(ac, cfg, N, 4000, abl, dur)
            imu_runs[sn][cn] = r
            out['imu'][sn][cn] = summarize(r)
        print(f"imu {sn} {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    # ---- Command-link attacks (A7 on the C2 uplink) and Layer 1 -------------
    link_att = {'A7j': ['A7j'], 'A7r': ['A7r'], 'A7i': ['A7i']}
    link_cfg = {'B1': ('B1', None), 'F-noL1': ('F', 'noL1'), 'F': ('F', None)}
    out['link'] = {}; link_runs = {}
    for an, ac in link_att.items():
        out['link'][an] = {}; link_runs[an] = {}
        for cn, (cfg, abl) in link_cfg.items():
            r = mc(ac, cfg, N, 5000, abl, dur)
            link_runs[an][cn] = r
            out['link'][an][cn] = link_summary(r)
        print(f"link {an} {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    abls = {
        'Full framework': None,
        '$-$ L1 (signing, validation)': 'noL1',
        '$-$ cross-sensor (L3)': 'noL3',
        '$-$ trust adaptation (L4)': 'noL4'
    }
    out['ablation'] = {}
    for lab, abl in abls.items():
        r = mc(['A1', 'A2c', 'A3'], 'F', N, 3000, abl, dur)
        lm, _ = ci95([x['latency'] for x in r])
        rm, _ = ci95([x['rmse'] for x in r])
        out['ablation'][lab] = dict(
            dr=float(np.mean([x['detected'] for x in r]) * 100),
            lat=lm,
            rmse=rm,
            comp=float(np.mean([x['completion'] for x in r]) * 100),
            corridor=float(np.mean([x['corridor_ok'] for x in r]) * 100)
        )

    out['overhead'] = overhead_bench()
    out['robustness'] = robustness(N, dur)
    out['calib_gen'] = calib_generalization(N, dur)
    print(f"robustness+calibgen {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    # figure data (fig_roc.pdf, fig_traj.pdf) consumed by scripts/make_figs.py
    n_roc = args.roc_N if args.roc_N is not None else N
    out['roc'] = roc_bench(n_roc, dur)
    out['traj'] = sample_trajectory(dur)
    print(f"roc+traj {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    # significance tests for the headline comparisons
    a2c = det_runs['A2c']
    s1 = mis_runs['S1']
    out['stats'] = {
        'A2c_F_vs_L2_mcnemar_p': mcnemar_p([x['detected'] for x in a2c['F']],
                                           [x['detected'] for x in a2c['L2']]),
        'A2c_F_vs_IF_mcnemar_p': mcnemar_p([x['detected'] for x in a2c['F']],
                                           [x['detected'] for x in a2c['IF']]),
        'S1_F_vs_B1_rmse_wilcoxon_p': float(wilcoxon([x['rmse'] for x in s1['F']],
                                                     [x['rmse'] for x in s1['B1']]).pvalue),
        'A4_F_vs_B1_mcnemar_p': mcnemar_p([x['detected'] for x in det_runs['A4']['F']],
                                          [x['detected'] for x in det_runs['A4']['B1']]),
        'A4s_F_vs_B1_mcnemar_p': mcnemar_p([x['detected'] for x in det_runs['A4s']['F']],
                                           [x['detected'] for x in det_runs['A4s']['B1']]),
        'S4_F_vs_FnoIMU_rmse_wilcoxon_p': float(wilcoxon(
            [x['rmse'] for x in mis_runs['S4']['F']],
            [x['rmse'] for x in mis_runs['S4']['F-noIMU']]).pvalue),
        'S4_F_vs_FnoIMU_integrity_mcnemar_p': mcnemar_p(
            [x['completion'] for x in mis_runs['S4']['F']],
            [x['completion'] for x in mis_runs['S4']['F-noIMU']]),
        'A3A4_F_vs_B1_rmse_wilcoxon_p': float(wilcoxon(
            [x['rmse'] for x in imu_runs['A3+A4']['F']],
            [x['rmse'] for x in imu_runs['A3+A4']['B1']]).pvalue),
        'A4_F_vs_B1_rmse_wilcoxon_p': float(wilcoxon(
            [x['rmse'] for x in imu_runs['A4']['F']],
            [x['rmse'] for x in imu_runs['A4']['B1']]).pvalue),
        'A4s_F_vs_B1_integrity_mcnemar_p': mcnemar_p(
            [x['completion'] for x in imu_runs['A4s']['F']],
            [x['completion'] for x in imu_runs['A4s']['B1']]),
        'A7r_F_vs_FnoL1_hijack_mcnemar_p': mcnemar_p(
            [x['hijacked'] for x in link_runs['A7r']['F']],
            [x['hijacked'] for x in link_runs['A7r']['F-noL1']]),
        'A7i_F_vs_FnoL1_hijack_mcnemar_p': mcnemar_p(
            [x['hijacked'] for x in link_runs['A7i']['F']],
            [x['hijacked'] for x in link_runs['A7i']['F-noL1']]),
        'A7j_F_vs_B1_abort_mcnemar_p': mcnemar_p(
            [x['link_abort'] for x in link_runs['A7j']['F']],
            [x['link_abort'] for x in link_runs['A7j']['B1']]),
    }

    # Holm-Bonferroni over the comparisons with a stochastic outcome
    tested = {k: v for k, v in out['stats'].items() if 'hijack' not in k}
    out['stats_holm'] = holm(tested)
    out['stats_note'] = ('hijack outcomes are deterministic by construction and are '
                         'excluded from the Holm-Bonferroni family')

    def sanitize_nan(val):
        if isinstance(val, dict):
            return {k: sanitize_nan(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [sanitize_nan(v) for v in val]
        elif isinstance(val, tuple):
            return [sanitize_nan(v) for v in val]
        elif isinstance(val, (float, np.floating)) and (np.isnan(val) or val != val):
            return None
        elif isinstance(val, np.floating):
            return float(val)
        elif isinstance(val, (np.bool_,)):
            return bool(val)
        elif isinstance(val, np.integer):
            return int(val)
        return val

    out = sanitize_nan(out)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"TOTAL {time.perf_counter()-t0:.0f}s -> {args.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
