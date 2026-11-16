"""Checkpointed version of maritime_uav.cli.main(): identical experiments, seeds
and summaries, but each unit is saved to parts/<unit>.json so the run can be
resumed across processes.  Usage:
    python3 run_parts.py run  <time_budget_s>
    python3 run_parts.py assemble <out.json>
"""
import json, os, sys, time
import numpy as np
from scipy.stats import wilcoxon
import maritime_uav as S
from maritime_uav.evaluation import summarize, link_summary

N, DUR = 100, 260.0
PART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parts')
os.makedirs(PART, exist_ok=True)

DET_ATT = {'A1': ['A1'], 'A2n': ['A2n'], 'A2c': ['A2c'], 'A3': ['A3'], 'A5': ['A5'],
           'A4': ['A4'], 'A4s': ['A4s']}
DET_CFG = {'B1': ('B1', None), 'B2': ('B2', None), 'L2': ('F', 'noL3'), 'F': ('F', None)}
SCEN = {'S1': ['A1', 'A3'], 'S2': ['A5', 'A6'], 'S3': ['A2c', 'A7'], 'S4': ['A1', 'A4']}
IMU_SCEN = {'A4': ['A4'], 'A4s': ['A4s'], 'A3+A4': ['A3', 'A4']}
IMU_CFG = {'B0': ('B0', None), 'B1': ('B1', None), 'B2': ('B2', None),
           'F-noIMU': ('F', 'noIMU'), 'F': ('F', None)}
LINK_ATT = {'A7j': ['A7j'], 'A7r': ['A7r'], 'A7i': ['A7i']}
LINK_CFG = {'B1': ('B1', None), 'F-noL1': ('F', 'noL1'), 'F': ('F', None)}
ABLS = {'Full framework': None, '$-$ L1 (signing, validation)': 'noL1',
        '$-$ cross-sensor (L3)': 'noL3', '$-$ trust adaptation (L4)': 'noL4'}

UNITS = ([f'det_{a}' for a in DET_ATT] + ['far'] + [f'mis_{s}' for s in SCEN]
         + [f'imu_{s}' for s in IMU_SCEN] + [f'link_{a}' for a in LINK_ATT]
         + ['ablation', 'overhead', 'robust_v', 'robust_pd', 'calibgen', 'roc', 'traj'])

_state = {}


def setup(need_if):
    if 'cal' not in _state:
        _state['cal'] = S.calibrate(DUR)
    if need_if and 'if' not in _state:
        _state['if'] = S.train_if(DUR)


def det_entry(r):
    lm, lh = S.ci95([x['latency'] for x in r])
    dd = [x['detected'] for x in r]
    return dict(dr=float(np.mean(dd)), dr_ci=S.binom_ci(int(np.sum(dd)), len(dd)), lat=lm, lat_ci=lh)


def mis_entry(r):
    rm, rh = S.ci95([x['rmse'] for x in r]); mm, mh = S.ci95([x['maxerr'] for x in r])
    vm, vh = S.ci95([x['recov'] for x in r]); comp = [x['completion'] for x in r]
    return dict(rmse=rm, rmse_ci=rh, maxerr=mm, maxerr_ci=mh, recov=vm, recov_ci=vh,
                completion=float(np.mean(comp) * 100),
                completion_ci=S.binom_ci(int(np.sum(comp)), len(comp)) * 100,
                safe_abort=float(np.mean([x['aborted'] for x in r]) * 100))


def vec(r, *keys):
    return {k: [None if (isinstance(x[k], float) and np.isnan(x[k])) else
                (bool(x[k]) if isinstance(x[k], (bool, np.bool_)) else x[k]) for x in r]
            for k in keys}


def run_unit(u):
    out = {}
    if u.startswith('det_'):
        an = u[4:]; ac = DET_ATT[an]
        setup(True); model, thr = _state['if']
        for cn, (cfg, abl) in DET_CFG.items():
            r = S.mc(ac, cfg, N, 1000, abl, DUR)
            e = det_entry(r)
            if an in ('A4', 'A4s'):
                im, ih = S.ci95([x['imu_latency'] for x in r])
                e.update(imu_dr=float(np.mean([x['imu_detected'] for x in r])),
                         imu_lat=im, imu_lat_ci=ih,
                         first_alarm_src=summarize(r)['first_alarm_src'])
            out[cn] = e
            out['_vec_' + cn] = vec(r, 'detected')
        ir = S.if_eval(model, thr, ac, N, 1000, DUR)
        dd = [d for d, _, _ in ir]; lm, lh = S.ci95([l for _, l, _ in ir])
        out['IF'] = dict(dr=float(np.mean(dd)), dr_ci=S.binom_ci(int(np.sum(dd)), len(dd)), lat=lm, lat_ci=lh)
        out['_vec_IF'] = {'detected': [bool(d) for d in dd]}
    elif u == 'far':
        setup(True); model, thr = _state['if']
        for cn, (cfg, abl) in DET_CFG.items():
            r = S.mc([], cfg, N, 7000, abl, DUR)
            out[cn] = float(np.mean([x['false_before'] for x in r]) / (DUR / 3600.0))
            out['_imu_' + cn] = float(np.mean([x['imu_detected'] for x in r]))
        ir = S.if_eval(model, thr, [], N, 7000, DUR)
        out['IF'] = float(np.mean([fa for _, _, fa in ir]) / (DUR / 3600.0))
    elif u.startswith('mis_'):
        setup(False); sn = u[4:]
        cfgs = [('B0', 'B0', None), ('B1', 'B1', None), ('B2', 'B2', None), ('F', 'F', None)]
        if sn == 'S4':
            cfgs.append(('F-noIMU', 'F', 'noIMU'))
        for cn, cfg, abl in cfgs:
            r = S.mc(SCEN[sn], cfg, N, 2000, abl, DUR)
            out[cn] = mis_entry(r)
            out['_vec_' + cn] = vec(r, 'rmse', 'completion', 'imu_detected', 'imu_latency')
    elif u.startswith('imu_'):
        setup(False); sn = u[4:]
        for cn, (cfg, abl) in IMU_CFG.items():
            r = S.mc(IMU_SCEN[sn], cfg, N, 4000, abl, DUR)
            out[cn] = summarize(r)
            out['_vec_' + cn] = vec(r, 'rmse', 'completion')
    elif u.startswith('link_'):
        setup(False); an = u[5:]
        for cn, (cfg, abl) in LINK_CFG.items():
            r = S.mc(LINK_ATT[an], cfg, N, 5000, abl, DUR)
            out[cn] = link_summary(r)
            out['_vec_' + cn] = vec(r, 'hijacked', 'link_abort')
    elif u == 'ablation':
        setup(False)
        for lab, abl in ABLS.items():
            r = S.mc(['A1', 'A2c', 'A3'], 'F', N, 3000, abl, DUR)
            lm, _ = S.ci95([x['latency'] for x in r]); rm, _ = S.ci95([x['rmse'] for x in r])
            out[lab] = dict(dr=float(np.mean([x['detected'] for x in r]) * 100), lat=lm, rmse=rm,
                            comp=float(np.mean([x['completion'] for x in r]) * 100))
    elif u == 'overhead':
        setup(False)
        out = S.overhead_bench()
    elif u == 'robust_v':            # = evaluation.robustness, vessel-count half
        setup(False)
        from maritime_uav.config import CFG
        base_v = CFG.n_vessels
        for nv in [2, 3, 4, 6, 8]:
            CFG.n_vessels = nv
            r = [S.run_once(['A2c'], 'F', 80000 + i, DUR) for i in range(N)]
            out[str(nv)] = float(np.mean([x['detected'] for x in r]))
        CFG.n_vessels = base_v
    elif u == 'robust_pd':           # = evaluation.robustness, radar-Pd half
        setup(False)
        from maritime_uav.config import CFG
        base_pd = CFG.pd_radar
        for pd in [0.5, 0.7, 0.9]:
            CFG.pd_radar = pd
            r = [S.run_once(['A2c'], 'F', 81000 + i, DUR) for i in range(N)]
            out[f'{pd:.1f}'] = float(np.mean([x['detected'] for x in r]))
        CFG.pd_radar = base_pd
    elif u == 'calibgen':
        setup(False)
        out = S.calib_generalization(N, DUR)
    elif u == 'roc':
        setup(False)
        out = S.roc_bench(N, DUR)
    elif u == 'traj':
        setup(False)
        out = S.sample_trajectory(DUR)
    return out


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


EST = {'far': 135, 'ablation': 160, 'overhead': 15, 'robust_v': 200, 'robust_pd': 120, 'calibgen': 125,
       'roc': 60, 'traj': 5}


def est(u):
    if u in EST:
        return EST[u]
    return {'det': 160, 'mis': 115, 'imu': 158, 'link': 115}[u.split('_')[0]] + (20 if u == 'mis_S4' else 0)


def run(budget):
    t0 = time.perf_counter()
    for u in UNITS:
        f = os.path.join(PART, u + '.json')
        if os.path.exists(f):
            continue
        setup_cost = 0 if 'cal' in _state else 25
        if time.perf_counter() - t0 + setup_cost + est(u) > budget:
            continue
        t1 = time.perf_counter()
        out = run_unit(u)
        json.dump(clean(out), open(f + '.tmp', 'w'))
        os.replace(f + '.tmp', f)
        print(f'{u} done in {time.perf_counter()-t1:.0f}s (total {time.perf_counter()-t0:.0f}s)', flush=True)
    left = [u for u in UNITS if not os.path.exists(os.path.join(PART, u + '.json'))]
    print('remaining:', left)
    if 'cal' in _state:
        json.dump(clean(_state['cal']), open(os.path.join(PART, '_calibration.json'), 'w'))


def assemble(dst):
    P = {u: json.load(open(os.path.join(PART, u + '.json'))) for u in UNITS}
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith('_')}
    out = {'N': N, 'duration': DUR,
           'calibration': json.load(open(os.path.join(PART, '_calibration.json')))}
    out['detection'] = {a: strip(P[f'det_{a}']) for a in DET_ATT}
    out['far'] = strip(P['far'])
    out['far_imu_alarm_rate'] = {k[5:]: v for k, v in P['far'].items() if k.startswith('_imu_')}
    out['mission'] = {s: strip(P[f'mis_{s}']) for s in SCEN}
    out['imu'] = {s: strip(P[f'imu_{s}']) for s in IMU_SCEN}
    out['link'] = {a: strip(P[f'link_{a}']) for a in LINK_ATT}
    out['ablation'] = P['ablation']; out['overhead'] = P['overhead']
    out['robustness'] = {'vessels': P['robust_v'], 'radar_pd': P['robust_pd']}; out['calib_gen'] = P['calibgen']
    out['roc'] = P['roc']; out['traj'] = P['traj']
    s4F = P['mis_S4']['_vec_F']
    lat = [x for x in s4F['imu_latency'] if x is not None]
    out['S4_F_imu'] = dict(imu_dr=float(np.mean(s4F['imu_detected'])),
                           imu_lat=float(np.mean(lat)) if lat else None)
    V = lambda u, c, k: [np.nan if x is None else x for x in P[u]['_vec_' + c][k]]
    mc_ = S.mcnemar_p
    out['stats'] = {
        'A2c_F_vs_L2_mcnemar_p': mc_(V('det_A2c', 'F', 'detected'), V('det_A2c', 'L2', 'detected')),
        'A2c_F_vs_IF_mcnemar_p': mc_(V('det_A2c', 'F', 'detected'), V('det_A2c', 'IF', 'detected')),
        'S1_F_vs_B1_rmse_wilcoxon_p': float(wilcoxon(V('mis_S1', 'F', 'rmse'), V('mis_S1', 'B1', 'rmse')).pvalue),
        'A4_F_vs_B1_mcnemar_p': mc_(V('det_A4', 'F', 'detected'), V('det_A4', 'B1', 'detected')),
        'A4s_F_vs_B1_mcnemar_p': mc_(V('det_A4s', 'F', 'detected'), V('det_A4s', 'B1', 'detected')),
        'S4_F_vs_FnoIMU_rmse_wilcoxon_p': float(wilcoxon(V('mis_S4', 'F', 'rmse'), V('mis_S4', 'F-noIMU', 'rmse')).pvalue),
        'S4_F_vs_FnoIMU_integrity_mcnemar_p': mc_(V('mis_S4', 'F', 'completion'), V('mis_S4', 'F-noIMU', 'completion')),
        'A3A4_F_vs_B1_rmse_wilcoxon_p': float(wilcoxon(V('imu_A3+A4', 'F', 'rmse'), V('imu_A3+A4', 'B1', 'rmse')).pvalue),
        'A4_F_vs_B1_rmse_wilcoxon_p': float(wilcoxon(V('imu_A4', 'F', 'rmse'), V('imu_A4', 'B1', 'rmse')).pvalue),
        'A4s_F_vs_B1_integrity_mcnemar_p': mc_(V('imu_A4s', 'F', 'completion'), V('imu_A4s', 'B1', 'completion')),
        'A7r_F_vs_FnoL1_hijack_mcnemar_p': mc_(V('link_A7r', 'F', 'hijacked'), V('link_A7r', 'F-noL1', 'hijacked')),
        'A7i_F_vs_FnoL1_hijack_mcnemar_p': mc_(V('link_A7i', 'F', 'hijacked'), V('link_A7i', 'F-noL1', 'hijacked')),
        'A7j_F_vs_B1_abort_mcnemar_p': mc_(V('link_A7j', 'F', 'link_abort'), V('link_A7j', 'B1', 'link_abort')),
    }
    json.dump(clean(out), open(dst, 'w'), indent=2)
    print('assembled ->', dst)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(float(sys.argv[2]))
    else:
        assemble(sys.argv[2])
