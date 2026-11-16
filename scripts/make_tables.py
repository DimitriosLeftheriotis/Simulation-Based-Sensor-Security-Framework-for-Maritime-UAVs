#!/usr/bin/env python3
"""Read results.json and generate the LaTeX table bodies of the manuscript (v5),
plus key_numbers.txt with the figures quoted in the text.  Each table body is
written to its own .txt file and all of them to all_tables.txt."""
import argparse
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / 'results.json'
DEFAULT_OUT_DIR = REPO_ROOT / 'artifacts' / 'tables'
COLS = ['B1', 'B2', 'L2', 'F', 'IF']
DET_ROWS = [('A1', 'A1 jump'), ('A2n', 'A2 drift, naive'), ('A2c', 'A2 drift, consistent'),
            ('A3', 'A3 jamming'), ('A5', 'A5 AIS position offset'),
            ('A4', 'A4 IMU bias'), ('A4s', 'A4 IMU scale')]
SNAMES = {'S1': 'S1 Coastal defence', 'S2': 'S2 Naval surveillance',
          'S3': 'S3 Border patrol', 'S4': 'S4 Offshore asset'}


def f(x, nd=1):
    return '--' if x is None or x != x else f'{x:.{nd}f}'


def wilson_half(k, n, z=1.96):
    """95% Wilson half-width (same formula as evaluation.binom_ci)."""
    if n == 0:
        return 0.0
    p = k / n
    return float(z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / (1 + z * z / n))


def cfg_label(c):
    return {'F-noIMU': 'F$-$IMU mon.', 'F-noL1': 'F$-$L1', 'B1': 'B1 (unsigned)'}.get(c, c)


def build_tab_detection(d):
    out = ['% ===== tab:detection (rate / latency s); columns B1 B2 L2 F IF =====']
    for a, lab in DET_ROWS:
        if a in d['detection']:
            cells = [f"${d['detection'][a][c]['dr']:.2f}$ / ${f(d['detection'][a][c]['lat'])}$" for c in COLS]
            out.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    far = d['far']
    out.append(r'\midrule')
    out.append(r'False-alarm rate (h$^{-1}$) & ' + ' & '.join(f"${far[c]['rate']:.1f}$" for c in COLS) + r'\\')
    out.append(f"Runs with an alarm (of {far['F']['runs']}) & " + ' & '.join(f"${far[c]['runs_with_alarm']}$" for c in COLS) + r'\\')
    return '\n'.join(out) + '\n'


def build_tab_mission(d):
    out = ['% ===== tab:mission (RMSE, max, recov, integrity %, corridor %, contingency %) =====']
    for s, lab in SNAMES.items():
        cfgs = [c for c in ['B0', 'B1', 'B2', 'F', 'F-noIMU'] if c in d['mission'][s]]
        for i, c in enumerate(cfgs):
            v = d['mission'][s][c]
            head = f"\\multirow{{{len(cfgs)}}}{{*}}{{{lab}}}" if i == 0 else ''
            cl = 'F$-$IMU mon.' if c == 'F-noIMU' else c
            cont = '--' if c.startswith('B') else f"${v['safe_abort']:.0f}$"
            out.append(f"{head} & {cl} & ${f(v['rmse'])}$ & ${f(v['maxerr'])}$ & ${f(v['recov'])}$ & "
                       f"${v['completion']:.0f}$ & ${f(v.get('corridor'), 0)}$ & {cont}" + r'\\')
        out.append(r'\midrule')
    return '\n'.join(out[:-1]) + '\n'


def build_tab_imu(d):
    out = ['% ===== tab:imu (alarm %, IMU attribution % / s, RMSE, max, integrity %, corridor %, contingency %) =====']
    for s, lab in [('A4', 'A4 bias'), ('A4s', 'A4 scale'), ('A3+A4', 'A3 + A4')]:
        for i, c in enumerate(['B0', 'B1', 'B2', 'F-noIMU', 'F']):
            v = d['imu'][s][c]
            head = f"\\multirow{{5}}{{*}}{{{lab}}}" if i == 0 else ''
            alarm = '--' if c == 'B0' else f"${v['detected'] * 100:.0f}$"
            attr = f"${v['imu_detected'] * 100:.0f}$\\,/\\,${f(v['imu_latency'])}$" if c == 'F' else '--'
            cont = '--' if c.startswith('B') else f"${v['aborted'] * 100:.0f}$"
            out.append(f"{head} & {cfg_label(c) if c != 'B1' else 'B1'} & {alarm} & {attr} & ${f(v['rmse'])}$ & "
                       f"${f(v['maxerr'])}$ & ${v['completion'] * 100:.0f}$ & ${v['corridor_ok'] * 100:.0f}$ & {cont}" + r'\\')
        out.append(r'\midrule')
    return '\n'.join(out[:-1]) + '\n'


def build_tab_link(d):
    out = ['% ===== tab:link (hijack %, rejected/run, breach %, max cross-track m, link-loss abort %) =====']
    for a, lab in [('A7j', 'A7 jamming'), ('A7r', 'A7 replay'), ('A7i', 'A7 injection')]:
        for i, c in enumerate(['B1', 'F-noL1', 'F']):
            v = d['link'][a][c]
            head = f"\\multirow{{3}}{{*}}{{{lab}}}" if i == 0 else ''
            rej = f"${sum(v['rejected_mean'].values()):.1f}$" if c == 'F' else '--'
            out.append(f"{head} & {cfg_label(c)} & ${v['hijacked'] * 100:.0f}$ & {rej} & "
                       f"${(1 - v['corridor_ok']) * 100:.0f}$ & ${f(v['xtrack_max'])}$ & ${v['link_abort'] * 100:.0f}$" + r'\\')
        out.append(r'\midrule')
    return '\n'.join(out[:-1]) + '\n'


def build_tab_ablation(d):
    out = ['% ===== tab:ablation (combined A1 + A2c + A3; detection %, latency s, RMSE m, integrity %, corridor %) =====']
    for lab, v in d['ablation'].items():
        out.append(f"{lab} & ${f(v['dr'], 0)}$ & ${f(v['lat'])}$ & ${f(v['rmse'])}$ & ${f(v['comp'], 0)}$ & "
                   f"${f(v.get('corridor'), 0)}$" + r'\\')
    return '\n'.join(out) + '\n'


def build_tab_overhead(d):
    o = d['overhead']
    rows = [('L1 signing', o['L1'], '--', f(o['L1'], 3)),
            ('L1 in-loop verification (signature, replay window)', o.get('L1_verify'), '--', f(o.get('L1_verify'), 3)),
            ('L2 filter and innovation tests', o['L2'], f(o['mem_mb'], 3), '--'),
            ('L2 IMU monitor', o.get('L2_imu'), '--', '--'),
            (f"L3 cross-sensor scan, amortised (per scan {f(o.get('L3_scan'), 3)}~ms)", o['L3'], '--', '--'),
            ('L4 trust and state machine', o['L4'], '--', '--')]
    out = ['% ===== tab:overhead (CPU ms per 10 Hz cycle, memory MB, added latency ms) =====']
    out += [f"{n} & ${f(c, 3)}$ & {m} & {l}" + r'\\' for n, c, m, l in rows]
    out.append(r'\midrule')
    out.append(f"Total & ${f(o['total'], 3)}$ (${f(o.get('total_ext'), 3)}$) & $<0.01$ & ${f(o['L1'], 3)}$" + r'\\')
    return '\n'.join(out) + '\n'


def build_tab_robustness(d):
    n = d['N']
    rv, rp = d['robustness']['vessels'], d['robustness']['radar_pd']
    out = ['% ===== tab:robustness (A2c detection by F; Wilson 95% half-widths) =====',
           'Vessels in the scene & ' + ' & '.join(f'${k}$' for k in rv) + r'\\',
           'Detection rate & ' + ' & '.join(f'${v:.2f}$' for v in rv.values()) + r'\\',
           r'95\% CI ($\pm$) & ' + ' & '.join(f'${wilson_half(round(v * n), n):.2f}$' for v in rv.values()) + r'\\',
           r'\midrule',
           'Radar $P_d$ & ' + ' & '.join(f'${k}$' for k in rp) + r'\\',
           'Detection rate & ' + ' & '.join(f'${v:.2f}$' for v in rp.values()) + r'\\',
           r'95\% CI ($\pm$) & ' + ' & '.join(f'${wilson_half(round(v * n), n):.2f}$' for v in rp.values()) + r'\\']
    return '\n'.join(out) + '\n'


def auc_pair(roc):
    from sklearn.metrics import roc_auc_score
    res = {}
    for name, a, b in [('L2 (peak CUSUM)', 'innov_att', 'innov_fre'), ('L3 (peak common mode)', 'mar_att', 'mar_fre')]:
        y = np.r_[np.ones(len(roc[a])), np.zeros(len(roc[b]))]
        res[name] = float(roc_auc_score(y, np.r_[roc[a], roc[b]]))
    return res


def build_stats(d):
    out = ['% ===== significance and text figures =====']
    holm = d.get('stats_holm', {})
    for k, p in d['stats'].items():
        tag = f"  Holm-adjusted {holm[k]:.3g}" if k in holm else '  (deterministic, not tested)'
        out.append(f"% {k}: p = {p:.3g}{tag}")
    if holm:
        out.append(f"% largest Holm-adjusted p = {max(holm.values()):.4f} over {len(holm)} tests")
    for name, a in auc_pair(d['roc']).items():
        out.append(f"% ROC AUC {name} = {a:.3f}")
    for lab, v in d['calib_gen'].items():
        out.append(f"% calibration sensitivity {lab}: {v['rate']:.2f} /h ({v['events']} alarm onsets in "
                   f"{v['hours']:.2f} h; 95% {v['rate_ci'][0]:.2f}-{v['rate_ci'][1]:.2f}), "
                   f"{v['runs_with_alarm']}/{v['runs']} runs with an alarm")
    return '\n'.join(out) + '\n'


def build_key_numbers(d):
    det, mis, imu, lnk, far, ov = d['detection'], d['mission'], d['imu'], d['link'], d['far'], d['overhead']
    L = ['Key numbers quoted in the manuscript text (regenerate after every run)', '']
    L.append('Detection (rate, mean latency s):')
    for a, lab in DET_ROWS:
        L.append(f"  {lab:24s} " + '  '.join(f"{c}={det[a][c]['dr']:.2f}/{f(det[a][c]['lat'])}" for c in COLS))
    L.append('  A4/A4s IMU attribution (F): ' + ', '.join(
        f"{a}: {det[a]['F']['imu_dr']:.2f} / {f(det[a]['F']['imu_lat'])} s, first alarm {det[a]['F']['first_alarm_src']}"
        for a in ('A4', 'A4s')))
    L.append('False alarms, attack-free (onsets per hour over the whole run):')
    for c in COLS:
        v = far[c]
        L.append(f"  {c:3s} {v['rate']:.2f} /h (95% {v['rate_ci'][0]:.2f}-{v['rate_ci'][1]:.2f}; "
                 f"{v['events']} onsets in {v['hours']:.2f} h; {v['runs_with_alarm']}/{v['runs']} runs)")
    L.append('Missions (RMSE m / integrity % / corridor % / contingency %):')
    for s in SNAMES:
        L.append(f"  {s}: " + '  '.join(
            f"{c}={f(v['rmse'])}/{v['completion']:.0f}/{f(v.get('corridor'), 0)}/{v['safe_abort']:.0f}"
            for c, v in mis[s].items()))
    v = mis['S4']['F']
    L.append(f"  S4 F IMU attribution: {f(v.get('imu_dr'), 2)} at {f(v.get('imu_lat'))} s")
    L.append('IMU experiments (alarm %, RMSE, integrity %, corridor %, contingency %):')
    for s in imu:
        L.append(f"  {s}: " + '  '.join(
            f"{c}={v['detected'] * 100:.0f}/{f(v['rmse'])}/{v['completion'] * 100:.0f}/{v['corridor_ok'] * 100:.0f}/{v['aborted'] * 100:.0f}"
            for c, v in imu[s].items()))
    L.append('Link (hijack %, breach %, max cross-track m, link-loss abort %):')
    for a in lnk:
        L.append(f"  {a}: " + '  '.join(
            f"{c}={v['hijacked'] * 100:.0f}/{(1 - v['corridor_ok']) * 100:.0f}/{f(v['xtrack_max'])}/{v['link_abort'] * 100:.0f}"
            for c, v in lnk[a].items()))
    L.append(f"Overhead: total {ov['total']:.3f} ms, with IMU monitor + verification {ov.get('total_ext', float('nan')):.3f} ms, "
             f"memory {ov['mem_mb'] * 1000:.1f} kB")
    L.append(f"Calibration: {json.dumps({k: round(v, 3) for k, v in d['calibration'].items()})}")
    return '\n'.join(L) + '\n'


def generate_tables(results_path=None, out_dir=None, print_stdout=False):
    results_path = Path(results_path) if results_path else DEFAULT_RESULTS
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(Path(results_path).read_text())
    tables = {
        'tab9_detection.txt': build_tab_detection(d),
        'tab10_mission.txt': build_tab_mission(d),
        'tab_imu.txt': build_tab_imu(d),
        'tab_link.txt': build_tab_link(d),
        'tab11_ablation.txt': build_tab_ablation(d),
        'tab12_overhead.txt': build_tab_overhead(d),
        'tab13_robustness.txt': build_tab_robustness(d),
        'stats_significance.txt': build_stats(d),
    }
    for name, content in tables.items():
        (out_dir / name).write_text(content, encoding='utf-8')
        print(f"wrote {out_dir / name}")
    combined = '\n\n'.join(c.strip() for c in tables.values()) + '\n'
    (out_dir / 'all_tables.txt').write_text(combined, encoding='utf-8')
    (out_dir / 'key_numbers.txt').write_text(build_key_numbers(d), encoding='utf-8')
    print(f"wrote {out_dir / 'all_tables.txt'} and key_numbers.txt")
    if print_stdout:
        print('\n' + combined)


def main():
    ap = argparse.ArgumentParser(description="Generate LaTeX tables from results.json into .txt files.")
    ap.add_argument('positional_results', nargs='?', default=None)
    ap.add_argument('--results', default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--stdout', action='store_true')
    a = ap.parse_args()
    generate_tables(a.positional_results or a.results, a.out_dir, a.stdout)


if __name__ == '__main__':
    main()
