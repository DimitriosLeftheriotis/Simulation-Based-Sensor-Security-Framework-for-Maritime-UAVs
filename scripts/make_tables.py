#!/usr/bin/env python3
"""Read results.json and generate LaTeX table bodies matching the manuscript.
Outputs individual .txt files for each table, plus a combined all_tables.txt."""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / 'results.json'
DEFAULT_OUT_DIR = REPO_ROOT / 'artifacts' / 'tables'


def f(x, nd=1):
    return '--' if x is None or x != x else f'{x:.{nd}f}'


def build_tab_detection(d):
    lines = ['% ===== tab:detection (DR / latency s); columns B1 B2 L2 F IF =====']
    names = {'A1': 'A1 jump', 'A2n': 'A2 drift, naive', 'A2c': 'A2 drift, consistent',
             'A3': 'A3 jamming', 'A5': 'A5 ghost vessel'}
    cols = ['B1', 'B2', 'L2', 'F', 'IF']
    for a, lab in names.items():
        cells = []
        for c in cols:
            v = d['detection'][a][c]
            lat = f(v['lat'], 1)
            cells.append(f"${v['dr']:.2f}$ / ${lat}$")
        lines.append(f"{lab} & " + " & ".join(cells) + r"\\")
    far = d['far']
    lines.append(r"\midrule")
    lines.append("False-alarm rate (per hour) & " +
                 " & ".join(f"${f(far[c],2)}$" for c in cols) + r"\\")
    return "\n".join(lines) + "\n"


def build_tab_mission(d):
    lines = ['% ===== tab:mission (RMSE, max, recov, integrity%, safe-abort%) =====']
    snames = {'S1': 'S1 Coastal defence', 'S2': 'S2 Naval surveillance',
              'S3': 'S3 Border patrol', 'S4': 'S4 Offshore asset'}
    for s, lab in snames.items():
        for i, c in enumerate(['B0', 'B1', 'B2', 'F']):
            v = d['mission'][s][c]
            head = f"\\multirow{{4}}{{*}}{{{lab}}}" if i == 0 else ""
            lines.append(f"{head} & {c} & ${f(v['rmse'])}$ & ${f(v['maxerr'])}$ & ${f(v['recov'])}$ "
                         f"& ${f(v['completion'],0)}$ & ${f(v.get('safe_abort'),0)}$\\\\")
        lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def build_tab_ablation(d):
    lines = ['% ===== tab:ablation =====']
    for lab, v in d['ablation'].items():
        lines.append(f"{lab} & ${f(v['dr'],0)}$ & ${f(v['lat'])}$ & ${f(v['rmse'])}$ & ${f(v['comp'],0)}$\\\\")
    return "\n".join(lines) + "\n"


def build_tab_overhead(d):
    lines = ['% ===== tab:overhead =====']
    o = d['overhead']
    for name, cpu, mem, lat in [('L1 signing and validation', o['L1'], '--', f(o['L1'], 3)),
                                ('L2 monitors', o['L2'], '--', '--'),
                                ('L3 cross-sensor consistency', o['L3'], '--', '--'),
                                ('L4 trust and state machine', o['L4'], '--', '--')]:
        lines.append(f"{name} & ${f(cpu,3)}$ & {mem} & {lat}\\\\")
    lines.append(f"\\midrule\nTotal & ${f(o['total'],3)}$ & $<0.01$ & ${f(o['L1'],3)}$\\\\")
    return "\n".join(lines) + "\n"


def build_tab_robustness(d):
    lines = ['% ===== tab:robustness (A2c maritime detection) =====']
    rv = d['robustness']['vessels']; rp = d['robustness']['radar_pd']
    lines.append("Vessels in view & " + " & ".join(f"${k}$" for k in rv) + r"\\")
    lines.append("Detection rate & " + " & ".join(f"${f(v,2)}$" for v in rv.values()) + r"\\")
    lines.append(r"\midrule")
    lines.append("Radar $P_d$ & " + " & ".join(f"${k}$" for k in rp) + r"\\")
    lines.append("Detection rate & " + " & ".join(f"${f(v,2)}$" for v in rp.values()) + r"\\")
    return "\n".join(lines) + "\n"


def build_stats_significance(d):
    lines = ['% ===== significance (for the text) =====']
    st = d['stats']
    lines.append(f"% A2c F vs L2 (McNemar) p = {st['A2c_F_vs_L2_mcnemar_p']:.3g}")
    lines.append(f"% A2c F vs IF (McNemar) p = {st['A2c_F_vs_IF_mcnemar_p']:.3g}")
    lines.append(f"% S1 F vs B1 RMSE (Wilcoxon) p = {st['S1_F_vs_B1_rmse_wilcoxon_p']:.3g}")
    return "\n".join(lines) + "\n"


def generate_tables(results_path=None, out_dir=None, print_stdout=False):
    results_path = Path(results_path) if results_path else DEFAULT_RESULTS
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'r') as f:
        d = json.load(f)

    tables = {
        'tab9_detection.txt': build_tab_detection(d),
        'tab10_mission.txt': build_tab_mission(d),
        'tab11_ablation.txt': build_tab_ablation(d),
        'tab12_overhead.txt': build_tab_overhead(d),
        'tab13_robustness.txt': build_tab_robustness(d),
        'stats_significance.txt': build_stats_significance(d),
    }

    all_content = []
    for filename, content in tables.items():
        file_path = out_dir / filename
        file_path.write_text(content, encoding='utf-8')
        all_content.append(content.strip())
        print(f"wrote {file_path}")

    combined_path = out_dir / 'all_tables.txt'
    combined_content = "\n\n".join(all_content) + "\n"
    combined_path.write_text(combined_content, encoding='utf-8')
    print(f"wrote {combined_path}")

    if print_stdout:
        print("\n" + combined_content)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results.json into .txt files.")
    parser.add_argument('positional_results', nargs='?', default=None, help="Path to results.json (positional)")
    parser.add_argument('--results', default=None, help="Path to results.json (flag)")
    parser.add_argument('--out-dir', default=None, help="Output directory for table .txt files (defaults to artifacts/tables)")
    parser.add_argument('--stdout', action='store_true', help="Also print tables to stdout")
    args = parser.parse_args()

    results_file = args.positional_results or args.results
    generate_tables(results_path=results_file, out_dir=args.out_dir, print_stdout=args.stdout)


if __name__ == '__main__':
    main()
