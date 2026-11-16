#!/usr/bin/env python3
"""Generate fig_roc.pdf, fig_traj.pdf, and fig_robust.pdf from results.json.
Colourblind-safe (Okabe-Ito), vector output, readable in print. Pure plotting script."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / 'results.json'
DEFAULT_OUT_DIR = REPO_ROOT / 'artifacts' / 'figures'

OI = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
      'red': '#D55E00', 'purple': '#CC79A7', 'grey': '#555555'}
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})


def roc_fig(full, out_path):
    roc = full['roc']
    innov_att = roc['innov_att']; mar_att = roc['mar_att']
    innov_fre = roc['innov_fre']; mar_fre = roc['mar_fre']
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    for name, att, fre, col in [('Innovation (L2)', innov_att, innov_fre, OI['orange']),
                                ('Maritime cross-check (L3)', mar_att, mar_fre, OI['blue'])]:
        y = np.r_[np.ones(len(att)), np.zeros(len(fre))]
        s = np.r_[att, fre]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC={auc(fpr,tpr):.2f})')
    ax.plot([0, 1], [0, 1], ':', color=OI['grey'], lw=1)
    ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')
    ax.set_title('Consistent GNSS drift (A2)', fontsize=9)
    ax.legend(loc='lower right', fontsize=7.5, frameon=False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(); fig.savefig(out_path); print(f'wrote {out_path}')


def traj_fig(full, out_path):
    r = full['traj']
    truth = np.array(r['truth'])
    xhat = np.array(r['xhat'])
    gnss = np.array(r['gnss'])
    dt = r.get('dt', 0.1)
    t_eval = r.get('t_eval', 40.0)
    k_on = r['k_on']
    ka = int(r['alarm_t'] / dt) if r.get('alarm_t') is not None else k_on
    sl = slice(max(0, k_on - 300), min(len(truth), k_on + int(t_eval / dt)))

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(truth[sl, 0], truth[sl, 1], color=OI['green'], lw=2, label='True path')
    ax.plot(xhat[sl, 0], xhat[sl, 1], '--', color=OI['blue'], lw=1.6, label='Estimate (framework)')
    ax.plot(gnss[k_on:k_on+int(t_eval/dt), 0], gnss[k_on:k_on+int(t_eval/dt), 1],
            '.', color=OI['red'], ms=2.5, alpha=0.5, label='Spoofed GNSS')
    ax.plot(*truth[k_on], 'o', color=OI['orange'], ms=7, label='Attack onset')
    ax.plot(*truth[ka], 's', color=OI['purple'], ms=7, label='Framework alarm')
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)'); ax.set_aspect('equal', 'datalim')
    ax.legend(loc='best', fontsize=7, frameon=False)
    ax.set_title('Representative run (S1, GNSS jump)', fontsize=9)
    fig.tight_layout(); fig.savefig(out_path)
    seed_str = f" (seed {r['seed']})" if 'seed' in r else ""
    print(f'wrote {out_path}{seed_str}')


def robust_fig(full, out_path):
    d = full['robustness']; innov = full['detection']['A2c']['L2']['dr']
    rv, rp = d['vessels'], d['radar_pd']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.6))
    a1.plot([int(k) for k in rv], list(rv.values()), 'o-', color=OI['blue'], lw=2)
    a1.set_xlabel('Vessels in view'); a1.set_ylabel('A2 detection rate')
    a1.set_ylim(-0.03, 1.03); a1.axhline(innov, ls=':', color=OI['orange'], lw=1)
    a1.text(max(int(k) for k in rv), innov - 0.03, 'innovation-only', fontsize=6.5,
            ha='right', va='top', color=OI['orange'])
    a2.plot([float(k) for k in rp], list(rp.values()), 's-', color=OI['green'], lw=2)
    a2.set_xlabel('Radar detection prob. $P_d$'); a2.set_ylabel('A2 detection rate')
    a2.set_ylim(-0.03, 1.03)
    fig.tight_layout(); fig.savefig(out_path); print(f'wrote {out_path}')


def generate_figures(results_path=None, out_dir=None):
    results_path = Path(results_path) if results_path else DEFAULT_RESULTS
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'r') as f:
        full = json.load(f)

    if 'roc' not in full or 'traj' not in full:
        raise SystemExit(
            f"Error: '{results_path}' does not contain 'roc' and 'traj' figure data.\n"
            f"Please regenerate results using:\n"
            f"    python run_sim.py --N 100 --out {results_path}"
        )

    roc_fig(full, out_dir / 'fig_roc.pdf')
    traj_fig(full, out_dir / 'fig_traj.pdf')
    robust_fig(full, out_dir / 'fig_robust.pdf')


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from results.json into PDF files.")
    parser.add_argument('positional_results', nargs='?', default=None, help="Path to results.json (positional)")
    parser.add_argument('--results', default=None, help="Path to results.json (flag)")
    parser.add_argument('--out-dir', default=None, help="Output directory for PDF figures (defaults to artifacts/figures)")
    args = parser.parse_args()

    results_file = args.positional_results or args.results
    generate_figures(results_path=results_file, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
