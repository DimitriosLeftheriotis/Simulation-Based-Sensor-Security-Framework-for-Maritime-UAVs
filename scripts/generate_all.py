#!/usr/bin/env python3
"""Convenience script to generate all tables (.txt) and figures (.pdf)
from results.json into the artifacts directory in one command."""
import argparse
import sys
from pathlib import Path

from make_tables import generate_tables
from make_figs import generate_figures

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / 'results.json'
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / 'artifacts'


def main():
    parser = argparse.ArgumentParser(description="Generate all tables and figures from results.json into artifacts/.")
    parser.add_argument('positional_results', nargs='?', default=None, help="Path to results.json (positional)")
    parser.add_argument('--results', default=None, help="Path to results.json (flag)")
    parser.add_argument('--artifacts-dir', default=None, help="Root directory for artifacts (defaults to artifacts/)")
    parser.add_argument('--stdout', action='store_true', help="Also print tables to stdout")
    args = parser.parse_args()

    results_file = args.positional_results or args.results or DEFAULT_RESULTS
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else DEFAULT_ARTIFACTS_DIR

    tables_dir = artifacts_dir / 'tables'
    figures_dir = artifacts_dir / 'figures'

    print("=" * 60)
    print("Generating Tables...")
    print("=" * 60)
    generate_tables(results_path=results_file, out_dir=tables_dir, print_stdout=args.stdout)

    print("\n" + "=" * 60)
    print("Generating Figures...")
    print("=" * 60)
    generate_figures(results_path=results_file, out_dir=figures_dir)

    print("\n" + "=" * 60)
    print(f"All artifacts successfully generated under: {artifacts_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
