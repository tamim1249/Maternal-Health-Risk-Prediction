#!/usr/bin/env python
"""Run the full OCH-MRNet study end to end.

Steps: duplicate audit -> Protocol B (leakage-free) on maternal and fetal data ->
Protocol A (naive split) leakage study on maternal data -> all tables, figures,
SHAP / counterfactual explanations and a final summary.

Example
-------
    python scripts/run_experiments.py --maternal "data/Maternal Health Risk Data Set.csv" \
        --fetal data/fetal_health.csv --out outputs

    # quick smoke test (1 repeat, short training)
    python scripts/run_experiments.py --fast
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use("Agg")


def parse_args():
    p = argparse.ArgumentParser(description="OCH-MRNet leakage-free maternal/fetal risk experiments")
    p.add_argument("--maternal", default="data/Maternal Health Risk Data Set.csv", help="path to the Maternal Health Risk CSV")
    p.add_argument("--fetal", default="data/fetal_health.csv", help="path to the Fetal Health (CTG) CSV")
    p.add_argument("--out", default="outputs", help="output folder for tables and figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--high-cost", type=float, default=2.0, help="extra cost (lambda) on the highest-risk class")
    p.add_argument("--fast", action="store_true", help="quick dry run: 1 repeat, short FT-Transformer training")
    p.add_argument("--no-fetal", action="store_true", help="skip the fetal CTG dataset")
    p.add_argument("--no-ablation", action="store_true", help="skip the retrained ablation variants")
    return p.parse_args()


def main():
    args = parse_args()
    from och_mrnet.config import CFG, configure, save_table

    configure(maternal_path=args.maternal, fetal_path=args.fetal, out=args.out, seed=args.seed,
              high_cost=args.high_cost, fast_mode=args.fast, run_fetal=not args.no_fetal,
              run_ablation=not args.no_ablation)

    from och_mrnet.data import prepare_data
    from och_mrnet.baselines import hyperparam_table
    from och_mrnet.experiment import run_protocol
    from och_mrnet.reporting import generate_report

    t0 = time.time()
    print(f"Output folder: {os.path.abspath(CFG['OUT'])}")
    DATA = prepare_data()
    save_table(hyperparam_table(), "T03_hyperparameters")

    RUNS = {("maternal", "B"): run_protocol(DATA, "maternal", "B"),
            ("maternal", "A"): run_protocol(DATA, "maternal", "A")}
    if CFG["RUN_FETAL"]:
        RUNS[("fetal", "B")] = run_protocol(DATA, "fetal", "B")

    generate_report(DATA, RUNS)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {os.path.abspath(CFG['OUT'])}")


if __name__ == "__main__":
    main()
