"""Command-line entry point: ``readmit run-all``."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="readmit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-all", help="train, calibrate, evaluate and audit; write results")
    r.add_argument("--data-dir", default="data/raw")
    r.add_argument("--out-dir", default="docs/results")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--n-boot", type=int, default=500)
    r.add_argument("--threshold", type=float, default=0.15)
    r.add_argument("--synthetic", action="store_true",
                   help="run on a small synthetic table (smoke test; numbers are meaningless)")
    e = sub.add_parser("experiment", help="bounded AUROC experiment (history, diagnoses, Optuna)")
    e.add_argument("--data-dir", default="data/raw")
    e.add_argument("--out-dir", default="docs/results")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-trials", type=int, default=50)
    e.add_argument("--n-boot", type=int, default=500)
    e.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "experiment":
        from .experiment import main_experiment

        raw = None
        if args.synthetic:
            from .synthetic import make_synthetic

            raw = make_synthetic(3000, seed=args.seed)
        res = main_experiment(args.data_dir, args.out_dir, seed=args.seed,
                              n_trials=args.n_trials, n_boot=args.n_boot, raw=raw,
                              fast=args.synthetic)
        for s in res["steps"]:
            print(s["step"], {k: round(v, 4) for k, v in s["auroc"].items()}, "kept" if s["kept"]
                  else "dropped", s.get("test_auroc_diff_vs_original", ""))

    if args.cmd == "run-all":
        from .pipeline import run_all

        raw = None
        if args.synthetic:
            from .synthetic import make_synthetic

            raw = make_synthetic(4000, seed=args.seed)
        m = run_all(args.data_dir, args.out_dir, seed=args.seed, n_boot=args.n_boot,
                    threshold=args.threshold, fast=args.synthetic, raw=raw)
        head = m["test_metrics"][m["selected_model"]][m["headline_calibrator"]]
        print(json.dumps({"selected_model": m["selected_model"],
                          "auroc": head["auroc"], "ece": head["ece"], "brier": head["brier"]},
                         indent=2))


if __name__ == "__main__":
    main()
