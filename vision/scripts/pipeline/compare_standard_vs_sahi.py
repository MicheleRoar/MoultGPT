"""Diagnostic comparison of YOLO-standard vs YOLO+SAHI on the same 125 val
images, using the CSVs already saved by final_end_to_end_eval.py and
sahi_tiled_eval.py -- no retraining, no re-inference.

Answers:
- branch distribution + per-branch accuracy, both runs
- true-class distribution inside each rule branch (are the rules just wrong?)
- is the accuracy delta real, or noise on n=125? (McNemar exact + paired
  bootstrap CI on the accuracy difference)

Usage:
    cd vision
    python scripts/pipeline/compare_standard_vs_sahi.py \
        --standard scripts/results/final_end_to_end_eval_yolo_detect_GROUPED_SPLIT.csv \
        --sahi scripts/results/sahi_tiled_eval_yolo_detect_GROUPED_SPLIT.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest

_cli = argparse.ArgumentParser()
_cli.add_argument("--standard", required=True)
_cli.add_argument("--sahi", required=True)
_cli.add_argument("--n_boot", type=int, default=10000)
_cli.add_argument("--seed", type=int, default=42)
_args = _cli.parse_args()


def branch_report(df, label):
    print(f"\n=== {label}: branch distribution + per-branch accuracy ===")
    d = df.copy()
    d["correct"] = d["true_stage"] == d["pred_detector"]
    summary = d.groupby("branch_detector").agg(n=("correct", "size"), accuracy=("correct", "mean"))
    print(summary.round(3))

    print(f"\n=== {label}: true-class distribution inside each branch ===")
    print(pd.crosstab(d["branch_detector"], d["true_stage"]))


def mcnemar_exact(b, c):
    """Exact McNemar test on discordant pairs b (std-right/sahi-wrong) and
    c (std-wrong/sahi-right) via two-sided binomial test, p=0.5."""
    n = b + c
    if n == 0:
        return float("nan")
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def paired_bootstrap_ci(correct_std, correct_sahi, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(correct_std)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = correct_sahi[idx].mean() - correct_std[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi


def main():
    std = pd.read_csv(_args.standard)
    sahi = pd.read_csv(_args.sahi)

    branch_report(std, "STANDARD")
    branch_report(sahi, "SAHI")

    merged = std[["filename", "true_stage", "pred_detector"]].merge(
        sahi[["filename", "pred_detector"]], on="filename", suffixes=("_std", "_sahi"))
    merged["correct_std"] = merged["true_stage"] == merged["pred_detector_std"]
    merged["correct_sahi"] = merged["true_stage"] == merged["pred_detector_sahi"]

    n = len(merged)
    acc_std, acc_sahi = merged["correct_std"].mean(), merged["correct_sahi"].mean()
    print(f"\n=== PAIRED COMPARISON (n={n} images matched by filename) ===")
    print(f"accuracy standard={acc_std:.3f}  accuracy sahi={acc_sahi:.3f}  delta={acc_sahi-acc_std:+.3f}")

    b = ((merged["correct_std"]) & (~merged["correct_sahi"])).sum()  # std right, sahi wrong
    c = ((~merged["correct_std"]) & (merged["correct_sahi"])).sum()  # std wrong, sahi right
    both_right = (merged["correct_std"] & merged["correct_sahi"]).sum()
    both_wrong = (~merged["correct_std"] & ~merged["correct_sahi"]).sum()
    print(f"\nboth correct={both_right}  both wrong={both_wrong}  "
          f"SAHI broke (std right, sahi wrong)={b}  SAHI fixed (std wrong, sahi right)={c}")

    p = mcnemar_exact(b, c)
    print(f"McNemar exact test (discordant pairs only, b={b} c={c}): p={p:.4f}"
          f"  {'(significant at 0.05)' if p < 0.05 else '(NOT significant at 0.05 -- could be noise)'}")

    mean_diff, lo, hi = paired_bootstrap_ci(
        merged["correct_std"].values, merged["correct_sahi"].values, _args.n_boot, _args.seed)
    print(f"\nPaired bootstrap accuracy delta: mean={mean_diff:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]"
          f"  {'(CI excludes 0, likely real)' if lo > 0 or hi < 0 else '(CI includes 0, not conclusive on this n)'}")


if __name__ == "__main__":
    main()
