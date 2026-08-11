"""
Print the canonical numbers behind the paper, straight from the committed CSVs
in results/, so that any value quoted in the paper can be traced to exactly one
row without downloading a dataset or re-running an experiment.
"""
import pandas as pd

print("=== NSL-KDD ablation summary (results/nslkdd_ablation_summary.csv) ===")
print(pd.read_csv("results/nslkdd_ablation_summary.csv").to_string(index=False))

print("\n=== NSL-KDD significance (V3 vs V0), results/nslkdd_significance.csv ===")
print(pd.read_csv("results/nslkdd_significance.csv").to_string(index=False))

print("\n=== NSL-KDD baselines, results/nslkdd_baselines.csv ===")
print(pd.read_csv("results/nslkdd_baselines.csv").to_string(index=False))

print("\n=== NSL-KDD per-class (V4), results/nslkdd_per_class_folds.csv (mean over folds) ===")
pc = pd.read_csv("results/nslkdd_per_class_folds.csv")
print(pc.groupby("cls")[["precision", "recall", "f1", "tnr"]].mean().to_string())

print("\n=== UNSW-NB15 binary, results/unsw_binary.csv ===")
print(pd.read_csv("results/unsw_binary.csv").to_string(index=False))

print("\n=== UNSW-NB15 multiclass, results/unsw_multiclass_summary.csv ===")
print(pd.read_csv("results/unsw_multiclass_summary.csv").to_string(index=False))
