""""
Produces:
  1. summary.csv — master table of all runs (best val, best epoch, final train)
  2. curves/{run}.png — training curves per run
  3. comparison_granularity.png — baseline vs pose vs pose-only across 3 granularities
  4. ablation_bar.png — element-level ablation bar chart
  5. confusion/{run}.png — confusion matrices for element-level runs
  6. per_class_f1/{run}.csv — per-class precision/recall/F1

Usage:
    python analysis.py --base_dir /path/to/dissertation_root
                       --val_video <path_to_val_videos>
                       --val_pose <path_to_val_poses>

The base directory is expected to contain an `experiments/` subfolder with
one folder per training run. Each run folder should contain either
`pose_training_history.npy` or `training_history.npy`, and (for element-level
runs used in confusion matrix generation) a `best_pose_model.pth` checkpoint.
"""
import os, sys, json, argparse, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)

from pose_enhanced_model import PoseEnhancedModel, PoseEnhancedDataset

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base_dir", type=str, default=".",
                    help="Dissertation root containing experiments/ (default: cwd)")
parser.add_argument("--val_video", type=str, default=None,
                    help="Path to validation videos folder "
                         "(e.g. dataset_element/val). "
                         "If omitted, confusion-matrix generation is skipped.")
parser.add_argument("--val_pose", type=str, default=None,
                    help="Path to validation pose tensors folder "
                         "(e.g. poses_element/val). "
                         "If omitted, confusion-matrix generation is skipped.")
args = parser.parse_args()

BASE = Path(args.base_dir).resolve()
OUT = BASE / "analysis"
(OUT / "curves").mkdir(parents=True, exist_ok=True)
(OUT / "confusion").mkdir(parents=True, exist_ok=True)
(OUT / "per_class_f1").mkdir(parents=True, exist_ok=True)

EXPERIMENTS_DIR = BASE / "experiments"
print(f"EXPERIMENTS_DIR = {EXPERIMENTS_DIR}")
print(f"Exists? {EXPERIMENTS_DIR.exists()}")

if not EXPERIMENTS_DIR.exists():
    sys.exit(f"[Error] {EXPERIMENTS_DIR} does not exist. "
             f"Pass --base_dir to point at your dissertation root.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Generating summary.csv ===")
runs = []
for d in sorted(EXPERIMENTS_DIR.iterdir()):
    if not d.is_dir(): continue
    # Try both history file conventions
    for hname in ["pose_training_history.npy", "training_history.npy"]:
        hpath = d / hname
        if hpath.exists():
            h = np.load(hpath, allow_pickle=True).item()
            val = h.get("val_acc", [])
            tr  = h.get("train_acc", [])
            if len(val) == 0: continue
            bi = int(np.argmax(val))
            runs.append({
                "run": d.name,
                "best_val_acc": val[bi],
                "best_epoch": bi + 1,
                "final_train_acc": tr[-1] if len(tr) else None,
                "n_epochs": len(val),
                "history_file": hname,
            })
            break

import csv
with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=runs[0].keys())
    w.writeheader()
    w.writerows(runs)
print(f"  Wrote {len(runs)} runs to summary.csv")


print("=== Generating training curves ===")
for d in sorted(EXPERIMENTS_DIR.iterdir()):
    if not d.is_dir(): continue
    for hname in ["pose_training_history.npy", "training_history.npy"]:
        hpath = d / hname
        if not hpath.exists(): continue
        h = np.load(hpath, allow_pickle=True).item()
        epochs = range(1, len(h["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, h["train_loss"], label="Train", marker="o", markersize=3)
        axes[0].plot(epochs, h["val_loss"],   label="Val",   marker="s", markersize=3)
        axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{d.name} — Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(epochs, [a*100 for a in h["train_acc"]], label="Train", marker="o", markersize=3)
        axes[1].plot(epochs, [a*100 for a in h["val_acc"]],   label="Val",   marker="s", markersize=3)
        axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title=f"{d.name} — Accuracy")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "curves" / f"{d.name}.png", dpi=120, bbox_inches="tight")
        plt.close()
        break
print(f"  Curves saved to {OUT}/curves/")


print("=== Generating granularity comparison plot ===")
def get_acc(folder, hname):
    p = EXPERIMENTS_DIR / folder / hname
    if not p.exists(): return None
    h = np.load(p, allow_pickle=True).item()
    return max(h["val_acc"]) * 100

data = {
    "Coarse (3)":  {
        "Baseline v2":  get_acc("v2_baseline_coarse_aug", "training_history.npy"),
        "Pose v2":      get_acc("v2_pose_coarse_aug", "pose_training_history.npy"),
        "Pose-only":    get_acc("ablation_A1_pose_only_coarse", "pose_training_history.npy"),
    },
    "Set (6)": {
        "Baseline v2":  get_acc("v2_baseline_set_aug", "training_history.npy"),
        "Pose v2":      get_acc("v2_pose_set_aug", "pose_training_history.npy"),
        "Pose-only":    get_acc("ablation_A1_pose_only_set", "pose_training_history.npy"),
    },
    "Element (28)": {
        "Baseline v2":  get_acc("v2_baseline_aug", "training_history.npy"),
        "Pose v2":      get_acc("v2_pose_aug", "pose_training_history.npy"),
        "Pose-only":    get_acc("ablation_A1_pose_only", "pose_training_history.npy"),
    },
}
fig, ax = plt.subplots(figsize=(10, 5))
gran_labels = list(data.keys())
model_labels = ["Baseline v2", "Pose v2", "Pose-only"]
x = np.arange(len(gran_labels))
width = 0.25
for i, m in enumerate(model_labels):
    vals = [data[g][m] if data[g][m] is not None else 0 for g in gran_labels]
    bars = ax.bar(x + (i-1)*width, vals, width, label=m)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}",
                    ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(gran_labels)
ax.set_ylabel("Best Validation Accuracy (%)")
ax.set_title("Model Comparison Across Granularities")
ax.legend(); ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(OUT / "comparison_granularity.png", dpi=120, bbox_inches="tight")
plt.close()
print("  comparison_granularity.png saved")

print("=== Generating ablation bar chart ===")
ablations = [
    ("Reference (full v2)",   get_acc("v2_pose_aug", "pose_training_history.npy")),
    ("A1: Pose-only",         get_acc("ablation_A1_pose_only", "pose_training_history.npy")),
    ("A2: No class weights",  get_acc("ablation_A2_no_class_weights", "pose_training_history.npy")),
    ("A2: NCW (seed 43)",     get_acc("ablation_A2_no_class_weights_seed43", "pose_training_history.npy")),
    ("A3: No quality head",   get_acc("ablation_A3_no_quality_head", "pose_training_history.npy")),
    ("A5: Late fusion",       get_acc("ablation_A5_late_fusion", "pose_training_history.npy")),
]
labels = [a[0] for a in ablations]
vals = [a[1] for a in ablations]
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2c3e50"] + ["#e74c3c"] + ["#3498db"]*2 + ["#9b59b6"] + ["#1abc9c"]
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.2f}%", ha="center", fontsize=10)
ax.axhline(vals[0], linestyle="--", color="gray", alpha=0.5, label=f"Reference ({vals[0]:.2f}%)")
ax.set_ylabel("Best Validation Accuracy (%)")
ax.set_title("Element-Level Ablation Studies")
ax.set_ylim(0, max(vals) * 1.15)
plt.xticks(rotation=20, ha="right")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT / "ablation_bar.png", dpi=120, bbox_inches="tight")
plt.close()
print("  ablation_bar.png saved")


print("=== Generating confusion matrices and per-class F1 ===")

if args.val_video is None or args.val_pose is None:
    print("  Skipping confusion matrices: --val_video and --val_pose not provided.")
    VAL_VIDEO = None
else:
    VAL_VIDEO = Path(args.val_video)
    VAL_POSE  = Path(args.val_pose)

if VAL_VIDEO is None or not VAL_VIDEO.exists():
    print(f"  Skipping confusion matrix generation.")
else:
    val_ds = PoseEnhancedDataset(
        video_root=VAL_VIDEO, pose_root=VAL_POSE,
        clip_length=16, granularity="element"
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    classes = val_ds.classes
    n_cls = len(classes)
    print(f"  Loaded {len(val_ds)} val samples, {n_cls} classes")

    element_runs = [
        ("v2_pose_aug",                       "best_pose_model.pth", {}),
        ("ablation_A1_pose_only",             "best_pose_model.pth", {"pose_only": True}),
        ("ablation_A2_no_class_weights",      "best_pose_model.pth", {}),
        ("ablation_A3_no_quality_head",       "best_pose_model.pth", {}),
        ("ablation_A5_late_fusion",           "best_pose_model.pth", {"fusion_mode": "late"}),
    ]

    for run_name, ckpt_name, model_kwargs in element_runs:
        ckpt_path = EXPERIMENTS_DIR / run_name / ckpt_name
        if not ckpt_path.exists():
            print(f"  Skipping {run_name}: checkpoint not found")
            continue
        print(f"  Processing {run_name}...")
        model = PoseEnhancedModel(num_classes=n_cls, **model_kwargs).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for clips, poses, labels in val_loader:
                clips = clips.to(DEVICE); poses = poses.to(DEVICE)
                logits, _ = model(clips, poses)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_cls)))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        fig, ax = plt.subplots(figsize=(15, 13))
        sns.heatmap(
            cm_norm, annot=False, cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            cbar_kws={"label": "Recall"}, ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=18, labelpad=10)
        ax.set_ylabel("True",      fontsize=18, labelpad=10)
        ax.set_title(
            f"Confusion Matrix (row-normalised) — {run_name}",
            fontsize=19, pad=14,
        )
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label("Recall", fontsize=15)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout(pad=0.5)
        plt.savefig(
            OUT / "confusion" / f"{run_name}.png",
            dpi=150, bbox_inches="tight", pad_inches=0.05,
        )
        plt.close()

        # Per-class F1
        p, r, f1, sup = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(n_cls)), zero_division=0
        )
        with open(OUT / "per_class_f1" / f"{run_name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "precision", "recall", "f1", "support"])
            for i, c in enumerate(classes):
                w.writerow([c, f"{p[i]:.4f}", f"{r[i]:.4f}", f"{f1[i]:.4f}", int(sup[i])])
        print(f"    Done — {run_name}")

print("\n=== All analysis complete. Output in ./analysis/ ===")
