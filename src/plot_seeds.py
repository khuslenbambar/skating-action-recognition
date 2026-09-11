"""
Plot training curves for multi-seed pose-enhanced runs.

Usage:
    python plot_seeds.py --run experiments/v2_pose_aug_seed44 --label seed44
    python plot_seeds.py --run experiments/v2_pose_aug_seed45 --label seed45
    
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
})


def plot_seed_run(run_dir: Path, label: str):
    history_path = run_dir / "pose_training_history.npy"
    if not history_path.exists():
        print(f"[Error] No training history at {history_path}")
        return
    h = np.load(history_path, allow_pickle=True).item()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(h["train_loss"], "o-", label="Train")
    ax1.plot(h["val_loss"],   "s-", label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{label} — Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot([x * 100 for x in h["train_acc"]], "o-", label="Train")
    ax2.plot([x * 100 for x in h["val_acc"]],   "s-", label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{label} — Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout(pad=0.5)
    out_path = f"{label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves for a seed run")
    parser.add_argument("--run", type=str, required=True,
                        help="Path to experiment folder (e.g. experiments/v2_pose_aug_seed44)")
    parser.add_argument("--label", type=str, required=True,
                        help="Label for plot filename and titles (e.g. seed44)")
    args = parser.parse_args()
    plot_seed_run(Path(args.run), args.label)


if __name__ == "__main__":
    main()