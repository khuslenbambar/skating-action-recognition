import os
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
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

FIGURES_DIR = "figures"
EXPERIMENTS_DIR = "experiments"

GRANULARITY_LABELS = {
    "coarse":  "Coarse (3 classes)",
    "set":     "Set-Level (7 classes)",
    "element": "Element-Level (28 classes)",
}
COLORS = {"coarse": "#2196F3", "set": "#FF9800", "element": "#4CAF50"}


def load_history(g):
    path = os.path.join(EXPERIMENTS_DIR, g, "training_history.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path, allow_pickle=True).item()


def plot_training_curves_single(g):
    h = load_history(g)
    if h is None:
        return
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    color, label = COLORS[g], GRANULARITY_LABELS[g]

    ax1.plot(epochs, h["train_loss"], "o-",  color=color, label="Train")
    ax1.plot(epochs, h["val_loss"],   "s--", color=color, label="Validation", alpha=0.75)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"Loss — {label}"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, h["train_acc"], "o-",  color=color, label="Train")
    ax2.plot(epochs, h["val_acc"],   "s--", color=color, label="Validation", alpha=0.75)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy — {label}"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIGURES_DIR, f"training_curves_{g}.png"))
    plt.close()


def plot_training_curves_overlay():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for g in ["coarse", "set", "element"]:
        h = load_history(g)
        if h is None: continue
        epochs = range(1, len(h["val_loss"]) + 1)
        ax1.plot(epochs, h["val_loss"], "o-", color=COLORS[g], label=GRANULARITY_LABELS[g])
        ax2.plot(epochs, h["val_acc"],  "o-", color=COLORS[g], label=GRANULARITY_LABELS[g])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation Loss")
    ax1.set_title("Validation Loss Across Granularities"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy Across Granularities"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIGURES_DIR, "training_curves_all.png"))
    plt.close()


def plot_accuracy_bar_chart():
    gs, accs, cols = [], [], []
    for g in ["coarse", "set", "element"]:
        h = load_history(g)
        if h is None: continue
        gs.append(GRANULARITY_LABELS[g]); accs.append(max(h["val_acc"]) * 100); cols.append(COLORS[g])
    if not gs: return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(gs, accs, color=cols, width=0.55, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=15)
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("R3D-18 Baseline Performance by Classification Granularity")
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIGURES_DIR, "comparison_accuracy_bar.png"))
    plt.close()


def plot_class_count_vs_accuracy():
    counts = {"coarse": 3, "set": 7, "element": 28}
    px, py, labs = [], [], []
    for g in ["coarse", "set", "element"]:
        h = load_history(g)
        if h is None: continue
        px.append(counts[g]); py.append(max(h["val_acc"]) * 100); labs.append(GRANULARITY_LABELS[g])
    if len(px) < 2: return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(px, py, "o-", color="#2196F3", linewidth=2.5, markersize=11, zorder=3)
    for x, y, lab in zip(px, py, labs):
        ax.annotate(f"{lab}\n{y:.1f}%", (x, y),
                    textcoords="offset points", xytext=(0, 18),
                    ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Classes"); ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Impact of Classification Granularity on Model Performance")
    ax.set_ylim(0, 110); ax.grid(True, alpha=0.3); ax.set_xticks(px)
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIGURES_DIR, "classes_vs_accuracy.png"))
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for g in ["coarse", "set", "element"]:
        plot_training_curves_single(g)
    plot_training_curves_overlay()
    plot_accuracy_bar_chart()
    plot_class_count_vs_accuracy()
    print(f"All figures saved to '{FIGURES_DIR}/'")


if __name__ == "__main__":
    main()