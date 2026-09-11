import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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


def plot_training_curves(history_path="training_history.npy",
                         save_path="training_curves.png"):
    history = np.load(history_path, allow_pickle=True).item()
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "s-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "s-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    print(f"[Plot] Saved training curves to '{save_path}'")


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    n_cls = len(class_names)
    figsize = (max(8, n_cls * 0.55), max(7, n_cls * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=(n_cls <= 10), fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"shrink": 0.85},
        annot_kws={"size": 13},
    )
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    ax.set_title("Confusion Matrix", pad=14)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    print(f"[Plot] Saved confusion matrix to '{save_path}'")


if __name__ == "__main__":
    history_path = "training_history.npy"
    if Path(history_path).exists():
        plot_training_curves(history_path)
    else:
        print(f"No training history found at '{history_path}'.")