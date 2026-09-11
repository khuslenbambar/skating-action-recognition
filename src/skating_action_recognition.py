"""
This module implements a baseline action recognition system for figure
skating videos. It uses a pretrained R3D-18 model (3D ResNet-18) from
torchvision with transfer learning. The system is designed to be modular
and extensible as it can later be enhanced with pose-based features (e.g., from MediaPipe or OpenPose).

Architecture Overview:
    1. Video clips are loaded and uniformly sampled to a fixed number of
       frames (default: 16).
    2. Frames are resized to 112x112 and normalised with ImageNet statistics.
    3. A pretrained R3D-18 backbone extracts spatiotemporal features.
    4. A custom classification head maps features to skating action classes.

Usage:
    python skating_action_recognition.py --data_dir dataset --epochs 30

Expected dataset layout:
    dataset/
        train/
            jump/
                video001.mp4
                video002.avi
                ...
            spin/
                video001.mp4
                ...
        val/
            jump/
                ...
            spin/
                ...

Dependencies:
    pip install torch torchvision opencv-python scikit-learn

This is the version 3 of the initial module with improvements. The initial version was not submitted as this is the final version.

"""

import os
import argparse
import random
import time
from pathlib import Path
from collections import defaultdict, Counter

import cv2
cv2.setNumThreads(0)  
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def _worker_init(worker_id):
    """Disable OpenCV threading inside each DataLoader worker process."""
    import cv2
    cv2.setNumThreads(0)


IMAGENET_MEAN = [0.43216, 0.394666, 0.37645]
IMAGENET_STD  = [0.22803, 0.22145, 0.216989]

GRANULARITY_COARSE = {
    "1Axel": "Jump", "1Flip": "Jump", "1Loop": "Jump", "1Lutz": "Jump",
    "1Salchow": "Jump", "1Toeloop": "Jump", "2Axel": "Jump", "2Flip": "Jump",
    "2Loop": "Jump", "2Lutz": "Jump", "2Salchow": "Jump", "2Toeloop": "Jump",
    "3Axel": "Jump", "3Flip": "Jump", "3Loop": "Jump", "3Lutz": "Jump",
    "3Salchow": "Jump", "3Toeloop": "Jump", "4Flip": "Jump", "4Loop": "Jump",
    "4Lutz": "Jump", "4Salchow": "Jump", "4Toeloop": "Jump",
    "CamelSpin": "Spin", "SitSpin": "Spin", "UprightSpin": "Spin",
    "NoBasic": "Other", "Sequence": "Other",
}

GRANULARITY_SET = {
    "1Axel": "Jump", "1Flip": "Jump", "1Loop": "Jump", "1Lutz": "Jump",
    "1Salchow": "Jump", "1Toeloop": "Jump", "2Axel": "Jump", "2Flip": "Jump",
    "2Loop": "Jump", "2Lutz": "Jump", "2Salchow": "Jump", "2Toeloop": "Jump",
    "3Axel": "Jump", "3Flip": "Jump", "3Loop": "Jump", "3Lutz": "Jump",
    "3Salchow": "Jump", "3Toeloop": "Jump", "4Flip": "Jump", "4Loop": "Jump",
    "4Lutz": "Jump", "4Salchow": "Jump", "4Toeloop": "Jump",
    "CamelSpin": "CamelSpin", "SitSpin": "SitSpin", "UprightSpin": "UprightSpin",
    "NoBasic": "NoBasic", "Sequence": "Sequence",
}

def get_granularity_map(name):
    if name == "coarse":
        return GRANULARITY_COARSE
    elif name == "set":
        return GRANULARITY_SET
    elif name == "element":
        return None  # identity
    else:
        raise ValueError(f"Unknown granularity: {name}")


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Section 1: Configuration
def get_default_config():
    return {
        # --- Data ---
        "clip_length": 16,         
        "frame_height": 112,        
        "frame_width": 112,         
        "batch_size": 8,           
        "num_workers": 8,          
        "freeze_backbone": True,    
        "unfreeze_last_n": 1,      
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 30,
        "early_stopping_patience": 5,
        "max_grad_norm": 1.0,
        "seed": 42,
    }


# Section 2: Dataset
class SkatingVideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading figure skating video clips.

    Each video is uniformly sampled to produce exactly `clip_length` frames.
    Frames are resized, converted to float tensors, and normalised with
    ImageNet statistics to match the pretrained R3D-18 input distribution.

    Returns
    -------
    clip : torch.Tensor
        Shape (C, T, H, W) — channels-first, as expected by torchvision
        video models.
    label : int
        Integer class index.
    """

    def __init__(
        self,
        root_dir: str,
        clip_length: int = 16,
        frame_height: int = 112,
        frame_width: int = 112,
        mean: list = None,
        std: list = None,
        train: bool = False,
        granularity: str = "element",
    ):
        """
        Parameters
        ----------
        root_dir : str
            Path to the split folder (e.g., "dataset/train"). Must contain
            one sub-folder per class.
        clip_length : int
            Number of frames to sample from each video.
        frame_height, frame_width : int
            Target spatial resolution after resizing.
        mean, std : list[float]
            Per-channel normalisation statistics (default: ImageNet).
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.clip_length = clip_length
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.mean = np.array(mean or IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(std or IMAGENET_STD, dtype=np.float32)
        self.train = train


        element_classes = sorted(
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        )
        self.granularity = granularity
        self.gran_map = get_granularity_map(granularity)
        if self.gran_map is None:
            self.classes = element_classes
        else:
            mapped_set = sorted(set(self.gran_map[c] for c in element_classes))
            self.classes = mapped_set
        self.element_to_target = {}
        for i, ec in enumerate(element_classes):
            target_name = self.gran_map[ec] if self.gran_map else ec
            self.element_to_target[ec] = self.classes.index(target_name)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = self._collect_samples()

        print(f"[Dataset] Loaded {len(self.samples)} videos from '{root_dir}'")
        print(f"[Dataset] Classes: {self.classes}")


    def _collect_samples(self):
        samples = []
        for element_name in sorted(self.element_to_target.keys()):
            target_idx = self.element_to_target[element_name]
            cls_dir = self.root_dir / element_name
            if not cls_dir.is_dir():
                continue
            for file_path in sorted(cls_dir.iterdir()):
                if file_path.suffix.lower() == ".npy":
                    samples.append((str(file_path), target_idx))
        return samples

    def _load_frames(self, video_path: str) -> list:

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Failed to read any frames from: {video_path}")

        return frames

    def _sample_frames(self, frames: list) -> list:
        total = len(frames)

        if total >= self.clip_length:
            indices = np.linspace(0, total - 1, self.clip_length, dtype=int)
        else:
            indices = list(range(total))
            indices += [total - 1] * (self.clip_length - total)

        return [frames[i] for i in indices]

    def _preprocess(self, frames: list) -> torch.Tensor:
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.mean) / self.std
            processed.append(frame)

        clip = np.stack(processed, axis=0)             
        clip = np.transpose(clip, (3, 0, 1, 2))       

        return torch.from_numpy(clip).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        clip = np.load(clip_path)                     

        if self.train:
            if np.random.rand() < 0.5:
                clip = clip[:, :, ::-1, :].copy()

            T, H, W, C = clip.shape
            crop_size = int(min(H, W) * 0.9)
            top = np.random.randint(0, H - crop_size + 1)
            left = np.random.randint(0, W - crop_size + 1)
            clip = clip[:, top:top+crop_size, left:left+crop_size, :]
            resized = np.empty((T, H, W, C), dtype=np.uint8)
            for t in range(T):
                resized[t] = cv2.resize(clip[t], (W, H))
            clip = resized

            brightness = np.random.uniform(0.85, 1.15)
            contrast = np.random.uniform(0.85, 1.15)
            clip = clip.astype(np.float32)
            mean_val = clip.mean()
            clip = (clip - mean_val) * contrast + mean_val * brightness
            clip = np.clip(clip, 0, 255).astype(np.uint8)

        clip = clip.astype(np.float32) / 255.0
        clip = (clip - self.mean) / self.std
        clip = np.transpose(clip, (3, 0, 1, 2))      
        return torch.from_numpy(clip).float(), label


# Section 3: Model

def build_model(num_classes: int, freeze_backbone: bool = True,
                unfreeze_last_n: int = 1) -> nn.Module:
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        layer_groups = [model.layer1, model.layer2, model.layer3, model.layer4]
        layers_to_unfreeze = layer_groups[-unfreeze_last_n:] if unfreeze_last_n > 0 else []

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True


    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] R3D-18 loaded with pretrained Kinetics-400 weights")
    print(f"[Model] Total parameters:     {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")
    print(f"[Model] Frozen parameters:    {total_params - trainable_params:,}")
    print(f"[Model] Output classes:        {num_classes}")

    return model


# Section 4: Training

class Trainer:

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config


        class_counts = Counter()
        for _, label in train_loader.dataset.samples:
            class_counts[label] += 1

        num_classes = len(class_counts)
        total_samples = sum(class_counts.values())


        class_weights = torch.zeros(num_classes)
        for cls_idx, count in class_counts.items():
            class_weights[cls_idx] = total_samples / (num_classes * count)

        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.to(device)

        print(f"\n[Class Weights] Computed from {total_samples} training samples:")
        min_w = class_weights.min().item()
        max_w = class_weights.max().item()
        print(f"  Range: {min_w:.3f} (most common) → {max_w:.3f} (rarest)")

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimiser = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode="min", factor=0.5, patience=3,
        )

        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        self.history = defaultdict(list)

    def train_one_epoch(self, epoch: int) -> tuple:

        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (clips, labels) in enumerate(self.train_loader):
            clips = clips.to(self.device)    # (B, C, T, H, W)
            labels = labels.to(self.device)  # (B,)

            outputs = self.model(clips)      # (B, num_classes)
            loss = self.criterion(outputs, labels)

            self.optimiser.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            self.optimiser.step()

            running_loss += loss.item() * clips.size(0)
            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % max(1, len(self.train_loader) // 3) == 0:
                print(
                    f"  [Batch {batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> tuple:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for clips, labels in self.val_loader:
            clips = clips.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(clips)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * clips.size(0)
            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(self):
        best_val_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0

        print("\n" + "=" * 65)
        print(" BASELINE MODEL — TRAINING START (IMPROVED)")
        print("=" * 65)
        print(f" Optimizer:       AdamW (weight_decay={self.config['weight_decay']})")
        print(f" LR Scheduler:    ReduceLROnPlateau (patience=3, factor=0.5)")
        print(f" Early Stopping:  patience={self.early_stopping_patience}")
        print(f" Grad Clipping:   max_norm={self.max_grad_norm}")
        print(f" Class Weights:   Yes (inverse frequency)")
        print("=" * 65)

        for epoch in range(1, self.config["epochs"] + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_one_epoch(epoch)

            val_loss, val_acc = self.validate()

            self.scheduler.step(val_loss)

            elapsed = time.time() - start_time
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            current_lr = self.optimiser.param_groups[0]['lr']

            print(
                f"\nEpoch [{epoch}/{self.config['epochs']}]  "
                f"({elapsed:.1f}s, lr={current_lr:.2e})\n"
                f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}\n"
                f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), "best_model.pth")
                print(f"  ★ New best model saved (val_acc={val_acc:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  — No improvement ({epochs_without_improvement}/{self.early_stopping_patience})")

                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n  Early stopping triggered — no improvement for "
                          f"{self.early_stopping_patience} epochs")
                    break

        print("\n" + "=" * 65)
        print(f" TRAINING COMPLETE — Best val accuracy: {best_val_acc:.4f} "
              f"(epoch {best_epoch})")
        print("=" * 65)

        return self.history


# Section 5: Evaluation

@torch.no_grad()
def evaluate_model(model, data_loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    for clips, labels in data_loader:
        clips = clips.to(device)
        outputs = model(clips)
        _, predicted = outputs.max(dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 65)
    print(" EVALUATION RESULTS")
    print("=" * 65)
    print(f"\n  Overall Accuracy:  {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  Weighted Precision: {precision:.4f}")
    print(f"  Weighted Recall:    {recall:.4f}")
    print(f"  Weighted F1-Score:  {f1:.4f}")

    print("\n  --- Per-Class Report ---")
    print(
        classification_report(
            all_labels, all_preds,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("  --- Confusion Matrix ---")
    # Print header
    header = "          " + "  ".join(f"{name:>8}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>8}" for val in row)
        print(f"  {class_names[i]:>8}  {row_str}")

    print("=" * 65)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "ground_truth": all_labels,
    }


# Section 6: Utility Functions

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU (training will be slow)")
    return device


def print_dataset_summary(dataset, split_name: str):
    counts = defaultdict(int)
    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1

    print(f"\n  {split_name} split — {len(dataset)} videos:")
    for cls_name in dataset.classes:
        print(f"    {cls_name}: {counts[cls_name]} videos")


# Section 7: Main Entry Point

def main():

    parser = argparse.ArgumentParser(
        description="Baseline Action Recognition for Figure Skating (R3D-18)",
    )
    parser.add_argument("--granularity", type=str, default="element",
                        choices=["element", "set", "coarse"],
                        help="Label granularity: element (28), set (6), or coarse (3)")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable training augmentation (for v1-style baseline runs)")
    parser.add_argument(
        "--data_dir", type=str, default="dataset",
        help="Root dataset directory containing 'train/' and 'val/' folders.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--clip_length", type=int, default=None, help="Frames per clip.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    config = get_default_config()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.clip_length is not None:
        config["clip_length"] = args.clip_length
    if args.seed is not None:
        config["seed"] = args.seed

    print("\n" + "=" * 65)
    print(" FIGURE SKATING ACTION RECOGNITION — BASELINE (R3D-18)")
    print("=" * 65)
    print("\n[Config]")
    for k, v in config.items():
        print(f"  {k}: {v}")

    set_seed(config["seed"])

    device = get_device()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: '{train_dir}'\n"
            f"Expected folder structure:\n"
            f"  {args.data_dir}/\n"
            f"    train/\n"
            f"      jump/\n"
            f"      spin/\n"
            f"    val/\n"
            f"      jump/\n"
            f"      spin/"
        )

    train_dataset = SkatingVideoDataset(
        root_dir=train_dir,
        clip_length=config["clip_length"],
        frame_height=config["frame_height"],
        frame_width=config["frame_width"],
        train=not args.no_augmentation,
        granularity=args.granularity,
    )
    val_dataset = SkatingVideoDataset(
        root_dir=val_dir,
        clip_length=config["clip_length"],
        frame_height=config["frame_height"],
        frame_width=config["frame_width"],
        granularity=args.granularity,
    )

    print_dataset_summary(train_dataset, "Train")
    print_dataset_summary(val_dataset, "Val")

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        worker_init_fn=_worker_init,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        worker_init_fn=_worker_init,
    )

    model = build_model(
        num_classes=num_classes,
        freeze_backbone=config["freeze_backbone"],
        unfreeze_last_n=config["unfreeze_last_n"],
    )

    trainer = Trainer(model, train_loader, val_loader, device, config)
    history = trainer.fit()

    print("\n[Eval] Loading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)

    results = evaluate_model(model, val_loader, class_names, device)

    history_path = "training_history.npy"
    np.save(history_path, dict(history))
    print(f"\n[Info] Training history saved to '{history_path}'")
    print("[Info] Best model saved to 'best_model.pth'")
    print("\nDone. You can now extend this baseline with pose-based features.")


if __name__ == "__main__":
    main()