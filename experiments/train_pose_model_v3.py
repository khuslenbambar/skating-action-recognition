"""
=============================================================================
Train Pose-Enhanced Model — Action Recognition + Quality Assessment
=============================================================================

IMPROVED VERSION — Changes from original:
    1. quality_loss_weight reduced from 0.5 to 0.1 (prevents Q-Loss destabilising training)
    2. Default epochs increased to 30 (with early stopping in model, this is safe)
    3. Default batch_size changed to 8 (matches your MSI runs)

Prerequisites:
    1. Dataset organised: dataset/train/Jump/, dataset/val/Jump/, etc.
    2. Poses extracted:   poses/train/Jump/video001.npy, etc.
       Run: python extract_poses.py --data_dir dataset --output_dir poses

Usage:
    python train_pose_model.py --data_dir dataset --pose_dir poses --epochs 30

=============================================================================
"""

import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from pose_enhanced_model import (
    PoseEnhancedModel,
    PoseEnhancedDataset,
    PoseEnhancedTrainer,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train pose-enhanced action recognition + quality model",
    )
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Video dataset directory (with train/ and val/)")
    parser.add_argument("--pose_dir", type=str, default="poses",
                        help="Pose data directory (with train/ and val/)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--granularity", type=str, default="element", choices=["element", "set", "coarse"])
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true",
                        help="Ablation A2: disable inverse-frequency class weighting")
    parser.add_argument("--no_quality_head", action="store_true",
                        help="Ablation A3: disable quality head (zero quality loss weight)")
    parser.add_argument("--fusion", type=str, default="concat",
                        choices=["concat", "late"],
                        help="Ablation A5: fusion strategy (default concat)")
    parser.add_argument("--pose_only", action="store_true",
                        help="Ablation A1: pose-only mode (bypass R3D-18)")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] CPU")

    # Datasets
    train_dataset = PoseEnhancedDataset(
        video_root=os.path.join(args.data_dir, "train"),
        pose_root=os.path.join(args.pose_dir, "train"),
        clip_length=args.clip_length,
        train=not args.no_augmentation,
        granularity=args.granularity,
    )
    val_dataset = PoseEnhancedDataset(
        video_root=os.path.join(args.data_dir, "val"),
        pose_root=os.path.join(args.pose_dir, "val"),
        clip_length=args.clip_length,
        granularity=args.granularity,
    )

    num_classes = len(train_dataset.classes)
    print(f"[Data] Classes: {train_dataset.classes}")
    print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    # Model — A1 (pose_only) and A5 (fusion) plumbed in here
    model = PoseEnhancedModel(
        num_classes=num_classes,
        fusion_mode=args.fusion,
        pose_only=args.pose_only,
    )

    # Config — CHANGED: quality_loss_weight 0.5 → 0.1
    config = {
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "action_loss_weight": 1.0,
        "quality_loss_weight": 0.1,
        "no_class_weights": args.no_class_weights,  # A2
        "no_quality_head": args.no_quality_head,    # A3
    }

    # Train
    trainer = PoseEnhancedTrainer(model, train_loader, val_loader, device, config)
    history = trainer.fit(args.epochs)

    # Save history
    np.save("pose_training_history.npy", dict(history))
    print(f"\n[Saved] Training history → pose_training_history.npy")
    print(f"[Saved] Best model → best_pose_model.pth")
    print(f"\nNext: python analyse.py --video test_videos/some_clip.mp4")


if __name__ == "__main__":
    main()