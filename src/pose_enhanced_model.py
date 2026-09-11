"""
Pose-Enhanced Action Recognition + Quality Assessment for Figure Skating

IMPROVED VERSION(v3) — Changes from initial version(v1):
    1. AdamW optimizer (proper weight decay decoupling)
    2. Early stopping with patience=5 on val accuracy
    3. Gradient clipping (max_norm=1.0) to prevent training instability
    4. Improved logging: shows LR and early stopping countdown
    5. Class-weighted CrossEntropyLoss to handle class imbalance

The initial version (v1) is not included in the submission as this is the latest version of the code.

"""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset
from torchvision.models.video import r3d_18, R3D_18_Weights
import cv2

IMAGENET_MEAN = [0.43216, 0.394666, 0.37645]
IMAGENET_STD  = [0.22803, 0.22145, 0.216989]
NUM_KEYPOINTS = 33
KEYPOINT_DIM = 3

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
    if name == "coarse": return GRANULARITY_COARSE
    elif name == "set": return GRANULARITY_SET
    elif name == "element": return None
    else: raise ValueError(f"Unknown granularity: {name}")


MP_POSE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18),
    (19, 20), (21, 22), (23, 24), (25, 26),
    (27, 28), (29, 30), (31, 32),
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# Section 1: Pose Encoder Network

class PoseEncoder(nn.Module):
    """
    Encodes a temporal sequence of skeleton keypoints into a fixed-length
    feature vector.

    The encoder flattens each frame's 17 keypoints × 3 values = 51 features,
    then processes the temporal sequence through an LSTM to capture motion
    dynamics, producing a 128-dimensional pose embedding.

    Input:  (batch, T, 17, 3) — temporal pose sequence
    Output: (batch, 128) — pose feature vector
    """

    def __init__(self, num_keypoints=17, keypoint_dim=3, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()

        input_size = num_keypoints * keypoint_dim  # 17 × 3 = 51

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.hidden_size = hidden_size

    def forward(self, pose_sequence):
        batch, T, num_kp, kp_dim = pose_sequence.shape

        x = pose_sequence.reshape(batch, T, -1)

        x = self.input_projection(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        pose_features = h_n[-1] 

        pose_features = self.output_projection(pose_features)

        return pose_features


# Section 2: Pose-Enhanced Model

class PoseEnhancedModel(nn.Module):
    """
    Dual-head model combining R3D-18 RGB features with pose features
    for both action classification and quality assessment.
    """

    def __init__(self, num_classes, rgb_feature_dim=512, pose_feature_dim=128,
                 freeze_backbone=True, unfreeze_last_n=1,
                 fusion_mode="concat", pose_only=False):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.pose_only = pose_only
        assert fusion_mode in ("concat", "late"), f"bad fusion_mode: {fusion_mode}"

        self.rgb_backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.rgb_backbone.parameters():
                param.requires_grad = False

            layer_groups = [
                self.rgb_backbone.layer1,
                self.rgb_backbone.layer2,
                self.rgb_backbone.layer3,
                self.rgb_backbone.layer4,
            ]
            for layer in layer_groups[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.rgb_backbone.fc = nn.Identity()

        self.pose_encoder = PoseEncoder(
            num_keypoints=NUM_KEYPOINTS,
            keypoint_dim=KEYPOINT_DIM,
            hidden_size=pose_feature_dim,
        )

        fused_dim = rgb_feature_dim + pose_feature_dim  # 512 + 128 = 640
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.action_head = nn.Linear(256, num_classes)

        self.rgb_only_head = nn.Sequential(
            nn.Linear(rgb_feature_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.pose_only_head = nn.Sequential(
            nn.Linear(pose_feature_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        self.quality_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self.num_classes = num_classes
        self.fused_dim = fused_dim

        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[PoseEnhancedModel] Total parameters:     {total:,}")
        print(f"[PoseEnhancedModel] Trainable parameters: {trainable:,}")
        print(f"[PoseEnhancedModel] Output classes:        {self.num_classes}")
        print(f"[PoseEnhancedModel] Quality score range:   1.0 - 10.0")

    def forward(self, video_clip, pose_sequence):
        if self.pose_only:
            pose_features = self.pose_encoder(pose_sequence)  
            action_logits = self.pose_only_head(pose_features)
            batch_size = pose_features.size(0)
            quality_score = torch.full(
                (batch_size, 1), 5.0, device=pose_features.device
            )
            return action_logits, quality_score

        rgb_features = self.rgb_backbone(video_clip)      
        pose_features = self.pose_encoder(pose_sequence)  

        if self.fusion_mode == "late":
            rgb_logits = self.rgb_only_head(rgb_features)
            pose_logits = self.pose_only_head(pose_features)
            action_logits = (rgb_logits + pose_logits) / 2.0
            fused = torch.cat([rgb_features, pose_features], dim=1)
            fused = self.fusion(fused)
            quality_raw = self.quality_head(fused)
            quality_score = torch.sigmoid(quality_raw) * 9.0 + 1.0
            return action_logits, quality_score

        fused = torch.cat([rgb_features, pose_features], dim=1) 
        fused = self.fusion(fused)                                
        action_logits = self.action_head(fused)
        quality_raw = self.quality_head(fused)
        quality_score = torch.sigmoid(quality_raw) * 9.0 + 1.0
        return action_logits, quality_score


# Section 3: Dataset

class PoseEnhancedDataset(Dataset):

    def __init__(self, video_root, pose_root, clip_length=16,
                 frame_height=112, frame_width=112, train=False, granularity="element"):
        self.clip_length = clip_length
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.train = train
        self.granularity = granularity
        self.mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
        self.std = np.array(IMAGENET_STD).reshape(1, 1, 3)

        element_classes = sorted([
            d for d in os.listdir(video_root)
            if os.path.isdir(os.path.join(video_root, d))
        ])
        gran_map = get_granularity_map(granularity)
        if gran_map is None:
            self.classes = element_classes
        else:
            self.classes = sorted(set(gran_map[c] for c in element_classes))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for element_name in element_classes:
            cls_video_dir = os.path.join(video_root, element_name)
            cls_pose_dir = os.path.join(pose_root, element_name)
            target_name = gran_map[element_name] if gran_map else element_name
            target_label = self.class_to_idx[target_name]
            for fname in sorted(os.listdir(cls_video_dir)):
                if fname.lower().endswith('.npy'):
                    clip_path = os.path.join(cls_video_dir, fname)
                    pose_path = os.path.join(cls_pose_dir, fname)
                    self.samples.append((clip_path, pose_path, target_label))

        print(f"[PoseDataset] {len(self.samples)} videos, "
              f"{len(self.classes)} classes from '{video_root}'")

    def _load_clip_uint8(self, clip_path):
        return np.load(clip_path)

    def _flip_pose_horizontal(self, pose):
        pose = pose.copy()
        pose[..., 0] = 1.0 - pose[..., 0]
        for left, right in MP_POSE_LR_PAIRS:
            tmp = pose[:, left, :].copy()
            pose[:, left, :] = pose[:, right, :]
            pose[:, right, :] = tmp
        return pose

    def _load_pose_raw(self, pose_path):
        if os.path.isfile(pose_path):
            pose = np.load(pose_path)
            if pose.shape == (self.clip_length, NUM_KEYPOINTS, KEYPOINT_DIM):
                return pose.astype(np.float32)
        return np.zeros((self.clip_length, NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, pose_path, label = self.samples[idx]
        clip = self._load_clip_uint8(clip_path)
        pose = self._load_pose_raw(pose_path)

        if self.train:
            if np.random.rand() < 0.5:
                clip = clip[:, :, ::-1, :].copy()
                pose = self._flip_pose_horizontal(pose)

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
        return torch.from_numpy(clip).float(), torch.from_numpy(pose).float(), label


# Section 4: Training Loop

class PoseEnhancedTrainer:

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        if config.get("no_class_weights", False):
            print("\n[Class Weights] DISABLED (ablation A2) — using uniform weights")
            self.action_criterion = nn.CrossEntropyLoss()
        else:
            class_counts = Counter()
            for _, _, label in train_loader.dataset.samples:
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

            self.action_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.quality_criterion = nn.MSELoss()

        self.action_weight = config.get("action_loss_weight", 1.0)
        if config.get("no_quality_head", False):
            self.quality_weight = 0.0
            print("[Quality Head] DISABLED (ablation A3) — quality_weight forced to 0")
        else:
            self.quality_weight = config.get("quality_loss_weight", 0.1)

        self.optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode="min", factor=0.5, patience=3,
        )

        self.early_stopping_patience = config.get("early_stopping_patience", 5)

        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_quality_loss": [], "val_quality_loss": [],
        }

    def _compute_quality_targets(self, action_logits, labels):
        with torch.no_grad():
            probs = torch.softmax(action_logits.detach(), dim=1)
            correct_probs = probs.gather(1, labels.unsqueeze(1))  
            quality_targets = correct_probs * 9.0 + 1.0
        return quality_targets

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_action_loss = 0.0
        total_quality_loss = 0.0
        correct = 0
        total = 0

        for clips, poses, labels in self.train_loader:
            clips = clips.to(self.device)
            poses = poses.to(self.device)
            labels = labels.to(self.device)

            action_logits, quality_scores = self.model(clips, poses)

            action_loss = self.action_criterion(action_logits, labels)

            quality_targets = self._compute_quality_targets(action_logits, labels)
            quality_loss = self.quality_criterion(quality_scores, quality_targets)

            loss = (self.action_weight * action_loss +
                    self.quality_weight * quality_loss)

            self.optimiser.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            self.optimiser.step()

            total_loss += loss.item() * clips.size(0)
            total_action_loss += action_loss.item() * clips.size(0)
            total_quality_loss += quality_loss.item() * clips.size(0)
            _, predicted = action_logits.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / total,
            "action_loss": total_action_loss / total,
            "quality_loss": total_quality_loss / total,
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_quality_loss = 0.0
        correct = 0
        total = 0

        for clips, poses, labels in self.val_loader:
            clips = clips.to(self.device)
            poses = poses.to(self.device)
            labels = labels.to(self.device)

            action_logits, quality_scores = self.model(clips, poses)

            action_loss = self.action_criterion(action_logits, labels)
            quality_targets = self._compute_quality_targets(action_logits, labels)
            quality_loss = self.quality_criterion(quality_scores, quality_targets)

            loss = (self.action_weight * action_loss +
                    self.quality_weight * quality_loss)

            total_loss += loss.item() * clips.size(0)
            total_quality_loss += quality_loss.item() * clips.size(0)
            _, predicted = action_logits.max(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / total,
            "quality_loss": total_quality_loss / total,
            "accuracy": correct / total,
        }

    def fit(self, epochs):
        best_val_acc = 0.0
        epochs_without_improvement = 0 

        print(f"\n{'=' * 65}")
        print(f" POSE-ENHANCED MODEL — TRAINING START (IMPROVED)")
        print(f"{'=' * 65}")
        print(f" Optimizer:       AdamW (weight_decay={self.config.get('weight_decay', 1e-4)})")
        print(f" LR Scheduler:    ReduceLROnPlateau (patience=3, factor=0.5)")
        print(f" Early Stopping:  patience={self.early_stopping_patience}")
        print(f" Grad Clipping:   max_norm={self.max_grad_norm}")
        print(f" Quality Weight:  {self.quality_weight}")
        print(f" Class Weights:   Yes (inverse frequency)")
        print(f"{'=' * 65}")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()
            self.scheduler.step(val_metrics["loss"])

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_quality_loss"].append(train_metrics["quality_loss"])
            self.history["val_quality_loss"].append(val_metrics["quality_loss"])

            current_lr = self.optimiser.param_groups[0]['lr']

            print(
                f"\nEpoch [{epoch}/{epochs}]  (lr={current_lr:.2e})\n"
                f"  Train — Loss: {train_metrics['loss']:.4f}  "
                f"Acc: {train_metrics['accuracy']:.4f}  "
                f"Q-Loss: {train_metrics['quality_loss']:.4f}\n"
                f"  Val   — Loss: {val_metrics['loss']:.4f}  "
                f"Acc: {val_metrics['accuracy']:.4f}  "
                f"Q-Loss: {val_metrics['quality_loss']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), "best_pose_model.pth")
                print(f"  ★ New best model saved (val_acc={val_metrics['accuracy']:.4f})")
            else:
                epochs_without_improvement += 1
                remaining = self.early_stopping_patience - epochs_without_improvement
                print(f"  — No improvement ({epochs_without_improvement}/{self.early_stopping_patience})")

                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n  Early stopping triggered — no improvement for "
                          f"{self.early_stopping_patience} epochs")
                    break

        print(f"\n{'=' * 65}")
        print(f" TRAINING COMPLETE — Best val accuracy: {best_val_acc:.4f}")
        print(f"{'=' * 65}")

        return self.history