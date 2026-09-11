"""
Skating Action Predictor — Single Video Inference

Takes a video file as input and predicts the skating action using the
trained R3D-18 model.

Usage:
    python predict.py --video path/to/video.mp4
    python predict.py --video path/to/video.mp4 --model best_model.pth
    python predict.py --video path/to/folder/  (processes all videos in folder)

Example:
    python predict.py --video test_videos/some_clip.mp4

Requirements:
    - A trained model file (best_model.pth) from skating_action_recognition.py
    - The same class names used during training

"""

import os
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


IMAGENET_MEAN = [0.43216, 0.394666, 0.37645]
IMAGENET_STD  = [0.22803, 0.22145, 0.216989]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Default class names — update this if you trained with different classes
# These match --granularity coarse (3 classes)
DEFAULT_CLASSES = ["Jump", "Other", "Spin"]


def load_video_frames(video_path: str) -> list:
    """Load all frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from: {video_path}")

    return frames


def sample_frames(frames: list, clip_length: int = 16) -> list:
    """Uniformly sample clip_length frames from the video."""
    total = len(frames)
    if total >= clip_length:
        indices = np.linspace(0, total - 1, clip_length, dtype=int)
    else:
        indices = list(range(total))
        indices += [total - 1] * (clip_length - total)
    return [frames[i] for i in indices]


def preprocess_clip(frames: list, height: int = 112, width: int = 112) -> torch.Tensor:
    """
    Resize, normalise, and convert frames to a model-ready tensor.

    Returns: torch.Tensor of shape (1, 3, T, H, W) — batched and ready.
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    processed = []
    for frame in frames:
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
        processed.append(frame)

    clip = np.stack(processed, axis=0)       
    clip = np.transpose(clip, (3, 0, 1, 2))    
    tensor = torch.from_numpy(clip).float()
    tensor = tensor.unsqueeze(0)               

    return tensor


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Rebuild the R3D-18 architecture and load trained weights.

    The architecture must match exactly what was used during training:
    same backbone, same classifier head structure.
    """
    model = r3d_18(weights=None) 
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[Model] Loaded weights from '{model_path}'")
    print(f"[Model] Classes: {num_classes}")

    return model

@torch.no_grad()
def predict_video(
    video_path: str,
    model: nn.Module,
    class_names: list,
    device: torch.device,
    clip_length: int = 16,
) -> dict:
    frames = load_video_frames(video_path)
    frames = sample_frames(frames, clip_length)
    clip = preprocess_clip(frames)
    clip = clip.to(device)

    output = model(clip)                             
    probabilities = torch.softmax(output, dim=1)[0]   
    predicted_idx = probabilities.argmax().item()
    confidence = probabilities[predicted_idx].item()

    all_scores = {}
    for i, name in enumerate(class_names):
        all_scores[name] = round(probabilities[i].item() * 100, 1)

    result = {
        "video": os.path.basename(video_path),
        "prediction": class_names[predicted_idx],
        "confidence": round(confidence * 100, 1),
        "all_scores": all_scores,
        "total_frames": len(load_video_frames(video_path)),
    }

    return result


def print_result(result: dict):
    print(f"\n{'─' * 50}")
    print(f"  Video:      {result['video']}")
    print(f"  Frames:     {result['total_frames']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  {'─' * 46}")
    print(f"  All scores:")

    sorted_scores = sorted(
        result["all_scores"].items(), key=lambda x: x[1], reverse=True
    )
    for class_name, score in sorted_scores:
        bar = "█" * int(score / 2)  
        marker = " ◄" if class_name == result["prediction"] else ""
        print(f"    {class_name:>10}: {score:5.1f}%  {bar}{marker}")

    print(f"{'─' * 50}")



def predict_folder(
    folder_path: str,
    model: nn.Module,
    class_names: list,
    device: torch.device,
    clip_length: int = 16,
) -> list:
    results = []
    video_files = sorted([
        f for f in Path(folder_path).rglob("*")
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        print(f"[Warning] No video files found in '{folder_path}'")
        return results

    print(f"\n[Batch] Processing {len(video_files)} videos from '{folder_path}'...")

    for i, video_path in enumerate(video_files, 1):
        try:
            result = predict_video(str(video_path), model, class_names, device, clip_length)
            results.append(result)
            print(f"  [{i}/{len(video_files)}] {result['video']} → "
                  f"{result['prediction']} ({result['confidence']}%)")
        except Exception as e:
            print(f"  [{i}/{len(video_files)}] {video_path.name} → ERROR: {e}")

    print(f"\n{'=' * 50}")
    print(f" BATCH RESULTS SUMMARY")
    print(f"{'=' * 50}")
    from collections import Counter
    predictions = Counter(r["prediction"] for r in results)
    for cls_name in class_names:
        count = predictions.get(cls_name, 0)
        pct = count / len(results) * 100 if results else 0
        print(f"  {cls_name}: {count} videos ({pct:.0f}%)")
    print(f"  Total: {len(results)} videos processed")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Predict skating actions from video using trained R3D-18",
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to a video file OR a folder of videos",
    )
    parser.add_argument(
        "--model", type=str, default="best_model.pth",
        help="Path to trained model weights (default: best_model.pth)",
    )
    parser.add_argument(
        "--classes", type=str, nargs="+", default=None,
        help="Class names in order (default: Jump Other Spin)",
    )
    parser.add_argument(
        "--clip_length", type=int, default=16,
        help="Number of frames per clip (must match training)",
    )
    parser.add_argument(
        "--save_json", type=str, default=None,
        help="Save results to a JSON file (e.g. results.json)",
    )
    args = parser.parse_args()

    class_names = args.classes or DEFAULT_CLASSES
    num_classes = len(class_names)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    if not os.path.isfile(args.model):
        print(f"\n[Error] Model file not found: '{args.model}'")
        print(f"  Make sure you've trained the model first by running:")
        print(f"  python skating_action_recognition.py --data_dir dataset --epochs 25")
        return

    model = load_model(args.model, num_classes, device)

    if os.path.isdir(args.video):
        results = predict_folder(args.video, model, class_names, device, args.clip_length)
    elif os.path.isfile(args.video):
        result = predict_video(args.video, model, class_names, device, args.clip_length)
        print_result(result)
        results = [result]
    else:
        print(f"\n[Error] Path not found: '{args.video}'")
        return

    if args.save_json and results:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] Results written to '{args.save_json}'")


if __name__ == "__main__":
    main()