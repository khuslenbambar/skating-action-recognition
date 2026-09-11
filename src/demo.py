"""
Runs a single video through the trained model and produces:
  1. Console output: predicted class, confidence, quality score
  2. An annotated video with skeleton overlay + prediction bar

Usage:
    # Basic — predict a single clip
    python demo.py --video clip.mp4

    # With skeleton overlay — produces an annotated output video
    python demo.py --video clip.mp4 --overlay --output demo_output.mp4

    # Baseline model (RGB-only)
    python demo.py --video clip.mp4 --mode baseline --checkpoint best_model.pth

    # Pose-enhanced model (default)
    python demo.py --video clip.mp4 --mode pose --checkpoint best_pose_model.pth

    # Process a whole folder
    python demo.py --video val/3Lutz/ --mode pose

"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
IMAGENET_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

ELEMENT_CLASSES = [
    '1Axel','1Flip','1Loop','1Lutz','1Salchow','1Toeloop',
    '2Axel','2Flip','2Loop','2Lutz','2Salchow','2Toeloop',
    '3Axel','3Flip','3Loop','3Lutz','3Salchow','3Toeloop',
    '4Flip','4Loop','4Lutz','4Salchow','4Toeloop',
    'CamelSpin','NoBasic','Sequence','SitSpin','UprightSpin',
]

SET_CLASSES = ['Axel','Flip','Loop','Lutz','NoBasic','Salchow','Sequence',
               'SitSpin','Toeloop','UprightSpin','CamelSpin']

COARSE_CLASSES = ['Jump', 'Other', 'Spin']

GRANULARITY_MAP = {
    'element': ELEMENT_CLASSES,
    'set': sorted(list(set(SET_CLASSES))),
    'coarse': COARSE_CLASSES,
}

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(15,17),(15,19),(15,21),
    (16,18),(16,20),(16,22),
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_video_frames(video_path):
    """Load all frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from: {video_path}")
    return frames


def sample_frames(frames, clip_length=16):
    """Uniformly sample clip_length frames."""
    total = len(frames)
    if total >= clip_length:
        indices = np.linspace(0, total - 1, clip_length, dtype=int)
    else:
        indices = list(range(total)) + [total-1] * (clip_length - total)
    return [frames[i] for i in indices], indices


def preprocess_clip(frames, height=112, width=112):
    """Resize, normalise, return tensor (1, C, T, H, W)."""
    processed = []
    for f in frames:
        f = cv2.resize(f, (width, height))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = f.astype(np.float32) / 255.0
        f = (f - IMAGENET_MEAN) / IMAGENET_STD
        processed.append(f)
    clip = np.stack(processed)  
    clip = np.transpose(clip, (3, 0, 1, 2))  
    return torch.FloatTensor(clip).unsqueeze(0)


def extract_pose_single_clip(frames):
    """Extract MediaPipe poses for sampled frames. Returns (1, T, 33, 3)."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
        keypoints = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                kp = np.array([[lm.x, lm.y, lm.visibility]
                               for lm in result.pose_landmarks.landmark])
            else:
                kp = np.zeros((33, 3))
            keypoints.append(kp)
        pose.close()
        arr = np.stack(keypoints).astype(np.float32)  # (T, 33, 3)
        return torch.FloatTensor(arr).unsqueeze(0)
    except ImportError:
        print("[Warning] MediaPipe not installed. Using zero poses.")
        return torch.zeros(1, len(frames), 33, 3)


def load_baseline_model(checkpoint_path, num_classes, device):
    from skating_action_recognition import build_model
    model = build_model(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def load_pose_model(checkpoint_path, num_classes, device):
    from pose_enhanced_model import PoseEnhancedModel
    model = PoseEnhancedModel(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def predict_single(video_path, model, mode, device, classes, clip_length=16):
    """Run inference on a single video. Returns dict of results."""
    frames = load_video_frames(video_path)
    sampled, indices = sample_frames(frames, clip_length)
    clip_tensor = preprocess_clip(sampled).to(device)

    with torch.no_grad():
        if mode == "baseline":
            out = model(clip_tensor)
            logits = out[0] if isinstance(out, tuple) else out
            quality = None
        else:
            poses = extract_pose_single_clip(sampled).to(device)
            out = model(clip_tensor, poses)
            logits = out[0] if isinstance(out, tuple) else out
            quality = out[1].item() if isinstance(out, tuple) and len(out) > 1 else None

    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    # Top-5 predictions
    top5_idx = probs.argsort()[::-1][:5]
    top5 = [(classes[i], float(probs[i])) for i in top5_idx]

    return {
        "video": os.path.basename(video_path),
        "prediction": pred_class,
        "confidence": confidence,
        "quality_score": quality,
        "top5": top5,
        "all_probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "n_frames": len(frames),
        "sampled_frames": sampled,
        "frame_indices": indices.tolist() if hasattr(indices, 'tolist') else list(indices),
    }


def print_result(result):
    """Pretty-print a prediction result."""
    print()
    print("=" * 60)
    print(f"  Video:        {result['video']}")
    print(f"  Frames:       {result['n_frames']}")
    print(f"  Prediction:   {result['prediction']}")
    print(f"  Confidence:   {result['confidence']*100:.1f}%")
    if result['quality_score'] is not None:
        print(f"  Quality:      {result['quality_score']:.2f} / 10.0")
    print("=" * 60)
    print("  Top-5 predictions:")
    for cls, prob in result['top5']:
        bar = "█" * int(prob * 40)
        marker = " ◄" if cls == result['prediction'] else ""
        print(f"    {cls:<14} {prob*100:5.1f}%  {bar}{marker}")
    print("=" * 60)
    print()


def draw_skeleton_on_frame(frame, keypoints, connections=POSE_CONNECTIONS):
    """Draw MediaPipe skeleton on a frame. keypoints shape: (33, 3)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for (i, j) in connections:
        if keypoints[i][2] > 0.3 and keypoints[j][2] > 0.3:
            pt1 = (int(keypoints[i][0] * w), int(keypoints[i][1] * h))
            pt2 = (int(keypoints[j][0] * w), int(keypoints[j][1] * h))
            cv2.line(overlay, pt1, pt2, (0, 255, 128), 2)

    for kp in keypoints:
        if kp[2] > 0.3:
            pt = (int(kp[0] * w), int(kp[1] * h))
            cv2.circle(overlay, pt, 4, (0, 200, 255), -1)
            cv2.circle(overlay, pt, 4, (0, 0, 0), 1)

    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)


def create_overlay_video(video_path, result, output_path, clip_length=16):
    """Create annotated video with skeleton overlay and prediction bar."""
    try:
        import mediapipe as mp
    except ImportError:
        print("[Error] MediaPipe required for overlay. pip install mediapipe")
        return

    frames = load_video_frames(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    h, w = frames[0].shape[:2]
    bar_h = 80
    out_h = h + bar_h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, out_h))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            kps = np.array([[lm.x, lm.y, lm.visibility]
                            for lm in res.pose_landmarks.landmark])
            frame = draw_skeleton_on_frame(frame, kps)

        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40) 

        pred = result['prediction']
        conf = result['confidence']
        text = f"Prediction: {pred}  ({conf*100:.1f}%)"
        cv2.putText(bar, text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if result['quality_score'] is not None:
            q_text = f"Quality: {result['quality_score']:.1f}/10"
            cv2.putText(bar, q_text, (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        bar_x = w - 220
        bar_w = 200
        cv2.rectangle(bar, (bar_x, 15), (bar_x + bar_w, 35), (80, 80, 80), -1)
        fill_w = int(bar_w * conf)
        color = (0, 200, 100) if conf > 0.7 else (0, 200, 255) if conf > 0.4 else (0, 80, 255)
        cv2.rectangle(bar, (bar_x, 15), (bar_x + fill_w, 35), color, -1)
        cv2.rectangle(bar, (bar_x, 15), (bar_x + bar_w, 35), (200, 200, 200), 1)

        combined = np.vstack([frame, bar])
        writer.write(combined)

    pose.close()
    writer.release()
    print(f"[Saved] Annotated video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Figure Skating Action Recognition Demo")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file or folder")
    parser.add_argument("--mode", type=str, default="pose",
                        choices=["baseline", "pose"],
                        help="Model type (default: pose)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--granularity", type=str, default="element",
                        choices=["element", "set", "coarse"])
    parser.add_argument("--overlay", action="store_true",
                        help="Create annotated output video with skeleton overlay")
    parser.add_argument("--output", type=str, default="demo_output.mp4",
                        help="Output path for overlay video")
    parser.add_argument("--clip_length", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    print(f"[Device] {device}")

    classes = GRANULARITY_MAP.get(args.granularity, ELEMENT_CLASSES)
    num_classes = len(classes)
    print(f"[Config] Mode: {args.mode}, Granularity: {args.granularity} ({num_classes} classes)")

    if args.checkpoint is None:
        args.checkpoint = "best_pose_model.pth" if args.mode == "pose" else "best_model.pth"

    if not os.path.isfile(args.checkpoint):
        print(f"\n[Error] Checkpoint not found: {args.checkpoint}")
        print("  Place the .pth file in the same directory, or use --checkpoint path/to/file.pth")
        return

    print(f"[Model] Loading {args.checkpoint}...")
    if args.mode == "baseline":
        model = load_baseline_model(args.checkpoint, num_classes, device)
    else:
        model = load_pose_model(args.checkpoint, num_classes, device)
    print("[Model] Ready.")

    if os.path.isdir(args.video):
        video_files = sorted([
            os.path.join(args.video, f) for f in os.listdir(args.video)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.npy'))
        ])
        print(f"\n[Batch] Processing {len(video_files)} files from {args.video}")
        correct = 0
        total = 0
        true_class = os.path.basename(os.path.normpath(args.video))
        for vf in video_files[:20]:  # limit to 20 for demo
            try:
                result = predict_single(vf, model, args.mode, device, classes, args.clip_length)
                is_correct = result['prediction'] == true_class
                total += 1
                if is_correct:
                    correct += 1
                status = "✓" if is_correct else "✗"
                q_str = f"  Q={result['quality_score']:.1f}" if result['quality_score'] else ""
                status = "✓" if is_correct else "✗"
                print(f"  {status} {result['video']:<30} → {result['prediction']:<14} "
                    f"({result['confidence']*100:.1f}%){q_str}")
            except Exception as e:
                print(f"  ! {os.path.basename(vf)}: {e}")
        if total > 0 and true_class in classes:
            print(f"\n  Accuracy on this folder: {correct}/{total} ({correct/total*100:.1f}%)")
    else:
        result = predict_single(args.video, model, args.mode, device, classes, args.clip_length)
        print_result(result)

        if args.overlay:
            print("[Overlay] Creating annotated video...")
            create_overlay_video(args.video, result, args.output, args.clip_length)


if __name__ == "__main__":
    main()