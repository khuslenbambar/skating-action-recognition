"""
Extracts 2D skeleton keypoints from skating video frames using MediaPipe
Pose. Produces a temporal sequence of keypoints for each video clip.

MediaPipe Pose outputs 33 keypoints per frame:
    0:  nose             1:  left_eye_inner   2:  left_eye
    3:  left_eye_outer   4:  right_eye_inner  5:  right_eye
    6:  right_eye_outer  7:  left_ear         8:  right_ear
    9:  mouth_left       10: mouth_right      11: left_shoulder
    12: right_shoulder   13: left_elbow       14: right_elbow
    15: left_wrist       16: right_wrist      17: left_pinky
    18: right_pinky      19: left_index       20: right_index
    21: left_thumb       22: right_thumb      23: left_hip
    24: right_hip        25: left_knee        26: right_knee
    27: left_ankle       28: right_ankle      29: left_heel
    30: right_heel       31: left_foot_index  32: right_foot_index

Each keypoint has (x, y, confidence), so output per frame is (33, 3).
For a clip of T frames, the output is (T, 33, 3).

Usage:
    python extract_poses.py --data_dir dataset --output_dir poses
    python extract_poses.py --video path/to/video.mp4 --output_dir poses

Dependencies:
    pip install mediapipe opencv-python numpy
    
"""

import os
import argparse
import numpy as np
import cv2

import mediapipe as mp

NUM_KEYPOINTS = 33
KEYPOINT_DIM = 3
DEFAULT_CLIP_LENGTH = 16


def create_pose_detector(static_image_mode=False, model_complexity=1,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Create a MediaPipe Pose detector."""
    pose = mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return pose


def extract_keypoints_from_frame(pose_detector, frame_rgb, frame_height, frame_width):
    """Extract 33 keypoints from a single RGB frame. Returns (33, 3) array."""
    results = pose_detector.process(frame_rgb)
    if results.pose_landmarks is None:
        return np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)

    keypoints = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
    for i, landmark in enumerate(results.pose_landmarks.landmark):
        keypoints[i, 0] = landmark.x * frame_width
        keypoints[i, 1] = landmark.y * frame_height
        keypoints[i, 2] = landmark.visibility
    return keypoints


def extract_poses_from_video(video_path, pose_detector, clip_length=DEFAULT_CLIP_LENGTH):
    """Extract pose keypoints from a video clip. Returns (T, 33, 3) array."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        print(f"  [Warning] No frames in {video_path}")
        return np.zeros((clip_length, NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)

    total = len(frames)
    if total >= clip_length:
        indices = np.linspace(0, total - 1, clip_length, dtype=int)
    else:
        indices = list(range(total))
        indices += [total - 1] * (clip_length - total)

    h, w = frames[0].shape[:2]
    pose_sequence = np.zeros((clip_length, NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)

    for i, frame_idx in enumerate(indices):
        frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        pose_sequence[i] = extract_keypoints_from_frame(pose_detector, frame_rgb, h, w)

    return pose_sequence


def normalise_keypoints(pose_sequence):
    """Normalise keypoints to [0,1] relative to skeleton bounding box. Returns (T, 33, 3)."""
    normalised = pose_sequence.copy()
    for t in range(normalised.shape[0]):
        frame_kps = normalised[t]
        valid_mask = frame_kps[:, 2] > 0.1
        if valid_mask.sum() < 2:
            continue
        valid_x = frame_kps[valid_mask, 0]
        valid_y = frame_kps[valid_mask, 1]
        x_min, x_max = valid_x.min(), valid_x.max()
        y_min, y_max = valid_y.min(), valid_y.max()
        x_range = max(x_max - x_min, 1.0)
        y_range = max(y_max - y_min, 1.0)
        normalised[t, :, 0] = (frame_kps[:, 0] - x_min) / x_range
        normalised[t, :, 1] = (frame_kps[:, 1] - y_min) / y_range
        normalised[t, ~valid_mask, 0] = 0.0
        normalised[t, ~valid_mask, 1] = 0.0
    return normalised


def extract_dataset_poses(data_dir, output_dir, clip_length=DEFAULT_CLIP_LENGTH):
    """Extract and save poses for an entire dataset (train/ and val/ splits)."""
    pose_detector = create_pose_detector(static_image_mode=False)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue

        print(f"\n[Poses] Processing {split} split...")
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            out_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            videos = sorted([f for f in os.listdir(class_dir)
                             if os.path.splitext(f)[1].lower() in video_extensions])
            print(f"  {class_name}: {len(videos)} videos")

            for vid_idx, vid_name in enumerate(videos):
                vid_path = os.path.join(class_dir, vid_name)
                out_name = os.path.splitext(vid_name)[0] + ".npy"
                out_path = os.path.join(out_class_dir, out_name)
                if os.path.isfile(out_path):
                    continue
                try:
                    pose_seq = extract_poses_from_video(vid_path, pose_detector, clip_length)
                    pose_seq = normalise_keypoints(pose_seq)
                    np.save(out_path, pose_seq)
                except Exception as e:
                    print(f"    [Error] {vid_name}: {e}")
                if (vid_idx + 1) % 50 == 0:
                    print(f"    Processed {vid_idx + 1}/{len(videos)}")

    pose_detector.close()
    print(f"\n[Poses] Done. Saved to '{output_dir}/'")


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose keypoints")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="poses")
    parser.add_argument("--clip_length", type=int, default=16)
    args = parser.parse_args()

    if args.data_dir:
        extract_dataset_poses(args.data_dir, args.output_dir, args.clip_length)
    elif args.video:
        detector = create_pose_detector(static_image_mode=True)
        pose_seq = extract_poses_from_video(args.video, detector, args.clip_length)
        pose_seq = normalise_keypoints(pose_seq)
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "single_video_pose.npy")
        np.save(out_path, pose_seq)
        print(f"[Poses] Shape: {pose_seq.shape}")
        print(f"[Poses] Saved to '{out_path}'")
        detector.close()
    else:
        print("Provide either --data_dir or --video")


if __name__ == "__main__":
    main()