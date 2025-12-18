import cv2
import os
import numpy as np
import sys

# -------------------------------
# Scene Detection
# -------------------------------
def detect_scenes(video_path, threshold=0.5, min_scene_len=50):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_hist = None
    scene_boundaries = [0]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold or (frame_idx - scene_boundaries[-1]) > min_scene_len:
                scene_boundaries.append(frame_idx)
        prev_hist = hist
        frame_idx += 1

    scene_boundaries.append(total_frames)
    cap.release()
    return scene_boundaries, fps

# -------------------------------
# Motion Intensity
# -------------------------------
def compute_motion_intensity(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_sum = 0.0
    count = 0

    for f in range(start_frame+1, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        motion_sum += np.mean(diff)
        prev_gray = gray
        count += 1

    cap.release()
    return motion_sum / max(count,1)

# -------------------------------
# Assign Scenes to Workers
# -------------------------------
def assign_scenes(video_path, scene_boundaries, max_interp=5):
    # Compute motion intensity for each scene
    scenes = []
    max_motion = 0
    for i in range(len(scene_boundaries)-1):
        start = scene_boundaries[i]
        end = scene_boundaries[i+1]
        motion = compute_motion_intensity(video_path, start, end)
        max_motion = max(max_motion, motion)
        scenes.append({'start': start, 'end': end, 'motion': motion})

    # Estimate number of frames to interpolate per scene (adaptive)
    for s in scenes:
        s['estimated_interp_frames'] = max(1, int(max_interp * (s['motion']/max_motion)))

    # Greedy assignment to balance load
    worker1, worker2 = [], []
    load1, load2 = 0, 0
    for s in scenes:
        if load1 <= load2:
            worker1.append(s)
            load1 += s['estimated_interp_frames']
        else:
            worker2.append(s)
            load2 += s['estimated_interp_frames']

    print(f"Worker 1 load (estimated frames): {load1}, scenes: {len(worker1)}")
    print(f"Worker 2 load (estimated frames): {load2}, scenes: {len(worker2)}")
    return worker1, worker2

# -------------------------------
# Save Scenes
# -------------------------------
def save_scenes(video_path, assigned_scenes, worker_name, output_dir="scenes"):
    os.makedirs(os.path.join(output_dir, worker_name), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for idx, s in enumerate(assigned_scenes,1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, s['start'])
        out_path = os.path.join(output_dir, worker_name, f"scene_{idx:03d}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        for f in range(s['start'], s['end']):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        print(f"Saved {out_path} (estimated interp frames: {s['estimated_interp_frames']})")

    cap.release()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scene_split_motion_workers.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Processing video: {video_path}")

    scene_boundaries, fps = detect_scenes(video_path)
    print(f"Detected scene boundaries: {scene_boundaries}")

    worker1, worker2 = assign_scenes(video_path, scene_boundaries)
    save_scenes(video_path, worker1, "worker1")
    save_scenes(video_path, worker2, "worker2")
    print("✅ Scene splitting, motion analysis, and worker assignment done!")

