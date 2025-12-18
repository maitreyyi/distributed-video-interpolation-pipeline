#!/usr/bin/env python3
"""
master_s3.py
Master: detect scenes, compute motion-adaptive interpolation estimates,
save scene clips, upload them to S3, write per-worker job JSONs to S3,
then wait for outputs, download and concat into final_output.mp4
"""

import os
import sys
import time
import json
import boto3
import cv2
import numpy as np
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# ========== CONFIG ==========
BUCKET = "s3-scene-dect-bucket"     # your bucket in us-west-2
JOB_PREFIX = "jobs/job_001"
SCENES_PREFIX = "scenes"
OUTPUTS_PREFIX = "outputs"
REGION = "us-west-2"                # all in us-west-2
MAX_INTERP = 5
POLL_INTERVAL = 5
WORKERS = ["scene-worker-1", "scene-worker-2"]
# ============================

#fps = 30
s3 = boto3.client("s3", region_name=REGION)

# ------------------------------
# Utilities
# ------------------------------
def s3_uri(bucket, key):
    return f"s3://{bucket}/{key}"

def upload_file_to_s3(local_path, s3_key):
    s3.upload_file(local_path, BUCKET, s3_key)
    return s3_uri(BUCKET, s3_key)

def download_file_from_s3(s3_uri_str, local_path):
    parsed = s3_uri_str.replace("s3://", "").split("/", 1)
    bucket = parsed[0]; key = parsed[1]
    s3.download_file(bucket, key, local_path)

# ------------------------------
# Scene detection (OpenCV histogram)
# ------------------------------
def detect_scenes(video_path, threshold=0.5, min_scene_len=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")
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

# ------------------------------
# Motion intensity per-scene (frame abs-diff)
# ------------------------------
def compute_motion_intensity(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release(); return 0.0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_sum = 0.0; count = 0
    for f in range(start_frame+1, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        motion_sum += float(np.mean(diff))
        prev_gray = gray
        count += 1
    cap.release()
    return motion_sum / max(count, 1)

# ------------------------------
# Save scene clip to local file
# ------------------------------
def save_scene_clip(video_path, start_frame, end_frame, out_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for f in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    out.release()
    cap.release()
    return out_path

# ------------------------------
# Assign scenes to workers (motion-adaptive)
# ------------------------------
def assign_scenes(video_path, scene_boundaries):
    scenes = []
    max_motion = 0.0
    for i in range(len(scene_boundaries)-1):
        s = scene_boundaries[i]; e = scene_boundaries[i+1]
        motion = compute_motion_intensity(video_path, s, e)
        scenes.append({"start": s, "end": e, "motion": motion})
        max_motion = max(max_motion, motion)    
    for sc in scenes:
        sc["est_interp"] = max(1, int(MAX_INTERP * (sc["motion"] / (max_motion or 1.0))))
        #if sc["est_interp"] < 3:
        #    sc["est_interp"] = 1
        #else:
        #    sc["est_interp"] = 2
        if sc["motion"] <= max_motion/3:
            sc["est_interp"] = 1
        elif sc["motion"] <= ((2*max_motion)/3):
            sc["est_interp"] = 2
        else:
            sc["est_interp"] = 3
    
    assign = {w: [] for w in WORKERS}
    loads = {w: 0 for w in WORKERS}
    for sc in scenes:
        target = min(WORKERS, key=lambda w: loads[w])
        # print(target,loads[target],sc["start"],sc["end"],"=================================================")
        assign[target].append(sc)
        loads[target] += (sc["end"] - sc["start"] + 1)
        #loads[target] += ((sc["end"] - sc["start"] + 1) * ((2**sc["est_interp"])-1))
        print(target,loads[target],sc["start"],sc["end"],"=================================================")
    print("Assigned loads:", loads)
    return assign

# ------------------------------
# Upload scene files and create job JSONs on S3
# ------------------------------
def upload_scenes_and_create_jobs(video_path, assignment, local_tmp_dir="tmp_scenes"):
    os.makedirs(local_tmp_dir, exist_ok=True)
    job_objects = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = []
        for worker_id, scenes in assignment.items():
            job_objects[worker_id] = {"scenes": []}
            for idx, sc in enumerate(scenes):
                start, end = sc["start"], sc["end"]
                local_name = f"scene_{start}_{end}.mp4"
                local_path = os.path.join(local_tmp_dir, local_name)
                futures.append(ex.submit(save_scene_clip, video_path, start, end, local_path))
                sc["_local_path"] = local_path
        for f in futures: f.result()

        upload_futures = []
        for worker_id, scenes in assignment.items():
            for sc in scenes:
                local_path = sc["_local_path"]
                s3_key = f"{SCENES_PREFIX}/{os.path.basename(local_path)}"
                upload_futures.append(ex.submit(upload_file_to_s3, local_path, s3_key))
                sc["_s3_uri"] = s3_uri(BUCKET, s3_key)
        for f in upload_futures: f.result()

        for worker_id, scenes in assignment.items():
            job_scenes = []
            for sc in scenes:
                job_scenes.append({"s3_uri": sc["_s3_uri"],"interp": sc["est_interp"] })
            job_key = f"{JOB_PREFIX}/{worker_id}.json"
            job_body = json.dumps({"scenes": job_scenes})
            s3.put_object(Bucket=BUCKET, Key=job_key, Body=job_body)

            print(f"Uploaded job JSON s3://{BUCKET}/{job_key} for {worker_id}")

            job_objects[worker_id] = {
                "job_s3": s3_uri(BUCKET, job_key),
                "scenes": job_scenes
            }

    return job_objects

# ------------------------------
# Poll for outputs and download them
# ------------------------------
def wait_for_and_download_outputs(job_objects, fps, local_output_dir="worker_outputs"):
    os.makedirs(local_output_dir, exist_ok=True)
    expected_outputs = []
    for worker, info in job_objects.items():
        for s3_scene in info["scenes"]:
            parsed = s3_scene["s3_uri"].replace("s3://", "").split("/", 1)
            key = parsed[1]
            base = os.path.splitext(os.path.basename(key))[0]
            out_key = f"{OUTPUTS_PREFIX}/{base}_interp.mp4"
            expected_outputs.append({"worker": worker, "s3_key": out_key, "start": int(base.split("_")[1])})
    print("Expecting outputs:", len(expected_outputs))
    remaining = expected_outputs.copy()
    while remaining:
        to_remove = []
        for item in remaining:
            try:
                s3.head_object(Bucket=BUCKET, Key=item["s3_key"])
                print("Found", item["s3_key"])
                local_path = os.path.join(local_output_dir, os.path.basename(item["s3_key"]))
                s3.download_file(BUCKET, item["s3_key"], local_path)
                item["local_path"] = local_path
                to_remove.append(item)
            except s3.exceptions.ClientError:
                pass
        for x in to_remove:
            remaining.remove(x)
        if remaining:
            print(f"{len(remaining)} outputs missing, sleeping {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)
    outputs_sorted = sorted([{"start": it["start"], "local": it["local_path"]} for it in expected_outputs], key=lambda x: x["start"])
    ordered_paths = [x["local"] for x in outputs_sorted]
    # -------------------------------------------
    # Convert each interpolated scene back to original FPS (slow motion)
    # -------------------------------------------
    print("Converting interpolated scenes back to original FPS for slow-motion...")

    converted_paths = []
    original_fps = fps  # obtained earlier from detect_scenes()
    print(f"ORIGINAL FPS VALUE--",original_fps) 
    for scene_path in ordered_paths:
        converted_path = scene_path.replace(".mp4", "_origfps.mp4")
        # cmd = f"ffmpeg -y -i {scene_path} -vf fps={original_fps} -c:v libx264 {converted_path}"
        cap3 = cv2.VideoCapture(scene_path)
        interp_fps = cap3.get(cv2.CAP_PROP_FPS)
        slowdown = interp_fps / original_fps
        cmd = (
        f"ffmpeg -y -i {scene_path} "
        f'-filter:v "setpts={slowdown}*PTS" '
        f"-r {original_fps} {converted_path}"
        )
        print("Running:", cmd)
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0:
            raise RuntimeError(f"FPS conversion failed for: {scene_path}")
        converted_paths.append(converted_path)

    # Now use converted paths for final concat
    ordered_paths = converted_paths

    return ordered_paths

# ------------------------------
# Concat using ffmpeg
# ------------------------------
def concat_videos(video_paths, out_path="final_output.mp4"):
    listfile = "concat_list.txt"
    with open(listfile, "w") as f:
        for p in video_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = f"ffmpeg -y -f concat -safe 0 -i {listfile} -c copy {out_path}"
    print("Running:", cmd)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg concat failed")
    print("Created", out_path)
    return out_path

# ------------------------------
# Main flow
# ------------------------------
def main(video_file):
    print("Detecting scenes...")
    st_1 = time.time()
    scene_boundaries, fps = detect_scenes(video_file)
    st_2 = time.time()
    print("Scene boundaries:", scene_boundaries)

    print("Assigning scenes to workers...")
    assignment = assign_scenes(video_file, scene_boundaries)
    st_3 = time.time()
    print("Saving scenes and uploading to S3, creating job JSONs...")
    job_objects = upload_scenes_and_create_jobs(video_file, assignment)
    st_4 = time.time()
    print("Waiting for worker outputs on S3...")
    ordered_outputs = wait_for_and_download_outputs(job_objects, fps)
    st_5 = time.time()
    print("Concatenating outputs...")
    final = concat_videos(ordered_outputs)
    st_6 = time.time()
    print("Total Time: ", st_6 - st_1, st_6 - st_5, st_5 - st_4, st_4 - st_3, st_3 - st_2, st_2 - st_1)
    print("Pipeline complete. Final video:", final)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 master_s3.py <input_video>")
        sys.exit(1)
    main(sys.argv[1])

