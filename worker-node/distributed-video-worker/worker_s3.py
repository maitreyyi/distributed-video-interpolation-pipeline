#!/usr/bin/env python3
"""
worker_s3.py
Worker: poll job JSON on S3 (jobs/job_001/<WORKER_ID>.json),
download each scene, run interpolation, upload interpolated results to s3://<BUCKET>/outputs/
"""

import os
import time
import json
import boto3
import subprocess
from urllib.parse import urlparse

# ========== CONFIG ==========
BUCKET = "s3-scene-dect-bucket"   # your bucket
JOB_PREFIX = "jobs/job_001"
OUTPUTS_PREFIX = "outputs"
WORKER_ID = "scene-worker-1"      # <-- CHANGE THIS ON WORKER 2
POLL_INTERVAL = 5
TMP_DIR = "/tmp/worker_scenes"
REGION = "us-west-2"
# ============================

os.makedirs(TMP_DIR, exist_ok=True)
s3 = boto3.client("s3", region_name=REGION)

def s3_to_local(s3_uri, local_path):
    parsed = s3_uri.replace("s3://","").split("/",1)
    s3.download_file(parsed[0], parsed[1], local_path)
    return local_path

def local_to_s3(local_path, s3_key):
    s3.upload_file(local_path, BUCKET, s3_key)
    return f"s3://{BUCKET}/{s3_key}"

# Replace with your actual interpolation model (RIFE/DAIN)
def run_interpolation_model(local_in, local_out):
    import shutil, time
    print("Simulating interpolation (copy)...")
    shutil.copy(local_in, local_out)
    time.sleep(1)
    print("Done simulated interpolation:", local_out)

def poll_for_job_and_process():
    job_key = f"{JOB_PREFIX}/{WORKER_ID}.json"
    while True:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=job_key)
            body = obj['Body'].read().decode('utf-8')
            job = json.loads(body)
            scenes = job.get("scenes", [])
            if not scenes:
                print("No scenes in job. Sleeping...")
                time.sleep(POLL_INTERVAL)
                continue

            print(f"Found job with {len(scenes)} scenes.")
            st1 = time.time()
            outputs = []

            for s3_scene in scenes:
                fname = os.path.basename(s3_scene["s3_uri"])
                local_in = os.path.join(TMP_DIR, fname)
                local_out = os.path.join(TMP_DIR, f"{os.path.splitext(fname)[0]}_interp.mp4")

                # print("Downloading:", s3_scene)
                # s3_to_local(s3_scene, local_in)

                print("Downloading:", s3_scene["s3_uri"])
                s3_to_local(s3_scene["s3_uri"], local_in)

                print("Running interpolation for", local_in)
                # run_interpolation_model(local_in, local_out)
                subprocess.run(["/home/ubuntu/ECCV2022-RIFE/rife_env/bin/python", "inference_video.py", "--video", local_in, "--exp", str(s3_scene["interp"]), "--output", local_out], check=True, cwd = "/home/ubuntu/ECCV2022-RIFE")

                out_key = f"{OUTPUTS_PREFIX}/{os.path.basename(local_out)}"
                print("Uploading:", out_key)
                local_to_s3(local_out, out_key)
                outputs.append(f"s3://{BUCKET}/{out_key}")
            end1 = time.time()
            print("Time take by this worker: ",end1-st1)
            print("Worker finished job; uploaded outputs:", outputs)
            return

        except s3.exceptions.NoSuchKey:
            print("No job file yet, sleeping...")
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print("Error polling/processing:", e)
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    print("Worker starting, ID:", WORKER_ID)
    poll_for_job_and_process()
    print("Worker done; exiting.")

