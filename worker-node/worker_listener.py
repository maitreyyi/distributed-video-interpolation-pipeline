#!/usr/bin/env python3
import boto3
import json
import os
import time

# --- AWS CONFIGURATION ---
SQS_REGION = "us-west-2"
S3_REGION = "us-west-2"
JOB_QUEUE_URL = "https://sqs.us-west-2.amazonaws.com/993471822684/scene-job"
STATUS_QUEUE_URL = "https://sqs.us-west-2.amazonaws.com/993471822684/scene-status"
BUCKET_NAME = "s3-scene-dect-bucket"

# --- AWS CLIENTS ---
sqs = boto3.client("sqs", region_name=SQS_REGION)
s3 = boto3.client("s3", region_name=S3_REGION)

print("🎯 Worker started.")
print(f"📬 Listening for messages from: {JOB_QUEUE_URL}")
print(f"💾 Will fetch files from bucket: {BUCKET_NAME} ({S3_REGION})")

# --- MAIN LOOP ---
while True:
    try:
        # Poll SQS for new jobs
        response = sqs.receive_message(
            QueueUrl=JOB_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20
        )

        messages = response.get("Messages", [])
        if not messages:
            print("⏳ No messages yet...")
            continue

        # Process one message
        msg = messages[0]
        body = json.loads(msg["Body"])
        receipt = msg["ReceiptHandle"]

        print("\n📩 Received job:", body)

        # Handle a 'download' action
        if body.get("action") == "download":
            s3_key = body["s3_key"]
            target_path = body.get("target_path", "/home/ubuntu/video.mp4")

            print(f"⬇️  Downloading s3://{BUCKET_NAME}/{s3_key} → {target_path}")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            s3.download_file(BUCKET_NAME, s3_key, target_path)
            print("✅ Download complete!")

            # Send success status back to the status queue
            status_msg = {
                "job_id": body["job_id"],
                "status": "downloaded",
                "file": s3_key,
                "worker": os.uname()[1],
            }
            sqs.send_message(QueueUrl=STATUS_QUEUE_URL, MessageBody=json.dumps(status_msg))
            print(f"📤 Status sent to scene-status queue: {status_msg}")

        else:
            print("⚠️ Unknown job action:", body.get("action"))

        # Delete processed message
        sqs.delete_message(QueueUrl=JOB_QUEUE_URL, ReceiptHandle=receipt)
        print("🗑️  Message deleted from job queue.\n")

    except Exception as e:
        print("❌ Error while processing message:", e)
        time.sleep(5)

