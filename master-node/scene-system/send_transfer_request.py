import boto3, json, uuid

sqs = boto3.client("sqs", region_name="us-west-2")

# change this to your actual SQS queue URL
QUEUE_URL = "https://sqs.us-west-2.amazonaws.com/993471822684/scene-job"
BUCKET = "s3-scene-dect-bucket"
VIDEO_KEY = "4114797-uhd_3840_2160_25fps.mp4"  # path in your S3 bucket

message = {
    "job_id": str(uuid.uuid4()),
    "action": "download",
    "s3_bucket": BUCKET,
    "s3_key": VIDEO_KEY,
    "target_path": "/home/ubuntu/video.mp4"
}

sqs.send_message(
    QueueUrl=QUEUE_URL,
    MessageBody=json.dumps(message)
)
print("Transfer request sent to worker via SQS.")
