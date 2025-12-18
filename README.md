# Distributed Video Frame Interpolation Pipeline

**A scene-aware, distributed system for cost-efficient long-video frame interpolation**

**Authors:**  
- Maitreyi Sinha (maitres@uci.edu)  
- Bharath Reddy (vreddem@uci.edu)  
- Anisha Mohanty (apmohant@uci.edu)

---

## Overview

Video frame interpolation is computationally expensive for long videos, especially when using deep learning models such as **RIFE**. Processing such videos on a single machine leads to long runtimes and poor utilization, while naïve distributed splitting by frame index can cause **load imbalance** and **temporal artifacts at scene boundaries**.

This project presents a **scene-aware distributed video frame interpolation pipeline** that splits videos by detected scene boundaries and dynamically assigns work to multiple worker nodes. The system improves **throughput, load balance, and scalability** while preserving temporal consistency in the final output.

---

## Key Contributions

- **Scene-aware partitioning:** Videos are split at scene boundaries rather than fixed frame ranges.
- **Distributed execution:** A master–worker architecture executes interpolation in parallel across multiple nodes.
- **Load-aware scheduling:** Scene complexity is estimated and used to balance workloads.
- **Reproducible execution:** Each run produces logs and artifacts describing assignments and execution order.
- **System-level evaluation:** Runtime improvements are measured against a single-node baseline.

---

## System Architecture

<img width="577" height="524" alt="System architecture diagram" src="https://github.com/user-attachments/assets/ae46e36c-a702-4439-ab84-429b1a7f023f" />

<img width="1222" height="222" alt="Distributed workflow diagram" src="https://github.com/user-attachments/assets/ef6dcd27-33c0-4032-a50d-e2ca9806279f" />

---

## Interpolation Model

This pipeline uses **RIFE (Real-Time Intermediate Flow Estimation)** as the underlying video frame interpolation model:

> Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou.  
> *Real-Time Intermediate Flow Estimation for Video Frame Interpolation.*  
> European Conference on Computer Vision (ECCV), 2022.

RIFE is treated as a **black-box interpolation module**. Our contributions focus on **distributed orchestration, scheduling, and system-level efficiency**, rather than modifying the interpolation model itself.

<img width="515" height="436" alt="RIFE interpolation example" src="https://github.com/user-attachments/assets/a3d90e89-0d05-437c-aca0-9c1dabe38ce7" />

---

## Deployment Setup

- **Nodes:** 3 total  
  - 1 master node  
  - 2 worker nodes
- **Execution model:** Master assigns scene-level tasks to workers
- **Interpolation model:** RIFE (ECCV 2022)
- **Environment:** Python, PyTorch, ffmpeg
- **Coordination layer:** Amazon S3

Each node logs its hostname at startup to verify true multi-node execution.

---

## Running the Pipeline

The system follows a **master–worker execution model**. The master node performs scene detection, scheduling, and output merging. Worker nodes perform RIFE-based interpolation on assigned scenes.

---

### Master Node

1. SSH into the master instance:
```bash
ssh ubuntu@<master-ip>
```

2. Activate the master environment:
```bash
source ~/venv/bin/activate
```

3. Navigate to the project directory:
```bash
cd distributed-video-interpolation
```

4. Run the master pipeline script:
```bash
python master_s3_backup.py ../cpu_input.mp4
```

The master script:
- Detects scene boundaries  
- Estimates scene-level workload  
- Uploads scene clips and job metadata to S3  
- Waits for worker outputs  
- Concatenates interpolated scenes into a final output video  

---

### Worker Nodes

Repeat the following steps on **each worker node**.

1. SSH into the worker instance:
```bash
ssh ubuntu@<worker-ip>
```

2. Navigate to the RIFE directory and activate the RIFE environment:
```bash
cd ECCV2022-RIFE
source rife_env/bin/activate
```

3. Navigate to the distributed worker directory:
```bash
cd ../distributed-worker
```

4. Run the worker process:
```bash
python worker.py
```

Each worker:
- Polls S3 for assigned jobs  
- Downloads scene clips  
- Runs RIFE interpolation  
- Uploads interpolated outputs back to S3  

---

## Evidence of Distributed Execution

A sample master log excerpt demonstrating distributed execution:

```text
[BOOT] role=MASTER host=ip-172-31-2-156
Assigned loads: {'scene-worker-1': 12, 'scene-worker-2': 12}
Waiting for worker outputs on S3...
Found outputs/scene_0_51_interp.mp4
Found outputs/scene_51_102_interp.mp4
...
Concatenating outputs...
Created final_output.mp4
Pipeline complete.
```

Each worker logs its hostname and task assignments, confirming execution across independent nodes.

---

## Experimental Results

<img width="876" height="406" alt="Runtime comparison results" src="https://github.com/user-attachments/assets/340d171d-7d41-402d-b1c4-614b43be0e3b" />

Results show improved throughput and reduced straggler effects compared to naïve single-node processing.

---

## Cleanup After Each Run

To ensure reproducibility and avoid stale artifacts, cleanup is required after **each successful run or failure**.

---

### S3 Cleanup (Master Node)

```bash
aws s3 rm s3://$BUCKET_NAME/scenes/ --recursive
aws s3 rm s3://$BUCKET_NAME/outputs/ --recursive
aws s3 rm s3://$BUCKET_NAME/jobs/ --recursive
```

---

### Local Cleanup (Master Node)

```bash
rm -rf tmp_files output_files
```

> ⚠️ Verify directory names before deletion.

---

## Limitations

- Scene complexity estimation is heuristic and may not perfectly correlate with inference cost.
- Network and disk I/O are not yet optimized.
- Evaluation focuses on runtime rather than perceptual quality metrics.

---

## Future Work

- Support heterogeneous hardware (CPUs, GPUs, accelerators)
- Adaptive auto-scaling based on workload intensity
- Detailed monitoring (metrics, logs, tracing)
- Fault tolerance via task reassignment and checkpointing

---

## Acknowledgments

This project uses the **RIFE** video frame interpolation model (ECCV 2022):  
https://github.com/hzwer/ECCV2022-RIFE

Our work focuses on **distributed systems design and orchestration** built on top of RIFE.
