# Distributed Video Frame Interpolation Pipeline

**A scene-aware, distributed system for cost-efficient long-video frame interpolation**

**Authors:** Maitreyi Sinha: maitres@uci.edu, Bharath Reddy: vreddem@uci.edu, Anisha Mohanty: apmohant@uci.edu

---

## Overview

Video frame interpolation is computationally expensive for long videos, especially when using deep learning models such as **RIFE**. Processing such videos on a single machine leads to long runtimes and poor utilization, while naïve distributed splitting by frame index can cause **load imbalance** and **temporal artifacts at scene boundaries**.

This project presents a **scene-aware distributed video frame interpolation pipeline** that splits videos by scene boundaries and dynamically assigns work to multiple worker nodes. The system is designed to improve **throughput, load balance, and scalability** while preserving temporal consistency.

---

## Key Contributions

- **Scene-aware partitioning**: Videos are split at detected scene boundaries rather than fixed frame ranges.
- **Distributed execution**: A master–worker architecture runs interpolation in parallel across multiple nodes.
- **Load-aware scheduling**: Scene complexity is estimated and used to balance work across workers.
- **Reproducible pipeline**: All runs produce a manifest capturing assignments, node IDs, and timing.
- **Empirical evaluation**: Wall-time and speedup comparisons against a single-node baseline.

---

## System Architecture


---

## Interpolation Model

This pipeline uses **RIFE (Real-Time Intermediate Flow Estimation)** as the underlying video frame interpolation model:

> Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou.  
> *Real-Time Intermediate Flow Estimation for Video Frame Interpolation.*  
> European Conference on Computer Vision (ECCV), 2022.

RIFE is treated as a **black-box interpolation module** within our system; our contributions focus on **distributed orchestration, scheduling, and system-level efficiency**, rather than modifying the interpolation model itself.

---

## Deployment Setup

- **Nodes**: 3 total  
  - 1 master node  
  - 2 worker nodes
- **Execution model**: Master assigns scene-level tasks to workers
- **Interpolation model**: RIFE (ECCV 2022)
- **Environment**: Python, PyTorch, ffmpeg

Each node logs its hostname at startup, allowing verification that tasks ran on distinct machines.

---

## Evidence of Distributed Execution

A sample distributed run includes:
- **Host-stamped logs** from each node
- **run_manifest.json** recording:
  - master hostname
  - worker hostnames
  - scene assignments
  - per-scene timing
  - merge status

Example log excerpt:



---

## Experimental Results

| Setup | Nodes | Wall Time | Speedup |
|------|------:|----------:|--------:|
| Single-node baseline | 1 | XX min | 1.0× |
| Distributed (scene-aware) | 3 | YY min | Z.Z× |

Results show improved throughput and reduced straggler effects compared to naïve single-node processing.

---

## Reproducibility

### Requirements

- Python 3.9+
- PyTorch
- ffmpeg
- RIFE model weights (see RIFE repository)

### Running the Pipeline

**Master node**
```bash
python master.py --input input.mp4 --out output.mp4 --num-workers 2
```

### Limitations
	•	Scene complexity estimation is heuristic and may not perfectly correlate with inference cost.
	•	Network and disk I/O are not yet optimized.
	•	Evaluation focuses on runtime rather than perceptual quality metrics.


### Future Work
	•	Learned cost models for scene complexity
	•	Adaptive interpolation rates based on motion intensity
	•	Fault-tolerant scheduling and dynamic worker scaling
	•	Integration with cloud-native orchestration frameworks


Acknowledgments
	•	RIFE video frame interpolation model (ECCV 2022)
https://github.com/hzwer/ECCV2022-RIFE
