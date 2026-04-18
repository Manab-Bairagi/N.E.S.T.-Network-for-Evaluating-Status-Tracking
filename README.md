<div align="center">

# 🏥 N.E.S.T.
### Network for Evaluating Status & Tracking

**An intelligent, real-time healthcare monitoring pipeline that fuses skeleton-based action recognition, multi-person tracking, and a conversational RAG-LLM interface for caregiving environments.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-0F9D58?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [System Architecture](#-system-architecture)
3. [Key Features](#-key-features)
4. [Technology Stack](#-technology-stack)
5. [Project Structure](#-project-structure)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [Pipeline Deep Dive](#-pipeline-deep-dive)
9. [Action Classes](#-action-classes)
10. [JSON Output Format](#-json-output-format)
11. [RAG-LLM Query Interface](#-rag-llm-query-interface)
12. [Model Weights](#-model-weights)
13. [Configuration Reference](#-configuration-reference)

---

## 🔍 Overview

**N.E.S.T.** (Network for Evaluating Status & Tracking) is a full-stack, real-time action recognition and monitoring system designed for **healthcare and caregiving environments**. It processes video footage from a scene and performs:

- **Multi-person detection and tracking** using a custom ReID model and BoT-SORT tracker
- **3D skeleton extraction** using Google's MediaPipe Pose Landmarker
- **Skeleton-based action recognition** using a fine-tuned **EfficientGCN-B0** model trained on **45 NTU RGB+D action classes**
- **Interaction detection** between patients and caregivers
- **Structured JSON event logging** with per-frame metadata
- **Natural language querying** over the event log using a RAG + LLM pipeline (Qwen via Ollama)

The goal is to provide automated, unobtrusive observational intelligence in healthcare scenarios — flagging distress actions (falling, chest pain, staggering), recognising routine activities, and enabling caregivers or supervisors to query what happened at any given moment.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT VIDEO                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │   YOLOv8  Detector  │  (person class only)
                  └──────────┬──────────┘
                             │ detections [x1,y1,x2,y2,conf]
                  ┌──────────▼──────────────────────────┐
                  │   MemoryEnhancedBoTSORT Tracker      │
                  │  ┌──────────┐  ┌─────────────────┐  │
                  │  │NSA Kalman│  │ Custom ReID CNN  │  │
                  │  │  Filter  │  │(ImprovedWideRes) │  │
                  │  └──────────┘  └─────────────────┘  │
                  │  ┌──────────┐  ┌─────────────────┐  │
                  │  │   GMC    │  │ Appearance Buffer│  │
                  │  │(ORB+RANSAC  │  (gallery/re-ID) │  │
                  │  └──────────┘  └─────────────────┘  │
                  └──────────┬──────────────────────────┘
                             │ stable track IDs + bboxes
          ┌──────────────────┼──────────────────────┐
          │                  │                      │
 ┌────────▼────────┐ ┌───────▼────────┐  ┌──────────▼──────────┐
 │ MediaPipe Pose  │ │  Interaction   │  │  Bounding Box       │
 │  Landmarker     │ │   Detector     │  │  Expansion (×1.5)   │
 │ (33 landmarks,  │ │(IoU+dist+persist│  └──────────┬──────────┘
 │   x, y, z)      │ │   temporal)    │             │
 └────────┬────────┘ └───────┬────────┘             │
          │                  │ interaction pairs      │
          │ [33,3] landmarks │                       │
 ┌────────▼────────┐         │                       │
 │ MP→NTU25 Mapper │         │                       │
 │ + OneEuro Filter│         │                       │
 │  (stabilisation)│         │                       │
 └────────┬────────┘         │                       │
          │ [25,3] joints    │                       │
 ┌────────▼──────────────────▼───┐                   │
 │         SkeletonBuffer        │                   │
 │  (90-frame rolling window)    │                   │
 │ ┌────────────────────────────┐│                   │
 │ │ generate_features() → 16ch ││                   │
 │ │  joint(6) + velocity(6)    ││                   │
 │ │  + bone(4)  [T,25,3→16ch]  ││                   │
 │ └────────────────────────────┘│                   │
 └────────┬──────────────────────┘                   │
          │ [1,16,T,25,M] tensor                     │
 ┌────────▼──────────────────┐                       │
 │    EfficientGCN-B0 (3D)   │                       │
 │  ┌───────┐ ┌────────────┐ │                       │
 │  │Spatial│ │STJoint Att │ │                       │
 │  │ Graph │ │ + CrossSt  │ │                       │
 │  │ Conv  │ │ Attention  │ │                       │
 │  └───────┘ └────────────┘ │                       │
 │  45-class softmax output  │                       │
 └────────┬──────────────────┘                       │
          │ (code, label, confidence)                │
 ┌────────▼──────────────────────────────────────────▼──┐
 │             Visualization & JSON Logging              │
 │  • Bounding boxes, skeletons, action labels on video  │
 │  • ActionEventLogger → structured JSON per second     │
 └────────┬──────────────────────────────────────────────┘
          │ output_action.mp4 + action_log.json
 ┌────────▼────────────────────────┐
 │       Post-Processing           │
 │  generate_metadata.py           │
 │  → metadata_output.jsonl        │
 └────────┬────────────────────────┘
          │
 ┌────────▼────────────────────────┐
 │     RAG + LLM Query Layer       │
 │  SentenceTransformers (MiniLM)  │
 │  + Qwen 4B via Ollama           │
 │  → Natural language Q&A         │
 └─────────────────────────────────┘
```

---

## ✨ Key Features

### 🎯 Multi-Person Detection & Tracking
- **YOLOv8-based** person detection (class 0 only) with configurable confidence threshold
- **MemoryEnhancedBoTSORT** — a custom extension of BoT-SORT with:
  - **NSA Kalman Filter** — Noise Scale Adaptive; dynamically scales measurement trust based on detection confidence
  - **GMC (Global Motion Compensation)** — uses ORB features + RANSAC to compensate for camera motion
  - **Custom ReID backbone** (`ImprovedDeepSortWideResNet`) — a squeeze-excitation ResNet with GeM pooling, trained to 128-dim appearance embeddings
  - **Appearance gallery** per track for robust re-identification after occlusion

### 🦴 3D Skeleton Extraction
- **MediaPipe Pose Landmarker Heavy** (30M+ params) extracts 33 landmarks per person with x, y, z
- **MP→NTU25 mapping** remaps MediaPipe's 33 landmarks to the NTU RGB+D 25-joint skeleton format
- Z-depth is scaled by crop width to match NTU's depth-proportional convention
- **OneEuro filtering** smooths skeleton sequences per track to eliminate jitter
- `snap_missing_to_spine` snaps undetected joints to the spine, preventing zero-valued features

### 🧠 Action Recognition — EfficientGCN-B0 (3D)
- Fine-tuned on **45 NTU RGB+D action classes** covering patient distress, daily activities, caregiver actions, and two-person interactions
- **Three-stream feature extraction** from skeleton sequences:
  - **Joint stream** (6ch): absolute (x,y,z) + spine-relative (x,y,z)
  - **Velocity stream** (6ch): fast Δ (2-frame) + slow Δ (1-frame) per axis
  - **Bone stream** (4ch): bone delta vector (dx, dy, dz) + 3D Euclidean bone length
- **Cross-stream attention** soft-fuses the three streams before final classification
- **ST-Joint Attention** applies spatial (per joint) and temporal attention simultaneously
- **PredictionCache** holds predictions for N frames to avoid redundant inference every frame

### 🤝 Interaction Detection
- Detects when two tracked individuals are in proximity using:
  - IoU overlap threshold between bounding boxes
  - Centre-to-centre pixel distance (with dynamic scaling by bounding box diagonal)
  - **Temporal persistence** gate — N consecutive frames required before interaction is declared active
- Separate skeleton buffers for `(idA, idB)` pairs enable *joint* action recognition for the interacting individuals

### 📋 Structured JSON Logging
- **ActionEventLogger** captures a full event log in JSON, sampled once per second:
  - Session metadata (video path, FPS, resolution, model info, device)
  - Per-frame: bounding boxes, action codes, labels, confidence, category, interactions (IoU, pixel distance, skeleton spine distance)
  - Summary: action distribution, unique track IDs, total recognitions
- 45 classes are taxonomised into four semantic categories: `patient_specific`, `caregiver_specific`, `interaction_based`, `common`

### 🗣 RAG-LLM Query Interface
- `generate_metadata.py` converts the raw JSON log into a clean `.jsonl` file, grouping events by second with role attribution
- `rag_retriever.py` uses `all-MiniLM-L6-v2` (SentenceTransformers) for semantic retrieval over the event log
- `rag_llm.py` + `rag_main.py` provide an interactive CLI where you can ask natural-language questions like:
  - *"Did anyone fall down?"*
  - *"What was the caregiver doing at 10:05 AM?"*
  - *"Were there any interactions between patients?"*

---

## 🔧 Technology Stack

| Component | Technology |
|---|---|
| Person Detection | YOLOv8 (Ultralytics) |
| Multi-Object Tracking | BoT-SORT (custom — NSA Kalman + GMC + ReID) |
| ReID Backbone | Custom WideResNet + SE blocks + GeM pooling |
| Pose Estimation | MediaPipe Pose Landmarker Heavy |
| Skeleton Smoothing | One Euro Filter |
| Action Recognition | EfficientGCN-B0 (3D, fine-tuned on NTU RGB+D 45-class) |
| Deep Learning | PyTorch 2.0+ |
| Video Processing | OpenCV 4.8+ |
| Metadata Generation | Python (json, datetime, collections) |
| Semantic Retrieval | SentenceTransformers `all-MiniLM-L6-v2` |
| LLM Backend | Qwen 4B via Ollama |
| Scientific Computing | NumPy, SciPy |

---

## 📁 Project Structure

```
NEST/
├── mediapipe_inference.py       # Main inference pipeline (detection → tracking →
│                                #   pose → action recognition → logging)
├── customtrackerfinal.py        # Custom BoT-SORT implementation
│                                #   (NSA Kalman, GMC, ReID backbone, Track class)
├── generate_metadata.py         # Post-processing: JSON log → structured JSONL
├── rag_retriever.py             # Semantic retrieval over JSONL (MiniLM embeddings)
├── rag_llm.py                   # Qwen LLM wrapper (Ollama API)
├── rag_main.py                  # Interactive RAG CLI entry point
├── requirements.txt             # Python dependencies
│
├── best_efficientgcn_b0_media_3d_81.pth   # Action recognition model weights
├── best_model.pth                          # ReID model weights (BoT-SORT)
├── pose_landmarker_heavy.task              # MediaPipe pose model
├── yolo26s.pt                              # YOLO detection model
│
├── output_action.mp4            # Annotated output video (generated)
├── output_action_action_log.json# Structured action event log (generated)
└── metadata_output.jsonl        # Cleaned metadata for RAG (generated)
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU recommended (CPU fallback is supported)
- [Ollama](https://ollama.com/) installed for the LLM query interface

### 1. Clone the repository

```bash
git clone https://github.com/Manab-Bairagi/N.E.S.T.-Network-for-Evaluating-Status-Tracking.git
cd N.E.S.T.-Network-for-Evaluating-Status-Tracking
```

### 2. Install PyTorch (GPU — CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> For CPU-only, just run `pip install torch torchvision torchaudio`

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe pose model

```bash
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
    'pose_landmarker_heavy.task'
)
print('Downloaded pose_landmarker_heavy.task')
"
```

### 5. Pull the Qwen model via Ollama (for RAG)

```bash
ollama pull qwen:4b
```

---

## 🚀 Usage

### Step 1 — Run the Inference Pipeline

```bash
python mediapipe_inference.py
```

> Edit the bottom of `mediapipe_inference.py` (Cell 11) to point `video_path` at your input video before running.

**Key parameters you can tune in Cell 11:**

```python
process_video_with_action_recognition(
    video_path      = 'your_video.mp4',   # Input video
    output_path     = 'output_action.mp4',# Annotated output
    reid_model_path = 'best_model.pth',   # ReID weights
    conf_threshold  = 0.5,                # YOLO detection confidence
    max_frames      = None,               # None = process all frames
    buffer_len      = 60,                 # Temporal window (frames)
    bbox_scale      = 1.5,                # Crop expansion for pose
    iou_thresh      = 0.20,               # Interaction IoU threshold
    dist_thresh     = 150,                # Interaction distance (px)
    persist_frames  = 10,                 # Frames to confirm interaction
    inference_every = 4,                  # Run model every N frames
)
```

**Outputs:**
- `output_action.mp4` — annotated video with bounding boxes, skeletons, action labels, and interaction overlays
- `output_action_action_log.json` — structured JSON log

---

### Step 2 — Generate Metadata for RAG

```bash
python generate_metadata.py
```

Reads `output_action_action_log.json` → produces `metadata_output.jsonl` (events grouped by second with role attribution).

---

### Step 3 — Launch the RAG Query Interface

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull and serve Qwen
ollama run qwen:4b

# Terminal 3: Launch the RAG CLI
python rag_main.py
```

**Example interaction:**
```
>> Did anyone fall down during the session?
🧠 Answer:
At 2026-03-25T10:00:15, a patient was observed falling down, 
followed by a caregiver providing support shortly after.
--------------------------------------------------

>> What actions did the caregiver perform?
🧠 Answer:
The caregiver was observed patting someone on the back 
and supporting somebody at multiple timestamps during the session.
--------------------------------------------------

>> exit
```

---

## 🔬 Pipeline Deep Dive

### Feature Extraction — Three-Stream Skeleton Encoding

Given a 90-frame skeleton sequence `[T, 25, 3]` per person, three feature streams are computed:

| Stream | Channels | Description |
|---|---|---|
| **Joint** | 6 | Absolute (x,y,z) normalized coords + Spine-relative (x,y,z) offsets |
| **Velocity** | 6 | Fast Δ (frame t+2 − t) + Slow Δ (frame t+1 − t), per axis |
| **Bone** | 4 | Bone delta vector (dx, dy, dz) + 3D Euclidean bone length |

All streams are concatenated into a `[16, T, 25, M]` tensor where M is the number of persons (1 for solo, 2 for an interacting pair).

### EfficientGCN-B0 Architecture

```
InitialBlock(6ch → 64ch)  ×3 (joint / velocity / bone streams)
        ↓
Stage1 GCNBlock(64 → 48ch) ×3 streams separately
        ↓
Stage2 GCNBlock(48 → 16ch) ×3 streams separately
        ↓
CrossStreamAttention → weighted soft-fusion (16ch per stream → 16ch fused)
        ↓
Concatenate [joint | velocity | bone | fused] → 64ch
        ↓
Stage3 GCNBlock(64 → 64ch, stride=2, depth=1)
        ↓
Stage4 GCNBlock(64 → 128ch, stride=2, depth=1)
        ↓
AdaptiveAvgPool → Dropout(0.30) → FC(128 → 45)
        ↓
Softmax → (class_code, label, confidence)
```

Each `GCNBlock` = `SpatialGraphConv` + `SGLayer` (depthwise temporal conv) + `STJointAtt` (spatial × temporal attention).

### Tracker — MemoryEnhancedBoTSORT

The tracker combines four key components:

1. **NSA Kalman Filter** — adapts measurement covariance R inversely to detection confidence. Low-confidence detections (occluded, blurry) are treated with high noise, trusting the motion prediction more.

2. **GMC (Global Motion Compensation)** — detects ORB keypoints on a downscaled grayscale frame, matches them across consecutive frames, estimates an affine transform via RANSAC, and compensates all track positions accordingly.

3. **Custom ReID backbone** — `ImprovedDeepSortWideResNet` with:
   - Efficient stem (32→32→MaxPool)
   - 3 residual stages (32→64→128) with Squeeze-and-Excitation
   - Generalized Mean Pooling (GeM) for robust global descriptor
   - BatchNorm neck + 128-dim L2-normalized embeddings

4. **Fused cost matrix** — matching uses `λ_IoU × IoU_cost + λ_ReID × ReID_cost` (default 0.98 / 0.02), prioritising spatial consistency while using appearance for disambiguation.

---

## 🎭 Action Classes

The model recognises **45 NTU RGB+D actions**, grouped into four healthcare-relevant categories:

### 🧑‍⚕️ Patient-Specific (25 actions)
`drink water` · `eat meal` · `brush teeth` · `reading` · `writing` · `put on glasses` · `take off glasses` · `jump up` · `sneeze/cough` · **`staggering`** · **`falling down`** · **`headache`** · **`chest pain`** · **`back pain`** · **`neck pain`** · **`nausea/vomiting`** · `fan self` · `squat down` · `apply cream on face` · `apply cream on hand` · `put object into bag` · `take object out of bag` · `open a box` · `move heavy objects` · `yawn`

### 👨‍⚕️ Caregiver-Specific (5 actions)
`pat on back` · `giving object` · `carry object` · `follow` · **`support somebody`**

### 🤝 Interaction-Based (10 actions)
`phone call` · `punch/slap` · `hugging` · `shaking hands` · `walking towards` · `walking apart` · `hit with object` · `wield knife` · `knock over` · `grab stuff`

### 🔄 Common (5 actions)
`drop` · `pick up` · `sit down` · `stand up` · `point finger`

> **Bold** = distress or clinically significant actions

---

## 📊 JSON Output Format

`output_action_action_log.json` structure:

```json
{
  "session": {
    "video_path": "Deepark.mp4",
    "output_video_path": "output_action.mp4",
    "processed_at": "2026-04-18T21:00:00",
    "fps": 25,
    "resolution": { "width": 1920, "height": 1080 },
    "model": "EfficientGCN-B0 (3D, NTU 45-class)",
    "device": "cuda",
    "num_action_classes": 45
  },
  "action_categories": {
    "patient_specific": [ { "code": "A043", "label": "falling down" }, "..." ],
    "caregiver_specific": [ "..." ],
    "interaction_based": [ "..." ],
    "common": [ "..." ]
  },
  "summary": {
    "total_frames_processed": 150,
    "unique_track_ids": [0, 1, 2],
    "total_recognitions": 87,
    "action_distribution": { "falling down": 12, "support somebody": 9 }
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "tracks": {
        "0": { "bbox": [120.0, 80.0, 320.0, 540.0], "has_skeleton": true }
      },
      "actions": {
        "0": { "code": "A043", "label": "falling down", "confidence": 0.82, "category": "patient_specific" }
      },
      "interactions": [
        {
          "track_a": 0, "track_b": 1, "interaction_active": true,
          "distance_px": 134.5, "iou": 0.21,
          "skeleton_spine_distance_px": 118.3
        }
      ]
    }
  ]
}
```

---

## 🗣 RAG-LLM Query Interface

The query system follows a **Retrieval-Augmented Generation (RAG)** pattern:

```
User Question
     │
     ▼
SentenceTransformer (all-MiniLM-L6-v2)
     │  encode query → 384-dim embedding
     ▼
Cosine Similarity over all event embeddings
     │  top-k most relevant events
     ▼
Prompt construction:
  "At <timestamp>, a <role> was performing <action>." × k
     │
     ▼
Qwen 4B (via Ollama local inference)
     │  natural language answer
     ▼
Answer printed to terminal
```

The LLM is instructed to:
- Answer descriptively (not yes/no)
- Stay grounded in the observed events only
- Clearly state when an event was **not** observed
- Keep answers concise (1–3 sentences)

---

## 🔑 Model Weights

| File | Description | Size |
|---|---|---|
| `best_efficientgcn_b0_media_3d_81.pth` | EfficientGCN-B0 action recognition weights | ~1.3 MB |
| `best_model.pth` | ReID backbone (BoT-SORT appearance model) | ~9.3 MB |
| `pose_landmarker_heavy.task` | MediaPipe Pose Landmarker Heavy | ~29 MB |
| `yolo26s.pt` | YOLOv8s person detection | ~19.5 MB |

> All weight files should be placed in the **project root directory** alongside the scripts.

---

## ⚙️ Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `conf_threshold` | `0.5` | YOLO detection confidence cutoff |
| `buffer_len` | `60` | Skeleton buffer length (frames) used as temporal window |
| `bbox_scale` | `1.5` | Bounding box expansion factor for MediaPipe crop |
| `iou_thresh` | `0.20` | Minimum IoU to consider two persons as interacting |
| `dist_thresh` | `150` | Maximum centre-to-centre distance (px) for interaction |
| `persist_frames` | `10` | Consecutive frames required to confirm an interaction |
| `inference_every` | `4` | Run action model every N frames (uses cache in between) |
| `CONFIDENCE_THRESHOLD` | `0.10` | Minimum softmax confidence to report a prediction |
| `MAX_FRAMES` | `90` | Model's temporal window (training-fixed) |
| `NUM_JOINTS` | `25` | NTU 25-joint skeleton |
| `DROPOUT` | `0.30` | EfficientGCN-B0 dropout rate |

---

## 👥 Team

**N.E.S.T.** — *Network for Evaluating Status & Tracking*

Built as a healthcare AI monitoring research project.

---

<div align="center">

*If you find this project useful, consider giving it a ⭐*

</div>
