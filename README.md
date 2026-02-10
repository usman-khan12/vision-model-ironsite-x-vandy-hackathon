# Object Permanence for Vision-Language Models

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-CUDA%20%7C%20MPS%20%7C%20CPU-blue)

> Enhances video-understanding VLMs with **object permanence** — the ability to track and reason about objects even when they become occluded (hidden behind other objects). Built for the **IronSite x Vanderbilt Hackathon**.

---

## Overview

Standard vision-language models lose track of objects the moment they disappear from view. This project adds a full **object permanence pipeline** on top of two popular video VLMs ([Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and [LLaVA-NeXT-Video](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf)) so they can reason about objects that are temporarily hidden behind other objects.

The pipeline draws inspiration from [Loci-Looped](https://arxiv.org/abs/2306.02012) and extends it with a novel **Spatial Memory Grid** that maintains "ghost" representations of occluded objects along with predictive occupancy maps.

---

## How It Works

Each video frame is processed through a seven-stage pipeline:

| Stage | Component               | Description                                                                                                                                                                       |
| :---: | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1   | **Feature Extraction**  | Projects model hidden states into per-object feature vectors via learned queries or simple projection + pooling.                                                                  |
|   2   | **Spatial Memory Grid** | Maintains a 2D grid of "ghost" representations for occluded objects with predictive occupancy maps. Re-identifies objects by cross-referencing new observations against the grid. |
|   3   | **Occlusion Detection** | Estimates per-object visibility from feature similarity, magnitude, and detection confidence through a weighted multi-signal MLP.                                                 |
|   4   | **State Prediction**    | Predicts features for the next frame using an MLP (or optional Transformer) with velocity-aware residual connections.                                                             |
|   5   | **Percept Gating**      | Learns separate position and appearance gates that decide when to trust observations vs. predictions, biased toward observation by default.                                       |
|   6   | **Temporal Fusion**     | Blends observed and predicted features via the learned gate values using linear interpolation.                                                                                    |
|   7   | **Prompt Enhancement**  | When occlusion is detected, augments the text prompt with spatial-memory-backed evidence so the VLM reasons about hidden objects.                                                 |

---

## Project Structure

```
.
├── object_permanence/           # Core object permanence module
│   ├── integration.py           #   Main ObjectPermanenceModule orchestrator
│   ├── feature_extractor.py     #   Object-centric feature extraction (attention + simple)
│   ├── spatial_memory_grid.py   #   2D spatial grid with predictive occupancy maps
│   ├── occlusion_detector.py    #   Multi-signal occlusion detection MLP
│   ├── object_predictor.py      #   Next-frame state prediction (MLP / Transformer)
│   ├── percept_gate_controller.py  # Learned observation-vs-prediction gating
│   ├── temporal_fusion.py       #   Gate-based feature fusion
│   └── object_tracker.py        #   Temporal state memory across frames
├── models/
│   ├── qwen.py                  # Qwen3-VL-8B-Instruct wrapper + CLI
│   └── llava.py                 # LLaVA-NeXT-Video-7B wrapper + CLI
├── utils/
│   ├── video.py                 # Shared video frame sampling (PyAV)
│   └── download.py              # HuggingFace model downloader with resume
├── videos/                      # Test videos
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/usman-khan12/vision-model-ironsite-x-vandy-hackathon.git
cd vision-model-ironsite-x-vandy-hackathon

# Create a virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Qwen3-VL may need transformers installed from source:
# pip install git+https://github.com/huggingface/transformers
```

### Requirements

| Package           | Version   |
| ----------------- | --------- |
| `torch`           | >= 2.0.0  |
| `transformers`    | >= 4.57.0 |
| `huggingface-hub` | latest    |
| `av` (PyAV)       | latest    |
| `numpy`           | latest    |
| `Pillow`          | latest    |
| `accelerate`      | latest    |

---

## Quick Start

### Qwen3-VL with object permanence (recommended)

```bash
python -m models.qwen \
    --video videos/video2.mp4 \
    --prompt "What happens to the bottle?" \
    --enable-permanence --show-permanence-info
```

### LLaVA-NeXT-Video with object permanence

```bash
python -m models.llava \
    --video videos/video2.mp4 \
    --prompt "What happens to the bottle?" \
    --enable-permanence --show-permanence-info
```

### Pre-download model weights

```bash
python -m utils.download --model "Qwen/Qwen3-VL-8B-Instruct"
python -m utils.download --model "llava-hf/LLaVA-NeXT-Video-7B-hf"
```

---

## CLI Options

Both `models.qwen` and `models.llava` accept:

| Flag                     | Default                              | Description                           |
| ------------------------ | ------------------------------------ | ------------------------------------- |
| `--video`                | _(required)_                         | Path to input video file              |
| `--prompt`               | `"What is happening in this video?"` | Question to ask the model             |
| `--num-frames`           | `8`                                  | Number of frames to sample            |
| `--max-tokens`           | `200` (Qwen) / `100` (LLaVA)         | Max tokens to generate                |
| `--device`               | `auto`                               | Device: `auto`, `mps`, `cuda`, `cpu`  |
| `--enable-permanence`    | `True`                               | Enable the object permanence pipeline |
| `--no-permanence`        | —                                    | Disable object permanence             |
| `--show-permanence-info` | `False`                              | Print occlusion diagnostics           |
| `--num-objects`          | `10`                                 | Number of objects to track            |
| `--feature-dim`          | `512`                                | Object feature dimensionality         |

LLaVA additionally supports `--use-4bit` for 4-bit quantisation.

---

## Test Videos

| File         | Description                                                |
| ------------ | ---------------------------------------------------------- |
| `video1.mp4` | General test                                               |
| `video2.mp4` | Bottle is tied and pulled behind a laptop (occlusion test) |
| `video3.mp4` | Control — bottle is stationary (no occlusion)              |

---

## Hardware Requirements

| Configuration      | VRAM     |
| ------------------ | -------- |
| 4-bit quantisation | >= 8 GB  |
| Full precision     | >= 16 GB |

Supported accelerators: **NVIDIA CUDA**, **Apple MPS (Metal)**, and **CPU** fallback.

---

## Architecture Details

### Spatial Memory Grid

The spatial memory grid (`SpatialMemoryGrid`) is the novel contribution of this project. It maintains a `[B, H, W, N, D]` tensor representing object features at each spatial cell, along with per-cell confidence and temporal counters. Key operations:

- **Update**: Writes observed features into grid cells using visibility-weighted blending with exponential confidence decay.
- **Re-identification**: Matches new observations against stored grid features using a learned cross-reference network.
- **Occupancy Prediction**: An MLP predicts future occupancy maps from current features, positions, and velocities.
- **Ghost Retrieval**: Retrieves stored features at positions where objects were last seen, providing "ghost" representations for occluded objects.

### Percept Gate Controller

The gating mechanism produces separate **position** and **appearance** gates per object. Gates are initialised with a positive bias (~3.0 through the final linear layer) so the system defaults to trusting observations, only falling back to predictions when the occlusion detector signals low visibility.

---

## Acknowledgements

- [Loci-Looped](https://arxiv.org/abs/2306.02012) for the object permanence gating framework
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and [LLaVA-NeXT-Video](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) as base VLMs
- Built at the **IronSite x Vanderbilt Hackathon**
