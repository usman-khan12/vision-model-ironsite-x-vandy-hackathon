# Object Permanence for Vision-Language Models

Enhances video-understanding VLMs with **object permanence** -- the ability to
track and reason about objects even when they become occluded (hidden behind
other objects).  Built for the IronSite x Vanderbilt Hackathon.

## Project Structure

```
object_permanence/   Core module (occlusion detection, prediction, gating, spatial grid)
models/
  llava.py           LLaVA-NeXT-Video-7B wrapper + CLI
  qwen.py            Qwen3-VL-8B-Instruct wrapper + CLI
utils/
  video.py           Shared video frame sampling
  download.py        HuggingFace model downloader
videos/              Test videos
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Qwen3-VL may need transformers from source:
# pip install git+https://github.com/huggingface/transformers
```

## Quick Start

```bash
# Qwen3-VL with object permanence (recommended)
python -m models.qwen \
    --video videos/video2.mp4 \
    --prompt "What happens to the bottle?" \
    --enable-permanence --show-permanence-info

# LLaVA-NeXT-Video with object permanence
python -m models.llava \
    --video videos/video2.mp4 \
    --prompt "What happens to the bottle?" \
    --enable-permanence --show-permanence-info

# Download model weights ahead of time
python -m utils.download --model "Qwen/Qwen3-VL-8B-Instruct"
```

## How It Works

The object permanence pipeline processes each video frame through:

1. **Feature Extraction** -- project model hidden states into object-level features
2. **Spatial Memory Grid** -- maintain a 2D grid of "ghost" representations for
   occluded objects with predictive occupancy maps
3. **Occlusion Detection** -- estimate per-object visibility from feature
   similarity, magnitude, and detection confidence
4. **State Prediction** -- predict features for the next frame (MLP or
   Transformer)
5. **Percept Gating** -- learn when to trust observations vs predictions
6. **Temporal Fusion** -- blend observed and predicted features via learned gates
7. **Prompt Enhancement** -- when occlusion is detected, augment the text prompt
   so the VLM reasons about hidden objects

## Test Videos

| File | Description |
|------|-------------|
| `video1.mp4` | General test |
| `video2.mp4` | Bottle is tied and pulled behind a laptop (occlusion test) |
| `video3.mp4` | Control -- bottle is stationary (no occlusion) |

## Hardware

- GPU with >= 8 GB VRAM (4-bit quantisation) or >= 16 GB (full precision)
- Supports CUDA, Apple MPS (Metal), and CPU fallback
