# Qwen3-VL with Object Permanence

This directory contains code to test the Qwen3-VL-8B-Instruct model with the Object Permanence module.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install latest transformers (required for Qwen3-VL):
```bash
pip install git+https://github.com/huggingface/transformers
```

3. Make sure the object permanence module is accessible:
   - The object permanence module should be in `/Users/ukhan2024/Desktop/object-permanence-module`
   - Or update the path in `test_qwen_with_permanence.py`

## Usage

```bash
python test_qwen_with_permanence.py \
    --video path/to/video.mp4 \
    --prompt "What is happening in this video?" \
    --enable-permanence \
    --show-permanence-info
```

## Model Information

- **Model**: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- **Architecture**: Vision-Language Model
- **Parameters**: 8B
- **License**: Apache 2.0

## Features

- Native video understanding
- Long context (256K tokens, expandable to 1M)
- Advanced spatial perception
- Enhanced multimodal reasoning

## Object Permanence Integration

The object permanence module enhances Qwen3-VL with:
- Object tracking through occlusions
- Spatial Memory Grid for spatial reasoning
- Learned gating for prediction vs. observation
- Temporal fusion of features

