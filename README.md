# LLaVA-NeXT-Video-7B Local Testing

This repository contains scripts to run the [LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) model locally and test it with your own videos.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- At least 14GB GPU memory (for full precision) or 8GB (for 4-bit quantization)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** If you want to use 4-bit quantization, make sure `bitsandbytes` is installed:
```bash
pip install bitsandbytes
```

**Note:** If you want to use Flash Attention 2, install it separately:
```bash
pip install flash-attn
```

## Usage

### Basic Usage

Run the model with a video file and a prompt:

```bash
python test_video.py --video path/to/your/video.mp4 --prompt "What is happening in this video?"
```

### Advanced Options

```bash
python test_video.py \
    --video path/to/your/video.mp4 \
    --prompt "Why is this video funny?" \
    --num-frames 16 \
    --max-tokens 200 \
    --device cuda \
    --use-4bit \
    --use-flash-attention
```

### Arguments

- `--video`: (Required) Path to your video file
- `--prompt`: (Optional) Question/prompt about the video (default: "What is happening in this video?")
- `--num-frames`: (Optional) Number of frames to sample from video (default: 8)
- `--max-tokens`: (Optional) Maximum number of tokens to generate (default: 100)
- `--device`: (Optional) Device to run on: `cuda` or `cpu` (default: `cuda`)
- `--use-4bit`: (Optional) Use 4-bit quantization to reduce memory usage
- `--use-flash-attention`: (Optional) Use Flash Attention 2 for faster generation

### Example Prompts

- "What is happening in this video?"
- "Why is this video funny?"
- "Describe the actions in this video."
- "What objects are visible in this video?"
- "What is the main subject of this video?"

## Memory Optimization

If you're running out of GPU memory, try:

1. **4-bit Quantization**: Add `--use-4bit` flag
   ```bash
   python test_video.py --video your_video.mp4 --use-4bit
   ```

2. **Reduce number of frames**: Use fewer frames (e.g., `--num-frames 4`)

3. **Use CPU**: If GPU memory is insufficient, use `--device cpu` (will be slower)

## Supported Video Formats

The script supports common video formats that PyAV can decode, including:
- MP4
- AVI
- MOV
- MKV
- And other formats supported by FFmpeg

## Object Permanence Module

This repository includes an **Object Permanence** module that enables tracking objects through occlusions using techniques inspired by Loci-Looped. The module:

- Tracks objects across frames
- Detects occlusions
- Predicts object states when occluded
- Fuses predictions with observations using learned gates

### Using Object Permanence

```bash
python test_video_with_permanence.py \
    --video path/to/your/video.mp4 \
    --prompt "What is happening in this video?" \
    --enable-permanence \
    --show-permanence-info \
    --num-objects 10
```

### Key Features

- **Occlusion Detection**: Automatically detects when objects are occluded
- **State Prediction**: Predicts object features when not visible
- **Learned Gating**: Learns when to trust predictions vs. observations
- **Temporal Tracking**: Maintains object identity through occlusions

See `object_permanence/README.md` for detailed documentation.

## Model Information

- **Model**: [llava-hf/LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf)
- **Base LLM**: lmsys/vicuna-7b-v1.5
- **Parameters**: 7B
- **License**: Llama 2 Community License

## Troubleshooting

1. **CUDA out of memory**: Use `--use-4bit` or reduce `--num-frames`
2. **Model download issues**: Make sure you have internet connection and sufficient disk space (~14GB)
3. **Video decoding errors**: Ensure your video file is not corrupted and is in a supported format

## Citation

If you use this model, please cite:

```bibtex
@misc{zhang2024llavanextvideo,
  title={LLaVA-NeXT: A Strong Zero-shot Video Understanding Model},
  url={https://llava-vl.github.io/blog/2024-04-30-llava-next-video/},
  author={Zhang, Yuanhan and Li, Bo and Liu, haotian and Lee, Yong jae and Gui, Liangke and Fu, Di and Feng, Jiashi and Liu, Ziwei and Li, Chunyuan},
  month={April},
  year={2024}
}
```

