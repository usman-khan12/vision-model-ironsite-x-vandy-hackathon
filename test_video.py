"""
Script to test LLaVA-NeXT-Video-7B-hf model with custom videos.
"""
import av
import torch
import numpy as np
import argparse
import os
from pathlib import Path
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from huggingface_hub import snapshot_download


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_model(model_id="llava-hf/LLaVA-NeXT-Video-7B-hf", device="auto", use_4bit=False, use_flash_attention=False):
    """
    Load the LLaVA-NeXT-Video model and processor.
    
    Args:
        model_id: Hugging Face model identifier
        device: Device to run on ('auto', 'mps', 'cuda', or 'cpu')
        use_4bit: Whether to use 4-bit quantization
        use_flash_attention: Whether to use Flash Attention 2
    """
    # Auto-detect best device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
            print("Detected MPS (Metal) - using GPU acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Detected CUDA - using GPU acceleration")
        else:
            device = "cpu"
            print("Using CPU")
    
    print(f"Loading model: {model_id}")
    print(f"Device: {device}")
    
    # Use float16 for MPS and CUDA, float32 for CPU
    if device in ["mps", "cuda"]:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model_kwargs = {
        "dtype": dtype,  # Use dtype instead of deprecated torch_dtype
        "low_cpu_mem_usage": True,
        "resume_download": True,  # Resume interrupted downloads
    }
    
    if use_4bit:
        model_kwargs["load_in_4bit"] = True
        print("Using 4-bit quantization")
    
    if use_flash_attention:
        model_kwargs["use_flash_attention_2"] = True
        print("Using Flash Attention 2")
    
    print("Downloading model files (this may take a while, ~14GB)...")
    print("Progress will be shown below:")
    
    try:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nTrying to download model files separately...")
        try:
            # Try downloading with snapshot_download for better progress tracking
            local_dir = snapshot_download(
                repo_id=model_id,
                resume_download=True,
                local_files_only=False
            )
            print(f"Model files downloaded to: {local_dir}")
            # Retry loading from local
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                local_dir,
                **{k: v for k, v in model_kwargs.items() if k != "resume_download"}
            )
        except Exception as e2:
            print(f"Download failed: {e2}")
            raise
    
    # Move model to device (skip if using 4-bit quantization)
    if not use_4bit:
        print(f"Moving model to {device}...")
        model = model.to(device)
    
    print("Loading processor...")
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    
    print("Model loaded successfully!")
    return model, processor, device


def process_video(video_path, num_frames=8):
    """
    Process a video file and extract frames.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample uniformly
    """
    print(f"Processing video: {video_path}")
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    
    if total_frames == 0:
        raise ValueError("Video has no frames or cannot be decoded")
    
    # Sample frames uniformly
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    indices = indices[:num_frames]  # Ensure we don't exceed num_frames
    
    print(f"Sampling {len(indices)} frames from {total_frames} total frames")
    clip = read_video_pyav(container, indices)
    container.close()
    
    return clip


def generate_response(model, processor, video_path, prompt, num_frames=8, max_new_tokens=100, device="mps"):
    """
    Generate a response from the model given a video and prompt.
    
    Args:
        model: The loaded model
        processor: The loaded processor
        video_path: Path to the video file
        prompt: Text prompt/question about the video
        num_frames: Number of frames to sample from video
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on
    """
    # Process video
    clip = process_video(video_path, num_frames)
    
    # Create conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        },
    ]
    
    # Format prompt
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(text=formatted_prompt, videos=clip, padding=True, return_tensors="pt")
    
    # Move to device (MPS, CUDA, or CPU)
    if device in ["mps", "cuda"]:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    else:
        inputs = {k: v if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    # Decode response
    response = processor.decode(output[0][2:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Test LLaVA-NeXT-Video model with custom videos")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--prompt", type=str, default="What is happening in this video?", help="Question/prompt about the video")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from video")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Device to run on (auto detects best available)")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--use-flash-attention", action="store_true", help="Use Flash Attention 2")
    
    args = parser.parse_args()
    
    # Check if video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    try:
        # Load model (device detection happens inside)
        model, processor, actual_device = load_model(
            device=args.device,
            use_4bit=args.use_4bit,
            use_flash_attention=args.use_flash_attention
        )
        
        # Generate response
        response = generate_response(
            model=model,
            processor=processor,
            video_path=str(video_path),
            prompt=args.prompt,
            num_frames=args.num_frames,
            max_new_tokens=args.max_tokens,
            device=actual_device
        )
        
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        print(response)
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

