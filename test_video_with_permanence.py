"""
Test script for LLaVA-NeXT-Video with Object Permanence
Tests video understanding with object tracking through occlusions
"""
import av
import torch
import numpy as np
import argparse
import os
from pathlib import Path
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from huggingface_hub import snapshot_download

from test_video import load_model, process_video
from video_model_with_permanence import VideoModelWithPermanence


def generate_response_with_permanence(
    model, 
    processor, 
    video_path, 
    prompt, 
    num_frames=8, 
    max_new_tokens=100, 
    device="mps",
    enable_permanence=True,
    show_permanence_info=False
):
    """
    Generate a response from the model with object permanence capabilities.
    
    Args:
        model: The loaded model (with or without permanence wrapper)
        processor: The loaded processor
        video_path: Path to the video file
        prompt: Text prompt/question about the video
        num_frames: Number of frames to sample from video
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on
        enable_permanence: Whether to use object permanence
        show_permanence_info: Whether to print permanence information
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
    
    # Process with object permanence if enabled
    if enable_permanence and isinstance(model, VideoModelWithPermanence):
        print("Processing with object permanence...")
        model.reset_permanence()
        
        # Store permanence info for display
        permanence_info_list = []
        
        # Process frames separately to get permanence info
        if 'videos' in inputs or 'pixel_values' in inputs:
            video_inputs = inputs.get('videos') or inputs.get('pixel_values')
            if video_inputs is not None and len(video_inputs.shape) >= 4:
                num_frames = video_inputs.shape[1] if len(video_inputs.shape) == 5 else 1
                
                # Process each frame to collect permanence info
                for frame_idx in range(num_frames):
                    # Extract frame features (simplified - in practice would extract from model)
                    # For now, we'll get info during generation
                    pass
        
        # Forward pass with permanence to initialize
        with torch.no_grad():
            try:
                outputs = model.forward_with_permanence(inputs, process_frames_separately=True)
                if hasattr(outputs, 'permanence_info'):
                    permanence_info_list = outputs.permanence_info if isinstance(outputs.permanence_info, list) else [outputs.permanence_info]
            except Exception as e:
                print(f"Note: Could not extract permanence info from forward pass: {e}")
        
        # Show permanence info if requested
        if show_permanence_info and permanence_info_list:
            print("\n" + "="*50)
            print("OBJECT PERMANENCE INFORMATION:")
            print("="*50)
            for frame_idx, frame_info in enumerate(permanence_info_list):
                print(f"\nFrame {frame_idx + 1}:")
                if isinstance(frame_info, dict):
                    if 'occlusion_factors' in frame_info:
                        avg_occlusion = frame_info['occlusion_factors'].mean().item()
                        print(f"  Average Occlusion: {avg_occlusion:.3f}")
                    if 'position_gates' in frame_info:
                        avg_gate = frame_info['position_gates'].mean().item()
                        print(f"  Average Gate Value: {avg_gate:.3f}")
                    if 'visibility_scores' in frame_info:
                        avg_visibility = frame_info['visibility_scores'].mean().item()
                        print(f"  Average Visibility: {avg_visibility:.3f}")
            print("="*50 + "\n")
        elif show_permanence_info:
            print("\nNote: Object permanence is enabled but info not available for display.\n")
        
        # Generate response
        print("Generating response...")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        # Standard generation
        print("Generating response...")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    # Decode response
    response = processor.decode(output[0][2:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Test LLaVA-NeXT-Video model with Object Permanence")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--prompt", type=str, default="What is happening in this video?", help="Question/prompt about the video")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from video")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Device to run on (auto detects best available)")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--use-flash-attention", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--enable-permanence", action="store_true", default=True, help="Enable object permanence module")
    parser.add_argument("--no-permanence", dest="enable_permanence", action="store_false", help="Disable object permanence module")
    parser.add_argument("--show-permanence-info", action="store_true", help="Show object permanence information")
    parser.add_argument("--num-objects", type=int, default=10, help="Number of objects to track")
    parser.add_argument("--feature-dim", type=int, default=512, help="Dimension for object features")
    
    args = parser.parse_args()
    
    # Check if video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    try:
        # Load base model
        base_model, processor, actual_device = load_model(
            device=args.device,
            use_4bit=args.use_4bit,
            use_flash_attention=args.use_flash_attention
        )
        
        # Wrap with object permanence if enabled
        if args.enable_permanence:
            print("Wrapping model with Object Permanence module...")
            model = VideoModelWithPermanence(
                base_model=base_model,
                feature_dim=args.feature_dim,
                num_objects=args.num_objects,
                enable_permanence=True
            )
            model = model.to(actual_device)
        else:
            model = base_model
        
        # Generate response
        response = generate_response_with_permanence(
            model=model,
            processor=processor,
            video_path=str(video_path),
            prompt=args.prompt,
            num_frames=args.num_frames,
            max_new_tokens=args.max_tokens,
            device=actual_device,
            enable_permanence=args.enable_permanence,
            show_permanence_info=args.show_permanence_info
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

