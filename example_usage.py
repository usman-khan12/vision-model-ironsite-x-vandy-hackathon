"""
Simple example of using the LLaVA-NeXT-Video model programmatically.
"""
from test_video import load_model, generate_response


def main():
    # Configuration
    video_path = "path/to/your/video.mp4"  # Change this to your video path
    prompt = "What is happening in this video?"
    device = "cuda"  # or "cpu"
    num_frames = 8
    max_tokens = 100
    
    # Optional: Use 4-bit quantization to save memory
    use_4bit = False
    use_flash_attention = False
    
    print("Loading model...")
    model, processor = load_model(
        device=device,
        use_4bit=use_4bit,
        use_flash_attention=use_flash_attention
    )
    
    print(f"\nProcessing video: {video_path}")
    print(f"Prompt: {prompt}\n")
    
    response = generate_response(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt=prompt,
        num_frames=num_frames,
        max_new_tokens=max_tokens,
        device=device
    )
    
    print("\n" + "="*50)
    print("MODEL RESPONSE:")
    print("="*50)
    print(response)
    print("="*50)


if __name__ == "__main__":
    main()

