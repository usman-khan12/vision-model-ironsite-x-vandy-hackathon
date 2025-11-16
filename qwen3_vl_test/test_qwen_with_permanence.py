"""
Test Qwen3-VL-235B-A22B-Instruct with Object Permanence Module
"""
import torch
import argparse
from pathlib import Path
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
import av
import numpy as np

# Import object permanence module
# Option 1: If object-permanence-module is in parent directory
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent / "object_permanence"
if parent_dir.exists():
    sys.path.insert(0, str(parent_dir))
else:
    # Option 2: Try absolute path
    sys.path.insert(0, '/Users/ukhan2024/Desktop/object-permanence-module')

from object_permanence import ObjectPermanenceModule
from object_permanence.feature_extractor import SimpleFeatureExtractor


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
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


def process_video(video_path, num_frames=8):
    """
    Process a video file and extract frames.
    Always includes the first and last frames to ensure we capture the full sequence.
    """
    print(f"Processing video: {video_path}")
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    
    if total_frames == 0:
        raise ValueError("Video has no frames or cannot be decoded")
    
    if total_frames <= num_frames:
        # If we have fewer frames than requested, use all frames
        indices = np.arange(0, total_frames).astype(int)
    else:
        # Always include first frame (index 0) and last frame (index total_frames - 1)
        # Sample remaining frames uniformly from the middle
        remaining_frames = num_frames - 2
        if remaining_frames > 0:
            # Sample uniformly from frames 1 to total_frames - 2 (excluding first and last)
            middle_indices = np.linspace(1, total_frames - 2, remaining_frames, dtype=int)
            indices = np.concatenate([[0], middle_indices, [total_frames - 1]])
        else:
            # If num_frames is 2 or less, just use first and last
            indices = np.array([0, total_frames - 1])[:num_frames]
    
    # Remove duplicates and sort
    indices = np.unique(indices)
    indices = np.sort(indices)
    
    print(f"Sampling {len(indices)} frames from {total_frames} total frames")
    print(f"Frame indices: {indices.tolist()} (first: {indices[0]}, last: {indices[-1]})")
    clip = read_video_pyav(container, indices)
    container.close()
    
    return clip


def load_qwen_model(device="auto", use_flash_attention=False):
    """
    Load Qwen3-VL model and processor.
    """
    # Auto-detect device
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
    
    print(f"Loading Qwen3-VL-235B-A22B-Instruct model...")
    print(f"Device: {device}")
    
    # Load model
    model_kwargs = {
        "dtype": "auto",
        "device_map": "auto" if device != "cpu" else None,
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    
    try:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
            **model_kwargs
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying without flash attention...")
        model_kwargs.pop("attn_implementation", None)
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
            **model_kwargs
        )
    
    # Move to device if not using device_map
    if device != "cpu" and "device_map" not in model_kwargs:
        model = model.to(device)
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")
    
    print("Model loaded successfully!")
    return model, processor, device


def generate_response_with_permanence(
    model,
    processor,
    permanence_module,
    video_path,
    prompt,
    num_frames=8,
    max_new_tokens=200,
    device="mps",
    show_permanence_info=False
):
    """
    Generate response with object permanence.
    """
    # Process video
    clip = process_video(video_path, num_frames)
    
    # Create messages for Qwen3-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": clip},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # Move to device
    if device in ["mps", "cuda"]:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    else:
        inputs = {k: v if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Process with object permanence if enabled
    # CRITICAL: Process frames SEQUENTIALLY to build up temporal state
    permanence_info = None
    if permanence_module is not None:
        print("Processing with object permanence (frame-by-frame)...")
        permanence_module.reset()
        
        # Extract features from the full video first, then process frame-by-frame
        # This avoids the single-frame limitation of Qwen3-VL's processor
        with torch.no_grad():
            was_training = model.training
            model.eval()
            
            try:
                # First, extract features from the full video clip
                # Use a dummy prompt for feature extraction
                feature_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": clip},
                            {"type": "text", "text": "Describe the video."},
                        ],
                    }
                ]
                
                # Prepare inputs for feature extraction
                feature_inputs = processor.apply_chat_template(
                    feature_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # Move to device
                if device in ["mps", "cuda"]:
                    feature_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feature_inputs.items()}
                
                # Clear cache
                if hasattr(model, 'model') and hasattr(model.model, 'past_key_values'):
                    model.model.past_key_values = None
                if hasattr(model, 'past_key_values'):
                    model.past_key_values = None
                
                # Get hidden states from full video
                video_outputs = model.model(**feature_inputs, output_hidden_states=True, use_cache=False)
                
                # Extract hidden states
                if hasattr(video_outputs, 'last_hidden_state'):
                    all_hidden_states = video_outputs.last_hidden_state
                elif isinstance(video_outputs, tuple):
                    all_hidden_states = video_outputs[0]
                else:
                    all_hidden_states = video_outputs
                
                # Now process each frame's features sequentially through permanence module
                # Split hidden states by frame (approximate - may need adjustment based on actual structure)
                num_frames_in_clip = clip.shape[0] if hasattr(clip, 'shape') else len(clip)
                permanence_outputs = []
                
                # If we have sequence-level features, we need to split them by frame
                # For now, we'll use the full sequence and process it as a single "frame"
                # Then split it into frame-like chunks
                if len(all_hidden_states.shape) == 3:
                    # [batch, seq_len, hidden_dim]
                    seq_len = all_hidden_states.shape[1]
                    # Approximate frames per sequence (this is a heuristic)
                    # We'll split the sequence into num_frames chunks
                    chunk_size = max(1, seq_len // num_frames_in_clip)
                    
                    for frame_idx in range(num_frames_in_clip):
                        start_idx = frame_idx * chunk_size
                        end_idx = min((frame_idx + 1) * chunk_size, seq_len)
                        
                        # Extract features for this "frame" (chunk of sequence)
                        frame_hidden_states = all_hidden_states[:, start_idx:end_idx, :]
                        
                        # Pool features for this frame
                        frame_features = torch.mean(frame_hidden_states, dim=1)  # [batch, hidden_dim]
                        
                        # Convert to float32
                        if frame_features.dtype != torch.float32:
                            frame_features = frame_features.float()
                        
                        # Process this frame through permanence module
                        # This builds up temporal state across frames
                        frame_permanence = permanence_module(frame_features, frame_idx=frame_idx)
                        permanence_outputs.append(frame_permanence)
                else:
                    # Fallback: use mean pooling and process as single frame
                    frame_features = torch.mean(all_hidden_states, dim=1) if len(all_hidden_states.shape) > 1 else all_hidden_states
                    if frame_features.dtype != torch.float32:
                        frame_features = frame_features.float()
                    frame_permanence = permanence_module(frame_features, frame_idx=0)
                    permanence_outputs.append(frame_permanence)
                
                # Get final permanence state (from last frame)
                permanence_info = permanence_outputs[-1] if permanence_outputs else None
                
                # Extract key information for prompt enhancement
                if permanence_info and show_permanence_info:
                    print("\n" + "="*50)
                    print("OBJECT PERMANENCE INFORMATION:")
                    print("="*50)
                    if 'occlusion_factors' in permanence_info:
                        avg_occlusion = permanence_info['occlusion_factors'].mean().item()
                        max_occlusion = permanence_info['occlusion_factors'].max().item()
                        print(f"Average Occlusion: {avg_occlusion:.3f}")
                        print(f"Max Occlusion: {max_occlusion:.3f}")
                        if max_occlusion > 0.5:
                            print("⚠️  High occlusion detected - objects may be hidden!")
                    if 'occupancy_maps' in permanence_info:
                        print("✓ Spatial occupancy maps generated")
                        print("✓ Objects tracked through spatial memory grid")
                    if 'object_positions' in permanence_info:
                        positions = permanence_info['object_positions']
                        print(f"✓ Tracking {positions.shape[1]} objects")
                    print("="*50 + "\n")
                
                # Enhance prompt with permanence information
                # Use spatial memory grid data to distinguish occlusion from removal
                if permanence_info:
                    occlusion_note = ""
                    should_suggest_occlusion = False
                    occlusion_confidence = 0.0
                    
                    if 'occlusion_factors' in permanence_info:
                        max_occlusion = permanence_info['occlusion_factors'].max().item()
                        avg_occlusion = permanence_info['occlusion_factors'].mean().item()
                        
                        # Check spatial memory grid data for better occlusion detection
                        grid_evidence_occlusion = False
                        grid_confidence_score = 0.0
                        
                        if 'occupancy_maps' in permanence_info and 'grid_match_confidence' in permanence_info:
                            # If grid shows high confidence at predicted locations, object is likely occluded
                            grid_match_conf = permanence_info['grid_match_confidence']
                            if grid_match_conf is not None and len(grid_match_conf.shape) > 0:
                                max_grid_confidence = grid_match_conf.max().item()
                                avg_grid_confidence = grid_match_conf.mean().item()
                                
                                # High grid confidence + high occlusion = likely occluded (not removed)
                                if max_grid_confidence > 0.5 and max_occlusion > 0.3:
                                    grid_evidence_occlusion = True
                                    grid_confidence_score = max_grid_confidence
                        
                        # Also check occupancy maps - if they predict object should be somewhere
                        if 'occupancy_maps' in permanence_info:
                            occupancy_maps = permanence_info['occupancy_maps']
                            if occupancy_maps is not None:
                                # Check if occupancy maps show objects at locations (even if not visible)
                                max_occupancy = occupancy_maps.max().item()
                                if max_occupancy > 0.4:  # Objects predicted to be at locations
                                    grid_evidence_occlusion = True
                        
                        # Decision logic:
                        # 1. High occlusion + high grid confidence = strong occlusion signal
                        # 2. High occlusion + low grid confidence = might be removed, be cautious
                        # 3. Medium occlusion + grid evidence = likely occluded
                        
                        if max_occlusion > 0.3:
                            if grid_evidence_occlusion:
                                # Strong evidence for occlusion from spatial grid
                                should_suggest_occlusion = True
                                occlusion_confidence = max(max_occlusion, grid_confidence_score)
                                occlusion_note = f" IMPORTANT: Object permanence analysis with spatial memory grid detected that objects are likely occluded (occlusion: {max_occlusion:.2f}, grid confidence: {grid_confidence_score:.2f}). The spatial memory grid shows objects are predicted to be at specific locations even though they're not visible, indicating they are hidden behind other objects, not removed from the scene."
                            elif max_occlusion > 0.5:
                                # High occlusion but no grid evidence - still suggest occlusion but less confidently
                                should_suggest_occlusion = True
                                occlusion_confidence = max_occlusion
                                occlusion_note = f" IMPORTANT: Object permanence analysis detected that objects may be occluded (occlusion factor: {max_occlusion:.2f}). Objects that disappear from view are likely hidden behind other objects, not removed from the scene."
                            else:
                                # Medium occlusion, no strong grid evidence - be more cautious
                                should_suggest_occlusion = True
                                occlusion_confidence = max_occlusion
                                occlusion_note = f" Note: Some objects may be occluded (occlusion factor: {max_occlusion:.2f}). Consider that objects might be hidden behind other objects rather than removed."
                    
                    # Add explicit instruction about object permanence
                    # Only add strong instruction if we have evidence for occlusion
                    if should_suggest_occlusion:
                        if occlusion_confidence > 0.5:
                            # Strong evidence - add strong instruction
                            enhanced_prompt = f"{prompt}{occlusion_note} Remember: Objects don't disappear - if an object was visible earlier and is not visible at the end, it is most likely occluded (hidden behind) another object. Apply object permanence: track objects through occlusions."
                        else:
                            # Weaker evidence - add more cautious instruction
                            enhanced_prompt = f"{prompt}{occlusion_note} Consider that objects may be occluded rather than removed, but verify based on the video content."
                    else:
                        # No strong occlusion signal - don't add strong permanence instruction
                        # This allows the model to naturally determine if objects are removed vs occluded
                        enhanced_prompt = prompt
                    
                    # Update the prompt in messages
                    messages[0]["content"][1]["text"] = enhanced_prompt
                    
                    # Re-prepare inputs with enhanced prompt
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    # Move to device again
                    if device in ["mps", "cuda"]:
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        
            except Exception as e:
                print(f"Warning: Could not process with object permanence: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clear all caches
                if hasattr(model, 'model') and hasattr(model.model, 'past_key_values'):
                    model.model.past_key_values = None
                if hasattr(model, 'past_key_values'):
                    model.past_key_values = None
                if hasattr(model, 'reset_cache'):
                    model.reset_cache()
                if hasattr(model, 'model') and hasattr(model.model, 'reset_cache'):
                    model.model.reset_cache()
                
                # Clear PyTorch cache
                torch.mps.empty_cache() if device == "mps" else None
                torch.cuda.empty_cache() if device == "cuda" else None
                
                # Restore training state
                if was_training:
                    model.train()
                else:
                    model.eval()
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        # Ensure clean generation - explicitly set use_cache and clear any residual state
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'use_cache': True,  # Allow cache for generation
        }
        
        # Clear any residual cache before generation
        if hasattr(model, 'model') and hasattr(model.model, 'past_key_values'):
            model.model.past_key_values = None
        if hasattr(model, 'past_key_values'):
            model.past_key_values = None
        
        generated_ids = model.generate(**inputs, **generation_kwargs)
    
    # Decode response
    input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs.input_ids
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-VL with Object Permanence")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--prompt", type=str, default="What is happening in this video?", help="Question/prompt")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--use-flash-attention", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--enable-permanence", action="store_true", default=True, help="Enable object permanence")
    parser.add_argument("--no-permanence", dest="enable_permanence", action="store_false", help="Disable object permanence")
    parser.add_argument("--show-permanence-info", action="store_true", help="Show permanence information")
    parser.add_argument("--num-objects", type=int, default=10, help="Number of objects to track")
    parser.add_argument("--feature-dim", type=int, default=512, help="Dimension for object features")
    
    args = parser.parse_args()
    
    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    try:
        # Load Qwen3-VL model
        model, processor, actual_device = load_qwen_model(
            device=args.device,
            use_flash_attention=args.use_flash_attention
        )
        
        # Setup object permanence if enabled
        permanence_module = None
        if args.enable_permanence:
            print("Setting up Object Permanence module...")
            
            # Get model hidden dimension - check actual model structure
            if hasattr(model, 'config'):
                # Try different possible attribute names
                model_hidden_dim = getattr(model.config, 'hidden_size', None)
                if model_hidden_dim is None:
                    model_hidden_dim = getattr(model.config, 'd_model', None)
                if model_hidden_dim is None:
                    model_hidden_dim = getattr(model.config, 'dim', None)
                if model_hidden_dim is None:
                    # Try to infer from model layers
                    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                        model_hidden_dim = model.model.embed_tokens.embedding_dim
                    elif hasattr(model, 'get_input_embeddings'):
                        model_hidden_dim = model.get_input_embeddings().embedding_dim
                    else:
                        model_hidden_dim = 4096  # Qwen3-VL default
            else:
                model_hidden_dim = 4096  # Qwen3-VL default
            
            print(f"Detected model hidden dimension: {model_hidden_dim}")
            
            # Create permanence module
            permanence_module = ObjectPermanenceModule(
                feature_dim=args.feature_dim,
                num_objects=args.num_objects,
                use_spatial_grid=True
            )
            # Use float32 for permanence module to avoid dtype issues
            permanence_module = permanence_module.to(actual_device).float()
            
            # Create feature extractor
            feature_extractor = SimpleFeatureExtractor(
                input_dim=model_hidden_dim,
                object_feature_dim=args.feature_dim,
                num_objects=args.num_objects
            )
            # Use float32 for feature extractor
            feature_extractor = feature_extractor.to(actual_device).float()
            permanence_module.set_feature_extractor(feature_extractor)
            
            print("Object permanence module ready!")
        
        # Generate response
        response = generate_response_with_permanence(
            model=model,
            processor=processor,
            permanence_module=permanence_module,
            video_path=str(video_path),
            prompt=args.prompt,
            num_frames=args.num_frames,
            max_new_tokens=args.max_tokens,
            device=actual_device,
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

