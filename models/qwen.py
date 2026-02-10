"""
Qwen3-VL-8B-Instruct with Object Permanence.

Loads the Qwen3-VL model, processes video frames through the object permanence
pipeline, and uses the permanence signal to enhance prompts before generation.

Usage:
    python -m models.qwen \
        --video videos/video2.mp4 \
        --prompt "What happens to the bottle?" \
        --enable-permanence --show-permanence-info
"""

import argparse
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from utils.video import process_video
from object_permanence import ObjectPermanenceModule
from object_permanence.feature_extractor import SimpleFeatureExtractor


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _detect_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        print("Detected MPS (Metal)")
        return "mps"
    if torch.cuda.is_available():
        print("Detected CUDA")
        return "cuda"
    print("Using CPU")
    return "cpu"


def load_model(device="auto", use_flash_attention=False):
    """
    Load Qwen3-VL model and processor.

    Returns:
        (model, processor, device_str)
    """
    device = _detect_device(device)
    print(f"Loading {MODEL_ID} on {device} ...")

    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto" if device != "cpu" else None,
    }
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **model_kwargs
        )
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **model_kwargs
        )

    if device != "cpu" and "device_map" not in model_kwargs:
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded.")
    return model, processor, device


def _get_model_hidden_dim(model):
    """Infer the language-model hidden dimension from a Qwen3-VL model."""
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "d_model", "dim"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return val
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.embedding_dim
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().embedding_dim
    return 4096  # Qwen3-VL default


# ---------------------------------------------------------------------------
# Permanence-enhanced prompt builder
# ---------------------------------------------------------------------------


def _build_enhanced_prompt(prompt, permanence_info):
    """
    Append an occlusion hint to the user prompt when the permanence
    module signals that objects are likely hidden behind something.
    """
    if permanence_info is None:
        return prompt

    occ = permanence_info.get("occlusion_factors")
    if occ is None:
        return prompt

    max_occ = occ.max().item()
    if max_occ < 0.3:
        return prompt  # no meaningful occlusion

    # Check spatial grid signals
    grid_conf = permanence_info.get("grid_match_confidence")
    occupancy = permanence_info.get("occupancy_maps")
    grid_evidence = False
    grid_score = 0.0

    if grid_conf is not None and grid_conf.numel() > 0:
        grid_score = grid_conf.max().item()
        if grid_score > 0.5 and max_occ > 0.3:
            grid_evidence = True
    if occupancy is not None and occupancy.max().item() > 0.4:
        grid_evidence = True

    if grid_evidence:
        note = (
            f" IMPORTANT: Object permanence analysis with spatial memory grid "
            f"detected that objects are likely occluded (occlusion: {max_occ:.2f}, "
            f"grid confidence: {grid_score:.2f}). The spatial memory grid shows "
            f"objects are predicted to be at specific locations even though they're "
            f"not visible, indicating they are hidden behind other objects, not "
            f"removed from the scene."
        )
    elif max_occ > 0.5:
        note = (
            f" IMPORTANT: Object permanence analysis detected that objects may be "
            f"occluded (occlusion factor: {max_occ:.2f}). Objects that disappear "
            f"from view are likely hidden behind other objects, not removed."
        )
    else:
        note = (
            f" Note: Some objects may be occluded (occlusion factor: {max_occ:.2f}). "
            f"Consider that objects might be hidden behind other objects."
        )

    if max_occ > 0.5:
        return (
            f"{prompt}{note} Remember: Objects don't disappear - if an object "
            f"was visible earlier and is not visible at the end, it is most "
            f"likely occluded (hidden behind) another object."
        )
    return f"{prompt}{note}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def generate_response(
    model,
    processor,
    video_path,
    prompt,
    permanence_module=None,
    num_frames=8,
    max_new_tokens=200,
    device="mps",
    show_permanence_info=False,
):
    clip = process_video(video_path, num_frames)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": clip},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    # ------- object permanence processing -------
    permanence_info = None
    if permanence_module is not None:
        print("Processing with object permanence (frame-by-frame) ...")
        permanence_module.reset()

        with torch.no_grad():
            was_training = model.training
            model.eval()
            try:
                # Extract hidden states from the full video
                feature_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": clip},
                            {"type": "text", "text": "Describe the video."},
                        ],
                    }
                ]
                feat_inputs = processor.apply_chat_template(
                    feature_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
                feat_inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in feat_inputs.items()
                }

                # Clear residual KV cache
                for obj in (model, getattr(model, "model", None)):
                    if obj is not None and hasattr(obj, "past_key_values"):
                        obj.past_key_values = None

                video_out = model.model(
                    **feat_inputs, output_hidden_states=True, use_cache=False
                )

                hidden = (
                    video_out.last_hidden_state
                    if hasattr(video_out, "last_hidden_state")
                    else video_out[0]
                    if isinstance(video_out, tuple)
                    else video_out
                )

                # Process frame-by-frame through permanence module
                num_clip_frames = clip.shape[0] if hasattr(clip, "shape") else len(clip)
                permanence_outputs = []

                if hidden.dim() == 3:
                    seq_len = hidden.shape[1]
                    chunk = max(1, seq_len // num_clip_frames)
                    for fi in range(num_clip_frames):
                        chunk_h = hidden[
                            :, fi * chunk : min((fi + 1) * chunk, seq_len), :
                        ]
                        feat = chunk_h.mean(dim=1).float()
                        permanence_outputs.append(permanence_module(feat, frame_idx=fi))
                else:
                    feat = (
                        hidden.mean(dim=1).float()
                        if hidden.dim() > 1
                        else hidden.float()
                    )
                    permanence_outputs.append(permanence_module(feat, frame_idx=0))

                permanence_info = permanence_outputs[-1] if permanence_outputs else None

                # Show info
                if show_permanence_info and permanence_info:
                    print("\n" + "=" * 50)
                    print("OBJECT PERMANENCE INFORMATION:")
                    print("=" * 50)
                    if "occlusion_factors" in permanence_info:
                        avg = permanence_info["occlusion_factors"].mean().item()
                        mx = permanence_info["occlusion_factors"].max().item()
                        print(f"  Average Occlusion: {avg:.3f}")
                        print(f"  Max Occlusion:     {mx:.3f}")
                    if "occupancy_maps" in permanence_info:
                        print("  Spatial occupancy maps generated")
                    if "object_positions" in permanence_info:
                        n = permanence_info["object_positions"].shape[1]
                        print(f"  Tracking {n} objects")
                    print("=" * 50 + "\n")

                # Enhance prompt and re-tokenize
                enhanced = _build_enhanced_prompt(prompt, permanence_info)
                if enhanced != prompt:
                    messages[0]["content"][1]["text"] = enhanced
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }

            except Exception as e:
                print(f"Warning: permanence processing failed: {e}")
                import traceback

                traceback.print_exc()
            finally:
                # Clean caches
                for obj in (model, getattr(model, "model", None)):
                    if obj is not None:
                        for attr in ("past_key_values",):
                            if hasattr(obj, attr):
                                setattr(obj, attr, None)
                        if hasattr(obj, "reset_cache"):
                            obj.reset_cache()
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
                model.train(was_training)

    # ------- generate -------
    print("Generating response ...")
    with torch.no_grad():
        for obj in (model, getattr(model, "model", None)):
            if obj is not None and hasattr(obj, "past_key_values"):
                obj.past_key_values = None

        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True
        )

    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    trimmed = [out[len(inp) :] for inp, out in zip(input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL + Object Permanence")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument(
        "--prompt", type=str, default="What is happening in this video?"
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--enable-permanence", action="store_true", default=True)
    parser.add_argument(
        "--no-permanence", dest="enable_permanence", action="store_false"
    )
    parser.add_argument("--show-permanence-info", action="store_true")
    parser.add_argument("--num-objects", type=int, default=10)
    parser.add_argument("--feature-dim", type=int, default=512)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {args.video}")
        return

    try:
        model, processor, actual_device = load_model(
            device=args.device,
            use_flash_attention=args.use_flash_attention,
        )

        permanence_module = None
        if args.enable_permanence:
            print("Setting up Object Permanence module ...")
            hidden_dim = _get_model_hidden_dim(model)
            print(f"  Model hidden dim: {hidden_dim}")

            permanence_module = (
                ObjectPermanenceModule(
                    feature_dim=args.feature_dim,
                    num_objects=args.num_objects,
                    use_spatial_grid=True,
                )
                .to(actual_device)
                .float()
            )

            extractor = (
                SimpleFeatureExtractor(
                    input_dim=hidden_dim,
                    object_feature_dim=args.feature_dim,
                    num_objects=args.num_objects,
                )
                .to(actual_device)
                .float()
            )
            permanence_module.set_feature_extractor(extractor)
            print("  Object permanence ready.")

        response = generate_response(
            model=model,
            processor=processor,
            video_path=str(video_path),
            prompt=args.prompt,
            permanence_module=permanence_module,
            num_frames=args.num_frames,
            max_new_tokens=args.max_tokens,
            device=actual_device,
            show_permanence_info=args.show_permanence_info,
        )

        print("\n" + "=" * 50)
        print("RESPONSE:")
        print("=" * 50)
        print(response)
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
