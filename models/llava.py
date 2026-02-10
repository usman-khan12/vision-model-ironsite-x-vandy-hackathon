"""
LLaVA-NeXT-Video-7B with Object Permanence.

Loads the LLaVA model, optionally wraps it with the object permanence module,
and runs inference on a video file.

Usage:
    python -m models.llava \
        --video videos/video2.mp4 \
        --prompt "What happens to the bottle?" \
        --enable-permanence --show-permanence-info
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from huggingface_hub import snapshot_download

from utils.video import process_video
from object_permanence import ObjectPermanenceModule
from object_permanence.feature_extractor import SimpleFeatureExtractor


MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class LlavaWithPermanence(nn.Module):
    """Wraps LLaVA-NeXT-Video with object permanence capabilities."""

    def __init__(
        self,
        base_model: LlavaNextVideoForConditionalGeneration,
        feature_dim: int = 512,
        num_objects: int = 10,
        enable_permanence: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.enable_permanence = enable_permanence

        if not enable_permanence:
            return

        model_hidden_dim = getattr(
            getattr(base_model, "config", None), "hidden_size", 4096
        )

        self.permanence_module = ObjectPermanenceModule(
            feature_dim=feature_dim,
            num_objects=num_objects,
        )
        self.feature_extractor = SimpleFeatureExtractor(
            input_dim=model_hidden_dim,
            object_feature_dim=feature_dim,
            num_objects=num_objects,
        )
        self.permanence_module.set_feature_extractor(self.feature_extractor)

    # -- helpers --

    def _extract_hidden(self, model_outputs, inputs):
        """Pull a [batch, seq, hidden] tensor from model outputs."""
        if isinstance(model_outputs, tuple):
            return model_outputs[0]
        if isinstance(model_outputs, torch.Tensor):
            return model_outputs
        batch = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
        device = inputs["input_ids"].device if "input_ids" in inputs else "cpu"
        return torch.zeros(batch, 1, 4096, device=device)

    # -- public API --

    def forward_with_permanence(self, inputs, process_frames_separately=True):
        """Forward pass that also processes frames through permanence pipeline."""
        if not self.enable_permanence:
            return self.base_model(**inputs)

        video_inputs = inputs.get("videos") or inputs.get("pixel_values")
        if video_inputs is None or not process_frames_separately:
            return self.base_model(**inputs)

        batch_size = video_inputs.shape[0]
        num_frames = video_inputs.shape[1] if video_inputs.dim() > 4 else 1
        permanence_outputs = []

        for frame_idx in range(num_frames):
            frame = (
                video_inputs[:, frame_idx] if video_inputs.dim() == 5 else video_inputs
            )
            frame_inputs = dict(inputs)
            for key in ("videos", "pixel_values"):
                if key in frame_inputs:
                    frame_inputs[key] = (
                        frame.unsqueeze(1) if frame.dim() == 4 else frame
                    )

            try:
                with torch.no_grad():
                    if hasattr(self.base_model, "model"):
                        out = self.base_model.model(
                            **frame_inputs, output_hidden_states=True
                        )
                    else:
                        out = (torch.zeros(batch_size, 1, 4096, device=frame.device),)
            except Exception:
                out = (torch.zeros(batch_size, 1, 4096, device=frame.device),)

            features = self._extract_hidden(out, frame_inputs)
            if features.dim() == 3:
                features = features.mean(dim=1)

            permanence_outputs.append(
                self.permanence_module(features, frame_idx=frame_idx)
            )

        outputs = self.base_model(**inputs)
        outputs.permanence_info = permanence_outputs
        return outputs

    def generate(self, **kwargs):
        """Generate with permanence pre-processing."""
        gen_keys = {
            "max_new_tokens",
            "max_length",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "num_beams",
            "pad_token_id",
            "eos_token_id",
        }
        gen_kw = {k: v for k, v in kwargs.items() if k in gen_keys}
        model_kw = {k: v for k, v in kwargs.items() if k not in gen_keys}

        if self.enable_permanence:
            self.permanence_module.reset()
            self.forward_with_permanence(model_kw, process_frames_separately=True)

        return self.base_model.generate(**model_kw, **gen_kw)

    def reset_permanence(self):
        if self.enable_permanence:
            self.permanence_module.reset()


# ---------------------------------------------------------------------------
# Loading helpers
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


def load_model(device="auto", use_4bit=False, use_flash_attention=False):
    """
    Load LLaVA-NeXT-Video model and processor.

    Returns:
        (model, processor, device_str)
    """
    device = _detect_device(device)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    model_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if use_4bit:
        model_kwargs["load_in_4bit"] = True
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Loading {MODEL_ID} on {device} ...")
    try:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            MODEL_ID, **model_kwargs
        )
    except Exception:
        print("Retrying via snapshot_download ...")
        local = snapshot_download(repo_id=MODEL_ID, resume_download=True)
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            local, **{k: v for k, v in model_kwargs.items() if k != "resume_download"}
        )

    if not use_4bit:
        model = model.to(device)

    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
    print("Model loaded.")
    return model, processor, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def generate_response(
    model,
    processor,
    video_path,
    prompt,
    num_frames=8,
    max_new_tokens=100,
    device="mps",
    enable_permanence=True,
    show_permanence_info=False,
):
    clip = process_video(video_path, num_frames)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        }
    ]
    formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=formatted, videos=clip, padding=True, return_tensors="pt")
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    print("Generating response ...")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    return processor.decode(output[0][2:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LLaVA-NeXT-Video + Object Permanence")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument(
        "--prompt", type=str, default="What is happening in this video?"
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument("--use-4bit", action="store_true")
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
        base_model, processor, actual_device = load_model(
            device=args.device,
            use_4bit=args.use_4bit,
            use_flash_attention=args.use_flash_attention,
        )

        if args.enable_permanence:
            print("Wrapping with Object Permanence module ...")
            model = LlavaWithPermanence(
                base_model=base_model,
                feature_dim=args.feature_dim,
                num_objects=args.num_objects,
            ).to(actual_device)
        else:
            model = base_model

        response = generate_response(
            model=model,
            processor=processor,
            video_path=str(video_path),
            prompt=args.prompt,
            num_frames=args.num_frames,
            max_new_tokens=args.max_tokens,
            device=actual_device,
            enable_permanence=args.enable_permanence,
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
