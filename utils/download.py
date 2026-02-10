"""
Shared model download helper.
Pre-downloads model weights with progress tracking so inference scripts
can start immediately.
"""

import sys
from huggingface_hub import snapshot_download
from transformers import AutoProcessor


# Known model configs: model_id -> (display_name, size_hint)
KNOWN_MODELS = {
    "llava-hf/LLaVA-NeXT-Video-7B-hf": ("LLaVA-NeXT-Video-7B", "~14GB"),
    "Qwen/Qwen3-VL-8B-Instruct": ("Qwen3-VL-8B-Instruct", "~16GB"),
}


def download_model(model_id):
    """
    Download model files and processor with resume support.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        Local directory path where files were saved.
    """
    info = KNOWN_MODELS.get(model_id, (model_id, "unknown size"))
    print(f"Downloading model: {info[0]} ({info[1]})")
    print("=" * 60)

    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            resume_download=True,
            local_files_only=False,
        )
        print(f"\nModel downloaded to: {local_dir}")

        print("Downloading processor...")
        AutoProcessor.from_pretrained(model_id)
        print("Processor ready.")

        return local_dir
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. Re-run to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        print("Re-run to retry / resume.")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download a model from HuggingFace")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()
    download_model(args.model)
