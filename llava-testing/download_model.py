"""
Helper script to pre-download the model with better progress tracking.
Run this separately to download the model before testing videos.
"""
from huggingface_hub import snapshot_download
from transformers import LlavaNextVideoProcessor
import sys

def download_model(model_id="llava-hf/LLaVA-NeXT-Video-7B-hf"):
    """
    Download the model files with progress tracking.
    """
    print(f"Downloading model: {model_id}")
    print("This may take a while (~14GB total)...")
    print("=" * 60)
    
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            resume_download=True,
            local_files_only=False,
            tqdm_class=None  # Use default progress bar
        )
        print("\n" + "=" * 60)
        print(f"✓ Model downloaded successfully!")
        print(f"Location: {local_dir}")
        
        # Also download processor
        print("\nDownloading processor...")
        processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        print("✓ Processor downloaded successfully!")
        
        return local_dir
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. You can resume by running this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nYou can try running this script again to resume the download.")
        raise

if __name__ == "__main__":
    download_model()

