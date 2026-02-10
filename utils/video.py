"""
Shared video processing utilities.
Used by both LLaVA and Qwen model runners.
"""

import av
import numpy as np


def read_video_pyav(container, indices):
    """
    Decode specific frames from a video using PyAV.

    Args:
        container: PyAV input container.
        indices: List of frame indices to decode.

    Returns:
        np.ndarray of shape (num_frames, height, width, 3) in RGB.
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
    Load a video and sample frames uniformly, always including
    the first and last frames.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        np.ndarray of shape (num_frames, height, width, 3).
    """
    print(f"Processing video: {video_path}")
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames

    if total_frames == 0:
        raise ValueError("Video has no frames or cannot be decoded")

    if total_frames <= num_frames:
        indices = np.arange(0, total_frames).astype(int)
    else:
        # Always include first and last frame; sample the rest uniformly
        remaining = num_frames - 2
        if remaining > 0:
            middle = np.linspace(1, total_frames - 2, remaining, dtype=int)
            indices = np.concatenate([[0], middle, [total_frames - 1]])
        else:
            indices = np.array([0, total_frames - 1])[:num_frames]

    indices = np.unique(indices)
    indices = np.sort(indices)

    print(
        f"Sampling {len(indices)} frames from {total_frames} total "
        f"(first: {indices[0]}, last: {indices[-1]})"
    )
    clip = read_video_pyav(container, indices)
    container.close()
    return clip
