"""
AutoScene – Video Processing
==============================
Handles:
  • Frame sampling from uploaded video
  • Per-frame segmentation
  • Recomposition into output video with overlay
"""

import os
import tempfile
import numpy as np
import cv2
from typing import Callable, Generator, Tuple
import torch

from utils.inference import predict
from utils.camvid_classes import overlay_mask


def iter_frames(video_path: str,
                sample_every: int = 1
                ) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, rgb_frame) for every `sample_every`-th frame.
    Frames are returned as uint8 RGB numpy arrays at original resolution.
    """
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def process_video(
    video_path: str,
    model: torch.nn.Module,
    alpha: float = 0.45,
    sample_every: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    """
    Segment every sampled frame and write an output video with overlay.

    Args:
        video_path        : path to the uploaded video file
        model             : loaded segmentation model
        alpha             : overlay opacity (0–1)
        sample_every      : use 1 for every frame, 2 for half, etc.
        progress_callback : optional fn(current_frame, total_frames)

    Returns:
        output_path : path to the processed output video (.mp4)
    """
    info = get_video_info(video_path)
    fps  = info["fps"] / sample_every  if sample_every > 1 else info["fps"]
    w, h = info["width"], info["height"]

    # Temporary output file
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "autoscene_segmented.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total = max(1, info["total_frames"] // sample_every)

    for i, (frame_idx, rgb_frame) in enumerate(
            iter_frames(video_path, sample_every=sample_every)):

        # Predict mask at original resolution
        mask = predict(model, rgb_frame)

        # Blend overlay at original resolution
        blended = overlay_mask(rgb_frame, mask, alpha=alpha)

        # Write as BGR
        writer.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        if progress_callback:
            progress_callback(i + 1, total)

    writer.release()
    return output_path
