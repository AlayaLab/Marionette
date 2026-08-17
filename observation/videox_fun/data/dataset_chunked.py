"""Helios-style chunked-AR training dataset.

Each item produces:
  - pixel_values         : 81 RGB frames to denoise (the "current chunk")
  - control_pixel_values : 81 pose frames aligned with chunk
  - history_pixel_values         : 162 prior RGB frames (clean, used as condition)
  - history_control_pixel_values : 162 prior pose frames
  - has_history (bool)   : False if cold-start (10% of samples) → history is zeroed
  - text, data_type, idx

When `enable_bucket=True` (matches train_control.py), the returned arrays are
raw numpy of shape [T, H, W, 3]. The collate_fn in train_control.py is
responsible for resize+normalize via the bucket transform.

The dataset reads chunk_start / hist_start explicitly from train_data_chunks.json
— there is no random offset selection at __getitem__ time. AR rollout at
inference will run consecutive chunks at fixed stride, so action-aligned
training starts would not help; we just stride-sample the whole segment.
"""
import json
import os
import random

import numpy as np
import torch

from func_timeout import FunctionTimedOut, func_timeout

from .dataset_image_video import (
    ImageVideoControlDataset,
    VideoReader_contextmanager,
    VIDEO_READER_TIMEOUT,
    get_video_reader_batch,
    resize_frame,
)


class ChunkedVideoControlDataset(ImageVideoControlDataset):
    """Extends ImageVideoControlDataset by reading a fixed chunk + history window
    per item (positions specified in metadata, not random)."""

    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=512,
        video_sample_stride=4,
        video_sample_n_frames=81,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0,   # ignored for chunked (we use explicit positions)
        video_length_drop_end=1.0,
        enable_inpaint=False,
        enable_camera_info=False,
        return_file_name=False,
        enable_subject_info=False,
        padding_subject_info=True,
        # Chunked-specific
        history_n_frames=162,
        cold_start_prob=0.10,
        anti_drift_noise_prob=0.50,
        anti_drift_noise_std=0.02,
        anti_drift_exposure_prob=0.30,
        anti_drift_exposure_range=(0.9, 1.1),
    ):
        super().__init__(
            ann_path=ann_path,
            data_root=data_root,
            video_sample_size=video_sample_size,
            video_sample_stride=video_sample_stride,
            video_sample_n_frames=video_sample_n_frames,
            image_sample_size=image_sample_size,
            video_repeat=video_repeat,
            text_drop_ratio=text_drop_ratio,
            enable_bucket=enable_bucket,
            video_length_drop_start=video_length_drop_start,
            video_length_drop_end=video_length_drop_end,
            enable_inpaint=enable_inpaint,
            enable_camera_info=enable_camera_info,
            return_file_name=return_file_name,
            enable_subject_info=enable_subject_info,
            padding_subject_info=padding_subject_info,
        )
        self.history_n_frames = history_n_frames
        self.cold_start_prob = cold_start_prob
        self.anti_drift_noise_prob = anti_drift_noise_prob
        self.anti_drift_noise_std = anti_drift_noise_std
        self.anti_drift_exposure_prob = anti_drift_exposure_prob
        self.anti_drift_exposure_range = anti_drift_exposure_range

    def _read_window(self, path, start, n_frames):
        """Read frames [start, start+n_frames) from a video file, return raw uint8 numpy [T,H,W,3]
        with each frame resized to short-side = larger_side_of_image_and_video.
        """
        with VideoReader_contextmanager(path, num_threads=2) as vr:
            total = len(vr)
            end = min(start + n_frames, total)
            indices = np.arange(start, end, dtype=int)
            if len(indices) < n_frames:
                # pad by repeating last frame (shouldn't happen if metadata is correct)
                pad = np.full(n_frames - len(indices), indices[-1])
                indices = np.concatenate([indices, pad])
            try:
                frames = func_timeout(
                    VIDEO_READER_TIMEOUT,
                    get_video_reader_batch,
                    args=(vr, indices),
                )
            except FunctionTimedOut:
                raise ValueError(f"timeout reading {path} at {start}")
            resized = [resize_frame(f, self.larger_side_of_image_and_video) for f in frames]
            return np.array(resized)

    def _anti_drift(self, history_np):
        """Pixel-space corruption on uint8 history frames (in-place style returns new array)."""
        if history_np is None:
            return history_np
        out = history_np.astype(np.float32)
        if random.random() < self.anti_drift_noise_prob:
            noise = np.random.randn(*out.shape).astype(np.float32) * (self.anti_drift_noise_std * 255.0)
            out = out + noise
        if random.random() < self.anti_drift_exposure_prob:
            lo, hi = self.anti_drift_exposure_range
            factor = random.uniform(lo, hi)
            out = out * factor
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def get_batch_chunked(self, idx):
        """Returns (chunk_rgb, chunk_pose, history_rgb, history_pose, has_history, text)
        with arrays as raw uint8 numpy [T,H,W,3] (enable_bucket=True path)."""
        data_info = self.dataset[idx % len(self.dataset)]
        rgb_path = data_info["file_path"]
        pose_path = data_info["control_file_path"]
        chunk_start = int(data_info["chunk_start"])
        hist_start = int(data_info["hist_start"])
        text = data_info.get("text", "")

        if self.data_root is not None:
            rgb_path = os.path.join(self.data_root, rgb_path)
            pose_path = os.path.join(self.data_root, pose_path)

        # Read chunk
        chunk_rgb = self._read_window(rgb_path, chunk_start, self.video_sample_n_frames)
        chunk_pose = self._read_window(pose_path, chunk_start, self.video_sample_n_frames)

        # Decide cold-start
        has_history = random.random() >= self.cold_start_prob

        if has_history:
            hist_rgb = self._read_window(rgb_path, hist_start, self.history_n_frames)
            hist_pose = self._read_window(pose_path, hist_start, self.history_n_frames)
            hist_rgb = self._anti_drift(hist_rgb)
            # pose history NOT corrupted — pose is geometric, corruption would mislead
        else:
            hist_rgb = np.zeros(
                (self.history_n_frames, chunk_rgb.shape[1], chunk_rgb.shape[2], 3),
                dtype=np.uint8,
            )
            hist_pose = np.zeros_like(hist_rgb)

        # Random text dropout (matches parent behavior)
        if random.random() < self.text_drop_ratio:
            text = ""

        return chunk_rgb, chunk_pose, hist_rgb, hist_pose, has_history, text

    def __getitem__(self, idx):
        while True:
            try:
                chunk_rgb, chunk_pose, hist_rgb, hist_pose, has_history, text = self.get_batch_chunked(idx)
                if not self.enable_bucket:
                    raise NotImplementedError(
                        "ChunkedVideoControlDataset requires --enable_bucket; the collate_fn "
                        "applies bucket-aware resize+normalize to all four tensors."
                    )
                sample = {
                    "pixel_values": chunk_rgb,                 # [81, H, W, 3] uint8
                    "control_pixel_values": chunk_pose,
                    "history_pixel_values": hist_rgb,          # [162, H, W, 3] uint8
                    "history_control_pixel_values": hist_pose,
                    "has_history": bool(has_history),
                    "subject_image": None,
                    "text": text,
                    "data_type": "video",
                    "idx": idx,
                }
                return sample
            except Exception as e:
                print(f"[ChunkedDataset] idx={idx} err={e}", flush=True)
                idx = random.randint(0, self.length - 1)
