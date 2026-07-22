"""Pure planning and assembly of file-bounded manual latent windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ManualLatentWindowPlan:
    """A complete decoder window anchored inside one source file."""

    anchor_frame: int
    file_index: int
    file_start: int
    file_end: int
    chunk_start: int
    window_size: int
    frame_indices: Tuple[int, ...]

    @property
    def file_length(self) -> int:
        return self.file_end - self.file_start

    @property
    def repeated_final_frames(self) -> int:
        return max(0, self.window_size - self.file_length)

    @property
    def is_short_file(self) -> bool:
        return self.file_length < self.window_size


def _validated_offsets(
    file_offsets: Sequence[int], frame_count: Optional[int]
) -> Tuple[np.ndarray, int]:
    offsets_input = np.asarray(file_offsets)
    if offsets_input.ndim != 1 or offsets_input.size < 2:
        raise ValueError("file_offsets must be a one-dimensional [num_files + 1] array")
    if not np.all(np.isfinite(offsets_input)):
        raise ValueError("file_offsets contains non-finite values")

    offsets = offsets_input.astype(np.int64)
    if not np.array_equal(offsets_input, offsets):
        raise ValueError("file_offsets must contain integer frame positions")
    if offsets[0] != 0:
        raise ValueError(f"file_offsets must start at 0, got {int(offsets[0])}")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("file_offsets must be nondecreasing")

    resolved_frame_count = int(offsets[-1]) if frame_count is None else int(frame_count)
    if resolved_frame_count <= 0:
        raise ValueError("the corpus must contain at least one latent frame")
    if int(offsets[-1]) != resolved_frame_count:
        raise ValueError(
            "file_offsets must end at frame_count: "
            f"{int(offsets[-1])} != {resolved_frame_count}"
        )
    return offsets, resolved_frame_count


def plan_file_bounded_latent_window(
    anchor_frame: int,
    window_size: int,
    file_offsets: Sequence[int],
    *,
    frame_count: Optional[int] = None,
) -> ManualLatentWindowPlan:
    """Clamp an anchor to a complete T-frame chunk within its source file.

    For files at least T frames long, every planned index is unique and the
    chunk shifts backward near the file end.  Only a file shorter than T is
    padded, by repeating that file's final frame.
    """

    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    offsets, resolved_frame_count = _validated_offsets(file_offsets, frame_count)
    anchor = int(np.clip(int(anchor_frame), 0, resolved_frame_count - 1))
    file_index = int(np.searchsorted(offsets, anchor, side="right") - 1)
    file_index = int(np.clip(file_index, 0, offsets.size - 2))
    file_start = int(offsets[file_index])
    file_end = int(offsets[file_index + 1])
    if not (file_start <= anchor < file_end):
        raise ValueError(
            f"anchor {anchor} does not belong to a non-empty source-file range"
        )

    latest_start = max(file_start, file_end - window_size)
    chunk_start = int(np.clip(anchor, file_start, latest_start))
    indices = np.minimum(
        chunk_start + np.arange(window_size, dtype=np.int64), file_end - 1
    )

    return ManualLatentWindowPlan(
        anchor_frame=anchor,
        file_index=file_index,
        file_start=file_start,
        file_end=file_end,
        chunk_start=chunk_start,
        window_size=window_size,
        frame_indices=tuple(int(index) for index in indices),
    )


def assemble_manual_latent_window(
    latents: np.ndarray,
    plan: ManualLatentWindowPlan,
    *,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gather a planned ``[T, D]`` window and optionally denormalize it."""

    latent_array = np.asarray(latents, dtype=np.float32)
    if latent_array.ndim != 2 or latent_array.shape[0] <= 0 or latent_array.shape[1] <= 0:
        raise ValueError(f"latents must be a non-empty [N, D] array, got {latent_array.shape}")
    if len(plan.frame_indices) != plan.window_size:
        raise ValueError("plan frame count does not match its window_size")

    indices = np.asarray(plan.frame_indices, dtype=np.intp)
    if np.any(indices < 0) or np.any(indices >= latent_array.shape[0]):
        raise ValueError("plan contains frame indices outside the latent array")
    if np.any(indices < plan.file_start) or np.any(indices >= plan.file_end):
        raise ValueError("plan crosses its declared source-file boundary")

    window = np.ascontiguousarray(latent_array[indices], dtype=np.float32)
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together")
    if mean is None:
        return window

    latent_dim = int(latent_array.shape[1])
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_array = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_array.shape != (latent_dim,) or std_array.shape != (latent_dim,):
        raise ValueError(
            f"mean and std must both be shape [{latent_dim}], got "
            f"{mean_array.shape} and {std_array.shape}"
        )
    return np.ascontiguousarray(
        window * std_array[None, :] + mean_array[None, :], dtype=np.float32
    )


def build_file_bounded_latent_window(
    latents: np.ndarray,
    file_offsets: Sequence[int],
    anchor_frame: int,
    window_size: int,
    *,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ManualLatentWindowPlan]:
    """Plan and assemble one complete file-local decoder input window."""

    latent_array = np.asarray(latents)
    if latent_array.ndim != 2:
        raise ValueError(f"latents must be [N, D], got {latent_array.shape}")
    plan = plan_file_bounded_latent_window(
        anchor_frame=anchor_frame,
        window_size=window_size,
        file_offsets=file_offsets,
        frame_count=int(latent_array.shape[0]),
    )
    return (
        assemble_manual_latent_window(
            latent_array,
            plan,
            mean=mean,
            std=std,
        ),
        plan,
    )
