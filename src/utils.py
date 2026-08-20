"""
utils.py — Shared helper functions for the fall detection pipeline.

Provides:
  - Video discovery across the Le2i dataset scene folders
  - Label extraction from annotation files and filename fallback
  - Frame reading (sequential-only, no seeking)
  - Progress / logging helpers
"""

import os
import re
import cv2
import logging
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s %(levelname)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

# Le2i scene folders expected after unzipping the Kaggle download
SCENE_FOLDERS = [
    "Home", "Coffee_room", "Office", "Lecture_room",
    "Home_01", "Home_02", "Coffee_room_01", "Coffee_room_02"
]


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def find_video_files(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    Walk the Le2i dataset directory and return a list of dicts:
        {
            "video_path": Path,
            "scene": str,
            "annotation_path": Path | None,
            "label": int,          # 1 = fall, 0 = not_fall
            "label_source": str,   # "annotation_fall", "annotation_adl", "filename_hint"
            "fall_start": int | None,
            "fall_end": int | None,
        }

    Labelling strategy:
      1. If a matching annotation .txt file exists:
         - start_frame > 0 and end_frame > 0  => label = 1 (fall)
         - start_frame == 0 and end_frame == 0 => label = 0 (not_fall / ADL)
      2. If no annotation file exists (e.g. Office / Lecture_room):
         - path contains 'fall' => label = 1
         - otherwise => label = 0 (ADL / not_fall)
    """
    logger = get_logger("utils.find_videos")
    entries = []

    search_roots = _find_scene_roots(data_dir)
    if not search_roots:
        logger.warning(
            "No scene folders found under %s. "
            "Make sure the dataset is downloaded and extracted.", data_dir
        )
        return entries

    for scene_root in search_roots:
        scene_name = scene_root.name
        video_dir = _find_subdir(scene_root, "Videos") or scene_root
        annot_dir = _find_subdir(scene_root, "Annotation_files") or _find_subdir(scene_root, "Annotations_files")

        video_files = sorted(
            [f for f in video_dir.rglob("*") if f.suffix.lower() in (".avi", ".mp4")]
        )
        logger.info("Scene %-18s  videos: %d", scene_name, len(video_files))

        for vp in video_files:
            entry = {
                "video_path": vp,
                "scene": scene_name,
                "annotation_path": None,
                "label": 0,
                "label_source": "filename_hint",
                "fall_start": None,
                "fall_end": None,
            }

            # --- Try annotation file first ---
            annot_path = None
            if annot_dir is not None:
                annot_path = _match_annotation(vp, annot_dir)
            if annot_path is None:
                # Try finding in scene_root recursively
                annot_path = _match_annotation(vp, scene_root)

            if annot_path is not None:
                start, end, parsed_label = _parse_annotation(annot_path)
                entry["annotation_path"] = annot_path
                entry["fall_start"] = start
                entry["fall_end"] = end
                entry["label"] = parsed_label
                entry["label_source"] = "annotation_fall" if parsed_label == 1 else "annotation_adl"

            # --- Fallback: filename / path heuristic ---
            if entry["label_source"] == "filename_hint":
                if re.search(r"fall", str(vp), re.IGNORECASE) and not re.search(r"adl|not_fall", str(vp), re.IGNORECASE):
                    entry["label"] = 1
                else:
                    entry["label"] = 0

            entries.append(entry)

    logger.info(
        "Total videos found: %d  (fall=%d, not_fall=%d)",
        len(entries),
        sum(e["label"] for e in entries),
        sum(1 - e["label"] for e in entries),
    )
    return entries


# ---------------------------------------------------------------------------
# Frame reading — sequential only, no seeking
# ---------------------------------------------------------------------------
def read_frames_uniform(
    video_path: Path,
    num_frames: int = 16,
    resize: tuple[int, int] = (160, 160),
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> np.ndarray | None:
    """
    Read *num_frames* uniformly-sampled frames from a video file.

    Key design choice: we read SEQUENTIALLY with cap.read() and keep only the
    frames at the pre-computed target indices.  We do NOT use
    cap.set(CAP_PROP_POS_FRAMES, idx) because it is unreliable on many .avi
    codecs and can crash silently or return garbled frames.

    Parameters
    ----------
    video_path : Path
        Absolute path to the video file.
    num_frames : int
        Number of frames to sample uniformly.
    resize : tuple[int, int]
        (height, width) to resize each frame.
    start_frame, end_frame : int | None
        If provided, only sample from this sub-range of the video (useful
        for clipping fall events from annotation data).  Frame indices are
        0-based.

    Returns
    -------
    np.ndarray of shape (num_frames, H, W, 3) dtype float32 [0,1], or None
    if the video could not be read.
    """
    logger = get_logger("utils.read_frames")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        logger.warning("Video reports 0 frames: %s", video_path)
        cap.release()
        return None

    # Determine the range of frames to consider
    lo = start_frame if start_frame is not None else 0
    hi = end_frame if end_frame is not None else total_frames - 1
    hi = min(hi, total_frames - 1)
    lo = max(lo, 0)
    span = hi - lo + 1

    if span < num_frames:
        # Not enough frames — duplicate last frame to fill
        target_indices = list(range(lo, hi + 1))
        while len(target_indices) < num_frames:
            target_indices.append(target_indices[-1])
    else:
        # Uniformly sample num_frames indices from [lo, hi]
        target_indices = [
            lo + int(round(i * (span - 1) / (num_frames - 1)))
            for i in range(num_frames)
        ]

    target_set = set(target_indices)
    collected: dict[int, np.ndarray] = {}

    frame_idx = 0
    # We only need to read up to max(target_indices)
    max_needed = max(target_indices)

    while frame_idx <= max_needed:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_set:
            # BGR → RGB, resize, normalise to [0, 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize[1], resize[0]))
            collected[frame_idx] = frame.astype(np.float32) / 255.0
        frame_idx += 1

    cap.release()

    if len(collected) == 0:
        logger.warning("No frames could be read from: %s", video_path)
        return None

    # Assemble in the order of target_indices (may have repeats)
    frames = []
    for idx in target_indices:
        if idx in collected:
            frames.append(collected[idx])
        elif collected:
            # Fall back to the last successfully read frame
            frames.append(list(collected.values())[-1])
    frames = np.stack(frames, axis=0)  # (num_frames, H, W, 3)
    return frames


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _find_scene_roots(data_dir: Path) -> list[Path]:
    """Locate scene folders, handling possible Kaggle nesting."""
    roots = []
    for scene in SCENE_FOLDERS:
        # Direct: data/<Scene>/
        if (data_dir / scene).is_dir():
            roots.append(data_dir / scene)
            continue
        # One level nested: data/<extra>/<Scene>/
        for child in data_dir.iterdir():
            if child.is_dir() and (child / scene).is_dir():
                roots.append(child / scene)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _find_subdir(parent: Path, name: str) -> Path | None:
    """Case-insensitive subdirectory lookup."""
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def _match_annotation(video_path: Path, search_dir: Path) -> Path | None:
    """
    Try to find an annotation .txt file that corresponds to *video_path*.
    Matching strategy: strip the video extension and look for a .txt file
    with the same stem (case-insensitive) anywhere under search_dir.
    """
    stem = video_path.stem.lower()
    for f in search_dir.rglob("*.txt"):
        if f.stem.lower() == stem:
            return f
    return None


def _parse_annotation(annot_path: Path) -> tuple[int | None, int | None, int]:
    """
    Parse a Le2i annotation file.
      - Line 1: start frame of fall
      - Line 2: end frame of fall
    Returns (start_frame, end_frame, label) where label is 1 (fall) or 0 (not_fall).
    """
    try:
        lines = [line.strip() for line in annot_path.read_text().splitlines() if line.strip()]
        if len(lines) >= 2:
            if lines[0].isdigit() and lines[1].isdigit():
                start = int(lines[0])
                end = int(lines[1])
                if start > 0 or end > 0:
                    return start, end, 1
                else:
                    return None, None, 0
    except Exception:
        pass
    return None, None, 0
