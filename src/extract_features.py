"""
extract_features.py — Read video clips, extract per-frame CNN features,
and cache them to disk for downstream Bi-LSTM training.

Architecture:
    MobileNetV2 (ImageNet, include_top=False, pooling='avg')
    → 1280-d feature vector per frame
    → output shape per clip: (num_frames, 1280)

Usage:
    python src/extract_features.py [--max-clips 120] [--num-frames 16]
                                   [--batch-size 32] [--force]

    --max-clips   Max total clips to process (roughly class-balanced).
                  Default: 120.  Keeps CPU runtime manageable.
    --num-frames  Frames sampled per clip.  Default: 16.
    --batch-size  Batch size for MobileNetV2 inference.  Default: 32.
    --force       Re-extract even if cached features exist.

Output:
    features/features.npz — contains:
        X : float32, shape (N, num_frames, 1280)
        y : int32,   shape (N,)
        video_paths : list of source video paths (for debugging)
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (
    find_video_files,
    read_frames_uniform,
    get_logger,
    DATA_DIR,
    FEATURES_DIR,
)

logger = get_logger("extract_features")

FEATURES_FILE = FEATURES_DIR / "features.npz"

# Frame resize — 160×160 is a good CPU/accuracy trade-off for MobileNetV2
FRAME_SIZE = (160, 160)


def build_feature_extractor():
    """Load MobileNetV2 as a frozen feature extractor (CPU-safe)."""
    import tensorflow as tf

    # Confirm CPU-only setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU(s) detected: %s — but this pipeline is CPU-targeted.", gpus)
    else:
        logger.info("No GPU detected — running on CPU as expected.")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(FRAME_SIZE[0], FRAME_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    logger.info("MobileNetV2 feature extractor loaded  (output dim: %d)", base.output_shape[-1])
    return base


def extract_clip_features(model, frames: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """
    Extract features for all frames of a single clip.

    Parameters
    ----------
    model : tf.keras.Model
        The frozen MobileNetV2 feature extractor.
    frames : np.ndarray
        Shape (num_frames, H, W, 3), float32, range [0, 1].
    batch_size : int
        How many frames to feed through the CNN at once.

    Returns
    -------
    np.ndarray of shape (num_frames, 1280)
    """
    import tensorflow as tf

    # MobileNetV2 expects pixels preprocessed via its dedicated function.
    # The function expects values in [0, 255], so we scale back.
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        frames * 255.0
    )

    features = model.predict(preprocessed, batch_size=batch_size, verbose=0)
    return features  # (num_frames, 1280)


def select_clips(entries: list[dict], max_clips: int) -> list[dict]:
    """
    Select up to *max_clips* entries, roughly class-balanced.
    Falls back to whatever is available if one class has fewer samples.
    """
    falls = [e for e in entries if e["label"] == 1]
    nonfalls = [e for e in entries if e["label"] == 0]

    rng = np.random.RandomState(42)
    rng.shuffle(falls)
    rng.shuffle(nonfalls)

    per_class = max_clips // 2
    selected_falls = falls[:per_class]
    selected_nonfalls = nonfalls[:per_class]

    # If one class is short, fill from the other
    remaining = max_clips - len(selected_falls) - len(selected_nonfalls)
    if remaining > 0:
        if len(selected_falls) < per_class:
            selected_nonfalls = nonfalls[:per_class + remaining]
        else:
            selected_falls = falls[:per_class + remaining]

    selected = selected_falls + selected_nonfalls
    rng.shuffle(selected)

    logger.info(
        "Selected %d clips  (fall=%d, not_fall=%d)",
        len(selected),
        sum(e["label"] for e in selected),
        sum(1 - e["label"] for e in selected),
    )
    return selected


def main():
    parser = argparse.ArgumentParser(description="Extract CNN features from video clips.")
    parser.add_argument("--max-clips", type=int, default=120,
                        help="Max clips to process (default: 120).")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Frames to sample per clip (default: 16).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="CNN inference batch size (default: 32).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cached features exist.")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if cached
    if FEATURES_FILE.exists() and not args.force:
        logger.info("Cached features found at %s. Use --force to re-extract.", FEATURES_FILE)
        data = np.load(str(FEATURES_FILE), allow_pickle=True)
        logger.info("  Loaded X shape: %s, y shape: %s", data["X"].shape, data["y"].shape)
        return

    # Discover videos
    entries = find_video_files(DATA_DIR)
    if not entries:
        logger.error("No videos found in %s. Run download_data.py first.", DATA_DIR)
        sys.exit(1)

    # Select subset
    selected = select_clips(entries, args.max_clips)

    # Load model
    model = build_feature_extractor()

    # Extract features
    all_features = []
    all_labels = []
    all_paths = []
    failed = 0
    t_start = time.time()

    for i, entry in enumerate(tqdm(selected, desc="Extracting features", unit="clip")):
        # Progress every 10 clips
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(selected) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "Clip %d/%d  |  elapsed: %.0fs  |  ETA: %.0fs",
                i + 1, len(selected), elapsed, remaining,
            )

        try:
            frames = read_frames_uniform(
                entry["video_path"],
                num_frames=args.num_frames,
                resize=FRAME_SIZE,
                start_frame=entry.get("fall_start"),
                end_frame=entry.get("fall_end") if entry["label"] == 1 else None,
            )
            if frames is None:
                logger.warning("Skipping (unreadable): %s", entry["video_path"])
                failed += 1
                continue

            feats = extract_clip_features(model, frames, batch_size=args.batch_size)
            all_features.append(feats)
            all_labels.append(entry["label"])
            all_paths.append(str(entry["video_path"]))

        except Exception as e:
            logger.warning("Skipping (error): %s — %s", entry["video_path"], e)
            failed += 1

    if not all_features:
        logger.error("No features were extracted. Check your data/ directory.")
        sys.exit(1)

    X = np.stack(all_features, axis=0).astype(np.float32)  # (N, T, 1280)
    y = np.array(all_labels, dtype=np.int32)                # (N,)

    logger.info("Extraction complete in %.0fs.", time.time() - t_start)
    logger.info("  Successful clips: %d  |  Failed: %d", len(all_features), failed)
    logger.info("  X shape: %s  |  y shape: %s", X.shape, y.shape)
    logger.info("  Class distribution — fall: %d, not_fall: %d",
                np.sum(y == 1), np.sum(y == 0))

    # Save
    np.savez_compressed(
        str(FEATURES_FILE),
        X=X,
        y=y,
        video_paths=np.array(all_paths, dtype=object),
    )
    logger.info("Features saved to %s (%.1f MB)",
                FEATURES_FILE, FEATURES_FILE.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
