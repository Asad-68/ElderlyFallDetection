"""
train.py — Train a Bidirectional LSTM on cached CNN features for
binary fall detection (fall vs. not_fall).

Architecture:
    Input (num_frames, 1280)
    → Bidirectional LSTM (64 units)
    → Dropout (0.4)
    → Dense (32, relu)
    → Dropout (0.3)
    → Dense (1, sigmoid)

Usage:
    python src/train.py [--epochs 50] [--batch-size 16] [--test-size 0.2]

Output:
    models/fall_detection_bilstm.keras   — trained model
    features/train_test_split.npz        — cached split indices (for evaluate.py)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import get_logger, FEATURES_DIR, MODELS_DIR, RESULTS_DIR

logger = get_logger("train")

FEATURES_FILE = FEATURES_DIR / "features.npz"
SPLIT_FILE    = FEATURES_DIR / "train_test_split.npz"
MODEL_FILE    = MODELS_DIR / "fall_detection_bilstm.keras"
HISTORY_PLOT  = RESULTS_DIR / "training_history.png"


def build_bilstm(input_shape: tuple) -> tf.keras.Model:
    """
    Build a lightweight Bi-LSTM binary classifier.

    Kept small on purpose — this is a CPU proof-of-concept and the dataset
    is small.  Scaling up: increase LSTM units, add a second LSTM layer,
    or unfreeze parts of MobileNetV2 for fine-tuning.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # Bi-LSTM: 64 units in each direction → 128-d output
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False)
        ),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Bi-LSTM for fall detection.")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for testing (default: 0.2).")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (default: 8).")
    args = parser.parse_args()

    # --- Load cached features ---
    if not FEATURES_FILE.exists():
        logger.error("Features file not found at %s. Run extract_features.py first.", FEATURES_FILE)
        sys.exit(1)

    data = np.load(str(FEATURES_FILE), allow_pickle=True)
    X = data["X"]  # (N, T, 1280)
    y = data["y"]  # (N,)
    logger.info("Loaded features: X=%s  y=%s", X.shape, y.shape)
    logger.info("Class counts — fall: %d, not_fall: %d", np.sum(y == 1), np.sum(y == 0))

    # --- Stratified train/test split ---
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )
    logger.info("Train: %d samples  |  Test: %d samples", len(X_train), len(X_test))

    # Save split indices so evaluate.py uses the exact same test set
    # (We save the actual data, not indices, for simplicity)
    np.savez_compressed(
        str(SPLIT_FILE),
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
    )
    logger.info("Train/test split saved to %s", SPLIT_FILE)

    # --- Build model ---
    model = build_bilstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary(print_fn=logger.info)

    # --- Callbacks ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --- Train ---
    logger.info("Starting training — %d epochs, batch size %d, patience %d",
                args.epochs, args.batch_size, args.patience)

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,   # 15 % of training data for validation
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,               # Epoch-by-epoch progress
    )

    # --- Save ---
    model.save(str(MODEL_FILE))
    logger.info("Model saved to %s", MODEL_FILE)

    # --- Plot training history ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs_range, history.history["loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history.history:
        ax1.plot(epochs_range, history.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (Binary Crossentropy)", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs_range, history.history["accuracy"], label="Train Accuracy", color="#2ca02c", lw=2)
    if "val_accuracy" in history.history:
        ax2.plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy", color="#d62728", lw=2, linestyle="--")
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(str(HISTORY_PLOT), dpi=150)
    plt.close(fig)
    logger.info("Training history plot saved to %s", HISTORY_PLOT)

    # Quick training summary
    best_epoch = np.argmin(history.history["val_loss"])
    logger.info(
        "Best epoch: %d  |  val_loss: %.4f  |  val_accuracy: %.4f",
        best_epoch + 1,
        history.history["val_loss"][best_epoch],
        history.history["val_accuracy"][best_epoch],
    )
    logger.info("[DONE] Training complete.")


if __name__ == "__main__":
    main()
