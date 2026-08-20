"""
evaluate.py — Evaluate the trained Bi-LSTM fall detection model.

Outputs:
    - Classification report (precision, recall, F1-score) printed to console
    - Confusion matrix saved as  results/confusion_matrix.png

Usage:
    python src/evaluate.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import get_logger, FEATURES_DIR, MODELS_DIR, RESULTS_DIR

logger = get_logger("evaluate")

SPLIT_FILE = FEATURES_DIR / "train_test_split.npz"
MODEL_FILE = MODELS_DIR / "fall_detection_bilstm.keras"


def main():
    # --- Validate prerequisites ---
    if not SPLIT_FILE.exists():
        logger.error("Split file not found at %s. Run train.py first.", SPLIT_FILE)
        sys.exit(1)
    if not MODEL_FILE.exists():
        logger.error("Trained model not found at %s. Run train.py first.", MODEL_FILE)
        sys.exit(1)

    # --- Load data and model ---
    import tensorflow as tf

    split = np.load(str(SPLIT_FILE), allow_pickle=True)
    X_test = split["X_test"]
    y_test = split["y_test"]
    logger.info("Test set: %d samples  (fall=%d, not_fall=%d)",
                len(y_test), np.sum(y_test == 1), np.sum(y_test == 0))

    model = tf.keras.models.load_model(str(MODEL_FILE))
    logger.info("Model loaded from %s", MODEL_FILE)

    # --- Predict ---
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    # --- Classification report ---
    target_names = ["not_fall", "fall"]
    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=target_names, zero_division=0
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    logger.info("Confusion matrix:\n%s", cm)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        cbar=False,
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Fall Detection — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    cm_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(str(cm_path), dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", cm_path)

    # --- Per-sample predictions (useful for debugging) ---
    print("\n" + "-" * 60)
    print("PER-SAMPLE PREDICTIONS (first 20)")
    print("-" * 60)
    for i in range(min(20, len(y_test))):
        status = "[OK]      " if y_pred[i] == y_test[i] else "[MISMATCH]"
        print(f"  {status}  true={target_names[y_test[i]]:>9s}  "
              f"pred={target_names[y_pred[i]]:>9s}  "
              f"prob={y_proba[i]:.3f}")

    # --- Summary ---
    acc = np.mean(y_pred == y_test)
    print(f"\nOverall accuracy: {acc:.2%}")
    print(f"Confusion matrix image: {cm_path}")
    logger.info("[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
