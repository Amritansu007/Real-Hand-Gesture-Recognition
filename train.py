"""
train.py - Leakage-Free Session-Based Model Training
====================================================
Trains two separate gesture classification pipelines:
1. Raw Coordinates Pipeline (63 features: x, y, z for 21 landmarks)
2. Invariant Features Pipeline (8 engineered scale- & rotation-invariant features)

CRITICAL ML DESIGN NOTE (Preventing Temporal Data Leakage):
---------------------------------------------------------
Adjacent video frames within the same recording session exhibit severe temporal
autocorrelation (near-identical hand positions). Randomly shuffling frames across
train and test partitions leads to catastrophic data leakage, artificially inflating
evaluation accuracy while failing in real-world deployment.

To ensure rigorous and leak-free evaluation:
- Training vs Cross-Session data is split strictly by `session_id`.
- Same-Session validation split is created via chronological/block holdout
  (e.g., the first 80% of recorded frames per gesture for training, the remaining
  20% reserved for same-session testing), preserving temporal independence.
- Preprocessing transformers (StandardScaler) are encapsulated inside `sklearn.pipeline.Pipeline`
  and fitted strictly on training data only.

Usage:
  python train.py
  python train.py --train-session session_1_baseline --classifier rf
  python train.py --train-session session_1_baseline --classifier svm
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from features import (
    INVARIANT_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    invariant_features,
    raw_features,
)

DATA_PATH = os.path.join("data", "gestures.csv")
MODELS_DIR = "models"
RAW_MODEL_PATH = os.path.join(MODELS_DIR, "raw_model.joblib")
INVARIANT_MODEL_PATH = os.path.join(MODELS_DIR, "invariant_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """Loads CSV dataset and performs validation & null checks."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Run data_collection.py first to record gesture sessions."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset at '{csv_path}' is empty.")

    # Check required columns
    required_cols = ["session_id", "gesture_label"] + RAW_FEATURE_NAMES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Null value inspection (ML Best Practice)
    null_counts = df[required_cols].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"[WARNING] Detected {total_nulls} null values. Dropping incomplete rows.")
        df = df.dropna(subset=required_cols).reset_index(drop=True)
    else:
        print(f"[INFO] Verified dataset cleanliness: 0 null values found across {len(df)} rows.")

    return df


def split_data_by_session(
    df: pd.DataFrame, train_session_name: str, same_session_val_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs leakage-free session splitting:
    - train_df: First (1 - same_session_val_ratio) portion of the training session.
    - val_same_df: Held-out last same_session_val_ratio portion of the training session.
    - cross_test_df: All frames belonging to non-training sessions.
    """
    unique_sessions = df["session_id"].unique().tolist()
    if train_session_name not in unique_sessions:
        raise ValueError(
            f"Specified train session '{train_session_name}' not found. Available sessions: {unique_sessions}"
        )

    session_train_all = df[df["session_id"] == train_session_name].copy()
    cross_test_df = df[df["session_id"] != train_session_name].copy()

    # Sort chronologically by timestamp if available to guarantee temporal block split
    if "timestamp" in session_train_all.columns:
        session_train_all = session_train_all.sort_values(by="timestamp").reset_index(drop=True)

    # Perform contiguous chronological hold-out split per gesture to prevent frame correlation leakage
    train_splits = []
    val_splits = []

    for gesture, group in session_train_all.groupby("gesture_label"):
        n_samples = len(group)
        split_idx = int(n_samples * (1.0 - same_session_val_ratio))
        train_splits.append(group.iloc[:split_idx])
        val_splits.append(group.iloc[split_idx:])

    train_df = pd.concat(train_splits, ignore_index=True)
    val_same_df = pd.concat(val_splits, ignore_index=True)

    print("\n" + "=" * 60)
    print(" DATA SPLIT SUMMARY (LEAKAGE-FREE PARTITIONING)")
    print("=" * 60)
    print(f"Training Session ID:        '{train_session_name}'")
    print(f"  - Train Set Samples:      {len(train_df)} frames")
    print(f"  - Same-Session Val Set:   {len(val_same_df)} frames (chronologically held-out {int(same_session_val_ratio*100)}%)")
    cross_sessions = cross_test_df["session_id"].unique().tolist()
    print(f"Cross-Session Test IDs:     {cross_sessions if cross_sessions else 'None recorded yet'}")
    print(f"  - Cross-Session Samples:  {len(cross_test_df)} frames")
    print("=" * 60)

    return train_df, val_same_df, cross_test_df


def extract_features_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts:
      - X_raw: (N, 63)
      - X_inv: (N, 8)
      - y: (N,) string labels
    """
    raw_coords = df[RAW_FEATURE_NAMES].to_numpy(dtype=np.float32)
    labels = df["gesture_label"].to_numpy()

    # Compute invariant features row by row
    inv_list = []
    for row in raw_coords:
        inv_feat = invariant_features(row)
        inv_list.append(inv_feat)

    X_inv = np.array(inv_list, dtype=np.float32)
    X_raw = raw_coords

    return X_raw, X_inv, labels


def build_pipeline(classifier_type: str = "svm") -> Pipeline:
    """
    Builds a scikit-learn Pipeline with strict scaling and estimator encapsulation.
    Optimized for ultra-low latency (< 1ms) real-time inference.
    """
    if classifier_type.lower() == "svm":
        clf = SVC(
            C=10.0,
            kernel="rbf",
            random_state=42,
        )
    elif classifier_type.lower() == "rf":
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=1,  # n_jobs=1 ensures fast in-process execution without joblib IPC dispatch overhead
        )
    elif classifier_type.lower() == "lr":
        clf = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported classifier type: '{classifier_type}'. Choose 'svm', 'rf', or 'lr'.")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])


def train_models(
    train_session: str = None,
    classifier_type: str = "svm",
    same_session_val_ratio: float = 0.20,
):
    """Orchestrates loading, session splitting, training, validation, and serialization."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_and_validate_data(DATA_PATH)

    unique_sessions = df["session_id"].unique().tolist()
    if train_session is None:
        # Default to first session or session_1
        candidate_sessions = [s for s in unique_sessions if "1" in s or "train" in s or "baseline" in s]
        train_session = candidate_sessions[0] if candidate_sessions else unique_sessions[0]
        print(f"[INFO] Auto-selected training session: '{train_session}'")

    train_df, val_same_df, cross_test_df = split_data_by_session(
        df, train_session, same_session_val_ratio=same_session_val_ratio
    )

    # Extract feature matrices
    X_train_raw, X_train_inv, y_train = extract_features_matrix(train_df)
    X_val_raw, X_val_inv, y_val = extract_features_matrix(val_same_df)

    classes = np.unique(y_train).tolist()
    print(f"\n[INFO] Gesture Classes ({len(classes)}): {classes}")

    # 1. Train Raw Coordinates Pipeline
    print("\n--> Training Raw Coordinates Pipeline (63 Features)...")
    pipe_raw = build_pipeline(classifier_type)
    pipe_raw.fit(X_train_raw, y_train)

    y_pred_val_raw = pipe_raw.predict(X_val_raw)
    acc_val_raw = accuracy_score(y_val, y_pred_val_raw)
    print(f"  [RESULT] Raw Pipeline Same-Session Val Accuracy: {acc_val_raw * 100:.2f}%")

    # 2. Train Invariant Features Pipeline
    print("\n--> Training Invariant Features Pipeline (8 Features)...")
    pipe_inv = build_pipeline(classifier_type)
    pipe_inv.fit(X_train_inv, y_train)

    y_pred_val_inv = pipe_inv.predict(X_val_inv)
    acc_val_inv = accuracy_score(y_val, y_pred_val_inv)
    print(f"  [RESULT] Invariant Pipeline Same-Session Val Accuracy: {acc_val_inv * 100:.2f}%")

    # Save models
    joblib.dump(pipe_raw, RAW_MODEL_PATH)
    joblib.dump(pipe_inv, INVARIANT_MODEL_PATH)
    print(f"\n[SAVED] Raw model saved to: {RAW_MODEL_PATH}")
    print(f"[SAVED] Invariant model saved to: {INVARIANT_MODEL_PATH}")

    # Save metadata
    cross_sessions = cross_test_df["session_id"].unique().tolist()
    metadata = {
        "train_session": train_session,
        "cross_sessions": cross_sessions,
        "classifier_type": classifier_type,
        "classes": classes,
        "n_train_samples": int(len(train_df)),
        "n_val_same_samples": int(len(val_same_df)),
        "n_cross_samples": int(len(cross_test_df)),
        "same_session_val_accuracy": {
            "raw_coordinates": float(acc_val_raw),
            "invariant_features": float(acc_val_inv),
        },
        "raw_feature_names": RAW_FEATURE_NAMES,
        "invariant_feature_names": INVARIANT_FEATURE_NAMES,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Metadata saved to: {METADATA_PATH}")

    print("\nDetailed Same-Session Validation Classification Report (Invariant Features):")
    print(classification_report(y_val, y_pred_val_inv))


def main():
    parser = argparse.ArgumentParser(description="Session-based Model Training for Hand Gesture Recognition")
    parser.add_argument(
        "--train-session",
        type=str,
        default=None,
        help="Session ID to use for training (defaults to first or baseline session)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["svm", "rf"],
        default="svm",
        help="Classifier algorithm: 'svm' (Support Vector Machine, sub-ms) or 'rf' (Random Forest)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Proportion of training session held out for same-session validation (default: 0.20)",
    )
    args = parser.parse_args()

    train_models(
        train_session=args.train_session,
        classifier_type=args.classifier,
        same_session_val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
