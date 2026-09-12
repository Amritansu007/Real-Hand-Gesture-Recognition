"""
deploy.py - Live Webcam Real-Time Hand Gesture Recognition Deployment
=====================================================================
Deploys the real-time gesture recognition pipeline on live webcam feed.
Automatically selects the winning model from the 2x2 cross-session generalization
matrix (`results/evaluation_summary.json`), with optional manual toggle.

Features:
  - Real-time MediaPipe Hand landmark tracking.
  - Bounding box computation with adaptive padding around hand landmarks.
  - Overlaid predicted gesture label, class confidence percentage, and FPS HUD.
  - Live model toggle (`[M]` key) between Invariant Features and Raw Coordinates
    to directly witness real-time generalization differences under rotation/translation.
  - Press `[Q]` or `[ESC]` to exit.

Usage:
  # Deploy using the winning feature model:
  python deploy.py

  # Force specific feature representation:
  python deploy.py --model invariant
  python deploy.py --model raw
  python deploy.py --cam 0
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np

from features import invariant_features, raw_features

MODELS_DIR = "models"
RAW_MODEL_PATH = os.path.join(MODELS_DIR, "raw_model.joblib")
INVARIANT_MODEL_PATH = os.path.join(MODELS_DIR, "invariant_model.joblib")
SUMMARY_JSON_PATH = os.path.join("results", "evaluation_summary.json")

# Color palette for distinct gestures (BGR format)
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "open_palm": (0, 215, 255),    # Vibrant Amber/Gold
    "fist": (50, 50, 255),         # Vivid Crimson
    "peace": (255, 100, 0),        # Electric Blue
    "thumbs_up": (50, 220, 50),    # Emerald Green
}
DEFAULT_BOX_COLOR = (0, 255, 180)


def get_default_winner_model() -> str:
    """Reads the evaluation summary to determine which feature representation won."""
    if os.path.isfile(SUMMARY_JSON_PATH):
        try:
            with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("winning_feature_type", "invariant")
        except Exception:
            pass
    return "invariant"


def load_model_pipeline(model_type: str):
    """Loads specified model pipeline from disk."""
    path = INVARIANT_MODEL_PATH if model_type == "invariant" else RAW_MODEL_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. Run `python train.py` first."
        )
    return joblib.load(path)


def compute_hand_bounding_box(
    landmarks, frame_w: int, frame_h: int, padding: int = 24
) -> Tuple[int, int, int, int]:
    """
    Computes (x_min, y_min, x_max, y_max) bounding box from 21 landmarks
    with safety padding and boundary clamping.
    """
    x_coords = [lm.x * frame_w for lm in landmarks.landmark]
    y_coords = [lm.y * frame_h for lm in landmarks.landmark]

    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(frame_w, int(max(x_coords)) + padding)
    y_max = min(frame_h, int(max(y_coords)) + padding)

    return x_min, y_min, x_max, y_max


def draw_styled_box_and_label(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: Optional[float],
    color: Tuple[int, int, int],
):
    """Renders a modern rounded corner bounding box and stylish label badge."""
    x1, y1, x2, y2 = box

    # Main bounding rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Accent corner highlights
    corner_len = min(22, (x2 - x1) // 3, (y2 - y1) // 3)
    thick = 4
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thick)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thick)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thick)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thick)

    # Label text
    clean_label = label.replace("_", " ").upper()
    text = f"{clean_label} ({confidence * 100:.1f}%)" if confidence is not None else clean_label

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Label background badge
    badge_y1 = max(0, y1 - th - 14)
    badge_y2 = y1
    badge_x2 = min(frame.shape[1], x1 + tw + 16)
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

    # Text inside badge
    cv2.putText(
        frame,
        text,
        (x1 + 8, badge_y2 - 6),
        font,
        font_scale,
        (10, 10, 10),  # Dark text for high contrast against vivid badges
        thickness,
        cv2.LINE_AA,
    )


def run_deployment(initial_model_type: str = "invariant", camera_idx: int = 0):
    """Main real-time inference loop on webcam feed."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot access webcam at index {camera_idx}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Preload both models into memory before entering the frame loop
    models_cache = {}
    for m_type in ["invariant", "raw"]:
        try:
            models_cache[m_type] = load_model_pipeline(m_type)
        except Exception as e:
            if m_type == initial_model_type:
                raise e

    current_model_type = initial_model_type
    current_pipe = models_cache.get(current_model_type, load_model_pipeline(current_model_type))

    fps_buffer: List[float] = []
    prev_time = time.perf_counter()

    print("\n" + "=" * 65)
    print(" LIVE HAND GESTURE RECOGNITION DEPLOYMENT")
    print("=" * 65)
    print(f"Active Feature Type:  {current_model_type.upper()}")
    print("Controls:")
    print("  - [M] : Toggle Model (Invariant <-> Raw Coordinates)")
    print("  - [Q] / [ESC] : Quit application")
    print("=" * 65)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # FPS calculation with moving average
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            fps_instant = 1.0 / dt if dt > 0 else 0.0
            fps_buffer.append(fps_instant)
            if len(fps_buffer) > 20:
                fps_buffer.pop(0)
            avg_fps = float(np.mean(fps_buffer))

            # MediaPipe detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            hand_detected = False
            pred_label = "None"
            confidence = None

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw subtle landmark skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Feature extraction matching active model
                if current_model_type == "invariant":
                    feat_vec = invariant_features(hand_landmarks).reshape(1, -1)
                else:
                    feat_vec = raw_features(hand_landmarks).reshape(1, -1)

                # Classifier inference
                pred_label = str(current_pipe.predict(feat_vec)[0])
                if hasattr(current_pipe, "predict_proba"):
                    probas = current_pipe.predict_proba(feat_vec)[0]
                    confidence = float(np.max(probas))
                elif hasattr(current_pipe, "decision_function"):
                    scores = np.asarray(current_pipe.decision_function(feat_vec)[0])
                    exp_scores = np.exp(scores - np.max(scores))
                    probas = exp_scores / np.sum(exp_scores)
                    confidence = float(np.max(probas))

                # Compute bounding box and render
                box = compute_hand_bounding_box(hand_landmarks, w, h)
                box_color = COLOR_PALETTE.get(pred_label, DEFAULT_BOX_COLOR)
                draw_styled_box_and_label(frame, box, pred_label, confidence, box_color)

            # Top Status HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            model_badge = f"MODEL: {current_model_type.upper()}"
            cv2.putText(
                frame,
                model_badge,
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "[M] Toggle Model  |  [Q] Exit",
                (w // 2 - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Gesture Recognition Live Deployment", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[INFO] Exiting deployment.")
                break
            elif key == ord("m") or key == ord("M"):
                # Toggle model representation from memory cache
                target_type = "raw" if current_model_type == "invariant" else "invariant"
                if target_type in models_cache:
                    current_model_type = target_type
                    current_pipe = models_cache[current_model_type]
                    print(f"\n[TOGGLE] Switched active model to: {current_model_type.upper()}")
                else:
                    try:
                        pipe = load_model_pipeline(target_type)
                        models_cache[target_type] = pipe
                        current_model_type = target_type
                        current_pipe = pipe
                        print(f"\n[TOGGLE] Switched active model to: {current_model_type.upper()}")
                    except Exception as e:
                        print(f"\n[ERROR] Failed to load {target_type} model: {e}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live Webcam Gesture Recognition Deployment")
    parser.add_argument(
        "--model",
        type=str,
        choices=["invariant", "raw"],
        default=None,
        help="Feature model to load (defaults to winner from evaluate.py)",
    )
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()

    active_model = args.model if args.model is not None else get_default_winner_model()
    run_deployment(initial_model_type=active_model, camera_idx=args.cam)


if __name__ == "__main__":
    main()
