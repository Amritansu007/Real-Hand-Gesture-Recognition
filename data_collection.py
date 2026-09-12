"""
data_collection.py - Webcam Hand Landmark Recording & Dataset Builder
=====================================================================
Captures live webcam frames, extracts 21 (x, y, z) hand landmarks using
MediaPipe Hands, and saves labeled samples tagged with session ID and gesture
class.

Usage:
  # Interactive mode (prompts for session and gesture):
  python data_collection.py

  # Direct CLI argument mode:
  python data_collection.py --session session_1_baseline --gesture open_palm --samples 300
  python data_collection.py --session session_1_baseline --gesture fist --samples 300
  python data_collection.py --session session_1_baseline --gesture peace --samples 300
  python data_collection.py --session session_1_baseline --gesture thumbs_up --samples 300

  # Cross-session recording (different distance, angle, or lighting):
  python data_collection.py --session session_2_cross --gesture open_palm --samples 200
"""

import argparse
import csv
import os
import sys
import time
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from features import RAW_FEATURE_NAMES

DEFAULT_GESTURES: List[str] = ["open_palm", "fist", "peace", "thumbs_up"]
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CONSOLIDATED_FILE = os.path.join(DATA_DIR, "gestures.csv")

CSV_HEADER: List[str] = ["session_id", "gesture_label", "timestamp"] + RAW_FEATURE_NAMES


def ensure_directories():
    """Ensures data and raw data directories exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


def prompt_session_and_gesture() -> tuple[str, str, int]:
    """Interactive CLI prompt if arguments are not provided."""
    print("=" * 60)
    print(" HAND GESTURE DATA COLLECTION")
    print("=" * 60)

    # Prompt Session ID
    print("\nEnter a Session ID (e.g. 'session_1_baseline' or 'session_2_cross'):")
    print("  - Session 1: Baseline lighting, neutral distance and angle.")
    print("  - Session 2: Changed lighting, distance, angle, or different hand/person.")
    session_id = input("Session ID [session_1_baseline]: ").strip()
    if not session_id:
        session_id = "session_1_baseline"

    # Prompt Gesture Class
    print("\nSelect a Gesture Class to record:")
    for idx, g in enumerate(DEFAULT_GESTURES, start=1):
        print(f"  [{idx}] {g}")
    print(f"  [{len(DEFAULT_GESTURES) + 1}] Custom gesture name")

    choice = input(f"Select option [1-{len(DEFAULT_GESTURES) + 1}]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(DEFAULT_GESTURES):
        gesture_label = DEFAULT_GESTURES[int(choice) - 1]
    elif choice.isdigit() and int(choice) == len(DEFAULT_GESTURES) + 1:
        gesture_label = input("Enter custom gesture name: ").strip().lower().replace(" ", "_")
        if not gesture_label:
            gesture_label = "custom_gesture"
    elif choice in DEFAULT_GESTURES:
        gesture_label = choice
    else:
        gesture_label = DEFAULT_GESTURES[0]

    # Prompt Sample Count
    samples_str = input("\nTarget number of frames to collect [300]: ").strip()
    target_samples = int(samples_str) if samples_str.isdigit() and int(samples_str) > 0 else 300

    return session_id, gesture_label, target_samples


def draw_hud(
    frame: np.ndarray,
    session_id: str,
    gesture_label: str,
    collected: int,
    target: int,
    state: str,
    countdown_val: Optional[float] = None,
):
    """Renders informative HUD overlay on the webcam feed."""
    h, w, _ = frame.shape

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Text headers
    cv2.putText(
        frame,
        f"SESSION: {session_id} | GESTURE: {gesture_label.upper()}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Progress bar
    progress = min(collected / max(target, 1), 1.0)
    bar_width = int((w - 30) * progress)
    cv2.rectangle(frame, (15, 42), (w - 15, 54), (60, 60, 60), -1)
    bar_color = (0, 220, 100) if state == "RECORDING" else (180, 180, 180)
    if bar_width > 0:
        cv2.rectangle(frame, (15, 42), (15 + bar_width, 54), bar_color, -1)

    cv2.putText(
        frame,
        f"Samples: {collected}/{target} ({int(progress * 100)}%)",
        (15, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # State badge & instructions
    if state == "WAITING":
        cv2.putText(
            frame,
            "PRESS [SPACE] TO START RECORDING  |  [Q] QUIT",
            (w // 2 - 240, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif state == "COUNTDOWN":
        countdown_text = f"STARTING IN: {countdown_val:.1f}s" if countdown_val is not None else "Press [SPACE] to begin"
        cv2.putText(
            frame,
            countdown_text,
            (w // 2 - 140, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 165, 255),
            3,
            cv2.LINE_AA,
        )
    elif state == "RECORDING":
        # Flashing red REC indicator
        rec_color = (0, 0, 255) if int(time.time() * 3) % 2 == 0 else (50, 50, 255)
        cv2.circle(frame, (w - 30, 25), 10, rec_color, -1)
        cv2.putText(
            frame,
            "REC",
            (w - 75, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            rec_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "RECORDING - HOLD GESTURE STEADY (SLIGHT MOVEMENTS ARE GOOD)",
            (w // 2 - 270, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 100),
            2,
            cv2.LINE_AA,
        )
    elif state == "FINISHED":
        cv2.putText(
            frame,
            "COLLECTION COMPLETE! SAVING...",
            (w // 2 - 180, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def record_gesture(session_id: str, gesture_label: str, target_samples: int, camera_idx: int = 0):
    """Main capture loop running MediaPipe Hands on webcam feed."""
    ensure_directories()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open webcam device at index {camera_idx}.")
        print("Please check that a webcam is connected and accessible.")
        return

    # Set camera resolution (640x480 standard for fast processing)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    recorded_rows: List[List[object]] = []
    state = "WAITING"  # WAITING -> COUNTDOWN -> RECORDING -> FINISHED
    countdown_start = 0.0
    countdown_duration = 3.0  # seconds

    print(f"\n[INFO] Starting webcam feed.")
    print(f"[INFO] Target: {target_samples} samples of '{gesture_label}' for session '{session_id}'.")
    print(f"[INFO] Align your hand in the window, then press [SPACE] to begin.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("\n[WARNING] Empty frame received from webcam. Skipping...")
                continue

            # Mirror view for natural intuitive interaction
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            hand_detected = False
            raw_landmark_coords: Optional[List[float]] = None

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw skeleton on feed
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract 63 raw coordinates
                raw_landmark_coords = []
                for lm in hand_landmarks.landmark:
                    raw_landmark_coords.extend([float(lm.x), float(lm.y), float(lm.z)])

            # State Machine
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("\n[INFO] User cancelled collection.")
                break

            if state == "WAITING":
                if key == ord(" ") or key == 13:  # SPACE or ENTER
                    state = "COUNTDOWN"
                    countdown_start = time.time()
                    draw_hud(
                        frame,
                        session_id,
                        gesture_label,
                        len(recorded_rows),
                        target_samples,
                        state,
                        countdown_val=countdown_duration,
                    )
                else:
                    draw_hud(frame, session_id, gesture_label, len(recorded_rows), target_samples, state)

            elif state == "COUNTDOWN":
                elapsed = time.time() - countdown_start
                remaining = max(0.0, countdown_duration - elapsed)
                if remaining <= 0.0:
                    state = "RECORDING"
                draw_hud(
                    frame,
                    session_id,
                    gesture_label,
                    len(recorded_rows),
                    target_samples,
                    state,
                    countdown_val=remaining,
                )

            elif state == "RECORDING":
                if hand_detected and raw_landmark_coords is not None:
                    timestamp_now = time.time()
                    row = [session_id, gesture_label, timestamp_now] + raw_landmark_coords
                    recorded_rows.append(row)

                draw_hud(frame, session_id, gesture_label, len(recorded_rows), target_samples, state)

                if len(recorded_rows) >= target_samples:
                    state = "FINISHED"

            elif state == "FINISHED":
                draw_hud(frame, session_id, gesture_label, len(recorded_rows), target_samples, state)
                cv2.imshow("Hand Gesture Data Collection", frame)
                cv2.waitKey(800)
                break

            cv2.imshow("Hand Gesture Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()

    if recorded_rows:
        # Save per-run raw CSV file
        timestamp_slug = time.strftime("%Y%m%d_%H%M%S")
        safe_session = session_id.replace(" ", "_").lower()
        safe_gesture = gesture_label.replace(" ", "_").lower()
        raw_filename = f"{safe_session}_{safe_gesture}_{timestamp_slug}.csv"
        raw_filepath = os.path.join(RAW_DATA_DIR, raw_filename)

        with open(raw_filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(recorded_rows)

        print(f"\n[SUCCESS] Saved {len(recorded_rows)} raw samples to: {raw_filepath}")

        # Append to consolidated dataset data/gestures.csv
        file_exists = os.path.isfile(CONSOLIDATED_FILE)
        with open(CONSOLIDATED_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(CONSOLIDATED_FILE) == 0:
                writer.writerow(CSV_HEADER)
            writer.writerows(recorded_rows)

        print(f"[SUCCESS] Appended {len(recorded_rows)} samples to consolidated dataset: {CONSOLIDATED_FILE}")
    else:
        print("\n[INFO] No samples recorded.")


def main():
    parser = argparse.ArgumentParser(
        description="Data Collection tool for Hand Gesture Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session", type=str, default=None, help="Session identifier (e.g. session_1_baseline)")
    parser.add_argument("--gesture", type=str, default=None, help=f"Gesture class label ({', '.join(DEFAULT_GESTURES)})")
    parser.add_argument("--samples", type=int, default=300, help="Target number of frames to record")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index")

    args = parser.parse_args()

    if args.session is None or args.gesture is None:
        session_id, gesture_label, target_samples = prompt_session_and_gesture()
    else:
        session_id = args.session.strip().replace(" ", "_").lower()
        gesture_label = args.gesture.strip().replace(" ", "_").lower()
        target_samples = args.samples

    record_gesture(
        session_id=session_id,
        gesture_label=gesture_label,
        target_samples=target_samples,
        camera_idx=args.cam,
    )


if __name__ == "__main__":
    main()
