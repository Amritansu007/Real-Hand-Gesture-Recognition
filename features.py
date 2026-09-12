"""
features.py - Feature Extraction Module for Hand Gesture Recognition
=====================================================================
Provides two distinct feature representations from MediaPipe Hand Landmarks:
1. raw_features: 63-dimensional vector of raw (x, y, z) coordinates.
2. invariant_features: 8-dimensional engineered vector invariant to scale,
   translation, and 3D rotation.

Mathematical Formulations & Invariance Analysis:
------------------------------------------------
Let p_i = (x_i, y_i, z_i) in R^3 be the 3D coordinate of landmark i (i in 0..20).
Key landmarks used:
  - Landmark 0:  WRIST (base of the hand)
  - Landmark 2:  THUMB_MCP
  - Landmark 4:  THUMB_TIP
  - Landmark 5:  INDEX_FINGER_MCP
  - Landmark 8:  INDEX_FINGER_TIP
  - Landmark 9:  MIDDLE_FINGER_MCP (rigid base of middle metacarpal)
  - Landmark 12: MIDDLE_FINGER_TIP
  - Landmark 16: RING_FINGER_TIP
  - Landmark 20: PINKY_TIP

Reference Scale Distance (Rigid Metacarpal Palm Segment):
  d_ref = ||p_9 - p_0||_2
  The distance from the wrist (0) to the middle finger MCP (9) represents
  the rigid palm bone structure. Unlike finger segments, this distance does
  not change during finger flexion or extension, providing an invariant scale
  normalizer for varying hand sizes and camera distances.

Invariant Features:
  1. Thumb extension ratio:
     f_1 = ||p_4 - p_0||_2 / d_ref
     Formula: Euclidean distance from thumb tip to wrist normalized by d_ref.

  2. Index extension ratio:
     f_2 = ||p_8 - p_0||_2 / d_ref
     Formula: Euclidean distance from index tip to wrist normalized by d_ref.

  3. Middle extension ratio:
     f_3 = ||p_12 - p_0||_2 / d_ref
     Formula: Euclidean distance from middle tip to wrist normalized by d_ref.

  4. Ring extension ratio:
     f_4 = ||p_16 - p_0||_2 / d_ref
     Formula: Euclidean distance from ring tip to wrist normalized by d_ref.

  5. Pinky extension ratio:
     f_5 = ||p_20 - p_0||_2 / d_ref
     Formula: Euclidean distance from pinky tip to wrist normalized by d_ref.

  6. Thumb-to-Index inter-finger angle (radians):
     v_thumb = p_4 - p_2,  v_index = p_8 - p_5
     f_6 = arccos( (v_thumb . v_index) / (||v_thumb||_2 * ||v_index||_2) )
     Formula: Inner-product angle between thumb direction and index direction.

  7. Index-to-Middle inter-finger angle (radians):
     v_index = p_8 - p_5,  v_middle = p_12 - p_9
     f_7 = arccos( (v_index . v_middle) / (||v_index||_2 * ||v_middle||_2) )
     Formula: Inner-product angle between index ray and middle ray (detects V-shape).

  8. Hand span aperture ratio:
     f_8 = ||p_8 - p_20||_2 / d_ref
     Formula: Distance between index tip and pinky tip normalized by d_ref.

Why Invariant?
  - Translation Invariance: All features depend strictly on difference vectors
    (p_i - p_j). Under translation p_i' = p_i + t, (p_i' - p_j') = (p_i - p_j).
  - Scale Invariance: Under scaling p' = s * p (s > 0), ||p_i' - p_j'|| = s * ||p_i - p_j||.
    The ratio ||p_i' - p_j'|| / d_ref' = (s * ||p_i - p_j||) / (s * d_ref) = ||p_i - p_j|| / d_ref.
  - Rotation Invariance: Under 3D orthogonal rotation R in SO(3), ||R * v||_2 = ||v||_2
    and (R * u) . (R * v) = u . v. Thus norms and angles are completely preserved.
"""

from typing import List, Sequence, Union
import numpy as np

# Landmark constants for MediaPipe Hands (21 landmarks)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Feature column names
RAW_FEATURE_NAMES: List[str] = [
    f"landmark_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")
]

INVARIANT_FEATURE_NAMES: List[str] = [
    "dist_thumb_wrist_norm",      # f_1: Thumb tip to wrist
    "dist_index_wrist_norm",      # f_2: Index tip to wrist
    "dist_middle_wrist_norm",     # f_3: Middle tip to wrist
    "dist_ring_wrist_norm",       # f_4: Ring tip to wrist
    "dist_pinky_wrist_norm",      # f_5: Pinky tip to wrist
    "angle_thumb_index_rad",      # f_6: Angle between thumb and index rays
    "angle_index_middle_rad",     # f_7: Angle between index and middle rays
    "aperture_index_pinky_norm",  # f_8: Aperture span from index to pinky tip
]


def _to_numpy_coords(landmarks: Union[np.ndarray, Sequence, object]) -> np.ndarray:
    """
    Converts various input formats into a standardized (21, 3) numpy float32 array.
    Supports:
      - MediaPipe NormalizedLandmarkList (object with .landmark attribute)
      - Iterable of objects with .x, .y, .z attributes
      - 1D numpy array of shape (63,)
      - 2D numpy array of shape (21, 3)
      - Nested python list of 21 (x, y, z) coordinates
    """
    if hasattr(landmarks, "landmark"):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    elif isinstance(landmarks, np.ndarray):
        if landmarks.shape == (21, 3):
            coords = landmarks.astype(np.float32)
        elif landmarks.shape == (63,):
            coords = landmarks.reshape(21, 3).astype(np.float32)
        else:
            raise ValueError(f"Expected numpy array of shape (21, 3) or (63,), got {landmarks.shape}")
    elif isinstance(landmarks, (list, tuple)):
        if len(landmarks) == 63 and not hasattr(landmarks[0], "x") and not isinstance(landmarks[0], (list, tuple)):
            coords = np.array(landmarks, dtype=np.float32).reshape(21, 3)
        elif len(landmarks) == 21:
            if hasattr(landmarks[0], "x"):
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            else:
                coords = np.array(landmarks, dtype=np.float32)
        else:
            raise ValueError(f"Expected sequence of 21 landmarks or 63 values, got length {len(landmarks)}")
    else:
        raise TypeError(f"Unsupported landmarks input type: {type(landmarks)}")

    if coords.shape != (21, 3):
        raise ValueError(f"Resulting coordinate array must be (21, 3), got {coords.shape}")

    return coords


def raw_features(landmarks: Union[np.ndarray, Sequence, object]) -> np.ndarray:
    """
    Extracts all 63 raw (x, y, z) values from the 21 MediaPipe hand landmarks.

    Parameters:
        landmarks: MediaPipe LandmarkList, or array-like of shape (21, 3) or (63,).

    Returns:
        np.ndarray of shape (63,) with dtype float32, representing [x0, y0, z0, ..., x20, y20, z20].
    """
    coords = _to_numpy_coords(landmarks)
    return coords.flatten().astype(np.float32)


def invariant_features(landmarks: Union[np.ndarray, Sequence, object], eps: float = 1e-6) -> np.ndarray:
    """
    Extracts 8 engineered features that are strictly invariant to scale,
    translation, and 3D rotation.

    Features:
      f_1: Thumb tip to wrist distance, normalized by wrist-to-middle-MCP distance.
      f_2: Index tip to wrist distance, normalized by wrist-to-middle-MCP distance.
      f_3: Middle tip to wrist distance, normalized by wrist-to-middle-MCP distance.
      f_4: Ring tip to wrist distance, normalized by wrist-to-middle-MCP distance.
      f_5: Pinky tip to wrist distance, normalized by wrist-to-middle-MCP distance.
      f_6: Angle in radians between thumb direction (p4 - p2) and index direction (p8 - p5).
      f_7: Angle in radians between index direction (p8 - p5) and middle direction (p12 - p9).
      f_8: Aperture span from index tip to pinky tip, normalized by wrist-to-middle-MCP distance.

    Parameters:
        landmarks: MediaPipe LandmarkList, or array-like of shape (21, 3) or (63,).
        eps: Small epsilon value to avoid division by zero or numerical instabilities.

    Returns:
        np.ndarray of shape (8,) with dtype float32.
    """
    coords = _to_numpy_coords(landmarks)

    # Reference scale distance: wrist (0) to middle finger MCP (9)
    # Metacarpal palm segment does not flex or curl, making it a stable scale benchmark.
    p_wrist = coords[WRIST]
    p_middle_mcp = coords[MIDDLE_MCP]
    d_ref = float(np.linalg.norm(p_middle_mcp - p_wrist))
    d_ref = max(d_ref, eps)

    # 1-5: Normalized Fingertip-to-Wrist Distances
    d_thumb = float(np.linalg.norm(coords[THUMB_TIP] - p_wrist)) / d_ref
    d_index = float(np.linalg.norm(coords[INDEX_TIP] - p_wrist)) / d_ref
    d_middle = float(np.linalg.norm(coords[MIDDLE_TIP] - p_wrist)) / d_ref
    d_ring = float(np.linalg.norm(coords[RING_TIP] - p_wrist)) / d_ref
    d_pinky = float(np.linalg.norm(coords[PINKY_TIP] - p_wrist)) / d_ref

    # 6: Angle between Thumb Ray (p4 - p2) and Index Ray (p8 - p5)
    v_thumb = coords[THUMB_TIP] - coords[THUMB_MCP]
    v_index = coords[INDEX_TIP] - coords[INDEX_MCP]
    norm_thumb = max(float(np.linalg.norm(v_thumb)), eps)
    norm_index = max(float(np.linalg.norm(v_index)), eps)
    cos_thumb_index = np.dot(v_thumb, v_index) / (norm_thumb * norm_index)
    cos_thumb_index = np.clip(cos_thumb_index, -1.0, 1.0)
    angle_thumb_index = float(np.arccos(cos_thumb_index))

    # 7: Angle between Index Ray (p8 - p5) and Middle Ray (p12 - p9)
    v_middle = coords[MIDDLE_TIP] - coords[MIDDLE_MCP]
    norm_middle = max(float(np.linalg.norm(v_middle)), eps)
    cos_index_middle = np.dot(v_index, v_middle) / (norm_index * norm_middle)
    cos_index_middle = np.clip(cos_index_middle, -1.0, 1.0)
    angle_index_middle = float(np.arccos(cos_index_middle))

    # 8: Hand Span Aperture: Index tip to Pinky tip normalized by d_ref
    d_aperture = float(np.linalg.norm(coords[INDEX_TIP] - coords[PINKY_TIP])) / d_ref

    features = np.array([
        d_thumb,
        d_index,
        d_middle,
        d_ring,
        d_pinky,
        angle_thumb_index,
        angle_index_middle,
        d_aperture,
    ], dtype=np.float32)

    return features
