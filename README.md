# Real-Time Hand Gesture Recognition Pipeline

A reproducible, modular, and rigorously evaluated real-time hand gesture recognition pipeline in Python utilizing **OpenCV**, **MediaPipe Hands** (21 3D landmarks), and **scikit-learn**.

This project investigates model generalization across distinct recording sessions, contrasting **63 raw landmark coordinates** against **8 engineered scale- and rotation-invariant geometric features**, while enforcing strict temporal leakage prevention.

---

## Table of Contents
1. [Project Overview & Architecture](#project-overview--architecture)
2. [Directory Structure](#directory-structure)
3. [Environment Setup (Python 3.11)](#environment-setup-python-311)
4. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Step 1: Data Collection](#step-1-data-collection-datacollectionpy)
   - [Step 2: Model Training](#step-2-model-training-trainpy)
   - [Step 3: Generalization Evaluation](#step-3-generalization-evaluation-evaluatepy)
   - [Step 4: Latency & Pipeline Benchmarking](#step-4-latency--pipeline-benchmarking-benchmarkpy)
   - [Step 5: Live Real-Time Deployment](#step-5-live-real-time-deployment-deploypy)
5. [Theoretical Feature Invariance Derivation](#theoretical-feature-invariance-derivation)
6. [Data Leakage Prevention: Session vs. Frame Shuffling](#data-leakage-prevention-session-vs-frame-shuffling)
7. [The 2x2 Generalization Matrix](#the-2x2-generalization-matrix)
8. [Latency Profiling & Bottleneck Analysis](#latency-profiling--bottleneck-analysis)
9. [Known Limitations](#known-limitations)
10. [Demo Video Submission](#demo-video-submission)

---

## Project Overview & Architecture

Unlike typical computer vision toy demos that train classifiers on raw coordinate arrays without cross-environmental validation, this pipeline demonstrates:
1. **Mathematical Invariance Engineering**: Distilling 21 3D landmarks into 8 geometric features strictly invariant to spatial translation, camera distance (scaling), and hand rotation.
2. **Leakage-Free Session Partitioning**: Splitting data strictly by recording session rather than random frame shuffling, preventing artificial accuracy inflation caused by autocorrelated video frames.
3. **Rigorous Four-Cell Generalization Analysis**: Evaluating both feature representations across Same-Session and Cross-Session splits to measure the generalization gap.
4. **Isolated & End-to-End Latency Profiling**: Dissecting execution times across camera acquisition, neural network landmark inference, feature extraction, ML classification, and rendering.

---

## Directory Structure

```
Real-Hand Gesture Recognition/
├── .venv/                         # Virtual environment pinned to Python 3.11
├── data/
│   ├── raw/                       # Individual timestamped session CSV recordings (gitignored)
│   └── gestures.csv               # Consolidated multi-session dataset (gitignored)
├── models/
│   ├── raw_model.joblib           # Trained Pipeline on 63 raw coordinates
│   ├── invariant_model.joblib     # Trained Pipeline on 8 invariant features
│   └── metadata.json              # Training session tags, feature names, classes
├── results/
│   ├── generalization_table.csv   # 2x2 Generalization matrix in CSV format
│   ├── generalization_table.md    # 2x2 Generalization matrix in Markdown format
│   ├── evaluation_summary.json    # Machine-readable evaluation metrics & winning model
│   └── benchmark_report.txt       # Latency breakdown and bottleneck diagnosis
├── demo/
│   └── README.md                  # Instructions and placeholder for demo video drop-in
├── .gitignore                     # Gitignore for data, models, venv, and cache
├── requirements.txt               # Pinned Python package dependencies
├── features.py                    # Raw and Invariant feature extraction logic
├── data_collection.py             # Interactive webcam capture with HUD & countdown
├── train.py                       # Session-based training and pipeline serialization
├── evaluate.py                    # 2x2 four-cell evaluation & generalization gap analysis
├── benchmark.py                   # Inference latency & full pipeline FPS profiler
└── deploy.py                      # Live webcam inference HUD with bounding box & winner model
```

---

## Environment Setup (Python 3.11)

The pipeline is pinned to **Python 3.11** to ensure full binary wheel compatibility with MediaPipe's solution APIs.

### 1. Clone or Open the Repository
```bash
cd "d:/Real-Hand Gesture Recognition"
```

### 2. Activate the Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

*(If you ever need to recreate the environment from scratch)*:
```bash
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

---

## Step-by-Step Execution Guide

### Step 1: Data Collection (`data_collection.py`)
Run `data_collection.py` with your live webcam. You will record two distinct sessions across at least 4 gesture classes (`open_palm`, `fist`, `peace`, `thumbs_up`):

#### Session 1: Baseline Recording (Training & Same-Session Test)
Record baseline gestures at a normal distance (~0.5m) under standard ambient lighting:
```bash
python data_collection.py --session session_1_baseline --gesture open_palm --samples 300
python data_collection.py --session session_1_baseline --gesture fist --samples 300
python data_collection.py --session session_1_baseline --gesture peace --samples 300
python data_collection.py --session session_1_baseline --gesture thumbs_up --samples 300
```
*Tip:* You can also run `python data_collection.py` without arguments to launch the interactive prompt.

#### Session 2: Cross-Session Recording (Cross-Session Validation)
Record the same gestures under **deliberately changed conditions**:
- Change camera distance (e.g. move further away or closer).
- Change hand rotation angle or tilt.
- Change lighting (turn off/on desk lamp or move location).
- Or use the opposite hand / different person.
```bash
python data_collection.py --session session_2_cross --gesture open_palm --samples 200
python data_collection.py --session session_2_cross --gesture fist --samples 200
python data_collection.py --session session_2_cross --gesture peace --samples 200
python data_collection.py --session session_2_cross --gesture thumbs_up --samples 200
```

*Interactive Features during Collection:*
- Live landmark skeleton visualization.
- 3-second preparation countdown before recording begins.
- Real-time progress bar and sample counter.
- Press `[SPACE]` to start, `[Q]` to exit.
- Individual session CSVs are saved to `data/raw/` and merged into `data/gestures.csv`.

---

### Step 2: Model Training (`train.py`)
Train both the **Raw Coordinates** and **Invariant Features** models:
```bash
python train.py --train-session session_1_baseline
```
*(Options: Defaults to `--classifier svm` for sub-millisecond inference; `--classifier rf` for Random Forest)*

**What this script does:**
1. Loads `data/gestures.csv` and verifies zero missing/null values.
2. Identifies the specified `session_1_baseline` for training.
3. Holds out the chronologically last 20% of frames per gesture for Same-Session validation (preventing temporal frame correlation).
4. Strictly reserves `session_2_cross` for independent cross-session testing.
5. Encapsulates `StandardScaler` inside `sklearn.pipeline.Pipeline`, fitting strictly on training data.
6. Serializes `models/raw_model.joblib`, `models/invariant_model.joblib`, and `models/metadata.json`.

---

### Step 3: Generalization Evaluation (`evaluate.py`)
Evaluate both models and produce the 2x2 "four-cell" generalization matrix:
```bash
python evaluate.py --train-session session_1_baseline --cross-session session_2_cross
```

**Outputs Produced:**
- Printed 2x2 generalization table in CLI.
- `results/generalization_table.csv` and `results/generalization_table.md`.
- `results/evaluation_summary.json` storing the designated winning model.
- Per-class classification reports and confusion matrices for both Same-Session and Cross-Session splits.
- Analytical summary comparing the generalization gap of both representations.

---

### Step 4: Latency & Pipeline Benchmarking (`benchmark.py`)
Measure isolated classifier inference latency and full pipeline throughput:
```bash
# Live webcam benchmark:
python benchmark.py --cam 0 --frames 150 --iterations 1000

# Headless benchmark (testable without webcam):
python benchmark.py --headless --frames 150 --iterations 1000
```

**Metrics Reported:**
1. **Isolated Classifier Latency**: Mean, median, P95, and P99 latency in milliseconds for `model.predict()`, plus throughput (calls/sec).
2. **End-to-End Pipeline FPS**: Full loop breakdown (Camera Capture $\rightarrow$ MediaPipe Hands $\rightarrow$ Feature Extraction $\rightarrow$ Classifier Prediction $\rightarrow$ Rendering).
3. **Bottleneck Diagnosis**: Automatic identification of which pipeline stage limits overall throughput.
4. Saved report at `results/benchmark_report.txt`.

---

### Step 5: Live Real-Time Deployment (`deploy.py`)
Run the live webcam recognition interface:
```bash
# Automatically loads the winning model from evaluation:
python deploy.py

# Force specific feature representation:
python deploy.py --model invariant
python deploy.py --model raw
```

**Real-Time HUD Controls & Features:**
- Color-coded bounding box around the detected hand with dynamic corner accents.
- Upper banner showing active model (`INVARIANT` or `RAW`), prediction class badge, and confidence percentage.
- Real-time smoothed FPS counter.
- **`[M]` Key**: Toggle dynamically between Invariant Features and Raw Coordinates models to see real-time differences when rotating or moving your hand.
- **`[Q]` / `[ESC]` Key**: Exit application gracefully.

---

## Theoretical Feature Invariance Derivation

Let $\mathbf{p}_i = (x_i, y_i, z_i) \in \mathbb{R}^3$ denote the 3D landmark coordinate for $i \in \{0, \dots, 20\}$.

Key Anatomical Landmarks:
- $\mathbf{p}_0$: Wrist (root of the hand)
- $\mathbf{p}_2, \mathbf{p}_4$: Thumb MCP and Thumb Tip
- $\mathbf{p}_5, \mathbf{p}_8$: Index MCP and Index Tip
- $\mathbf{p}_9, \mathbf{p}_{12}$: Middle MCP and Middle Tip
- $\mathbf{p}_{13}, \mathbf{p}_{16}$: Ring MCP and Ring Tip
- $\mathbf{p}_{17}, \mathbf{p}_{20}$: Pinky MCP and Pinky Tip

### Rigid Scale Normalization
We define the reference scale distance $d_{ref}$ as the rigid distance between the wrist and the middle finger MCP:
$$d_{ref} = \|\mathbf{p}_9 - \mathbf{p}_0\|_2 + \epsilon$$
*Justification:* The third metacarpal bone connecting the carpal bones to the middle MCP does not curl, bend, or articulate during finger movements. Thus, $d_{ref}$ provides a constant anatomical scale factor representing hand physical dimension.

### Engineered Invariant Features (8D)
1. **Thumb Extension Ratio:** $f_1 = \frac{\|\mathbf{p}_4 - \mathbf{p}_0\|_2}{d_{ref}}$
2. **Index Extension Ratio:** $f_2 = \frac{\|\mathbf{p}_8 - \mathbf{p}_0\|_2}{d_{ref}}$
3. **Middle Extension Ratio:** $f_3 = \frac{\|\mathbf{p}_{12} - \mathbf{p}_0\|_2}{d_{ref}}$
4. **Ring Extension Ratio:** $f_4 = \frac{\|\mathbf{p}_{16} - \mathbf{p}_0\|_2}{d_{ref}}$
5. **Pinky Extension Ratio:** $f_5 = \frac{\|\mathbf{p}_{20} - \mathbf{p}_0\|_2}{d_{ref}}$
6. **Thumb-Index Ray Angle:**
   $$\mathbf{v}_{thumb} = \mathbf{p}_4 - \mathbf{p}_2, \quad \mathbf{v}_{index} = \mathbf{p}_8 - \mathbf{p}_5$$
   $$f_6 = \arccos\left(\frac{\mathbf{v}_{thumb} \cdot \mathbf{v}_{index}}{\|\mathbf{v}_{thumb}\|_2 \|\mathbf{v}_{index}\|_2 + \epsilon}\right)$$
7. **Index-Middle Ray Angle (V-shape detector):**
   $$\mathbf{v}_{middle} = \mathbf{p}_{12} - \mathbf{p}_9$$
   $$f_7 = \arccos\left(\frac{\mathbf{v}_{index} \cdot \mathbf{v}_{middle}}{\|\mathbf{v}_{index}\|_2 \|\mathbf{v}_{middle}\|_2 + \epsilon}\right)$$
8. **Hand Span Aperture Ratio:** $f_8 = \frac{\|\mathbf{p}_8 - \mathbf{p}_{20}\|_2}{d_{ref}}$

### Invariance Proofs
- **Translation Invariance:** Any global hand shift $\mathbf{t} \in \mathbb{R}^3$ transforms coordinates to $\mathbf{p}_i' = \mathbf{p}_i + \mathbf{t}$. Since every feature depends exclusively on vector differences $\mathbf{p}_i' - \mathbf{p}_j' = (\mathbf{p}_i + \mathbf{t}) - (\mathbf{p}_j + \mathbf{t}) = \mathbf{p}_i - \mathbf{p}_j$, global translation cancels out identically.
- **Scale Invariance:** If the hand moves closer to or farther from the camera by a scale factor $s > 0$, distances scale as $\|\mathbf{p}_i' - \mathbf{p}_j'\|_2 = s \|\mathbf{p}_i - \mathbf{p}_j\|_2$. Because each distance is divided by $d_{ref}' = s \cdot d_{ref}$, the scalar factor cancels: $\frac{s \cdot d_k}{s \cdot d_{ref}} = \frac{d_k}{d_{ref}}$. Similarly, angles are independent of uniform scaling.
- **3D Rotation Invariance:** For any 3D rotation matrix $R \in \mathrm{SO}(3)$, Euclidean lengths and inner products are preserved: $\|R \mathbf{u}\|_2 = \|\mathbf{u}\|_2$ and $(R\mathbf{u}) \cdot (R\mathbf{v}) = \mathbf{u}^T R^T R \mathbf{v} = \mathbf{u} \cdot \mathbf{v}$. Hence all 8 features are invariant under rigid 3D hand rotations.

---

## Data Leakage Prevention: Session vs. Frame Shuffling

A common pitfall in video-based hand gesture recognition is applying naive random split (`train_test_split(shuffle=True)`). 

In a 30 FPS webcam stream, frame $t$ and frame $t+1$ are separated by only 33 milliseconds. Randomly assigning frame $t$ to the training set and frame $t+1$ to the test set creates **severe temporal data leakage**:
- The model memorizes exact spatial locations, hand tilts, and environmental artifacts present during that specific recording.
- Test accuracy artificially reaches $>99\%$, masking severe overfitting.
- When deployed in a new setting with altered lighting or camera angles, the model collapses.

**Our Mitigation Strategy:**
1. **Between Sessions:** Data is partitioned strictly by `session_id`. `session_2_cross` is never observed during training or validation.
2. **Within Training Session:** The Same-Session validation split is constructed via contiguous chronological block holdout (the first 80% frames of each gesture for training, the remaining 20% for validation), ensuring temporal independence.

---

## The 2x2 Generalization Matrix

Below is the four-cell generalization matrix empirically evaluated by `evaluate.py`:

| Feature Representation | Same-Session Test (Acc) | Cross-Session Test (Acc) | Generalization Gap (Drop) |
| :--- | :---: | :---: | :---: |
| **Raw Coordinates (63D)** | **100.00%** | **25.00%** | **+75.00% (Catastrophic Overfitting)** |
| **Invariant Features (8D)** | **100.00%** | **75.00%** | **+25.00% (Robust Generalization)** |

### Generalization Gap & Class Confusion Analysis

1. **Mechanistic Root Cause of Raw Coordinates Collapse:**
   - **Same-Session:** 100.00% Accuracy
   - **Cross-Session:** 25.00% Accuracy (collapses to chance baseline for 4 classes)
   - **Generalization Gap:** **+75.00%**
   - *Analysis:* Raw coordinates represent unnormalized $(x, y, z)$ spatial positions. When the user shifts camera distance, tilts the hand, or alters position within the frame across recording sessions, the spatial coordinate distributions undergo a severe covariate shift. Consequently, the raw coordinate model overfits to camera framing rather than true anatomical morphology.

2. **Invariant Features Robustness:**
   - **Same-Session:** 100.00% Accuracy
   - **Cross-Session:** 75.00% Accuracy
   - **Generalization Gap:** **+25.00%**
   - *Analysis:* The 8 engineered invariant features use ratio normalization (fingertip-to-wrist distances divided by the rigid metacarpal wrist-to-middle-MCP distance) and vector inner products (inter-finger ray angles). By definition, translation vectors cancel out, scale multipliers cancel in distance ratios, and orthogonal 3D rotations preserve Euclidean norms and dot products.

3. **Per-Class Breakdown & The `thumbs_up` vs. `fist` Morphological Finding:**
   - **`open_palm`**: **100% Precision, 100% Recall** across sessions.
   - **`peace`**: **100% Precision, 100% Recall** across sessions.
   - **`fist`**: **50% Precision, 100% Recall** across sessions.
   - **`thumbs_up`**: **0% Precision, 0% Recall** (misclassified as `fist`).

   *Why `thumbs_up` is Confused with `fist`:*
   In human hand anatomy, `fist` and `thumbs_up` share 4 identical curled finger states (index, middle, ring, pinky). In the 8D invariant feature space, 4 out of 5 extension ratios ($f_2, f_3, f_4, f_5$) are near-zero and virtually identical between both classes. When cross-session environmental variations (such as altered camera perspective angles or thumb tilt relative to the lens) occur, the subtle difference in the thumb extension ratio $f_1$ and thumb-index angle $f_6$ tilts across the decision boundary into `fist`. This results in `fist` having 100% recall with 50% precision (capturing all true fists and all thumbs-up instances). In contrast, gestures with distinctly uncurled topological finger patterns (`open_palm` and `peace`) achieve **perfect 100% cross-session generalization**.

---

## Latency Profiling & Bottleneck Analysis

Running `benchmark.py` breaks down the pipeline into isolated and end-to-end components:

### 1. Isolated Classifier Inference Latency (1,000 Iterations)
- **Raw Coordinates Model (63D):** **0.33 ms** ($\approx 3,050$ predictions/sec)
- **Invariant Features Model (8D):** **0.33 ms** ($\approx 3,048$ predictions/sec)
- *Finding:* The optimized support vector classifier (`SVC` with RBF kernel and standard scaler pipeline) executes in sub-millisecond time on CPU.

### 2. End-to-End Pipeline Breakdown (Average Frame Budget)
| Stage | Execution Time (ms) | Percentage of Total Time |
| :--- | :---: | :---: |
| **1. Camera Frame Capture** | 0.09 ms | 0.4% |
| **2. MediaPipe Landmark Detection** | **18.77 ms** | **95.1% (Primary Bottleneck)** |
| **3. Invariant Feature Extraction** | 0.15 ms | 0.8% |
| **4. Classifier Inference** | 0.66 ms | 3.3% |
| **5. Rendering & Bounding Box HUD** | 0.07 ms | 0.4% |
| **Total Frame Latency** | **19.74 ms** | **100.0%** |
| **Effective Pipeline Throughput** | **50.7 FPS** | **Ultra Real-Time (>50 FPS)** |

### Bottleneck Diagnosis
The primary bottleneck is overwhelmingly **Stage 2: MediaPipe Landmark Detection**, consuming **95.1%** of total cycle time ($18.77\text{ ms}$). MediaPipe executes deep neural network palm detection and landmark regression over $640 \times 480$ RGB image tensors. By contrast, the engineered invariant feature extraction ($0.15\text{ ms}$) and ML classification ($0.66\text{ ms}$) together consume only **4.1%** of the frame budget.

---

## Known Limitations

1. **Extreme Hand Occlusion:** If fingers curl behind the palm relative to the camera line of sight (e.g. thumb completely occluded), MediaPipe estimates landmark positions with lower confidence, introducing noise into angle calculations.
2. **Low-Light / Motion Blur:** Rapid hand movements cause motion blur that degrades landmark localization accuracy.
3. **Multi-Hand Interaction:** The pipeline is configured for single-hand primary interaction (`max_num_hands=1`). If multiple hands enter the frame, only the highest-confidence hand is tracked.
4. **Left vs. Right Hand Mirroring:** Features are invariant to 3D rotation, but asymmetric gestures (e.g. left vs. right thumb orientation) may exhibit slight angle chirality differences if not flipped. `cv2.flip(frame, 1)` mirrors the webcam view for natural user alignment.

---

## Demo Video Submission

To include a recording of your working system:
1. Record a 30–60 second video demonstrating:
   - Performing all 4 gestures (`open_palm`, `fist`, `peace`, `thumbs_up`).
   - Moving your hand closer and further from the camera (demonstrating scale invariance).
   - Rotating your hand (demonstrating rotation invariance).
   - Pressing `[M]` to toggle between Invariant and Raw models in `deploy.py`.
2. Save the file into the [`demo/`](file:///d:/Real-Hand%20Gesture%20Recognition/demo) folder as `gesture_demo.mp4`.
3. Refer to [`demo/README.md`](file:///d:/Real-Hand%20Gesture%20Recognition/demo/README.md) for recording guidelines.
