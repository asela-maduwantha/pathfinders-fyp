# Tree Species Identification & Maturity Estimation Local Application

A research web application for automated tree trunk detection, instance segmentation, bark-based
tree species identification (open-set verification and multi-ROI consensus), and DBH/maturity
estimation (monocular depth + cylinder geometry + a species-aware maturity model).

The app has two integrated modules, exposed as tabs in a single web UI:

- **Module 1 — Species identification**: Tree Trunk Detection (DAACT-YOLO) → Bark ROI extraction →
  RAGLO-Bark species classification with calibrated open-set rejection. Tabs: *Integrated Pipeline*,
  *Tree Detector Only*, *Bark Classifier Only*.
- **Module 2 — DBH & maturity estimation**: Trunk detection/segmentation → UniDepth monocular depth →
  cylinder-projection DBH → species-aware maturity model. Tabs: *Maturity: Complete Pipeline*
  (multi-tree queue) and *Maturity: Single Estimation* (one tree, one photo).
- **Full Pipeline: Species → Maturity**: a 4th tab that chains both modules automatically — Module 1
  identifies the species from an uploaded photo, and that species feeds straight into Module 2's
  DBH/maturity estimation on the same photo, with no manual species entry.

---

## 1. Application Purpose & Architecture

This application processes forest and plantation images through an 11-stage deep learning pipeline:

1. **Tree Trunk Detection**: DAACT-YOLO instance segmentation (`models/tree_trunk_detection_best.pt`).
2. **Assessability Gate**: Evaluates trunk geometry at 1024px letterboxed resolution ($width \ge 40\text{px}, height \ge 150\text{px}, area \ge 5000\text{px}^2$).
3. **Multi-Candidate Bark ROI Extraction**: Extracts 3–5 candidate bark regions vertically within the trunk mask, excluding extreme top/bottom edges.
4. **Bark Quality Gate**: Filters candidates by occupancy ($\ge 0.40$), sharpness ($\ge 30.0$), contrast ($\ge 15.0$), and brightness ($25.0 \le B \le 230.0$). Ranks accepted candidates and selects top 1–3 ROIs.
5. **Exact RAGLO Preprocessing**: Extracts 4 local $58\%$ corner patches from the *original selected bark ROI before resizing*, then resizes/pads global to $384\times384$ and local patches to $224\times224$ with ImageNet-neutral padding $(127, 127, 127)$ and ImageNet normalization.
6. **RAGLO Neural Classification**: Executes dual-branch DINOv2 + ConvNeXt V2 model (`models/tree_classification_best.pt`) on multi-scale representations.
7. **Calibrated Open-Set Validation**: Computes prototype Euclidean distance $Z$-score and energy score $Z$-score to calculate combined unknownness:
   $$\text{combined\_unknownness} = \alpha \cdot Z_{\text{proto}} + (1 - \alpha) \cdot Z_{\text{energy}}$$
   Compares against open-set threshold ($\text{threshold} = -0.5624$). Candidates exceeding threshold are rejected as **Unknown / Unrecognized species**.
8. **Multi-ROI Species Consensus**: Only candidates passing open-set cast votes. Evaluates consensus across ROIs (Known, Unknown, or Uncertain).
9. **Commercial Flagging**: Applies preliminary species-based commercial indication for verified KNOWN species (Mahogany, Pine, Rubber, Teak).
10. **Result Visualization**: Displays interactive Light Mode interface with annotated segmentation polygon outlines, real-time 11-stage progress timeline, and expandable multi-candidate process views.

---

## 2. Trained Model Artifacts

Module 1 depends on two deployment model files located in `models/`:

- `models/tree_trunk_detection_best.pt`: DAACT-YOLO instance-segmentation model.
- `models/tree_classification_best.pt`: RAGLO dual-branch classifier & open-set deployment artifact.

Module 2 (DBH + maturity estimation) depends on:

- `models/maturity/smooth_direct_residual_deployment_v3.joblib`: species-aware maturity model bundle.
- `data/Combined_Tree_Maturity_Dataset_v2.csv`: source dataset used to populate optional category
  dropdowns (climatic zone, spacing, site class, management intensity, intended product).

Module 2's vision models (Grounding DINO / YOLO-World for trunk detection, SAM 2.1 / FastSAM for
segmentation, UniDepth for monocular depth) are downloaded automatically from Hugging Face /
Ultralytics on first use — the first run of any Maturity tab requires internet access. They are
lazily loaded (not loaded at server startup), so using only Module 1's tabs does not require them.

*Note: Model files and weights are treated as immutable sources of truth and are never retrained or modified.*

### Module 2 configuration (optional environment variables)

Module 2's behavior can be tuned via `SMARTTIMBER_*` environment variables (all optional, sensible
defaults are baked in) — e.g. `SMARTTIMBER_DEVICE` (`auto`/`cpu`/`cuda`/`mps`), `SMARTTIMBER_DETECTOR_BACKEND`
(`yolo_world`/`dino`), `SMARTTIMBER_SEGMENTER_BACKEND` (`fastsam`/`sam2`), `SMARTTIMBER_INTRINSICS_MODE`,
`SMARTTIMBER_MAX_LONG_SIDE`. See `app/maturity/config.py` for the full list.

---

## 3. Environment Setup & Local Execution

### Step 1: Create Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```
This installs both modules' dependencies, including a `git+https://...UniDepth.git` install for
Module 2's monocular depth model — this step needs internet access and can take a while due to the
combined size of the vision/depth model stacks.

*(Optional GPU Acceleration)*: If an NVIDIA CUDA GPU is available:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Model Checkpoints
```powershell
.venv\Scripts\python.exe verify_models.py
```

### Step 4: Run Application Server
```powershell
.venv\Scripts\python.exe app.py
```

Open your web browser at:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 4. Research Diagnostic & Calibration Tools

### Model Metadata Verification
```powershell
.venv\Scripts\python.exe verify_models.py
```

### Same-Tree Bark Representation Diagnostic Tool
```powershell
.venv\Scripts\python.exe tools/compare_bark_inputs.py --clean path/to/clean.jpg --trunk path/to/trunk.jpg --auto path/to/auto.jpg
```

### Open-Set Threshold Calibration Tool
```powershell
.venv\Scripts\python.exe tools/calibrate_open_set.py --val-dir integration_validation
```

---

## 5. Integration Logging

Candidate inference metrics, z-scores, and decisions are appended to:
`logs/raglo_integration_results.csv`
