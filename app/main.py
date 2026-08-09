"""
FastAPI Main Application Entrypoint for Tree Trunk Segmentation & Bark Species Identification.
"""
import uuid
import logging
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    STATIC_DIR,
    TEMPLATES_DIR,
    UPLOADS_DIR,
    RESULTS_DIR,
    DETECTOR_MODEL_NAME,
    CLASSIFIER_MODEL_NAME,
    DETECTOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    TREE_IMGSZ,
    TREE_CONFIDENCE,
    TREE_IOU,
    TREE_MAX_DET
)
from app.inference.tree_detector import TreeDetector
from app.inference.raglo_classifier import RAGLOClassifier
from app.inference.integrated_pipeline import IntegratedPipeline
from app.utils.image_utils import validate_image_file, load_pil_image_safe, save_image_disk
from app.inference_lock import INFERENCE_LOCK
from app.maturity.config import OUTPUTS_DIR as MATURITY_OUTPUTS_DIR
from app.maturity.routes import router as maturity_router
from app.timber.config import MODELS_DIR as TIMBER_MODELS_DIR, TIMBERVISION_MODEL_PATH
from app.timber.services.cropper import TrunkCropper
from app.timber.services.fusion_model import TimberFusionService
from app.timber.routes import router as timber_router
from app.auction.config import MODEL_PATH as AUCTION_MODEL_PATH, PREPROCESSOR_PATH as AUCTION_PREPROCESSOR_PATH
from app.auction.services.valuation import AuctionValuationService
from app.auction.routes import router as auction_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

# Global model state
models_store: Dict[str, Any] = {}
jobs_store: Dict[str, Dict[str, Any]] = {}


async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Loads models ONCE at server startup onto GPU (CUDA) or CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n==================================================")
    print(f" Inference device: {device}")
    print(f"==================================================")

    print(f"\nTree detector:\nmodels/tree_trunk_detection_best.pt\n")
    print(f"Tree classifier:\nmodels/tree_classification_best.pt\n")

    logger.info(f"Tree detector: models/{DETECTOR_MODEL_NAME}")
    logger.info(f"Tree classifier: models/{CLASSIFIER_MODEL_NAME}")

    logger.info(f"Loading Tree Detector checkpoint from {DETECTOR_MODEL_PATH}...")
    tree_detector = TreeDetector(DETECTOR_MODEL_PATH, device=device)

    logger.info(f"Loading RAGLO Classifier checkpoint from {CLASSIFIER_MODEL_PATH}...")
    raglo_classifier = RAGLOClassifier(CLASSIFIER_MODEL_PATH, device=device)

    pipeline = IntegratedPipeline(tree_detector, raglo_classifier)

    models_store["tree_detector"] = tree_detector
    models_store["raglo_classifier"] = raglo_classifier
    models_store["pipeline"] = pipeline
    models_store["device"] = device
    models_store["det_path"] = str(DETECTOR_MODEL_PATH)
    models_store["clf_path"] = str(CLASSIFIER_MODEL_PATH)

    # Timber Grading (Module 3) - fully independent from the pipeline above:
    # its own models, own state, never fed by or into tree species/maturity data.
    logger.info(f"Loading Timber Grading models from {TIMBER_MODELS_DIR}...")
    models_store["timber_cropper"] = TrunkCropper(TIMBERVISION_MODEL_PATH, device=device)
    models_store["timber_fusion"] = TimberFusionService(TIMBER_MODELS_DIR)

    # Auction Bid Price (Module 4) - independent backend; only the frontend bridges
    # a graded crop's measurements into this module's form.
    logger.info(f"Loading Auction Bid Price model from {AUCTION_MODEL_PATH}...")
    models_store["auction_valuation"] = AuctionValuationService(AUCTION_MODEL_PATH, AUCTION_PREPROCESSOR_PATH)

    logger.info("All deep learning models loaded successfully.")
    yield

    # Cleanup at server shutdown
    models_store.clear()
    logger.info("Server shutdown completed.")



app = FastAPI(
    title="Tree Species Identification System",
    description="DAACT-YOLO26 Tree Trunk Segmentation and RAGLO-Bark Species Identification Web API",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files and Jinja2 templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(MATURITY_OUTPUTS_DIR), check_dir=False), name="outputs")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Module 2 (DBH + maturity estimation) routes: /api/maturity/config, /api/maturity/analyze,
# /api/maturity/analyze/auto
app.include_router(maturity_router)

# Timber Grading (Module 3) routes: /api/timber/status, /api/timber/detect, /api/timber/grade.
# Independent module - no data or models shared with the pipeline above.
app.include_router(timber_router)

# Auction Bid Price (Module 4) routes: /api/auction/config, /api/auction/predict,
# /api/auction/history, /api/auction/report/{id}. Independent backend.
app.include_router(auction_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main research web interface page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
async def health_check():
    """Verify backend server and model initialization status."""
    is_loaded = "pipeline" in models_store
    return {
        "status": "online" if is_loaded else "initializing",
        "device": models_store.get("device", "unknown"),
        "models_loaded": is_loaded,
        "detector_checkpoint": models_store.get("det_path"),
        "classifier_checkpoint": models_store.get("clf_path")
    }


def process_job_task(job_id: str, upload_path: Path):
    """Background task orchestrating full 11-stage integrated pipeline execution."""
    pipeline: IntegratedPipeline = models_store["pipeline"]

    def update_progress(stage_idx: int, stage_name: str, status: str, message: str):
        jobs_store[job_id]["stage_index"] = stage_idx
        jobs_store[job_id]["stage_name"] = stage_name
        jobs_store[job_id]["status"] = status
        jobs_store[job_id]["message"] = message

    try:
        with INFERENCE_LOCK:
            results = pipeline.run_pipeline(
                job_id=job_id,
                image_path=upload_path,
                results_dir=RESULTS_DIR,
                progress_callback=update_progress
            )

        jobs_store[job_id]["status"] = "completed"
        jobs_store[job_id]["stage_index"] = 11
        jobs_store[job_id]["message"] = "Analysis complete."
        jobs_store[job_id]["results"] = results
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs_store[job_id]["status"] = "error"
        jobs_store[job_id]["message"] = f"Inference pipeline error: {str(e)}"


@app.post("/api/analyze")
async def analyze_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload forest image and launch asynchronous background analysis job.
    """
    if "pipeline" not in models_store:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please retry in a few seconds.")

    file_bytes = await file.read()
    valid, err_msg = validate_image_file(file_bytes, file.filename)
    if not valid:
        raise HTTPException(status_code=400, detail=err_msg)

    job_id = str(uuid.uuid4())
    upload_ext = Path(file.filename).suffix.lower()
    upload_path = UPLOADS_DIR / f"{job_id}{upload_ext}"
    upload_path.write_bytes(file_bytes)

    # Initialize job progress state
    jobs_store[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "stage_index": 1,
        "stage_name": "Image uploaded and validated",
        "message": "Upload accepted. Starting pipeline...",
        "results": None
    }

    # Queue background task
    background_tasks.add_task(process_job_task, job_id, upload_path)

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll job processing progress stage."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    job = jobs_store[job_id]
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage_index": job["stage_index"],
        "stage_name": job["stage_name"],
        "message": job["message"]
    }


@app.get("/api/results/{job_id}")
async def get_job_results(job_id: str):
    """Retrieve final structured analysis JSON result."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    job = jobs_store[job_id]
    if job["status"] != "completed" or job["results"] is None:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not yet completed. Current status: {job['status']}"
        )

    return JSONResponse(content=job["results"])


@app.post("/api/analyze/detector")
async def analyze_tree_detector(file: UploadFile = File(...)):
    """
    Run DAACT-YOLO Tree Trunk Instance-Segmentation Model independently (Single-Model Diagnostic Tab).
    """
    if "tree_detector" not in models_store:
        raise HTTPException(status_code=503, detail="Tree Detector model is still initializing.")

    file_bytes = await file.read()
    valid, err_msg = validate_image_file(file_bytes, file.filename)
    if not valid:
        raise HTTPException(status_code=400, detail=err_msg)

    job_id = str(uuid.uuid4())
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pil_img = load_pil_image_safe(file_bytes)
    img_np_rgb = np.array(pil_img)
    orig_h, orig_w = img_np_rgb.shape[:2]

    orig_filename = "original.jpg"
    save_image_disk(pil_img, job_dir / orig_filename)
    orig_rel_path = f"/static/results/{job_id}/{orig_filename}"

    tree_detector: TreeDetector = models_store["tree_detector"]
    with INFERENCE_LOCK:
        raw_detections = tree_detector.predict(
            img_np_rgb,
            imgsz=TREE_IMGSZ,
            conf=TREE_CONFIDENCE,
            iou=TREE_IOU,
            max_det=TREE_MAX_DET
        )

    from app.inference.assessability import assess_trunk_visibility
    from app.utils.visualization import draw_annotated_results

    detections = []
    for cand in raw_detections:
        ass_res = assess_trunk_visibility(cand["mask"], orig_w, orig_h)
        if ass_res["passed"]:
            if ass_res.get("cleaned_mask") is not None:
                cand["mask"] = ass_res["cleaned_mask"]
            detections.append(cand)

    detections.sort(key=lambda d: (d["centroid_x"], d["centroid_y"]))
    for i, det in enumerate(detections, start=1):
        det["tree_id"] = i
        det["display_name"] = f"Tree {i}"

    annotated_np = draw_annotated_results(img_np_rgb, detections)
    annotated_filename = "annotated_detector.jpg"
    save_image_disk(annotated_np, job_dir / annotated_filename)
    annotated_rel_path = f"/static/results/{job_id}/{annotated_filename}"

    serialized_trees = []
    total_area = 0

    for tree in detections:
        t_id = tree["tree_id"]
        t_dir_name = f"tree_{t_id:03d}"
        t_dir = job_dir / t_dir_name

        bbox = [round(float(v), 2) for v in tree["bbox"]]
        x1, y1, x2, y2 = [int(v) for v in tree["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        crop_rgb = img_np_rgb[y1:y2, x1:x2]
        crop_path = t_dir / "crop.png"
        save_image_disk(crop_rgb, crop_path)

        mask = tree["mask"]
        area = int(mask.sum())
        total_area += area

        serialized_trees.append({
            "tree_id": t_id,
            "display_name": tree["display_name"],
            "confidence": round(float(tree["confidence"]), 4),
            "bbox": bbox,
            "polygon": tree.get("polygon"),
            "polygon_vertex_count": len(tree["polygon"]) if tree.get("polygon") else 0,
            "centroid": [round(float(tree["centroid_x"]), 1), round(float(tree["centroid_y"]), 1)],
            "area_px": area,
            "crop_url": f"/static/results/{job_id}/{t_dir_name}/crop.png"
        })

    avg_conf = float(np.mean([t["confidence"] for t in detections])) if detections else 0.0

    return {
        "job_id": job_id,
        "status": "completed",
        "summary": {
            "trees_detected": len(detections),
            "avg_confidence": round(avg_conf, 4),
            "total_mask_area_px": total_area,
            "image_dimensions": f"{orig_w}x{orig_h} px"
        },
        "image": {
            "original": orig_rel_path,
            "annotated": annotated_rel_path
        },
        "trees": serialized_trees
    }


@app.post("/api/analyze/classifier")
async def analyze_raglo_classifier(file: UploadFile = File(...)):
    """
    Run RAGLO-Bark Species Classifier & Open-Set Engine independently (Single-Model Diagnostic Tab).
    """
    if "raglo_classifier" not in models_store:
        raise HTTPException(status_code=503, detail="RAGLO Classifier model is still initializing.")

    file_bytes = await file.read()
    valid, err_msg = validate_image_file(file_bytes, file.filename)
    if not valid:
        raise HTTPException(status_code=400, detail=err_msg)

    job_id = str(uuid.uuid4())
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pil_img = load_pil_image_safe(file_bytes)
    save_image_disk(pil_img, job_dir / "original_bark.jpg")
    orig_rel_path = f"/static/results/{job_id}/original_bark.jpg"

    from app.inference.raglo_preprocessing import prepare_raglo_images, prepare_raglo_tensors
    global_img, local_imgs, patch_boxes = prepare_raglo_images(pil_img)
    global_tensor, local_tensor = prepare_raglo_tensors(global_img, local_imgs)

    save_image_disk(global_img, job_dir / "global_384.png")
    global_rel_path = f"/static/results/{job_id}/global_384.png"

    patch_rel_paths = []
    for idx, p_img in enumerate(local_imgs, start=1):
        p_filename = f"patch_{idx}_224.png"
        save_image_disk(p_img, job_dir / p_filename)
        patch_rel_paths.append(f"/static/results/{job_id}/{p_filename}")

    raglo_classifier: RAGLOClassifier = models_store["raglo_classifier"]
    with INFERENCE_LOCK:
        clf_res = raglo_classifier.classify_candidate(global_tensor, local_tensor, candidate_id=1, tree_id=1)

    return {
        "job_id": job_id,
        "status": "completed",
        "result": clf_res,
        "artifacts": {
            "original": orig_rel_path,
            "global_384": global_rel_path,
            "patches_224": patch_rel_paths
        }
    }
