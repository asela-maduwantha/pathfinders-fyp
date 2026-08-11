"""
Integrated Research Inference Pipeline Module.
Orchestrates end-to-end 11-stage processing:
YOLO Tree Segmentation -> Assessability Check -> Multi-Bark Candidate Extraction -> Quality Filter ->
RAGLO Preprocessing -> Multi-ROI Neural Classification -> Calibrated Open-Set Validation ->
Multi-ROI Species Consensus -> Commercial Flagging -> Integration Logging -> Result Rendering.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from PIL import Image

from app.config import (
    BASE_DIR,
    TREE_IMGSZ,
    TREE_CONFIDENCE,
    TREE_PROPOSAL_CONF,
    TREE_NMS_IOU,
    TREE_IOU,
    TREE_MAX_DET,
    TREE_MIN_CONFIDENCE_FOR_CLASSIFICATION,
    DEBUG_TREE_DETECTION,
    DEBUG_ASSESSABILITY,
    DEBUG_TREE_FILTERING
)
from app.inference.tree_detector import TreeDetector
from app.inference.assessability import assess_trunk_visibility, evaluate_assessability
from app.inference.bark_roi import generate_candidate_rois, select_top_bark_candidates
from app.inference.raglo_preprocessing import prepare_raglo_images, prepare_raglo_tensors
from app.inference.raglo_classifier import RAGLOClassifier
from app.inference.species_consensus import evaluate_multi_roi_consensus
from app.utils.image_utils import load_pil_image_safe, save_image_disk
from app.utils.visualization import (
    draw_annotated_results,
    draw_debug_raw_yolo,
    draw_debug_after_duplicates,
    draw_debug_final_trunks,
    draw_debug_all_candidates
)
from app.utils.result_serializer import serialize_pipeline_results
from app.utils.integration_logger import log_candidate_inference

logger = logging.getLogger(__name__)

# Load commercial species configuration JSON
COMMERCIAL_JSON_PATH = BASE_DIR / "app" / "data" / "commercial_species.json"
COMMERCIAL_DATA: Dict[str, Any] = {}
if COMMERCIAL_JSON_PATH.exists():
    try:
        with open(COMMERCIAL_JSON_PATH, "r", encoding="utf-8") as f:
            COMMERCIAL_DATA = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load commercial species JSON: {e}")

PIPELINE_STAGES = [
    {"index": 1, "name": "Image uploaded and validated"},
    {"index": 2, "name": "Detecting tree trunks"},
    {"index": 3, "name": "Checking trunk assessability"},
    {"index": 4, "name": "Extracting bark candidates"},
    {"index": 5, "name": "Evaluating bark quality"},
    {"index": 6, "name": "Preparing classification inputs"},
    {"index": 7, "name": "Executing species classification"},
    {"index": 8, "name": "Performing open-set validation"},
    {"index": 9, "name": "Performing multi-ROI consensus"},
    {"index": 10, "name": "Applying commercial flag"},
    {"index": 11, "name": "Rendering final results"},
]


class IntegratedPipeline:
    """End-to-End Multi-ROI Research Inference Pipeline."""

    def __init__(self, tree_detector: TreeDetector, raglo_classifier: RAGLOClassifier):
        self.tree_detector = tree_detector
        self.raglo_classifier = raglo_classifier

    def run_pipeline(
        self,
        job_id: str,
        image_path: Path,
        results_dir: Path,
        progress_callback: Optional[Callable[[int, str, str, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Run complete 11-stage research inference pipeline on uploaded image.
        """
        job_dir = results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        def update_stage(idx: int, name: str, status: str = "processing", message: str = ""):
            if progress_callback:
                progress_callback(idx, name, status, message)
            logger.info(f"[Job {job_id}] Stage {idx}/11 ({name}): {status} - {message}")

        # Stage 1: Load and validate image with EXIF orientation fix
        update_stage(1, "Image uploaded and validated", "processing", "Loading target image...")
        pil_img = load_pil_image_safe(image_path)
        img_np_rgb = np.array(pil_img)
        orig_h, orig_w = img_np_rgb.shape[:2]

        orig_filename = "original.jpg"
        orig_save_path = job_dir / orig_filename
        save_image_disk(pil_img, orig_save_path)
        orig_rel_path = f"/static/results/{job_id}/{orig_filename}"

        update_stage(1, "Image uploaded and validated", "completed", f"Image dimensions: {orig_w}x{orig_h} px")

        # Stage 2: Tree Trunk Instance Segmentation & Proposal Generation
        update_stage(2, "Detecting tree trunks", "processing", f"Running proposal segmentation at conf={TREE_PROPOSAL_CONF:.2f}...")
        raw_detections = self.tree_detector.predict(
            img_np_rgb,
            imgsz=TREE_IMGSZ,
            conf=TREE_PROPOSAL_CONF,
            iou=TREE_NMS_IOU,
            max_det=TREE_MAX_DET
        )

        # Only trunks detected with enough confidence get shown and proceed to
        # classification - low-confidence proposals are dropped here.
        raw_detections = [
            d for d in raw_detections if d["confidence"] > TREE_MIN_CONFIDENCE_FOR_CLASSIFICATION
        ]
        num_trees = len(raw_detections)

        # Debug outputs: save debug_raw_yolo.jpg and debug_after_duplicates.jpg
        is_debug = DEBUG_TREE_DETECTION or DEBUG_ASSESSABILITY or DEBUG_TREE_FILTERING
        if is_debug:
            raw_debug_np = draw_debug_raw_yolo(img_np_rgb, raw_detections)
            save_image_disk(raw_debug_np, job_dir / "debug_raw_yolo.jpg")

            dedup_debug_np = draw_debug_after_duplicates(img_np_rgb, raw_detections)
            save_image_disk(dedup_debug_np, job_dir / "debug_after_duplicates.jpg")

        update_stage(2, "Detecting tree trunks", "completed", f"{num_trees} raw proposal candidate(s) generated")

        # Section 30: Handle zero tree detections gracefully without crashing
        if num_trees == 0:
            annotated_filename = "annotated.jpg"
            annotated_save_path = job_dir / annotated_filename
            save_image_disk(pil_img, annotated_save_path)
            annotated_rel_path = f"/static/results/{job_id}/{annotated_filename}"

            for idx in range(3, 12):
                stage_info = PIPELINE_STAGES[idx - 1]
                update_stage(idx, stage_info["name"], "completed", "Skipped (0 tree trunks detected)")

            summary_counts = {
                "trees_detected": 0,
                "assessable": 0,
                "bark_ready": 0,
                "identified": 0,
                "unknown": 0,
                "uncertain": 0,
                "not_assessable": 0,
                "commercial_flagged": 0,
                "raw_yolo_candidates": 0,
                "visibility_rejected": 0
            }

            return serialize_pipeline_results(
                job_id=job_id,
                original_rel_path=orig_rel_path,
                annotated_rel_path=annotated_rel_path,
                trees_processed=[],
                summary_counts=summary_counts
            )

        # Stage 3: Light Bark-Usability Safety Gate & Prominent Foreground Protection
        update_stage(3, "Checking trunk assessability", "processing", f"Evaluating {num_trees} proposal candidate(s)...")

        accepted_trees = []
        rejected_trees = []
        prominent_fg_count = 0

        for cand_idx, candidate in enumerate(raw_detections, start=1):
            mask = candidate["mask"]
            ass_res = assess_trunk_visibility(mask, orig_w, orig_h)
            candidate["assessability"] = ass_res
            candidate["assessable"] = ass_res["passed"]
            if ass_res.get("cleaned_mask") is not None:
                candidate["mask"] = ass_res["cleaned_mask"]

            decision_str = "ACCEPTED" if ass_res["passed"] else "REJECTED"

            if ass_res["passed"]:
                accepted_trees.append(candidate)
                if ass_res.get("is_prominent_foreground"):
                    prominent_fg_count += 1
            else:
                candidate["status"] = "visibility_rejected"
                rejected_trees.append(candidate)

            # Print diagnostic information for every candidate in DEBUG mode (Requirement 17)
            if is_debug:
                print(f"\nCandidate {cand_idx}")
                print(f"confidence = {candidate['confidence']:.4f}")
                print(f"bbox = {[round(v, 1) for v in candidate['bbox']]}")
                print(f"width_ratio = {ass_res.get('width_ratio', 0.0):.4f}")
                print(f"height_ratio = {ass_res.get('height_ratio', 0.0):.4f}")
                print(f"bbox_area_ratio = {ass_res.get('bbox_area_ratio', 0.0):.4f}")
                print(f"mask_area_ratio = {ass_res.get('mask_area_ratio', 0.0):.4f}")
                print(f"median_mask_width_ratio = {ass_res.get('median_mask_width_ratio', 0.0):.4f}")
                print(f"mask_occupancy = {ass_res.get('mask_occupancy', 0.0):.4f}")
                print(f"largest_component_ratio = {ass_res.get('largest_component_ratio', 0.0):.4f}")
                print(f"touches_frame_edge = {ass_res.get('touches_frame_edge', False)}")
                print(f"is_prominent_foreground = {ass_res.get('is_prominent_foreground', False)}")
                print(f"duplicate_status = {candidate.get('duplicate_status', 'KEPT')}")
                print(f"\ndecision:\n{decision_str}")
                print(f"Reason: {ass_res.get('reason_str', 'Passed')}")

        if is_debug:
            final_debug_np = draw_debug_final_trunks(img_np_rgb, accepted_trees, rejected_trees)
            save_image_disk(final_debug_np, job_dir / "debug_final_trunks.jpg")
            save_image_disk(final_debug_np, job_dir / "debug_filtered_candidates.jpg")

        # Sort ACCEPTED trees deterministically left-to-right by centroid X and assign sequential Tree IDs
        accepted_trees.sort(key=lambda d: (d["centroid_x"], d["centroid_y"]))
        for i, tree in enumerate(accepted_trees, start=1):
            tree["tree_id"] = i
            tree["display_name"] = f"Tree {i}"

        assessable_count = len(accepted_trees)
        rejected_count = len(rejected_trees)

        update_stage(
            3,
            "Checking trunk assessability",
            "completed",
            f"{assessable_count} clear trunk(s) accepted ({prominent_fg_count} prominent foreground), {rejected_count} candidate(s) excluded"
        )


        # Section 30: Handle zero accepted trees gracefully without crashing
        if assessable_count == 0:
            annotated_filename = "annotated.jpg"
            annotated_save_path = job_dir / annotated_filename
            save_image_disk(pil_img, annotated_save_path)
            annotated_rel_path = f"/static/results/{job_id}/{annotated_filename}"

            if DEBUG_TREE_FILTERING or DEBUG_ASSESSABILITY:
                debug_np = draw_debug_all_candidates(img_np_rgb, [], rejected_trees)
                save_image_disk(debug_np, job_dir / "debug_filtered_candidates.jpg")

            for idx in range(4, 12):
                stage_info = PIPELINE_STAGES[idx - 1]
                update_stage(idx, stage_info["name"], "completed", "Skipped (0 clear foreground trunks accepted)")

            summary_counts = {
                "trees_detected": 0,
                "assessable": 0,
                "bark_ready": 0,
                "identified": 0,
                "unknown": 0,
                "uncertain": 0,
                "not_assessable": 0,
                "commercial_flagged": 0,
                "raw_yolo_candidates": num_trees,
                "visibility_rejected": rejected_count
            }

            return serialize_pipeline_results(
                job_id=job_id,
                original_rel_path=orig_rel_path,
                annotated_rel_path=annotated_rel_path,
                trees_processed=[],
                summary_counts=summary_counts
            )

        trees_processed = accepted_trees

        # Stage 4 & 5: Bark Candidate Generation & Quality Gate Filter
        update_stage(4, "Extracting bark candidates", "processing", "Generating 3-5 candidate bark ROIs per trunk...")
        update_stage(5, "Evaluating bark quality", "processing", "Filtering by occupancy, sharpness, contrast, exposure...")

        bark_ready_count = 0

        for tree in trees_processed:
            tree_id = tree["tree_id"]
            tree_dir_name = f"tree_{tree_id:03d}"
            tree_dir = job_dir / tree_dir_name
            artifacts_dict = {}

            if not tree["assessable"]:
                tree["status"] = "not_assessable"
                tree["bark"] = {
                    "bark_ready": False,
                    "candidate_count": 0,
                    "reason": tree["assessability"]["reason"]
                }
                tree["raglo"] = {
                    "candidates": [],
                    "consensus": {},
                    "status": "not_evaluated",
                    "final_species": "Not evaluated"
                }
                tree["commercial"] = {
                    "evaluated": False,
                    "commercial_flag": None,
                    "status": "Not evaluated",
                    "category": "N/A",
                    "note": None
                }
                tree["artifacts"] = artifacts_dict
                continue

            mask = tree["mask"]
            bbox = tree["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            crop_rgb = img_np_rgb[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]

            tree_crop_path = tree_dir / "tree_crop.png"
            save_image_disk(crop_rgb, tree_crop_path)
            artifacts_dict["tree_crop"] = f"/static/results/{job_id}/{tree_dir_name}/tree_crop.png"

            masked_crop_np = np.empty_like(crop_rgb)
            masked_crop_np[:, :] = (127, 127, 127)
            masked_crop_np[crop_mask] = crop_rgb[crop_mask]

            masked_tree_path = tree_dir / "masked_tree.png"
            save_image_disk(masked_crop_np, masked_tree_path)
            artifacts_dict["masked_tree"] = f"/static/results/{job_id}/{tree_dir_name}/masked_tree.png"

            # Generate candidate ROIs
            raw_cands = generate_candidate_rois(img_np_rgb, mask, target_count=5)
            bark_ready, selected_cands, fail_reason = select_top_bark_candidates(raw_cands, max_select=3)

            tree["bark"] = {
                "bark_ready": bark_ready,
                "candidate_count": len(selected_cands),
                "reason": fail_reason
            }

            if not bark_ready or len(selected_cands) == 0:
                tree["status"] = "quality_failed"
                tree["raglo"] = {
                    "candidates": [],
                    "consensus": {},
                    "status": "quality_failed",
                    "final_species": None
                }
                tree["commercial"] = {
                    "evaluated": False,
                    "commercial_flag": None,
                    "status": "Not evaluated",
                    "category": "N/A",
                    "note": None
                }
                tree["artifacts"] = artifacts_dict
                continue

            tree["artifacts"] = artifacts_dict
            tree["selected_candidates"] = selected_cands
            bark_ready_count += 1

        update_stage(4, "Extracting bark candidates", "completed", "Candidate bark ROIs generated")
        update_stage(5, "Evaluating bark quality", "completed", f"{bark_ready_count} bark-ready trunk(s)")

        # Stage 6: RAGLO Exact Preprocessing Contract (Global 384 + 4 Local 224 corner patches)
        update_stage(6, "Preparing classification inputs", "processing", "Extracting 4 corner patches from original ROIs...")

        for tree in trees_processed:
            if not tree.get("bark", {}).get("bark_ready"):
                continue

            tree_id = tree["tree_id"]
            tree_dir_name = f"tree_{tree_id:03d}"
            tree_dir = job_dir / tree_dir_name

            for cand in tree["selected_candidates"]:
                c_id = cand["candidate_id"]
                cand_dir_name = f"candidate_{c_id:02d}"
                cand_dir = tree_dir / cand_dir_name

                roi_pil = Image.fromarray(cand["masked_crop"])
                global_img, local_imgs, patch_boxes = prepare_raglo_images(
                    roi_pil, roi_mask=cand.get("crop_mask")
                )
                global_t, local_t = prepare_raglo_tensors(global_img, local_imgs)

                cand["raglo_tensors"] = {
                    "global": global_t,
                    "local": local_t
                }

                # Save intermediate artifact images
                save_image_disk(roi_pil, cand_dir / "bark_roi.png")
                save_image_disk(global_img, cand_dir / "global_384.png")

                cand_artifacts = {
                    "bark_roi": f"/static/results/{job_id}/{tree_dir_name}/{cand_dir_name}/bark_roi.png",
                    "global_384": f"/static/results/{job_id}/{tree_dir_name}/{cand_dir_name}/global_384.png"
                }

                for p_idx, p_img in enumerate(local_imgs, start=1):
                    p_filename = f"patch_{p_idx}_224.png"
                    save_image_disk(p_img, cand_dir / p_filename)
                    cand_artifacts[f"patch_{p_idx}_224"] = f"/static/results/{job_id}/{tree_dir_name}/{cand_dir_name}/{p_filename}"

                cand["artifacts"] = cand_artifacts

                # Attach first candidate's artifacts to tree for top-level preview
                if c_id == 1:
                    tree["artifacts"]["selected_bark_roi"] = cand_artifacts["bark_roi"]
                    tree["artifacts"]["global_384"] = cand_artifacts["global_384"]
                    for k, v in cand_artifacts.items():
                        if k.startswith("patch_"):
                            tree["artifacts"][k] = v

        update_stage(6, "Preparing classification inputs", "completed", "Exact RAGLO input tensors created")

        # Stage 7 & 8: RAGLO Neural Classification & Calibrated Open-Set Validation
        update_stage(7, "Executing species classification", "processing", "Running dual-branch neural network...")
        update_stage(8, "Performing open-set validation", "processing", "Computing prototype distance & energy z-scores...")

        for tree in trees_processed:
            if not tree.get("bark", {}).get("bark_ready"):
                continue

            tree_id = tree["tree_id"]
            cand_results = []

            for cand in tree["selected_candidates"]:
                g_t = cand["raglo_tensors"]["global"]
                l_t = cand["raglo_tensors"]["local"]

                cand_res = self.raglo_classifier.classify_candidate(
                    global_tensor=g_t,
                    local_tensor=l_t,
                    candidate_id=cand["candidate_id"],
                    tree_id=tree_id,
                    quality_metrics=cand["metrics"]
                )
                cand_res["artifacts"] = cand["artifacts"]
                cand_results.append(cand_res)

            tree["cand_results"] = cand_results

        update_stage(7, "Executing species classification", "completed", "Multi-ROI neural inference complete")
        update_stage(8, "Performing open-set validation", "completed", "Calibrated open-set unknownness evaluated")

        # Stage 9: Multi-ROI Species Consensus
        update_stage(9, "Performing multi-ROI consensus", "processing", "Aggregating candidate votes...")

        identified_count = 0
        unknown_count = 0
        uncertain_count = 0

        for tree in trees_processed:
            if not tree.get("bark", {}).get("bark_ready"):
                continue

            cand_results = tree["cand_results"]
            consensus = evaluate_multi_roi_consensus(cand_results)

            status = consensus["status"]
            final_species = consensus["final_species"]

            tree["status"] = status
            tree["raglo"] = {
                "candidates": cand_results,
                "consensus": consensus,
                "status": status,
                "final_species": final_species
            }

            if status == "known":
                identified_count += 1
            elif status == "open_set_rejected":
                unknown_count += 1
            else:
                uncertain_count += 1

            # Log candidate inference to CSV
            for cand_res in cand_results:
                log_candidate_inference(
                    job_id=job_id,
                    tree_id=tree["tree_id"],
                    candidate_id=cand_res["candidate_id"],
                    quality_metrics=cand_res.get("quality", {}),
                    clf_dict=cand_res.get("classification", {}),
                    open_set_dict=cand_res.get("open_set", {}),
                    consensus_status=status,
                    final_species=final_species or "Unknown"
                )

        update_stage(9, "Performing multi-ROI consensus", "completed", f"{identified_count} identified, {unknown_count} unknown, {uncertain_count} uncertain")

        # Stage 10: Commercial Flagging (Section 19)
        update_stage(10, "Applying commercial flag", "processing", "Evaluating commercial species criteria...")

        commercial_flagged_count = 0

        for tree in trees_processed:
            status = tree.get("status")
            final_species = tree.get("raglo", {}).get("final_species")

            if (
                tree.get("assessable")
                and tree.get("bark", {}).get("bark_ready")
                and status == "known"
                and final_species
            ):
                species_key = final_species.lower()
                comm_data = COMMERCIAL_DATA.get(species_key, {
                    "commercial_flag": True,
                    "status": "Commercial species",
                    "category": "Commercial timber species",
                    "note": "Preliminary species-based commercial indication only."
                })
                is_comm = comm_data.get("commercial_flag", False)
                if is_comm:
                    commercial_flagged_count += 1

                tree["commercial"] = {
                    "evaluated": True,
                    "commercial_flag": is_comm,
                    "status": comm_data.get("status", "Commercial species"),
                    "category": comm_data.get("category", "Commercial timber species"),
                    "scientific_name": comm_data.get("scientific_name"),
                    "growing_regions": comm_data.get("growing_regions"),
                    "typical_uses": comm_data.get("typical_uses"),
                    "note": comm_data.get("note", "Preliminary species-based commercial indication only.")
                }
            else:
                tree["commercial"] = {
                    "evaluated": False,
                    "commercial_flag": None,
                    "status": "Not evaluated",
                    "category": "N/A",
                    "scientific_name": None,
                    "growing_regions": None,
                    "typical_uses": None,
                    "note": None
                }

        update_stage(10, "Applying commercial flag", "completed", f"{commercial_flagged_count} commercial species flagged")

        # Stage 11: Render Final Annotated Image & Response Payload
        update_stage(11, "Rendering final results", "processing", "Generating mask polygon overlays...")

        annotated_np = draw_annotated_results(img_np_rgb, trees_processed)
        annotated_filename = "annotated.jpg"
        annotated_save_path = job_dir / annotated_filename
        save_image_disk(annotated_np, annotated_save_path)
        annotated_rel_path = f"/static/results/{job_id}/{annotated_filename}"

        if DEBUG_TREE_FILTERING or DEBUG_ASSESSABILITY:
            debug_np = draw_debug_all_candidates(img_np_rgb, trees_processed, rejected_trees)
            save_image_disk(debug_np, job_dir / "debug_filtered_candidates.jpg")

        summary_counts = {
            "trees_detected": assessable_count,  # User-facing count of accepted clear foreground trunks
            "assessable": assessable_count,
            "bark_ready": bark_ready_count,
            "identified": identified_count,
            "unknown": unknown_count,
            "uncertain": uncertain_count,
            "not_assessable": 0,
            "commercial_flagged": commercial_flagged_count,
            "raw_yolo_candidates": num_trees,
            "visibility_rejected": rejected_count
        }

        final_response = serialize_pipeline_results(
            job_id=job_id,
            original_rel_path=orig_rel_path,
            annotated_rel_path=annotated_rel_path,
            trees_processed=trees_processed,
            summary_counts=summary_counts
        )

        update_stage(11, "Rendering final results", "completed", "Analysis successfully finished")
        return final_response
