"""
Result Serializer Module.
Formats integrated multi-ROI pipeline inference outputs into standard structured JSON responses.
"""
from typing import Dict, Any, List
import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert NumPy data types to standard Python primitive types."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    return obj


def serialize_pipeline_results(
    job_id: str,
    original_rel_path: str,
    annotated_rel_path: str,
    trees_processed: List[Dict[str, Any]],
    summary_counts: Dict[str, int]
) -> Dict[str, Any]:
    """
    Construct standardized research JSON response payload for frontend rendering.
    """
    serialized_trees = []

    for tree in trees_processed:
        assess_info = tree.get("assessability", {})
        bark_info = tree.get("bark", {})
        raglo_info = tree.get("raglo", {})
        comm_info = tree.get("commercial", {})

        t_data = {
            "tree_id": int(tree["tree_id"]),
            "display_name": f"Tree {tree['tree_id']}",
            "status": tree.get("status", "unknown"),

            "detection": {
                "confidence": round(float(tree.get("confidence", 0.0)), 4),
                "bbox": [round(float(v), 2) for v in tree.get("bbox", [])],
                "polygon": tree.get("polygon"),
                "polygon_vertex_count": len(tree.get("polygon")) if tree.get("polygon") else 0
            },

            "assessability": {
                "passed": bool(tree.get("assessable", False)),
                "width_1024": float(assess_info.get("width_1024", 0.0)),
                "height_1024": float(assess_info.get("height_1024", 0.0)),
                "area_1024": float(assess_info.get("area_1024", 0.0)),
                "reason": assess_info.get("reason")
            },

            "bark_quality": {
                "passed": bool(bark_info.get("bark_ready", False)),
                "reason": bark_info.get("reason")
            },

            "bark": {
                "bark_ready": bool(bark_info.get("bark_ready", False)),
                "candidate_count": int(bark_info.get("candidate_count", 0)),
                "reason": bark_info.get("reason")
            },

            "raglo": {
                "candidates": raglo_info.get("candidates", []),
                "consensus": raglo_info.get("consensus", {}),
                "status": raglo_info.get("status", "not_evaluated"),
                "final_species": raglo_info.get("final_species")
            },

            "commercial": {
                "evaluated": bool(comm_info.get("evaluated", False)),
                "commercial_flag": comm_info.get("commercial_flag"),
                "status": comm_info.get("status", "Not evaluated"),
                "category": comm_info.get("category", "N/A"),
                "scientific_name": comm_info.get("scientific_name"),
                "growing_regions": comm_info.get("growing_regions"),
                "typical_uses": comm_info.get("typical_uses"),
                "note": comm_info.get("note")
            },

            "artifacts": tree.get("artifacts", {})
        }
        serialized_trees.append(t_data)

    response = {
        "job_id": job_id,
        "status": "completed",
        "image": {
            "original": original_rel_path,
            "annotated": annotated_rel_path
        },
        "summary": summary_counts,
        "trees": serialized_trees
    }

    return convert_numpy_types(response)
