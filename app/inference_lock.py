"""Process-wide lock serializing heavy ML inference across Module 1 and Module 2 pipelines.

Both modules load large vision models (YOLO, DINOv2/ConvNeXt, Grounding DINO/YOLO-World,
FastSAM/SAM2.1, UniDepth). Running two heavy pipelines concurrently on the same GPU/CPU risks
resource contention (e.g. CUDA OOM), so all inference entry points acquire this lock before
running a pipeline.
"""
import threading

INFERENCE_LOCK = threading.Lock()
