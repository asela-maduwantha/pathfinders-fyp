"""
Modal deployment entrypoint for the Tree Detection / Classification / Maturity /
Timber Grading / Auction Bid Price application.

This file only describes how to build the container image and serve the
existing FastAPI app (app.main:app) on Modal's infrastructure - no application
code depends on Modal, and nothing here changes local dev (`python app.py`).

Deploy with:
    modal deploy modal_app.py

See DEPLOYMENT.md for the full step-by-step guide (account setup, CLI auth,
volume/secret creation, first deploy, verification).
"""
from pathlib import Path

import modal

app = modal.App("pathfinders-fyp")

REPO_ROOT = Path(__file__).parent

# Persistent storage for uploaded photos, annotated/result images, maturity run
# outputs, and the auction prediction history + PDF reports. Mounted directly
# onto the exact relative paths the application code already writes to
# (app/static/uploads, app/static/results, outputs/) - no application code
# needs to know it's running on Modal.
uploads_volume = modal.Volume.from_name("pathfinders-uploads", create_if_missing=True)
results_volume = modal.Volume.from_name("pathfinders-results", create_if_missing=True)
outputs_volume = modal.Volume.from_name("pathfinders-outputs", create_if_missing=True)

# Excluded from the image: local dev artifacts, generated/runtime data (that's
# what the volumes above are for), and two legacy model checkpoints that
# aren't referenced by any config (tree_trunk_detection_best-old.pt,
# tree_classification_best_old.pt).
IGNORE_PATTERNS = [
    ".venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "app/static/uploads/**",
    "app/static/results/**",
    "outputs/**",
    "integration_validation/**",
    "notebook/**",
    "*.db",
    "models/tree_trunk_detection_best-old.pt",
    "models/tree_classification_best_old.pt",
    ".git/**",
]


def _prewarm_models():
    """
    Runs once at image build time so every deployed container starts with these
    weights already cached in the image, instead of downloading them from
    Hugging Face Hub / Ultralytics on every cold start (the app logged
    unauthenticated-HF-Hub-request warnings during local testing - repeating
    that download on every scale-to-zero cold start would be slow and
    rate-limit-prone).
    """
    from transformers import AutoModel

    AutoModel.from_pretrained("facebook/dinov2-small")

    from unidepth.models import UniDepthV2

    UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitb14")

    from ultralytics import YOLO

    YOLO("yolov8s-worldv2.pt")
    YOLO("FastSAM-s.pt")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",  # needed for the `git+https://...UniDepth.git` line in requirements.txt
        # UniDepth's dependency chain imports the full opencv-python (not just
        # our own opencv-python-headless), which needs these system libraries -
        # absent from the minimal debian_slim base image.
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .add_local_dir(str(REPO_ROOT), remote_path="/root", ignore=IGNORE_PATTERNS, copy=True)
    .run_function(_prewarm_models)
    # If the build fails during _prewarm_models with a Hugging Face rate-limit
    # error, create a secret (see DEPLOYMENT.md step 3) and change the line
    # above to:
    #   .run_function(_prewarm_models, secrets=[modal.Secret.from_name("huggingface-secret")])
)


@app.function(
    image=image,
    gpu="T4",
    volumes={
        "/root/app/static/uploads": uploads_volume,
        "/root/app/static/results": results_volume,
        "/root/outputs": outputs_volume,
    },
    timeout=600,
    min_containers=0,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as web_app

    return web_app
