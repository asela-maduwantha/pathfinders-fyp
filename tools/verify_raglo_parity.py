"""
RAGLO Parity Verification & Diagnostic Tool.
Audits RAGLO inference parity against training calculations.
Computes both Cosine Distance (1 - cosine_similarity) and Euclidean Distance (torch.cdist)
to demonstrate the exact mathematical scale mismatch causing false open-set rejection.

Run with:
    .venv\\Scripts\\python.exe tools/verify_raglo_parity.py [--image path/to/image.png]
"""
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import CLASSIFIER_MODEL_PATH
from app.inference.raglo_classifier import RAGLOClassifier
from app.inference.raglo_preprocessing import prepare_raglo_images, prepare_raglo_tensors
from app.utils.image_utils import load_pil_image_safe


def run_parity_diagnostic(image_path: Path):
    print("=" * 70)
    print("       RAGLO INFERENCE PARITY & DISTANCE DIAGNOSTIC")
    print("=" * 70)
    print(f"Target Image: {image_path}")

    if not image_path.exists():
        print(f"[ERROR] Image path does not exist: {image_path}")
        return

    classifier = RAGLOClassifier(CLASSIFIER_MODEL_PATH, device="cpu")
    ckpt = torch.load(str(CLASSIFIER_MODEL_PATH), map_location="cpu", weights_only=False)

    print("\n--- 1. ARTIFACT METADATA ---")
    print("Class Names        :", classifier.class_names)
    print("Open-Set Config    :", classifier.open_set_config)
    print("Patch Settings     :", classifier.patch_settings)

    # Preprocessing & Forward Pass
    pil_img = load_pil_image_safe(image_path)
    g_img, l_imgs, _ = prepare_raglo_images(pil_img)
    g_t, l_t = prepare_raglo_tensors(g_img, l_imgs)

    classifier.model.eval()
    with torch.no_grad():
        outputs = classifier.model(g_t, l_t)

    fused_logits = outputs["fused_logits"]
    probs = F.softmax(fused_logits, dim=1)
    raw_conf, raw_class_idx = probs.max(dim=1)
    raw_pred_species = classifier.class_names[int(raw_class_idx[0])]

    embedding = outputs["embedding"]  # Shape (1, 256)
    embedding_norm = float(torch.norm(embedding, dim=-1)[0])

    prototypes = classifier.prototypes
    if not isinstance(prototypes, torch.Tensor):
        prototypes = torch.tensor(prototypes, dtype=torch.float32)

    emb_normed = F.normalize(embedding, dim=-1)
    proto_normed = F.normalize(prototypes, dim=-1)

    # 1. Cosine Distance (Training Method)
    cosine_dists = 1.0 - torch.mm(emb_normed, proto_normed.T)  # (1, C)
    cos_min_dist, cos_nearest_idx = torch.min(cosine_dists, dim=1)
    cos_dist_val = float(cos_min_dist[0])

    # 2. Euclidean Distance (Current Application Method)
    euclidean_dists = torch.cdist(emb_normed, proto_normed, p=2.0)  # (1, C)
    euc_min_dist, euc_nearest_idx = torch.min(euclidean_dists, dim=1)
    euc_dist_val = float(euc_min_dist[0])

    # Mathematical identity check for unit vectors: cosine = euclidean^2 / 2
    converted_euc_to_cos = (euc_dist_val ** 2) / 2.0

    print("\n--- 2. MODEL PREDICTIONS & EMBEDDINGS ---")
    print(f"Raw Logits             : {[round(v, 4) for v in fused_logits[0].tolist()]}")
    print(f"Softmax Probabilities  : {[round(v, 4) for v in probs[0].tolist()]}")
    print(f"Raw Predicted Class    : {raw_pred_species}")
    print(f"Raw Confidence         : {float(raw_conf[0])*100:.2f}%")
    print(f"Embedding Shape        : {tuple(embedding.shape)}")
    print(f"Embedding Norm         : {embedding_norm:.6f}")
    print(f"Prototypes Shape       : {tuple(prototypes.shape)}")

    print("\n--- 3. DISTANCE METHOD COMPARISON ---")
    print(f"Application Method (Euclidean torch.cdist) : {euc_dist_val:.6f}")
    print(f"Training Method    (Cosine 1 - cos_sim)    : {cos_dist_val:.6f}")
    print(f"Mathematical Check (Euclidean^2 / 2)       : {converted_euc_to_cos:.6f}")
    print(f"Distance Scale Ratio (Euclidean / Cosine)  : {euc_dist_val / max(cos_dist_val, 1e-8):.2f}x higher!")

    # Calculate Open-Set Metrics under BOTH methods
    open_cfg = classifier.open_set_config
    alpha = float(open_cfg.get("alpha", 0.95))
    thresh = float(open_cfg.get("threshold", -0.0323))
    p_mean = float(open_cfg.get("prototype_mean", 0.0463))
    p_std = float(open_cfg.get("prototype_std", 0.1060))
    e_mean = float(open_cfg.get("energy_mean", -2.9503))
    e_std = float(open_cfg.get("energy_std", 0.2945))
    temp = float(open_cfg.get("energy_temperature", 1.0))

    # Energy
    energy_val = float(-temp * torch.logsumexp(fused_logits / temp, dim=1)[0])
    energy_z = (energy_val - e_mean) / max(e_std, 1e-8)

    # Z-scores under Euclidean
    euc_p_z = (euc_dist_val - p_mean) / max(p_std, 1e-8)
    euc_comb = alpha * euc_p_z + (1.0 - alpha) * energy_z
    euc_margin = euc_comb - thresh
    euc_rejected = euc_margin >= 0.0

    # Z-scores under Cosine
    cos_p_z = (cos_dist_val - p_mean) / max(p_std, 1e-8)
    cos_comb = alpha * cos_p_z + (1.0 - alpha) * energy_z
    cos_margin = cos_comb - thresh
    cos_rejected = cos_margin >= 0.0

    print("\n--- 4. OPEN-SET CALIBRATION EVALUATION ---")
    print(f"Energy Score           : {energy_val:.4f} (Z: {energy_z:.4f})")
    print(f"Alpha                  : {alpha:.4f} (Prototype Contribution: {alpha*100:.1f}%, Energy: {(1-alpha)*100:.1f}%)")
    print(f"Threshold              : {thresh:.4f}")
    print("-" * 70)
    print(f"Metric                       Application (Euclidean)     Training Parity (Cosine)")
    print("-" * 70)
    print(f"Prototype Distance           {euc_dist_val:<27.4f} {cos_dist_val:<25.4f}")
    print(f"Prototype Z-Score            {euc_p_z:<27.4f} {cos_p_z:<25.4f}")
    print(f"Combined Unknownness         {euc_comb:<27.4f} {cos_comb:<25.4f}")
    print(f"Decision Margin              {euc_margin:<27.4f} {cos_margin:<25.4f}")
    print(f"Final Decision               {'REJECTED' if euc_rejected else 'KNOWN':<27} {'REJECTED' if cos_rejected else 'KNOWN':<25}")
    print("-" * 70)

    attn_weights = outputs["attention_weights"][0].tolist()
    print("\n--- 5. PATCH ATTENTION WEIGHTS ---")
    print("Attention Weights (P1..P4):", [round(w, 4) for w in attn_weights])

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RAGLO Inference Parity Diagnostic Tool")
    parser.add_argument("--image", type=str, help="Path to target bark image")
    args = parser.parse_args()

    if args.image:
        target_img = Path(args.image)
    else:
        # Search for a sample candidate image in results or test_output
        samples = list(Path("app/static/results").rglob("bark_roi.png"))
        if samples:
            target_img = samples[0]
        else:
            target_img = Path("test_output/real_forest.jpg")

    run_parity_diagnostic(target_img)


if __name__ == "__main__":
    main()
