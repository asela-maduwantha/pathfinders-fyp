"""
RAGLO Bark Classifier & Calibrated Open-Set Rejection Engine Module.
Loads the RAGLO-Bark A6 deployment artifact, reconstructs dual-branch global-local model,
and applies exact artifact-calibrated open-set rejection decision math.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None

from app.config import DEBUG_RAGLO

logger = logging.getLogger(__name__)

DINO_MODEL_ID = "facebook/dinov2-small"
DEFAULT_CLASSES = ["mahogany", "pine", "rubber", "teak"]


class RAGLOBarkModel(nn.Module):
    """
    RAGLO-Bark Dual-Branch Global-Local Model Architecture.
    Global: DINOv2-Small encoder (384x384)
    Local: ConvNeXt V2 Atto encoder (4 x 224x224 corner patches)
    Fusion: Softmax Patch Attention + Gated Feature Fusion
    """

    def __init__(
        self,
        num_classes: int = 4,
        embedding_dim: int = 256,
        pretrained: bool = False,
        global_encoder: Optional[nn.Module] = None,
        local_encoder: Optional[nn.Module] = None,
        use_attention: bool = True,
        fusion_mode: str = "gated",
        dropout: float = 0.2
    ):
        super().__init__()
        self.use_attention = use_attention
        self.fusion_mode = fusion_mode

        if global_encoder is not None:
            self.global_encoder = global_encoder
        elif AutoModel is not None:
            try:
                self.global_encoder = AutoModel.from_pretrained(DINO_MODEL_ID)
            except Exception as e:
                logger.warning(f"Could not load HuggingFace model '{DINO_MODEL_ID}' online ({e}). Constructing Dinov2Model from default config...")
                from transformers import Dinov2Config, Dinov2Model
                config = Dinov2Config(
                    image_size=518,
                    hidden_size=384,
                    num_hidden_layers=12,
                    num_attention_heads=6,
                    patch_size=14,
                )
                self.global_encoder = Dinov2Model(config)
        else:
            from transformers import Dinov2Config, Dinov2Model
            config = Dinov2Config(
                image_size=518,
                hidden_size=384,
                num_hidden_layers=12,
                num_attention_heads=6,
                patch_size=14,
            )
            self.global_encoder = Dinov2Model(config)

        if local_encoder is not None:
            self.local_encoder = local_encoder
        elif timm is not None:
            pattern = "convnextv2_atto*"
            candidates = timm.list_models(pattern, pretrained=pretrained)
            conv_name = sorted(candidates, key=lambda n: ("fcmae" not in n, n))[0] if candidates else "convnextv2_atto"
            self.local_encoder = timm.create_model(conv_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        else:
            self.local_encoder = None

        global_dim = 384 * 2  # DINOv2 small hidden size 384 (CLS + mean token pool)
        local_dim = getattr(self.local_encoder, "num_features", 320) if self.local_encoder else 320

        self.global_projection = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.local_projection = nn.Sequential(
            nn.Linear(local_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.patch_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        self.fusion_norm = nn.LayerNorm(256)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fused_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU()
        )
        self.embedding_head = nn.Linear(256, embedding_dim)
        self.fused_classifier = nn.Linear(256, num_classes)
        self.global_classifier = nn.Linear(256, num_classes)
        self.local_classifier = nn.Linear(256, num_classes)

    def forward(self, global_image: torch.Tensor, local_patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.global_encoder(pixel_values=global_image, return_dict=True).last_hidden_state
        global_feature = self.global_projection(torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1))

        batch, patches, channels, height, width = local_patches.shape
        local_raw = self.local_encoder(local_patches.reshape(batch * patches, channels, height, width))
        local_features = self.local_projection(local_raw).reshape(batch, patches, 256)

        if self.use_attention:
            attention = F.softmax(self.patch_attention(local_features).squeeze(-1), dim=1)
        else:
            attention = torch.full((batch, patches), 1.0 / patches, device=local_features.device)

        local_feature = (attention.unsqueeze(-1) * local_features).sum(dim=1)

        if self.fusion_mode == "average":
            gate = torch.full_like(global_feature, 0.5)
        else:
            gate = torch.sigmoid(self.gate_mlp(torch.cat([global_feature, local_feature], dim=1)))

        fused_feature = self.fused_projection(
            self.fusion_dropout(self.fusion_norm(gate * global_feature + (1.0 - gate) * local_feature))
        )
        embedding = F.normalize(self.embedding_head(fused_feature), dim=1)

        return {
            "fused_logits": self.fused_classifier(fused_feature),
            "global_logits": self.global_classifier(global_feature),
            "local_logits": self.local_classifier(local_feature),
            "embedding": embedding,
            "attention_weights": attention,
            "fusion_gate": gate
        }


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Calculate energy score; higher (less negative) indicates higher unknownness/uncertainty."""
    if temperature <= 0:
        temperature = 1.0
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def prototype_distances(embedding: torch.Tensor, prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate Cosine distance (1 - cosine_similarity) between embedding (N, D) and prototypes (C, D)."""
    embedding = F.normalize(embedding, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    distances = 1.0 - torch.mm(embedding, prototypes.T)  # (N, C)
    min_dist, nearest = torch.min(distances, dim=-1)
    return nearest, min_dist, distances


def calculate_calibrated_open_set(
    proto_distance: float,
    energy: float,
    open_set_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate artifact-calibrated open-set unknownness z-scores, combined unknownness, and decision margin.
    """
    alpha = float(open_set_config.get("alpha", 1.0))
    threshold = float(open_set_config.get("threshold", -0.5624))

    p_mean = float(open_set_config.get("prototype_mean", 0.1345))
    p_std = max(float(open_set_config.get("prototype_std", 0.1581)), 1e-8)

    e_mean = float(open_set_config.get("energy_mean", -2.7576))
    e_std = max(float(open_set_config.get("energy_std", 0.3758)), 1e-8)

    score_dir = open_set_config.get("score_direction", "higher_is_more_unknown")

    prototype_z = (proto_distance - p_mean) / p_std
    energy_z = (energy - e_mean) / e_std

    combined_unknownness = alpha * prototype_z + (1.0 - alpha) * energy_z
    decision_margin = combined_unknownness - threshold

    if score_dir == "lower_is_more_unknown":
        open_set_rejected = decision_margin <= 0.0
    else:
        # Standard higher_is_more_unknown rule
        open_set_rejected = decision_margin >= 0.0

    return {
        "prototype_distance": round(proto_distance, 4),
        "prototype_z": round(prototype_z, 4),
        "energy": round(energy, 4),
        "energy_z": round(energy_z, 4),
        "alpha": round(alpha, 4),
        "combined_unknownness": round(combined_unknownness, 4),
        "threshold": round(threshold, 4),
        "decision_margin": round(decision_margin, 4),
        "open_set_rejected": open_set_rejected,
        "score_direction": score_dir
    }


class RAGLOClassifier:
    """Wrapper around trained RAGLO-Bark model and open-set calibration engine."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.class_names = DEFAULT_CLASSES
        self.prototypes = None
        self.open_set_config = None
        self.preprocessing_config = None
        self.patch_settings = None
        self._load_deployment_artifact()

    def _load_deployment_artifact(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"RAGLO model checkpoint not found: {self.model_path}")

        logger.info(f"Loading RAGLO deployment model from {self.model_path} onto {self.device}...")
        checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict):
            if "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]
            elif "classes" in checkpoint:
                self.class_names = checkpoint["classes"]

            if "class_prototypes" in checkpoint:
                self.prototypes = checkpoint["class_prototypes"]
            elif "prototypes" in checkpoint:
                self.prototypes = checkpoint["prototypes"]

            if "open_set_configuration" in checkpoint:
                self.open_set_config = checkpoint["open_set_configuration"]
            elif "open_set_config" in checkpoint:
                self.open_set_config = checkpoint["open_set_config"]

            if "preprocessing" in checkpoint:
                self.preprocessing_config = checkpoint["preprocessing"]

            if "patch_settings" in checkpoint:
                self.patch_settings = checkpoint["patch_settings"]

            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
        else:
            state_dict = checkpoint

        num_classes = len(self.class_names)
        self.model = RAGLOBarkModel(num_classes=num_classes, pretrained=False)

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "").replace("_orig_mod.", "")
            cleaned_state_dict[key] = v

        try:
            self.model.load_state_dict(cleaned_state_dict, strict=True)
            logger.info("Successfully loaded state dict into RAGLOBarkModel with strict key matching.")
        except Exception as e:
            logger.warning(f"State dict load strict warning: {e}. Retrying strict=False...")
            self.model.load_state_dict(cleaned_state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        if self.open_set_config is None:
            raise RuntimeError(
                f"RAGLO checkpoint '{self.model_path}' has no 'open_set_configuration' "
                "(or 'open_set_config') entry. Open-set decisions cannot be made safely "
                "with placeholder thresholds. Recalibrate and re-export the checkpoint "
                "(see tools/calibrate_open_set.py) rather than deploying it as-is."
            )

    @torch.inference_mode()
    def classify_candidate(
        self,
        global_tensor: torch.Tensor,
        local_tensor: torch.Tensor,
        candidate_id: int = 1,
        tree_id: int = 1,
        quality_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run inference for a single bark ROI candidate and evaluate calibrated open-set decision.
        """
        if self.model is None:
            raise RuntimeError("RAGLO model is not loaded.")

        global_tensor = global_tensor.to(self.device)
        local_tensor = local_tensor.to(self.device)

        outputs = self.model(global_tensor, local_tensor)
        fused_logits = outputs["fused_logits"]
        probabilities = F.softmax(fused_logits, dim=1)

        raw_conf, raw_class_idx = probabilities.max(dim=1)
        raw_conf_val = float(raw_conf[0].cpu())
        raw_class_idx_val = int(raw_class_idx[0].cpu())
        raw_predicted_species = self.class_names[raw_class_idx_val]

        class_probs = {
            name: round(float(probabilities[0][i].cpu()), 4)
            for i, name in enumerate(self.class_names)
        }

        embedding = outputs["embedding"]
        attention_weights = [round(float(v), 4) for v in outputs["attention_weights"][0].cpu().numpy()]

        temp = float(self.open_set_config.get("energy_temperature", 1.0))
        energy_val = float(energy_score(fused_logits, temp)[0].cpu())

        proto_dist_val = 0.0
        if self.prototypes is not None:
            proto_tensor = self.prototypes.to(self.device) if isinstance(self.prototypes, torch.Tensor) else torch.tensor(self.prototypes, device=self.device)
            nearest_idx, min_dist, _ = prototype_distances(embedding, proto_tensor)
            proto_dist_val = float(min_dist[0].cpu())

        open_set_res = calculate_calibrated_open_set(proto_dist_val, energy_val, self.open_set_config)
        open_set_rejected = open_set_res["open_set_rejected"]

        if open_set_rejected:
            final_candidate_species = "Unknown / Unrecognized species"
            status = "open_set_rejected"
        else:
            final_candidate_species = raw_predicted_species
            status = "known"

        candidate_result = {
            "candidate_id": candidate_id,
            "status": status,
            "final_species": final_candidate_species,

            "quality": quality_metrics or {},

            "classification": {
                "raw_predicted_class": raw_predicted_species,
                "raw_confidence": round(raw_conf_val, 4),
                "class_probabilities": class_probs
            },

            "open_set": open_set_res,
            "attention": {
                "attention_weights": attention_weights
            }
        }

        if DEBUG_RAGLO:
            print(f"[DEBUG RAGLO] Tree {tree_id} Candidate {candidate_id} | "
                  f"Raw: {raw_predicted_species} ({raw_conf_val*100:.1f}%) | "
                  f"ProtoDist: {proto_dist_val:.4f} (Z: {open_set_res['prototype_z']:.4f}) | "
                  f"Energy: {energy_val:.4f} (Z: {open_set_res['energy_z']:.4f}) | "
                  f"Combined: {open_set_res['combined_unknownness']:.4f} | "
                  f"Thresh: {open_set_res['threshold']:.4f} | "
                  f"Margin: {open_set_res['decision_margin']:.4f} | "
                  f"Status: {status.upper()}")

        return candidate_result
