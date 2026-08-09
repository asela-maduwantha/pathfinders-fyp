from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image, ImageOps


try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_.") or "item"


def load_rgb(path: Path | str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def resize_for_processing(image: Image.Image, max_long_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(1.0, max_long_side / float(max(width, height)))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    if (new_width, new_height) == (width, height):
        return image.copy(), 1.0

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS), scale


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _read_exif(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        exif = image.getexif()
        items: dict[str, Any] = {}

        for key, value in exif.items():
            items[str(ExifTags.TAGS.get(key, key))] = value

        ifd_enum = getattr(ExifTags, "IFD", None)
        if ifd_enum is not None:
            for ifd_name in ("Exif", "GPSInfo"):
                ifd = getattr(ifd_enum, ifd_name, None)
                if ifd is None:
                    continue
                try:
                    nested = exif.get_ifd(ifd)
                except Exception:
                    nested = {}
                for key, value in nested.items():
                    tag_map = ExifTags.GPSTAGS if ifd_name == "GPSInfo" else ExifTags.TAGS
                    items[str(tag_map.get(key, key))] = value

        return items


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_camera_metadata(path: Path, width: int, height: int) -> dict[str, Any]:
    exif = _read_exif(path)
    metadata: dict[str, Any] = {
        "image_width": int(width),
        "image_height": int(height),
        "make": _clean_text(exif.get("Make")),
        "model": _clean_text(exif.get("Model")),
        "lens_model": _clean_text(exif.get("LensModel")),
        "focal_length_mm": _to_float(exif.get("FocalLength")),
        "focal_length_35mm": _to_float(exif.get("FocalLengthIn35mmFilm")),
        "digital_zoom_ratio": _to_float(exif.get("DigitalZoomRatio")),
        "focal_plane_x_resolution": _to_float(exif.get("FocalPlaneXResolution")),
        "focal_plane_y_resolution": _to_float(exif.get("FocalPlaneYResolution")),
        "focal_plane_resolution_unit": _to_float(exif.get("FocalPlaneResolutionUnit")),
        "orientation": _clean_text(exif.get("Orientation")),
    }
    metadata["has_focal_metadata"] = bool(
        metadata["focal_length_mm"] or metadata["focal_length_35mm"]
    )
    return metadata


def _resolution_unit_to_mm(unit: float | None) -> float | None:
    if unit is None:
        return None
    rounded = int(round(unit))
    if rounded == 2:
        return 25.4
    if rounded == 3:
        return 10.0
    if rounded == 4:
        return 1.0
    if rounded == 5:
        return 0.001
    return None


def estimate_intrinsics_from_metadata(
    metadata: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    focal_mm = metadata.get("focal_length_mm")
    focal_35mm = metadata.get("focal_length_35mm")
    unit_mm = _resolution_unit_to_mm(metadata.get("focal_plane_resolution_unit"))
    x_resolution = metadata.get("focal_plane_x_resolution")
    y_resolution = metadata.get("focal_plane_y_resolution")
    original_width = float(metadata.get("image_width") or width)
    original_height = float(metadata.get("image_height") or height)

    fx = fy = None
    source = None

    if focal_mm and unit_mm and x_resolution and y_resolution:
        px_per_mm_x = float(x_resolution) / unit_mm
        px_per_mm_y = float(y_resolution) / unit_mm
        fx = float(focal_mm) * px_per_mm_x * (float(width) / max(original_width, 1.0))
        fy = float(focal_mm) * px_per_mm_y * (float(height) / max(original_height, 1.0))
        source = "exif_focal_plane"
    elif focal_35mm:
        full_frame_diagonal_mm = math.sqrt(36.0**2 + 24.0**2)
        image_diagonal_px = math.sqrt(float(width) ** 2 + float(height) ** 2)
        focal_px = float(focal_35mm) / full_frame_diagonal_mm * image_diagonal_px
        fx = focal_px
        fy = focal_px
        source = "exif_35mm_equivalent"

    if fx is None or fy is None or fx <= 0 or fy <= 0:
        return {
            "available": False,
            "source": "none",
            "reason": "No usable EXIF focal-plane or 35mm-equivalent focal metadata",
            "K": None,
        }

    K = np.array(
        [
            [fx, 0.0, float(width) / 2.0],
            [0.0, fy, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return {
        "available": True,
        "source": source,
        "reason": "Estimated from image EXIF focal metadata",
        "K": K,
    }
