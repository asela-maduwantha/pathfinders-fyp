"""
Image Processing Utilities Module.
Handles image loading, EXIF orientation correction, format validation, and conversion helpers.
"""
import io
from pathlib import Path
from typing import Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB max upload size


def validate_image_file(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate image file extension, size, and integrity.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({len(file_bytes) / 1024 / 1024:.1f} MB) exceeds maximum allowed size (20 MB)."

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception as e:
        return False, f"Corrupted or invalid image content: {e}"

    return True, None


def load_pil_image_safe(image_source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """
    Load PIL Image with EXIF orientation handling and RGB conversion.
    Applies EXIF transpose and detaches EXIF metadata to prevent double rotation.
    """
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, (str, Path)):
        img = Image.open(image_source)
    else:
        raise ValueError("Unsupported image source type.")

    # Fix EXIF orientation (e.g. phone photos taken sideways/upside-down)
    img = ImageOps.exif_transpose(img)

    # Copy pixels onto a clean RGB canvas without lingering EXIF tags
    clean_img = Image.new("RGB", img.size)
    clean_img.paste(img.convert("RGB"))
    return clean_img


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR numpy array."""
    rgb_arr = np.array(pil_img)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL RGB image."""
    rgb_arr = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_arr)


def save_image_disk(image: Union[Image.Image, np.ndarray], destination: Path) -> str:
    """
    Save PIL Image or NumPy array to disk. Creates parent directories if missing.
    Returns relative path for API serving.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, Image.Image):
        image.save(destination, quality=95)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            # Assume RGB array
            Image.fromarray(image).save(destination, quality=95)
        else:
            cv2.imwrite(str(destination), image)
    else:
        raise ValueError("Unsupported image type for saving.")

    return str(destination)
