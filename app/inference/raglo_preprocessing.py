"""
RAGLO Spatial & Image Preprocessing Module.
Prepares global representation (384x384) from full Bark ROI and
local representations (4 x 224x224 corner sub-patches, 2x2 layout, fraction 0.58).
"""
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
from PIL import Image
import torch

NEUTRAL_PAD_RGB = (127, 127, 127)
RAGLO_GLOBAL_SIZE = 384
RAGLO_LOCAL_SIZE = 224
RAGLO_PATCH_COUNT = 4
MIN_PATCH_OCCUPANCY = 0.35
PATCH_SEARCH_STEPS = 6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def aspect_fit_neutral_pad(
    image: Image.Image,
    target_size: int = RAGLO_GLOBAL_SIZE
) -> Image.Image:
    """
    Aspect-fit resize image to target_size x target_size canvas padded with neutral gray (127,127,127).
    Prevents image distortion and stretching.
    """
    image = image.convert("RGB")
    width, height = image.size

    scale = min(float(target_size) / max(width, 1), float(target_size) / max(height, 1))
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))

    resized_img = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (target_size, target_size), NEUTRAL_PAD_RGB)
    paste_x = (target_size - resized_w) // 2
    paste_y = (target_size - resized_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))

    return canvas


def _occupancy(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    region = mask[y1:y2, x1:x2]
    return float(region.mean()) if region.size else 0.0


def _best_occupancy_box(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    box_w: int,
    box_h: int,
    corner: str,
    steps: int = PATCH_SEARCH_STEPS
) -> Tuple[int, int, int, int]:
    """
    Search within a corner's home quadrant for the box_w x box_h window with the
    highest mask occupancy, so a patch is not silently placed mostly on
    NEUTRAL_PAD_RGB filler when the trunk mask is irregular (leaning/curved trunks,
    imperfect segmentation).
    """
    x_max = max(0, img_w - box_w)
    y_max = max(0, img_h - box_h)

    if corner == "tl":
        x_lo, x_hi = 0, min(x_max, max(0, img_w // 2))
        y_lo, y_hi = 0, min(y_max, max(0, img_h // 2))
    elif corner == "tr":
        x_lo, x_hi = max(0, img_w // 2 - box_w), x_max
        y_lo, y_hi = 0, min(y_max, max(0, img_h // 2))
    elif corner == "bl":
        x_lo, x_hi = 0, min(x_max, max(0, img_w // 2))
        y_lo, y_hi = max(0, img_h // 2 - box_h), y_max
    else:  # "br"
        x_lo, x_hi = max(0, img_w // 2 - box_w), x_max
        y_lo, y_hi = max(0, img_h // 2 - box_h), y_max

    default_x = x_lo if corner in ("tl", "bl") else x_hi
    default_y = y_lo if corner in ("tl", "tr") else y_hi

    best_box = (default_x, default_y, default_x + box_w, default_y + box_h)
    best_occ = _occupancy(mask, *best_box)

    if best_occ >= MIN_PATCH_OCCUPANCY:
        return best_box

    xs = np.linspace(x_lo, x_hi, max(1, steps)).astype(int) if x_hi > x_lo else [x_lo]
    ys = np.linspace(y_lo, y_hi, max(1, steps)).astype(int) if y_hi > y_lo else [y_lo]

    for x1 in xs:
        for y1 in ys:
            candidate = (int(x1), int(y1), int(x1) + box_w, int(y1) + box_h)
            occ = _occupancy(mask, *candidate)
            if occ > best_occ:
                best_occ = occ
                best_box = candidate
                if best_occ >= MIN_PATCH_OCCUPANCY:
                    return best_box

    return best_box


def prepare_raglo_images(
    original_bark_roi: Union[Image.Image, np.ndarray],
    patch_fraction: float = 0.58,
    roi_mask: Optional[np.ndarray] = None
) -> Tuple[Image.Image, List[Image.Image], List[Tuple[int, int, int, int]]]:
    """
    Prepare global 384 image and 4 local 224 sub-patch images for RAGLO classifier.
    Extracts 4 local sub-patches from the Bark ROI using a 2x2 corner grid layout
    (top-left, top-right, bottom-left, bottom-right) with area fraction (default 0.58),
    aspect-fit padded to 224x224.

    If `roi_mask` (boolean array, same H x W as the ROI, True = real bark pixel) is
    provided, each corner patch is repositioned within its home quadrant to avoid
    landing mostly on neutral-gray padding when the trunk silhouette is irregular.
    """
    if isinstance(original_bark_roi, np.ndarray):
        roi_pil = Image.fromarray(original_bark_roi)
    else:
        roi_pil = original_bark_roi

    roi_pil = roi_pil.convert("RGB")
    w, h = roi_pil.size

    # Step 1: Aspect-fit resize global representation to 384x384 using full Bark ROI
    global_image = aspect_fit_neutral_pad(roi_pil, RAGLO_GLOBAL_SIZE)

    # Step 2: Extract 4 local sub-patches (2x2 corner grid layout, fraction 0.58)
    pw = max(1, min(w, int(round(w * patch_fraction))))
    ph = max(1, min(h, int(round(h * patch_fraction))))

    default_boxes = [
        (0, 0, pw, ph),                      # Top-Left patch
        (w - pw, 0, w, ph),                  # Top-Right patch
        (0, h - ph, pw, h),                  # Bottom-Left patch
        (w - pw, h - ph, w, h)               # Bottom-Right patch
    ]

    if roi_mask is not None and roi_mask.shape[:2] == (h, w):
        corners = ["tl", "tr", "bl", "br"]
        roi_boxes = [
            _best_occupancy_box(roi_mask, w, h, pw, ph, corner)
            for corner in corners
        ]
    else:
        roi_boxes = default_boxes

    local_images = []
    for box in roi_boxes:
        patch_crop = roi_pil.crop(box)
        local_img = aspect_fit_neutral_pad(patch_crop, RAGLO_LOCAL_SIZE)
        local_images.append(local_img)

    return global_image, local_images, roi_boxes


def image_to_normalized_tensor(
    image: Image.Image,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch Tensor with ImageNet normalization (3, H, W).
    """
    arr = np.array(image, dtype=np.float32) / 255.0  # (H, W, C)
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    arr = (arr - mean_arr) / std_arr
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor


def prepare_raglo_tensors(
    global_image: Image.Image,
    local_images: List[Image.Image],
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert global and local images into batched model inputs:
    global_tensor: (1, 3, 384, 384)
    local_tensor: (1, 4, 3, 224, 224)
    """
    global_t = image_to_normalized_tensor(global_image, mean, std).unsqueeze(0)

    local_t_list = [image_to_normalized_tensor(img, mean, std) for img in local_images]
    local_t = torch.stack(local_t_list, dim=0).unsqueeze(0)

    return global_t, local_t
