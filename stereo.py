import cv2
import numpy as np


def _prepare_layers(image_bgr: np.ndarray, depth_map: np.ndarray, num_layers: int = 12, max_shift: int = 20) -> dict:
    """Prepare a background plate and depth-sorted layers for compositing."""
    h, w = image_bgr.shape[:2]
    depth_f = depth_map.astype(np.float32) / 255.0
    median_depth = np.median(depth_f)

    fg_binary = (depth_f > median_depth).astype(np.uint8)
    strip_width = max_shift + 10
    dilated = cv2.dilate(fg_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strip_width * 2 + 1, strip_width * 2 + 1)))
    border_strip = ((dilated - fg_binary) * 255).astype(np.uint8)
    bg_plate = cv2.inpaint(image_bgr, border_strip, inpaintRadius=8, flags=cv2.INPAINT_TELEA)

    thresholds = np.linspace(0, 1.0, num_layers + 1)
    margin = 0.5 / num_layers
    layers = []
    for i in range(num_layers):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = ((depth_f >= lo) & (depth_f < hi)).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        mask[depth_f < (lo - margin)] = 0
        mask[depth_f > (hi + margin)] = 0
        mid_depth = (lo + hi) / 2.0
        layers.append({"mask": mask, "depth": mid_depth})

    return {"bg_plate": bg_plate, "layers": layers, "median_depth": median_depth}


def generate_shifted_view(
    image_bgr: np.ndarray, depth_map: np.ndarray, shift_fraction: float, max_shift: int = 20,
    _cache: dict | None = None,
) -> np.ndarray:
    """Generate a novel view using multi-plane compositing over a clean background."""
    if _cache is not None and "data" in _cache:
        data = _cache["data"]
    else:
        data = _prepare_layers(image_bgr, depth_map, max_shift=max_shift)
        if _cache is not None:
            _cache["data"] = data

    h, w = image_bgr.shape[:2]
    median = data["median_depth"]

    bg_shift = int((0 - median) * max_shift * shift_fraction * 2)
    M_bg = np.float32([[1, 0, bg_shift], [0, 1, 0]])
    result = cv2.warpAffine(data["bg_plate"], M_bg, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

    for layer in data["layers"]:
        shift_px = int((layer["depth"] - median) * max_shift * shift_fraction * 2)
        M = np.float32([[1, 0, shift_px], [0, 1, 0]])
        shifted_img = cv2.warpAffine(image_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        shifted_mask = cv2.warpAffine(layer["mask"], M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        alpha = np.stack([shifted_mask] * 3, axis=-1)
        result = result * (1 - alpha) + shifted_img * alpha

    return result.astype(np.uint8)


def generate_stereo_pair(
    image_bgr: np.ndarray, depth_map: np.ndarray, max_shift: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left and right eye views."""
    cache = {}
    left = generate_shifted_view(image_bgr, depth_map, -1.0, max_shift, cache)
    right = generate_shifted_view(image_bgr, depth_map, 1.0, max_shift, cache)
    return left, right
