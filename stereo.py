import cv2
import numpy as np


def _make_foreground_mask(depth_map: np.ndarray) -> np.ndarray:
    """Create a soft foreground mask from the depth map using Otsu thresholding."""
    _, binary = cv2.threshold(depth_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    soft = cv2.GaussianBlur(binary, (21, 21), 0)
    return soft.astype(np.float32) / 255.0


def _inpaint_background(image_bgr: np.ndarray, mask_float: np.ndarray) -> np.ndarray:
    """Fill in the foreground region to create a clean background plate."""
    inpaint_mask = (mask_float > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel)
    return cv2.inpaint(image_bgr, inpaint_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)


def _shift_layer(layer: np.ndarray, dx: int) -> np.ndarray:
    """Translate a layer horizontally by dx pixels."""
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(layer, M, (layer.shape[1], layer.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def generate_shifted_view(
    image_bgr: np.ndarray, depth_map: np.ndarray, shift_fraction: float, max_shift: int = 30,
    _cache: dict | None = None,
) -> np.ndarray:
    """Generate a shifted view using layer-based compositing.

    Foreground and background are separated, shifted in opposite directions, then composited.
    """
    if _cache is not None and "fg_mask" in _cache:
        fg_mask = _cache["fg_mask"]
        bg_plate = _cache["bg_plate"]
    else:
        fg_mask = _make_foreground_mask(depth_map)
        bg_plate = _inpaint_background(image_bgr, fg_mask)
        if _cache is not None:
            _cache["fg_mask"] = fg_mask
            _cache["bg_plate"] = bg_plate

    fg_shift = int(max_shift * shift_fraction)
    bg_shift = int(-max_shift * 0.3 * shift_fraction)

    bg_shifted = _shift_layer(bg_plate, bg_shift)
    fg_shifted = _shift_layer(image_bgr, fg_shift)
    mask_shifted = _shift_layer((fg_mask * 255).astype(np.uint8), fg_shift).astype(np.float32) / 255.0

    mask_3ch = np.stack([mask_shifted] * 3, axis=-1)
    composite = (fg_shifted.astype(np.float32) * mask_3ch + bg_shifted.astype(np.float32) * (1 - mask_3ch))
    return composite.astype(np.uint8)


def generate_stereo_pair(
    image_bgr: np.ndarray, depth_map: np.ndarray, max_shift: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left and right eye views."""
    cache = {}
    left = generate_shifted_view(image_bgr, depth_map, -1.0, max_shift, cache)
    right = generate_shifted_view(image_bgr, depth_map, 1.0, max_shift, cache)
    return left, right
