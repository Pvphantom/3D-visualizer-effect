import cv2
import imageio
import numpy as np

from stereo import generate_shifted_view


def create_wiggle_gif(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    output_path: str,
    max_shift: int = 30,
    num_frames: int = 20,
    frame_duration_ms: int = 80,
    num_loops: int = 0,
) -> None:
    """Create a smooth wiggle GIF using layer-based compositing."""
    h, w = image_bgr.shape[:2]
    crop = max_shift + 5
    cache = {}
    frames = []

    for i in range(num_frames):
        t = i / num_frames
        shift_fraction = np.sin(2 * np.pi * t)
        shifted = generate_shifted_view(image_bgr, depth_map, shift_fraction, max_shift, cache)
        cropped = shifted[:, crop:w - crop]
        frames.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    duration_s = frame_duration_ms / 1000.0
    imageio.mimsave(output_path, frames, duration=duration_s, loop=num_loops)
