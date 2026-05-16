import cv2
import numpy as np
import torch


def estimate_depth(image_bgr: np.ndarray) -> np.ndarray:
    """Run MiDaS DPT_Hybrid depth estimation. Returns a depth map normalized to [0, 255]."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    model.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.dpt_transform

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
    return depth.astype(np.uint8)
