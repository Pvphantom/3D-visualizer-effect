import argparse
import os
import sys

import cv2
import numpy as np

from depth import estimate_depth
from gif import create_wiggle_gif
from stereo import generate_stereo_pair


def load_image(source: str) -> np.ndarray:
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: cannot open webcam")
            sys.exit(1)
        print("Press SPACE to capture, ESC to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read from webcam")
                sys.exit(1)
            cv2.imshow("Webcam - Press SPACE to capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                cap.release()
                cv2.destroyAllWindows()
                return frame
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
    else:
        image = cv2.imread(source)
        if image is None:
            print(f"Error: cannot read image '{source}'")
            sys.exit(1)
        return image


def main():
    parser = argparse.ArgumentParser(
        description="Monocular Depth-Based 3D Effect Generator"
    )
    parser.add_argument(
        "source",
        help="Path to input image, or 'webcam' to capture from camera",
    )
    parser.add_argument(
        "--output-dir", "-o", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "--max-shift", type=int, default=30, help="Max parallax shift in pixels (default: 30)"
    )
    parser.add_argument(
        "--gif-speed", type=int, default=80, help="GIF frame duration in ms (default: 80)"
    )
    parser.add_argument(
        "--num-frames", type=int, default=20, help="Number of frames in wiggle GIF (default: 20)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading image...")
    image = load_image(args.source)

    print("Estimating depth with MiDaS...")
    depth_map = estimate_depth(image)

    print("Generating stereo pair...")
    left_view, right_view = generate_stereo_pair(image, depth_map, args.max_shift)

    original_path = os.path.join(args.output_dir, "original.png")
    depth_path = os.path.join(args.output_dir, "depth_map.png")
    left_path = os.path.join(args.output_dir, "left_view.png")
    right_path = os.path.join(args.output_dir, "right_view.png")
    gif_path = os.path.join(args.output_dir, "wiggle_3d.gif")

    cv2.imwrite(original_path, image)
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    cv2.imwrite(depth_path, depth_colormap)
    cv2.imwrite(left_path, left_view)
    cv2.imwrite(right_path, right_view)

    print("Creating wiggle GIF...")
    create_wiggle_gif(image, depth_map, gif_path, args.max_shift, args.num_frames, args.gif_speed)

    print(f"\nOutputs saved to '{args.output_dir}/':")
    print(f"  - {original_path}")
    print(f"  - {depth_path}")
    print(f"  - {left_path}")
    print(f"  - {right_path}")
    print(f"  - {gif_path}")


if __name__ == "__main__":
    main()
