import argparse
import os
import sys

from PIL import Image

from style_transfer import run_style_transfer, load_image, tensor_to_image


def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer (Gatys et al.)")
    parser.add_argument("content", help="Path to content image")
    parser.add_argument("style", help="Path to style image")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps (default: 300)")
    parser.add_argument("--content-weight", type=float, default=1.0, help="Content weight (default: 1.0)")
    parser.add_argument("--style-weight", type=float, default=1e6, help="Style weight (default: 1e6)")
    parser.add_argument("--max-size", type=int, default=512, help="Max image dimension in pixels (default: 512)")
    parser.add_argument("--save-progress", action="store_true", help="Save intermediate steps as images")
    args = parser.parse_args()

    for path in [args.content, args.style]:
        if not os.path.isfile(path):
            print(f"Error: cannot find '{path}'")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    progress_dir = os.path.join(args.output_dir, "progress")
    if args.save_progress:
        os.makedirs(progress_dir, exist_ok=True)

    def on_step(step, total, pastiche_tensor):
        if args.save_progress:
            img = tensor_to_image(pastiche_tensor)
            img.save(os.path.join(progress_dir, f"step_{step:04d}.png"))

    print(f"Content: {args.content}")
    print(f"Style:   {args.style}")
    print(f"Running {args.steps} optimization steps...\n")

    result = run_style_transfer(
        content_path=args.content,
        style_path=args.style,
        num_steps=args.steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        max_size=args.max_size,
        on_step=on_step,
    )

    output_path = os.path.join(args.output_dir, "stylized.png")
    result.save(output_path)

    content_copy = Image.open(args.content).convert("RGB")
    content_copy.save(os.path.join(args.output_dir, "content.png"))
    style_copy = Image.open(args.style).convert("RGB")
    style_copy.save(os.path.join(args.output_dir, "style.png"))

    print(f"\nOutputs saved to '{args.output_dir}/':")
    print(f"  - {output_path}")
    if args.save_progress:
        print(f"  - {progress_dir}/ (intermediate steps)")


if __name__ == "__main__":
    main()
