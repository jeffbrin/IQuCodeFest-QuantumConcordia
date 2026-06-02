from PIL import Image
import numpy as np

# 4x4 binary image with duplicate 2x2 chunks:
#
# chunk 0 == chunk 3
#
# [[1,0 | 0,1],
#  [1,1 | 1,0],
#  -----+-----
#  [0,0 | 1,0],
#  [1,0 | 1,1]]

arr = np.array([
    [255,   0,   0, 255, 0, 255, 0, 0],
    [255, 255, 255,   0, 0, 0, 255, 255],
    [  0,   0, 255,   0, 255, 255, 255, 0],
    [255,   0, 255, 255, 0, 0, 255, 255],
    [255,   255,   0, 255, 0, 255, 0, 0],
    [255, 255, 0,   0, 0, 0, 255, 255],
    [  0,   0, 255,   0, 255, 0, 0, 0],
    [255,   0, 0, 255, 255, 0, 0, 255],
], dtype=np.uint8)

Image.fromarray(arr, mode="L").save("all_pairs_similarity_analysis/test_8x8.png")

#!/usr/bin/env python3
"""
Resize an image to a specific resolution.

Example:
    python resize_image.py --input input.png --output output.png --width 512 --height 512
"""

import argparse
from pathlib import Path
from PIL import Image


def resize_image(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    keep_aspect_ratio: bool = False,
) -> None:
    """
    Resize an image to the requested resolution.

    If keep_aspect_ratio is False:
        The image is stretched/squeezed exactly to width x height.

    If keep_aspect_ratio is True:
        The image is resized to fit inside width x height while preserving aspect ratio,
        then padded with white background to exactly width x height.
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")

    img = Image.open(input_file).convert("RGB")

    if keep_aspect_ratio:
        # Resize while preserving aspect ratio.
        img.thumbnail((width, height), Image.Resampling.LANCZOS)

        # Create a white canvas of the target size.
        canvas = Image.new("RGB", (width, height), "white")

        # Center the resized image on the canvas.
        x = (width - img.width) // 2
        y = (height - img.height) // 2
        canvas.paste(img, (x, y))

        result = canvas
    else:
        # Resize exactly to the requested dimensions.
        result = img.resize((width, height), Image.Resampling.LANCZOS)

    result.save(output_path)
    print(f"Saved resized image to: {output_path}")
    print(f"Final resolution: {result.width}x{result.height}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize an image to a specific resolution.")

    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--output", required=True, help="Path to save the resized image.")
    parser.add_argument("--width", type=int, required=True, help="Target width in pixels.")
    parser.add_argument("--height", type=int, required=True, help="Target height in pixels.")

    parser.add_argument(
        "--keep-aspect-ratio",
        action="store_true",
        help="Preserve aspect ratio and pad to the requested resolution.",
    )

    args = parser.parse_args()

    resize_image(
        input_path=args.input,
        output_path=args.output,
        width=args.width,
        height=args.height,
        keep_aspect_ratio=args.keep_aspect_ratio,
    )


if __name__ == "__main__":
    main()