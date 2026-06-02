from typing import List, Tuple
def highlight_matching_chunks(
    image_path: str,
    positions: List[Tuple[int, int]],
    pair: Tuple[int, int],
    chunk_size: int,
    output_path: str = "highlighted_match.png",
    show: bool = True,
    scale: int = 40,
) -> None:
    """
    Highlight two matched chunks in the original image.

    This version is better for tiny demo images, because it scales the image up
    before drawing boxes. For a 4x4 image, scale=40 makes it 160x160.
    """
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    i, j = pair

    if i < 0 or i >= len(positions):
        raise ValueError(f"Invalid chunk index i={i}.")
    if j < 0 or j >= len(positions):
        raise ValueError(f"Invalid chunk index j={j}.")

    # Load and scale the image so tiny images are visible.
    img = Image.open(image_path).convert("RGB")

    width, height = img.size
    scaled = img.resize((width * scale, height * scale), resample=Image.Resampling.NEAREST)

    draw = ImageDraw.Draw(scaled)

    boxes = [
        (i, positions[i], "red"),
        (j, positions[j], "cyan"),
    ]

    for chunk_index, (row, col), color in boxes:
        left = col * scale
        top = row * scale
        right = (col + chunk_size) * scale - 1
        bottom = (row + chunk_size) * scale - 1

        # Draw a box that is visible but not enormous.
        line_width = max(1, scale // 10)

        for offset in range(line_width):
            draw.rectangle(
                [left + offset, top + offset, right - offset, bottom - offset],
                outline=color,
            )

        draw.text(
            (left + line_width + 2, top + line_width + 2),
            str(chunk_index),
            fill=color,
        )

    scaled.save(output_path)
    print(f"Saved highlighted image to: {output_path}")

    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(scaled)
        plt.axis("off")
        plt.title(f"Matched chunks: {i} and {j}")
        plt.show()