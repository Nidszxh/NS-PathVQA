"""Visualization utilities for interpretability artifacts."""

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.cm as cm


def render_bounding_boxes(
    image: Image.Image, boxes: List[List[float]], labels: Optional[List[str]] = None, color: str = "red", width: int = 3
) -> Image.Image:
    """Draw normalized bounding boxes [x1, y1, x2, y2] over the base PIL Image.

    Args:
        image: Base PIL Image.
        boxes: List of bounding boxes, where each is [x1, y1, x2, y2] in normalized coordinates [0, 1].
        labels: Optional list of strings to draw alongside the boxes.
        color: Color of the bounding box.
        width: Line width of the bounding box.

    Returns:
        A new Image with bounding boxes overlaid.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        abs_box = [x1 * w, y1 * h, x2 * w, y2 * h]
        draw.rectangle(abs_box, outline=color, width=width)

        if labels and i < len(labels):
            # Default font: robust across systems without a bundled TTF.
            draw.text((abs_box[0] + 5, abs_box[1] + 5), labels[i], fill=color)

    return img_copy


def render_attention_heatmap(
    image: Image.Image, heatmap: List[List[float]], alpha: float = 0.5, colormap: str = "jet"
) -> Image.Image:
    """Blend a 2D grid heatmap over the original image.

    Args:
        image: Base PIL Image.
        heatmap: 2D array-like (H_grid, W_grid) of attention scores.
        alpha: Blending factor (0.0 = only image, 1.0 = only heatmap).
        colormap: Matplotlib colormap name.

    Returns:
        A new PIL Image with the heatmap blended.
    """
    w, h = image.size

    hm_arr = np.array(heatmap)

    # Min-max normalize to [0, 1]; a flat map becomes all zeros.
    hm_min, hm_max = hm_arr.min(), hm_arr.max()
    if hm_max > hm_min:
        hm_norm = (hm_arr - hm_min) / (hm_max - hm_min)
    else:
        hm_norm = np.zeros_like(hm_arr)

    # Apply colormap (matplotlib >= 3.9 moved/removed cm.get_cmap)
    legacy_get_cmap = getattr(cm, "get_cmap", None)
    if legacy_get_cmap is not None:
        cmap = legacy_get_cmap(colormap)
    else:
        import matplotlib

        cmap = matplotlib.colormaps[colormap]

    hm_colored = cmap(hm_norm)  # (H_grid, W_grid, 4)

    # Convert to PIL Image, drop alpha channel for resizing
    hm_img = Image.fromarray((hm_colored[:, :, :3] * 255).astype(np.uint8))

    # Resize to match original image using bicubic interpolation
    # Image.BICUBIC is deprecated in Pillow 10+, use Image.Resampling.BICUBIC
    resample_filter = getattr(Image, "Resampling", Image).BICUBIC
    hm_resized = hm_img.resize((w, h), resample=resample_filter)

    img_copy = image.copy().convert("RGB")
    blended = Image.blend(img_copy, hm_resized, alpha=alpha)

    return blended


def render_symbolic_trace(
    trace: Dict[str, Any], size: Tuple[int, int] = (400, 400), bg_color: str = "white", text_color: str = "black"
) -> Image.Image:
    """Render a textual representation of the hierarchical reasoning trace onto an Image.

    Args:
        trace: Dictionary representing the reasoning trace.
        size: (width, height) of the output image.
        bg_color: Background color.
        text_color: Text color.

    Returns:
        A PIL Image containing the trace text.
    """
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), json.dumps(trace, indent=2), fill=text_color)
    return img


def render_composite_view(
    image_with_boxes: Image.Image,
    heatmap_img: Image.Image,
    trace_img: Image.Image,
    padding: int = 10,
    bg_color: str = "white",
) -> Image.Image:
    """Concatenates the three images side-by-side into a single composite view.

    Args:
        image_with_boxes: Image with bounding boxes overlay.
        heatmap_img: Image with attention heatmap overlay.
        trace_img: Image containing the text trace.
        padding: Pixels of padding between images.
        bg_color: Background color of the composite canvas.

    Returns:
        A wide PIL Image combining all three inputs.
    """
    widths = [img.width for img in (image_with_boxes, heatmap_img, trace_img)]
    heights = [img.height for img in (image_with_boxes, heatmap_img, trace_img)]

    total_width = sum(widths) + padding * 4
    max_height = max(heights) + padding * 2

    composite = Image.new("RGB", (total_width, max_height), color=bg_color)

    x_offset = padding
    for img in (image_with_boxes, heatmap_img, trace_img):
        # Center vertically
        y_offset = (max_height - img.height) // 2
        composite.paste(img, (x_offset, y_offset))
        x_offset += img.width + padding

    return composite
