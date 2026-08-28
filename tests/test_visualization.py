"""PIL rendering helpers: bounding boxes, heatmaps, traces, composite views."""

from PIL import Image
from pathvqa.utils.visualization import (
    render_bounding_boxes,
    render_attention_heatmap,
    render_symbolic_trace,
    render_composite_view,
)


def test_render_bounding_boxes():
    img = Image.new("RGB", (224, 224), color="black")
    boxes = [[0.1, 0.1, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9]]
    labels = ["box1", "box2"]

    out = render_bounding_boxes(img, boxes, labels)

    assert isinstance(out, Image.Image)
    assert out.size == (224, 224)


def test_render_attention_heatmap():
    img = Image.new("RGB", (224, 224), color="black")
    heatmap = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    out = render_attention_heatmap(img, heatmap)

    assert isinstance(out, Image.Image)
    assert out.size == (224, 224)


def test_render_symbolic_trace():
    trace = {"prediction": "lung", "confidence": 0.95, "reasoning_trace": {"step_1": {"target": "lung"}}}

    out = render_symbolic_trace(trace, size=(300, 300))

    assert isinstance(out, Image.Image)
    assert out.size == (300, 300)


def test_render_composite_view():
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="green")
    img3 = Image.new("RGB", (200, 200), color="blue")

    out = render_composite_view(img1, img2, img3, padding=10)

    assert isinstance(out, Image.Image)
    # Width: 100 + 100 + 200 + 10*4 = 440
    # Height: 200 + 10*2 = 220
    assert out.size == (440, 220)
