from data.pathvqa_dataset import (
    normalize_text, resize_keep_aspect, lookup_answer_idx,
    cache_key, ImageCache, IMAGENET_MEAN, IMAGENET_STD,
)
from PIL import Image
import torch
import pytest


def test_normalize_text_strips_whitespace():
    assert normalize_text("  yes  ") == "yes"
    assert normalize_text("gastrointestinal system") == "gastrointestinal system"


def test_normalize_text_collapses_internal_spaces():
    assert normalize_text("bone ,  calvarium") == "bone , calvarium"


def test_normalize_text_lowercases():
    assert normalize_text("Adenocarcinoma") == "adenocarcinoma"


def test_normalize_text_idempotent():
    t = "  foci   of  fat necrosis  "
    assert normalize_text(normalize_text(t)) == normalize_text(t)


def _img(w, h):
    return Image.new("RGB", (w, h), color=(128, 64, 32))


@pytest.mark.parametrize("w,h", [
    (492, 369), (309, 272), (210, 267), (177, 275), (640, 320),
    (100, 50), (320, 240), (526, 528), (45, 67), (2000, 1000),
])
def test_resize_keep_aspect_exact_output_size(w, h):
    out = resize_keep_aspect(_img(w, h), (320, 240))
    assert out.size == (320, 240)


@pytest.mark.parametrize("w,h", [
    (492, 369), (309, 272), (210, 267), (177, 275), (640, 320),
    (100, 50), (526, 528), (45, 67), (2000, 1000), (320, 240),
])
def test_resize_keep_aspect_no_distortion(w, h):
    """A single scalar scale applied to both axes = uniform, no stretching."""
    out_w, out_h = 320, 240
    scale = min(w / out_w, h / out_h)
    new_w = max(1, round(w / scale))
    new_h = max(1, round(h / scale))
    assert new_w / new_h == pytest.approx(w / h, rel=2e-2)


def test_resize_keep_aspect_center_crop_content():
    """A landscape 2:1 image: output is a 4:3 center window, not a stretch."""
    img = Image.new("RGB", (640, 320), color="red")
    for x in range(240, 400):
        for y in range(320):
            img.putpixel((x, y), (0, 0, 255))
    out = resize_keep_aspect(img, (320, 240))
    assert out.getpixel((50, 120)) == (255, 0, 0)
    assert out.getpixel((200, 120)) == (0, 0, 255)
    assert out.getpixel((280, 120)) == (255, 0, 0)


def test_lookup_answer_idx_known_answer():
    vocab = {"yes": 0, "no": 1, "<UNK>": 2}
    assert lookup_answer_idx("yes", vocab) == 0


def test_lookup_answer_idx_unseen_maps_to_unk():
    vocab = {"yes": 0, "no": 1, "<UNK>": 2}
    assert lookup_answer_idx("gastrointestinal system", vocab) == 2


def test_lookup_answer_idx_raises_without_unk():
    vocab = {"yes": 0, "no": 1}
    with pytest.raises(KeyError):
        lookup_answer_idx("gastrointestinal system", vocab)


def test_cache_key_encodes_image_size_and_norm():
    assert cache_key((320, 240)) == "320x240_imagenet"
    assert cache_key((224, 224), norm="clip") == "224x224_clip"
    # Phase 3's CLIP-norm + 224×224 swap must not collide with the ResNet cache.
    assert cache_key((320, 240)) != cache_key((224, 224), norm="clip")


def test_image_cache_writes_and_reloads_uint8(tmp_path):
    img = Image.new("RGB", (640, 320), color=(128, 64, 32))
    cache = ImageCache(tmp_path, "train", (320, 240))
    t = cache.get(0, img)
    assert t.dtype == torch.uint8, "cache must store uint8, not fp32"
    assert t.shape == (3, 240, 320)
    # Second call hits disk (no rewrite) and must be byte-identical.
    t2 = cache.get(0, img)
    assert torch.equal(t, t2)
    assert (tmp_path / "320x240_imagenet" / "train" / "0000000.pt").exists()


def test_image_cache_keys_by_norm(tmp_path):
    img = Image.new("RGB", (320, 240))
    ImageCache(tmp_path, "val", (320, 240), norm="imagenet").get(0, img)
    ImageCache(tmp_path, "val", (320, 240), norm="clip").get(1, img)
    assert (tmp_path / "320x240_imagenet" / "val" / "0000000.pt").exists()
    assert (tmp_path / "320x240_clip" / "val" / "0000001.pt").exists()


def test_image_cache_atomic_write(tmp_path):
    img = Image.new("RGB", (320, 240), color=(10, 20, 30))
    cache = ImageCache(tmp_path, "train", (320, 240))
    cache.get(5, img)
    files = list((tmp_path / "320x240_imagenet" / "train").iterdir())
    assert all(".tmp" not in f.name for f in files), "no temp files left behind"


def test_imagenet_stats_unchanged():
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]
