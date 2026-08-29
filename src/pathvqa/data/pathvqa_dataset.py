"""PathVQA dataset: HuggingFace loader with CLIP preprocessing and caching."""

import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Repo root (this file lives at <root>/src/pathvqa/data/).
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Keep HF cache inside the repo's gitignored data/ dir.
os.environ.setdefault(
    "HF_HOME",
    str(_REPO_ROOT / "data" / "hf-cache"),
)

from datasets import load_dataset  # noqa: E402  (needs HF_HOME set above)

# Suppress harmless truncated-TIFF warnings from PathVQA images.
warnings.filterwarnings("ignore", message="Truncated File Read")

DATASET_REPO_ID = "flaviagiammarino/path-vqa"
# Resolve to the exact cached Hub commit used for the reproduced metrics.
DATASET_REVISION = "1685832883334b5bb5beaf4e4b333fdeecaa4ad9"


# ImageNet stats — legacy fallback; CLIP norm is preferred (loaded at runtime).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_CLIP_PROCESSOR: Optional[Any] = None


def get_clip_normalize() -> Any:
    """Return a torchvision Normalize using CLIP's mean/std from the model config."""
    global _CLIP_PROCESSOR
    if _CLIP_PROCESSOR is None:
        from transformers import CLIPImageProcessor

        model_name = "openai/clip-vit-base-patch32"
        try:
            _CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained(model_name, local_files_only=True)
        except OSError:
            # Keep first-run setup working when the processor is not cached yet.
            _CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained(model_name)
    return transforms.Normalize(mean=_CLIP_PROCESSOR.image_mean, std=_CLIP_PROCESSOR.image_std)


def get_normalize(norm: str) -> Any:
    """Return the normalization transform selected by ``Config.data.norm``.

    Single source of truth shared by the dataset and the serving layer: using
    CLIP statistics for a checkpoint trained with ImageNet normalization (or
    vice versa) silently degrades every prediction.
    """
    if norm == "clip":
        return get_clip_normalize()
    if norm == "imagenet":
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    raise ValueError(f"Unknown normalization {norm!r}; expected 'clip' or 'imagenet'")


# Bump when preprocessing semantics change (resize interpolation, crop policy,
# dataset revision) so stale cached tensors are never silently reused.
TRANSFORM_VERSION = "v1"


def cache_key(image_size: Tuple[int, int], norm: str = "imagenet") -> str:
    """Stable cache key for preprocessed images, keyed by (image_size, norm, transform version)."""
    return f"{image_size[0]}x{image_size[1]}_{norm}_{TRANSFORM_VERSION}"


def normalize_text(text: str) -> str:
    """Normalize answer text: strip and collapse whitespace, lowercase."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def resize_keep_aspect(img: PIL.Image.Image, size: Tuple[int, int] = (320, 240)) -> PIL.Image.Image:
    """Resize preserving aspect ratio, then center-crop to ``size``."""
    out_w, out_h = size
    w, h = img.size
    scale = min(w / out_w, h / out_h)
    new_w = max(1, round(w / scale))
    new_h = max(1, round(h / scale))
    img = img.resize((new_w, new_h), PIL.Image.Resampling.BILINEAR)
    left = (new_w - out_w) // 2
    top = (new_h - out_h) // 2
    return img.crop((left, top, left + out_w, top + out_h))


class ImageCache:
    """On-disk cache of resized, center-cropped uint8 image tensors.

    Stores (3, H, W) uint8 tensors after resize+crop but before augmentation
    and normalization. Writes are atomic (temp file + rename). Keyed by
    ``(image_size, norm, TRANSFORM_VERSION)`` so any preprocessing change
    rebuilds cleanly instead of reusing stale tensors.
    """

    def __init__(self, cache_dir: Union[str, Path], split: str, image_size: Tuple[int, int], norm: str = "imagenet"):
        self.split = split
        self.key = cache_key(image_size, norm)
        self.dir = Path(cache_dir) / self.key / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self._transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: resize_keep_aspect(img, image_size)),
                transforms.PILToTensor(),  # uint8 (3, H, W)
            ]
        )

    def _path(self, idx: int) -> Path:
        return self.dir / f"{idx:07d}.pt"

    def get(self, idx: int, pil_image: PIL.Image.Image) -> torch.Tensor:
        """Return cached uint8 tensor for ``idx``, building it if absent."""
        path = self._path(idx)
        if path.exists():
            cached: torch.Tensor = torch.load(path, weights_only=True)
            return cached
        tensor: torch.Tensor = self._transform(pil_image)
        # Atomic write: temp file + rename.
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(tensor, f)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return tensor


def lookup_answer_idx(answer: str, answer_to_idx: Dict[str, int]) -> int:
    """Index a normalized answer, mapping unseen answers to <UNK>."""
    idx = answer_to_idx.get(answer)
    if idx is not None:
        return idx
    idx = answer_to_idx.get("<UNK>")
    if idx is not None:
        return idx
    raise KeyError(
        f"Answer {answer!r} is not in the answer vocabulary and no <UNK> class "
        "is reserved. Pass an answer_to_idx that includes <UNK> (e.g. the train "
        "vocabulary) or build the vocabulary from this split."
    )


class PathVQADataset(Dataset[Dict[str, Any]]):
    """PathVQA HuggingFace dataset with CLIP preprocessing and optional caching."""

    def __init__(
        self,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        max_samples: Optional[int] = None,
        answer_to_idx: Optional[Dict[str, int]] = None,
        use_randaugment: bool = True,
        use_cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        norm: str = "clip",
        randaugment_num_ops: int = 2,
        randaugment_magnitude: int = 9,
        image_source_indices: Optional[List[int]] = None,
    ):
        print(f"Loading PathVQA {split} split from HuggingFace...")
        split_map = {"train": "train", "val": "validation", "test": "test"}
        if split not in split_map:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(split_map)}")
        hf_split = split_map[split]
        self.data = load_dataset(DATASET_REPO_ID, split=hf_split, revision=DATASET_REVISION)
        # ``is not None`` so max_samples=0 selects an empty slice instead of
        # silently loading the full split.
        if max_samples is not None:
            if max_samples < 0:
                raise ValueError(f"max_samples must be non-negative, got {max_samples}")
            self.data = self.data.select(range(min(max_samples, len(self.data))))
        self.image_source_indices = image_source_indices
        if image_source_indices is not None and len(image_source_indices) != len(self.data):
            raise ValueError("image_source_indices must have one entry per selected dataset row")
        if image_source_indices is not None and any(
            source < 0 or source >= len(self.data) for source in image_source_indices
        ):
            raise ValueError("image_source_indices entries must refer to selected dataset rows")

        if answer_to_idx is None:
            self.answers = sorted(set(normalize_text(a) for a in self.data["answer"]))
            self.answer_to_idx = {a: i for i, a in enumerate(self.answers)}
        else:
            # Use caller's mapping (e.g. train vocab) for target alignment on val/test.
            self.answer_to_idx = answer_to_idx
            self.answers = sorted(answer_to_idx)
        self.idx_to_answer = {i: a for a, i in self.answer_to_idx.items()}
        self.cache: Optional[ImageCache]

        # Cache: uint8 tensors after resize+crop, before aug + norm.
        if use_cache:
            default_cache_dir = _REPO_ROOT / "data" / "cache"
            self.cache = ImageCache(cache_dir or default_cache_dir, split, image_size, norm)
        else:
            self.cache = None
        self._base_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: resize_keep_aspect(img, image_size)),
                transforms.PILToTensor(),  # uint8 (3, H, W)
            ]
        )
        self.augment = use_randaugment and split == "train"
        self.randaugment = (
            transforms.RandAugment(
                num_ops=randaugment_num_ops,
                magnitude=randaugment_magnitude,
            )
            if self.augment
            else None
        )
        self.normalize = get_normalize(norm)

        print(f"Loaded {len(self.data)} samples, answer vocab: {len(self.answers)}")

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and transform one (image, question, answer) sample."""
        item = self.data[idx]
        image_row = self.image_source_indices[idx] if self.image_source_indices is not None else idx
        image = self.data[image_row]["image"]
        if image is None:
            raise ValueError(f"Row {idx}: image could not be decoded (missing or corrupt file).")
        image = image.convert("RGB")
        if self.cache is not None:
            img_u8 = self.cache.get(image_row, image)
        else:
            img_u8 = self._base_transform(image)
        # RandAugment on uint8 (train only), then float + normalize.
        if self.randaugment is not None:
            img_u8 = self.randaugment(img_u8)
        image = self.normalize(img_u8.to(torch.float32) / 255.0)
        answer = normalize_text(item["answer"])
        answer_idx = lookup_answer_idx(answer, self.answer_to_idx)
        return {
            "image": image,
            "question": item["question"],
            "answer": answer,
            "answer_idx": answer_idx,
        }


@dataclass
class CollateOutput:
    """Batched samples as produced by ``collate_fn``."""

    images: torch.Tensor  # (B, 3, H, W) float, normalized
    targets: torch.Tensor  # (B,) int64 answer indices
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)


@dataclass
class BatchData:
    """Device-resident batch as produced by ``prepare_batch``."""

    input_ids: torch.Tensor  # (B, L)
    attention_mask: torch.Tensor  # (B, L)
    targets: torch.Tensor  # (B,)
    images: torch.Tensor  # (B, 3, H, W)
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)


def collate_fn(batch: List[Dict[str, Any]]) -> CollateOutput:
    """Stack batch items into tensors."""
    return CollateOutput(
        images=torch.stack([b["image"] for b in batch]),
        targets=torch.tensor([b["answer_idx"] for b in batch]),
        questions=[b["question"] for b in batch],
        answers=[b["answer"] for b in batch],
    )


def prepare_batch(
    batch: Union[CollateOutput, Dict[str, Any]],
    tokenizer: Callable[..., Any],
    device: torch.device,
    max_seq_len: int = 64,
) -> BatchData:
    """Tokenize questions and move batch tensors to device.

    Accepts a :class:`CollateOutput` (or a dict with the same fields).
    """
    if isinstance(batch, dict):
        images = batch["images"]
        targets = batch.get("targets", batch.get("answer_indices"))
        questions = batch["questions"]
        answers = batch.get("answers", [])
        if targets is None:
            raise KeyError("Batch is missing 'targets' (or legacy 'answer_indices').")
    else:
        images = batch.images
        targets = batch.targets
        questions = batch.questions
        answers = batch.answers
    images = images.to(device)
    enc = tokenizer(
        questions,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )
    return BatchData(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        targets=targets.to(device),
        images=images,
        questions=list(questions),
        answers=list(answers),
    )


def pathvqa_dataloader(
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    max_samples: Optional[int] = None,
    answer_to_idx: Optional[Dict[str, int]] = None,
    use_randaugment: bool = True,
    use_cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    norm: str = "clip",
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    image_source_indices: Optional[List[int]] = None,
) -> DataLoader[Any]:
    """Create a configured PathVQA DataLoader. Pass ``answer_to_idx`` for val/test
    to align targets with the training vocabulary."""
    dataset = PathVQADataset(
        split,
        image_size,
        max_samples,
        answer_to_idx,
        use_randaugment,
        use_cache,
        cache_dir,
        norm,
        randaugment_num_ops,
        randaugment_magnitude,
        image_source_indices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


if __name__ == "__main__":
    import tempfile

    print("Testing PathVQADataset...")
    with tempfile.TemporaryDirectory() as tmp:
        ds = PathVQADataset(split="train", image_size=(224, 224), max_samples=10, cache_dir=tmp, norm="clip")
        item = ds[0]
        print(f"Image shape: {item['image'].shape}, Question: {item['question'][:50]}..., Answer: {item['answer']}")
        print(f"Answer vocab size: {len(ds.answers)}")
        batch = collate_fn([ds[i] for i in range(4)])
        print(f"Batch images shape: {batch.images.shape}")
        cached_files = list((Path(tmp) / cache_key((224, 224), norm="clip") / "train").glob("*.pt"))
        print(f"Cache files written: {len(cached_files)}")
    # RandAugment applied on train only
    print(f"Train augment enabled: {ds.augment}, RandAugment: {ds.randaugment is not None}")
    print("Dataset test passed!")
