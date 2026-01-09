"""PathVQA dataset loader from HuggingFace with image transforms."""

from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset


class PathVQADataset(Dataset):
    """Wraps the PathVQA HuggingFace dataset with standard image preprocessing.

    On initialization, loads the specified split, builds the answer vocabulary
    from the data, and sets up ResNet-compatible image transforms.
    """

    def __init__(self, split: str = "train", image_size: tuple = (320, 240),
                 max_samples: Optional[int] = None):
        print(f"Loading PathVQA {split} split from HuggingFace...")
        split_map = {"train": "train", "val": "validation", "test": "test"}
        hf_split = split_map[split]
        self.data = load_dataset("flaviagiammarino/path-vqa", split=hf_split)
        if max_samples:
            self.data = self.data.select(range(min(max_samples, len(self.data))))

        # Build answer vocabulary from the full split
        self.answers = sorted(set(self.data["answer"]))
        self.answer_to_idx = {a: i for i, a in enumerate(self.answers)}
        self.idx_to_answer = {i: a for a, i in self.answer_to_idx.items()}

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"Loaded {len(self.data)} samples, answer vocab: {len(self.answers)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        image = self.transform(item["image"].convert("RGB"))
        return {
            "image": image,
            "question": item["question"],
            "answer": item["answer"],
            "answer_idx": self.answer_to_idx[item["answer"]],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of dataset items into a batched dict."""
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "answer_indices": torch.tensor([b["answer_idx"] for b in batch]),
    }


def prepare_batch(batch: Dict, vocab: "QuestionVocabulary", device: torch.device
                 ) -> tuple:
    """Convert a raw batch dict into model-ready tensors.

    Returns:
        (padded_question_indices, lengths, target_indices, images)
    """
    images = batch["images"].to(device)
    q_indices, q_lengths = [], []
    for q in batch["questions"]:
        idxs = vocab.encode(q)
        q_indices.append(idxs)
        q_lengths.append(len(idxs))
    max_len = max(q_lengths) if q_lengths else 0
    padded = torch.zeros(len(q_indices), max_len, dtype=torch.long)
    for i, idxs in enumerate(q_indices):
        padded[i, :len(idxs)] = torch.tensor(idxs)
    return padded.to(device), torch.tensor(q_lengths).to(device), batch["answer_indices"].to(device), images


def pathvqa_dataloader(split: str = "train", batch_size: int = 32,
                       shuffle: bool = True, num_workers: int = 4,
                       image_size: tuple = (320, 240),
                       max_samples: Optional[int] = None) -> DataLoader:
    """Convenience function to create a configured PathVQA DataLoader."""
    dataset = PathVQADataset(split, image_size, max_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
