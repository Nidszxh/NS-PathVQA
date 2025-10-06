from importlib.resources import path
import json
from pathlib import Path
from typing import List, Dict, Iterator


class CLEVRProcessor:
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw JSON files ---
    def load_json(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r") as f:
            return json.load(f)

    def load_scenes(self, split: str = "train") -> List[Dict]:
        path = self.raw_data_dir / "scenes" / f"CLEVR_{split}_scenes.json"
        return self.load_json(path)["scenes"]

    def load_questions(self, split: str = "train") -> List[Dict]:
        path = self.raw_data_dir / "questions" / f"CLEVR_{split}_questions.json"
        return self.load_json(path)["questions"]

    # --- Preprocessing ---
    def preprocess_scenes(self, split: str = "train") -> Iterator[Dict]:
        for scene in self.load_scenes(split):
            yield {
                "image_id": scene["image_index"],
                "objects": [
                    {
                        "color": obj["color"],
                        "shape": obj["shape"],
                        "size": obj["size"],
                        "material": obj["material"],
                        "position": obj["3d_coords"]
                    }
                    for obj in scene["objects"]
                ]
            }

    def preprocess_questions(self, split: str = "train") -> Iterator[Dict]:
        for q in self.load_questions(split):
            yield {
                "image_id": q["image_index"],
                "question": q["question"],
                "program": q.get("program"),
                "answer": q["answer"]
            }

    # --- Save JSONL using a generator to save memory ---
    def save_jsonl(self, data: Iterator[Dict], out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    # --- Full pipeline for a split ---
    def process_split(self, split: str = "train"):
        print(f"Processing {split} split...")

        self.save_jsonl(
            self.preprocess_scenes(split),
            self.processed_data_dir / f"processed_{split}_scenes.jsonl"
        )
        self.save_jsonl(
            self.preprocess_questions(split),
            self.processed_data_dir / f"processed_{split}_questions.jsonl"
        )

        print(f"{split} split done. Files saved in {self.processed_data_dir}")

    def load_scenes(self, split: str = "train") -> List[Dict]:
        path = self.raw_data_dir / "scenes" / f"CLEVR_{split}_scenes.json"
        if not path.exists():
            raise FileNotFoundError(f"Scene file for '{split}' split not found at: {path}")
        return self.load_json(path)["scenes"]

    def load_questions(self, split: str = "train") -> List[Dict]:
        path = self.raw_data_dir / "questions" / f"CLEVR_{split}_questions.json"
        if not path.exists():
            raise FileNotFoundError(f"Question file for '{split}' split not found at: {path}")
        return self.load_json(path)["questions"]

if __name__ == "__main__":
    RAW_DIR = Path("/home/nidszxh/Neuro-Symbolic-Visual-Reasoning-System/data/clevr/CLEVR_v1.0")
    PROCESSED_DIR = Path("/home/nidszxh/Neuro-Symbolic-Visual-Reasoning-System/data/processed")

    processor = CLEVRProcessor(RAW_DIR, PROCESSED_DIR)

    # NSVR only needs train + val
    for split in ["train", "val"]:
        processor.process_split(split)

    print("All preprocessing done for NSVR training and validation!")