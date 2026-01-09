"""PathVQA entry point — currently a no-op that points users to train.py.

The actual dataloader, model, training, and evaluation logic lives in:
  - train.py
  - evaluate.py
  - src/data_loaders/pathvqaDataset.py
  - src/models/
  - src/symbolic/
"""


def main():
    """Print usage info and exit (entry point redirects to train.py)."""
    print("=" * 60)
    print("PathVQA Dataset — HuggingFace")
    print("=" * 60)
    print("\nDataset is downloaded on first use by the dataloader.")
    print("Run `python train.py` to start training.")
    print("\nIf you want to inspect the dataset first:")
    print("  python -c \"from datasets import load_dataset; d = load_dataset('flaviagiammarino/path-vqa', split='train'); print(d[0])\"")
    print()


if __name__ == "__main__":
    main()
