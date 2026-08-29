"""Hash the release surface for a reproducible submission artifact.

Records SHA-256 digests for the source tree, entry points, scripts, dependency
pins, and (optionally) the evaluated checkpoint, together with the pinned
dataset revision and runtime versions. The manifest is a provenance record, not
a substitute for archiving the files themselves: checkpoints and datasets are
not committed to Git.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE_SUFFIXES = {".py", ".toml", ".txt", ".yml", ".yaml", ".md"}
EXCLUDED_PARTS = {"__pycache__", ".git", "data", "checkpoints", "logs", "outputs", "scratch", ".pytest_cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield tracked-looking source/config files, skipping data and artifacts."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in INCLUDE_SUFFIXES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts):
            continue
        yield path


def _git_commit(root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def build_manifest(
    root: Path,
    checkpoint: Optional[Path] = None,
    dataset_repo: str = "flaviagiammarino/path-vqa",
    dataset_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the manifest dict for ``root``."""
    files = {
        str(path.relative_to(root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in iter_source_files(root)
    }
    manifest: Dict[str, Any] = {
        "root": str(root),
        "git_commit": _git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": {"repo_id": dataset_repo, "hub_revision": dataset_revision},
        "files": files,
        "file_count": len(files),
    }
    if checkpoint is not None:
        manifest["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/release_manifest.json"))
    parser.add_argument("--dataset-revision", type=str, default=None)
    args = parser.parse_args()

    if args.dataset_revision is None:
        sys.path.insert(0, str(REPO_ROOT))
        from pathvqa.data.pathvqa_dataset import DATASET_REVISION

        args.dataset_revision = DATASET_REVISION

    manifest = build_manifest(args.root, args.checkpoint, dataset_revision=args.dataset_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Manifest for {manifest['file_count']} files written to {args.output}")


if __name__ == "__main__":
    main()
