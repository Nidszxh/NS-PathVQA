"""Structured logging setup for entrypoints."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: int = logging.INFO,
                  log_file: Optional[Path] = None) -> logging.Logger:
    """Configure the root ``pathvqa`` logger with optional file output."""
    logger = logging.getLogger("pathvqa")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Return a namespaced child logger (e.g. ``pathvqa.train``)."""
    return logging.getLogger("pathvqa" + (f".{name}" if name else ""))


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging test passed!")