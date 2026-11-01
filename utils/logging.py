import logging
import time

import torch
from tqdm import tqdm

log = logging.getLogger("abliteration")


class TqdmLoggingHandler(logging.Handler):
    """Write log records through tqdm.write so they don't corrupt progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            print(self.format(record))


def setup_logging(log_file: str | None = None, verbose: bool = False) -> logging.Logger:
    """Log progress to stderr, and to `log_file` when provided. Returns the shared logger."""
    handlers: list[logging.Handler] = [TqdmLoggingHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for handler in handlers:
        handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    return log


def fmt_duration(seconds: float) -> str:
    """Render elapsed seconds as mm:ss or h:mm:ss."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def log_vram(stage: str = "") -> None:
    """Log current/peak VRAM usage when running on CUDA."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 2**30
        peak = torch.cuda.max_memory_allocated() / 2**30
        suffix = f" [{stage}]" if stage else ""
        log.info(f"VRAM{suffix}: allocated {allocated:.2f} GiB, peak {peak:.2f} GiB")
