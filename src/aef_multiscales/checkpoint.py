"""Checkpoint/resume for long-running generate jobs.

Tracks which (time, row, col) tiles have been written. On restart,
completed tiles are skipped. Checkpoint is flushed to disk periodically.
"""

import json
import time
from pathlib import Path


class Checkpoint:
    def __init__(self, path: str | Path, flush_interval: float = 60.0):
        self._path = Path(path)
        self._flush_interval = flush_interval
        self._completed: set[tuple[int, int, int]] = set()
        self._last_flush = time.monotonic()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            self._completed = {tuple(t) for t in data["completed"]}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"completed": [list(t) for t in sorted(self._completed)]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.rename(self._path)
        self._last_flush = time.monotonic()

    def is_done(self, time_idx: int, row: int, col: int) -> bool:
        return (time_idx, row, col) in self._completed

    def mark_done(self, time_idx: int, row: int, col: int) -> None:
        self._completed.add((time_idx, row, col))
        if time.monotonic() - self._last_flush >= self._flush_interval:
            self._flush()

    def flush(self) -> None:
        self._flush()

    def __len__(self) -> int:
        return len(self._completed)


class NoCheckpoint:
    """No-op checkpoint for when checkpointing is disabled."""

    def is_done(self, time_idx: int, row: int, col: int) -> bool:
        return False

    def mark_done(self, time_idx: int, row: int, col: int) -> None:
        pass

    def flush(self) -> None:
        pass

    def __len__(self) -> int:
        return 0
