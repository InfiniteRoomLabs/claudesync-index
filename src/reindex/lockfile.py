"""
Single-instance lockfile via flock(). Prevents overlapping cron runs from
stepping on each other (state file, cost log, INDEX.md writes).

Usage:
    with single_instance(export_root):
        ... pipeline ...

Releases on context exit, even on crash. Stale locks are auto-cleared by
flock() since the lock dies with the process holding the fd.
"""

from __future__ import annotations

import errno
import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE = ".reindex.lock"


@contextmanager
def single_instance(export_root: Path) -> Iterator[None]:
    """Acquire an exclusive flock on .reindex.lock for the duration of the block.

    Raises RuntimeError if another instance holds the lock.
    """
    lock_path = export_root / LOCK_FILE
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise RuntimeError(
                    f"another reindex instance holds {lock_path}; refusing to run"
                ) from e
            raise
        # Write our PID for diagnostics. Doesn't affect the lock.
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
