"""Lockfile: single-instance contention behavior."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from reindex.lockfile import LOCK_FILE, single_instance


def test_acquire_release(tmp_path: Path):
    with single_instance(tmp_path):
        assert (tmp_path / LOCK_FILE).is_file()
    # After exit the file remains but is unlocked. We'll re-acquire to prove that.
    with single_instance(tmp_path):
        pass


def test_pid_written_to_lock_file(tmp_path: Path):
    with single_instance(tmp_path):
        content = (tmp_path / LOCK_FILE).read_text(encoding="utf-8").strip()
        assert content == str(os.getpid())


def test_concurrent_acquire_raises(tmp_path: Path):
    """Second acquire while first holds → RuntimeError."""
    with single_instance(tmp_path):
        with pytest.raises(RuntimeError, match="another reindex instance"):
            # Use a separate process to truly contend the flock; same-process
            # re-entry on the same fd would succeed (flock is per-fd).
            script = f"""
            import sys
            sys.path.insert(0, {repr(str(Path(__file__).resolve().parent.parent / "src"))})
            from reindex.lockfile import single_instance
            from pathlib import Path
            try:
                with single_instance(Path({repr(str(tmp_path))})):
                    pass
            except RuntimeError as e:
                print('LOCKED:' + str(e))
                sys.exit(0)
            print('UNEXPECTEDLY ACQUIRED')
            sys.exit(1)
            """
            proc = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(script)],
                capture_output=True, text=True, timeout=5,
            )
            if "LOCKED:" not in proc.stdout:
                pytest.fail(f"contention not detected: stdout={proc.stdout} stderr={proc.stderr}")
            raise RuntimeError("another reindex instance held the lock (expected)")


def test_lock_released_after_block(tmp_path: Path):
    with single_instance(tmp_path):
        pass
    # Now spawn a child that tries to acquire — should succeed.
    script = f"""
    import sys
    sys.path.insert(0, {repr(str(Path(__file__).resolve().parent.parent / "src"))})
    from reindex.lockfile import single_instance
    from pathlib import Path
    with single_instance(Path({repr(str(tmp_path))})):
        print('OK')
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, timeout=5,
    )
    assert proc.returncode == 0
    assert "OK" in proc.stdout


def test_lock_released_on_exception(tmp_path: Path):
    """If the with-block raises, the lock must still be released."""
    with pytest.raises(ValueError):
        with single_instance(tmp_path):
            raise ValueError("oops")

    # Should be able to acquire again immediately (in another process to be sure).
    script = f"""
    import sys
    sys.path.insert(0, {repr(str(Path(__file__).resolve().parent.parent / "src"))})
    from reindex.lockfile import single_instance
    from pathlib import Path
    with single_instance(Path({repr(str(tmp_path))})):
        print('OK')
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, timeout=5,
    )
    assert proc.returncode == 0
