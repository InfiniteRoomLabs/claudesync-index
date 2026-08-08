"""
Two-tier graceful shutdown handler for SIGINT (Ctrl-C).

Tier 1 (1st press):
  - Sets `shutdown_event` (asyncio.Event) so cooperative checks bail out.
  - Cancels all running asyncio tasks except the current one. Workers receive
    CancelledError at next `await`. State-file writes (sync) complete.
  - Stderr message tells user how many more presses force-quit.

Tier 2 (3 presses within 2s):
  - `os._exit(130)` -- kills threads and process immediately. State file may
    contain a half-written batch entry. Trade-off the user explicitly opted into.

Usage:
    controller = install(asyncio.get_running_loop())
    if controller.is_shutting_down():
        ...  # cooperative bail
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections import deque

_FORCE_QUIT_PRESSES = 3
_FORCE_QUIT_WINDOW_S = 2.0


class ShutdownController:
    """Owns the shutdown_event and tracks recent SIGINT presses."""

    def __init__(self) -> None:
        self.shutdown_event = asyncio.Event()
        self._presses: deque[float] = deque()
        self._installed = False

    def is_shutting_down(self) -> bool:
        return self.shutdown_event.is_set()

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._installed:
            return
        self._installed = True
        try:
            loop.add_signal_handler(signal.SIGINT, self._handle)
            loop.add_signal_handler(signal.SIGTERM, self._handle_sigterm)
        except NotImplementedError:
            # Windows event loop doesn't support add_signal_handler -- best-effort.
            signal.signal(signal.SIGINT, lambda *_: self._handle())
            signal.signal(signal.SIGTERM, lambda *_: self._handle_sigterm())

    def uninstall(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._installed:
            return
        try:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        except (NotImplementedError, ValueError):
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        self._installed = False

    def _request_graceful(self, message: str) -> None:
        """Set the shutdown event and cancel in-flight tasks (tier-1 path)."""
        loop = asyncio.get_running_loop()
        self.shutdown_event.set()
        sys.stderr.write(message)
        sys.stderr.flush()
        # Cancel running tasks so awaits raise CancelledError.
        current = asyncio.current_task(loop)
        for task in asyncio.all_tasks(loop):
            if task is not current and not task.done():
                task.cancel()

    def _handle_sigterm(self) -> None:
        """SIGTERM (docker stop / k8s): single-shot graceful, no force tier.

        The supervisor escalates to SIGKILL on its own grace timeout, so
        press-counting makes no sense here. Idempotent on repeat delivery.
        """
        if self.shutdown_event.is_set():
            return
        self._request_graceful(
            "\n[reindex] SIGTERM received. Finishing in-flight work and "
            "persisting state.\n"
        )

    def _handle(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()

        # Prune presses outside the rolling window.
        while self._presses and now - self._presses[0] > _FORCE_QUIT_WINDOW_S:
            self._presses.popleft()
        self._presses.append(now)

        if len(self._presses) >= _FORCE_QUIT_PRESSES:
            sys.stderr.write(
                f"\n[reindex] {_FORCE_QUIT_PRESSES}x Ctrl-C in "
                f"{_FORCE_QUIT_WINDOW_S}s -- force-quit. State may be partial.\n"
            )
            sys.stderr.flush()
            os._exit(130)

        remaining = _FORCE_QUIT_PRESSES - len(self._presses)
        if not self.shutdown_event.is_set():
            self._request_graceful(
                "\n[reindex] Shutdown requested. Finishing in-flight work and "
                f"persisting state. Press Ctrl-C {remaining} more time(s) within "
                f"{_FORCE_QUIT_WINDOW_S}s to force-quit.\n"
            )
        else:
            sys.stderr.write(
                f"\n[reindex] Already shutting down. {remaining} more Ctrl-C "
                "to force-quit.\n"
            )
            sys.stderr.flush()


# Module-level singleton. Lifecycle managed by `install()` / `uninstall()`.
_controller: ShutdownController | None = None


def install(loop: asyncio.AbstractEventLoop) -> ShutdownController:
    global _controller
    if _controller is None:
        _controller = ShutdownController()
    _controller.install(loop)
    return _controller


def uninstall() -> None:
    global _controller
    if _controller is None:
        return
    try:
        loop = asyncio.get_running_loop()
        _controller.uninstall(loop)
    except RuntimeError:
        pass  # No running loop; nothing to uninstall.
    _controller = None


def get() -> ShutdownController | None:
    return _controller


def is_shutting_down() -> bool:
    return _controller is not None and _controller.is_shutting_down()
