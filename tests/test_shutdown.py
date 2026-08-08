"""ShutdownController: signal handling state machine."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from reindex import shutdown
from reindex.shutdown import ShutdownController


def _new_controller() -> ShutdownController:
    return ShutdownController()


@pytest.mark.asyncio
async def test_first_press_sets_event_and_cancels_tasks():
    """1st Ctrl-C: shutdown_event set, in-flight tasks cancelled."""
    ctl = _new_controller()
    ctl.install(asyncio.get_running_loop())

    async def long_task():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            return "cancelled"

    task = asyncio.create_task(long_task())
    await asyncio.sleep(0)  # let it start

    assert not ctl.is_shutting_down()
    ctl._handle()  # simulate SIGINT
    assert ctl.is_shutting_down()

    # The task should be cancelled.
    result = await task
    assert result == "cancelled"

    ctl.uninstall(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_second_press_increments_counter_no_force():
    """2nd press: still no force, tells user how many more remain."""
    ctl = _new_controller()
    ctl.install(asyncio.get_running_loop())

    ctl._handle()
    assert ctl.is_shutting_down()
    # Second press should NOT call os._exit.
    with patch("reindex.shutdown.os._exit") as mock_exit:
        ctl._handle()
        mock_exit.assert_not_called()

    ctl.uninstall(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_three_presses_within_window_force_quits():
    """3 presses within 2s -> os._exit(130)."""
    ctl = _new_controller()
    ctl.install(asyncio.get_running_loop())

    with patch("reindex.shutdown.os._exit") as mock_exit:
        ctl._handle()  # 1
        ctl._handle()  # 2
        ctl._handle()  # 3 -- force quit
        mock_exit.assert_called_once_with(130)

    ctl.uninstall(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_presses_outside_window_dont_force_quit(monkeypatch):
    """If presses are spaced > 2s apart, the deque prunes them and 3rd press is a 'first press' again."""
    ctl = _new_controller()
    ctl.install(asyncio.get_running_loop())

    loop = asyncio.get_running_loop()
    # Manipulate loop.time so we can simulate spaced presses.
    fake_now = [0.0]
    monkeypatch.setattr(loop, "time", lambda: fake_now[0])

    with patch("reindex.shutdown.os._exit") as mock_exit:
        ctl._handle()  # t=0
        fake_now[0] = 3.0
        ctl._handle()  # t=3 -- first press is older than 2s, deque pruned
        fake_now[0] = 6.0
        ctl._handle()  # t=6 -- only one in window
        mock_exit.assert_not_called()

    ctl.uninstall(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_install_idempotent():
    ctl = _new_controller()
    loop = asyncio.get_running_loop()
    ctl.install(loop)
    ctl.install(loop)  # no-op second time
    assert ctl._installed
    ctl.uninstall(loop)


@pytest.mark.asyncio
async def test_uninstall_safe_when_not_installed():
    ctl = _new_controller()
    ctl.uninstall(asyncio.get_running_loop())  # no exception


@pytest.mark.asyncio
async def test_module_helpers():
    """install() / uninstall() / get() / is_shutting_down() at module level."""
    loop = asyncio.get_running_loop()
    assert shutdown.get() is None
    assert not shutdown.is_shutting_down()

    ctl = shutdown.install(loop)
    assert shutdown.get() is ctl
    assert not shutdown.is_shutting_down()

    ctl._handle()
    assert shutdown.is_shutting_down()

    shutdown.uninstall()
    assert shutdown.get() is None


@pytest.mark.asyncio
async def test_sigterm_sets_event_and_cancels_tasks():
    """SIGTERM: same graceful behavior as SIGINT tier 1."""
    ctl = _new_controller()
    ctl.install(asyncio.get_running_loop())

    async def long_task():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            return "cancelled"

    task = asyncio.create_task(long_task())
    await asyncio.sleep(0)

    ctl._handle_sigterm()
    assert ctl.is_shutting_down()
    assert await task == "cancelled"

    ctl.uninstall(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_sigterm_never_force_quits():
    """Repeated SIGTERM must NOT escalate to os._exit."""
    ctl = _new_controller()
    ctl.install(asyncio.get_running_loop())

    with patch("reindex.shutdown.os._exit") as mock_exit:
        for _ in range(5):
            ctl._handle_sigterm()
        mock_exit.assert_not_called()
    assert ctl.is_shutting_down()

    ctl.uninstall(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_install_registers_sigterm_handler():
    """A real SIGTERM delivered to the process reaches the controller."""
    import os as _os
    import signal as _signal

    ctl = _new_controller()
    ctl.install(asyncio.get_running_loop())

    _os.kill(_os.getpid(), _signal.SIGTERM)
    try:
        await asyncio.sleep(0.05)  # let the loop dispatch the handler
    except asyncio.CancelledError:
        pass  # handler cancels in-flight tasks including this sleep; that's correct
    assert ctl.is_shutting_down()

    ctl.uninstall(asyncio.get_running_loop())
