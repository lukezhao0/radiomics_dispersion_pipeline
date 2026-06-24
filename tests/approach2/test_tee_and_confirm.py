"""Tests for Tee and interactive cost confirmation under redirected stdout."""

from __future__ import annotations

import contextlib
import io
import sys

import pytest

from approach2.api.cost import confirm_cost_estimate_or_exit
from approach2.logging_setup import Tee


def test_tee_isatty_delegates_to_primary():
    primary = io.StringIO()
    primary.isatty = lambda: True  # type: ignore[method-assign]
    tee = Tee(primary, io.StringIO())
    assert tee.isatty() is True


def test_confirm_cost_accepts_lowercase_yes(monkeypatch):
    fake_stdin = io.StringIO("yes\n")
    fake_stdin.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "__stdin__", fake_stdin)
    monkeypatch.setattr(sys, "__stdout__", io.StringIO())
    confirm_cost_estimate_or_exit({"model": "gpt-5-nano"}, assume_yes=False)


def test_confirm_cost_uses_real_stdin_under_redirected_stdout(monkeypatch):
    fake_stdin = io.StringIO("y\n")
    fake_stdin.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "__stdin__", fake_stdin)

    real_stdout = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", real_stdout)

    log = io.StringIO()
    tee = Tee(real_stdout, log)
    with contextlib.redirect_stdout(tee):
        # Would fail with broken Tee/input() before the fix.
        confirm_cost_estimate_or_exit({"model": "gpt-5-nano"}, assume_yes=False)

    assert "Continue with LLM extraction calls?" in real_stdout.getvalue()


def test_confirm_cost_rejects_unrecognized_reply(monkeypatch):
    fake_stdin = io.StringIO("maybe\n")
    fake_stdin.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "__stdin__", fake_stdin)
    monkeypatch.setattr(sys, "__stdout__", io.StringIO())
    with pytest.raises(SystemExit):
        confirm_cost_estimate_or_exit({"model": "gpt-5-nano"}, assume_yes=False)


def test_confirm_cost_non_tty_requires_yes_flag(monkeypatch):
    fake_stdin = io.StringIO()
    fake_stdin.isatty = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "__stdin__", fake_stdin)

    with pytest.raises(RuntimeError, match="stdin is not a TTY"):
        confirm_cost_estimate_or_exit({"model": "gpt-5-nano"}, assume_yes=False)
