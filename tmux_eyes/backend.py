"""Multiplexer backend selection.

Factory that returns a :class:`MultiplexerClientProto` instance based on
config + environment, so the rest of the daemon doesn't care whether the
underlying terminal multiplexer is tmux or wezterm.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING

from .config import Config
from .types import MultiplexerClientProto

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BackendError(RuntimeError):
    """No usable multiplexer backend could be selected."""


def detect_backend() -> str:
    """Best-effort autodetection: returns 'tmux' or 'wezterm'.

    Order of precedence:
      1. ``$TMUX`` env var set → tmux (we're inside a tmux session right now).
      2. ``$WEZTERM_PANE`` set → wezterm (we're inside a wezterm pane).
      3. ``tmux`` binary on PATH → tmux.
      4. ``wezterm`` binary on PATH → wezterm.
      5. Raise BackendError.
    """
    if os.environ.get("TMUX"):
        return "tmux"
    if os.environ.get("WEZTERM_PANE"):
        return "wezterm"
    if shutil.which("tmux"):
        return "tmux"
    if shutil.which("wezterm"):
        return "wezterm"
    raise BackendError(
        "No multiplexer detected. Set TMUX_EYES_BACKEND=tmux or =wezterm, "
        "or install one of them."
    )


def select_backend(config: Config) -> MultiplexerClientProto:
    """Build a multiplexer client based on the resolved backend name."""
    name = (config.backend or "auto").lower()
    if name == "auto":
        name = detect_backend()
        logger.info("backend autodetected: %s", name)

    if name == "tmux":
        from .tmux_io import TmuxClient

        return TmuxClient(socket=config.tmux_socket)

    if name == "wezterm":
        from .wezterm_io import WeztermClient

        return WeztermClient()

    raise BackendError(
        f"Unknown backend {name!r}. Use 'tmux', 'wezterm', or 'auto'."
    )
