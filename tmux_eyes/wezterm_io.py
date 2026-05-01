"""WezTerm CLI IPC client.

Design rationale (sol_arch.md §7):
  WezTerm exposes a stateless CLI rather than a persistent control-mode socket,
  so each call shells out fresh via `wezterm cli list --format json`. At ~10 ms
  per call this is acceptable because __main__.py caches pane geometry for 1 s.

  No persistent subprocess is maintained; close() and the context-manager
  methods are no-ops included only for TmuxClientProto compliance.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from tmux_eyes.types import PaneInfo


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class WeztermError(RuntimeError):
    """Raised when the wezterm CLI reports an error or output is unparseable."""


# ---------------------------------------------------------------------------
# Pure parsing helper
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("pane_id", "left_col", "top_row", "size", "is_active")
_SIZE_FIELDS = ("cols", "rows")


def parse_wezterm_list(json_text: str) -> dict[str, PaneInfo]:
    """Parse the stdout of `wezterm cli list --format json` into a mapping.

    Raises WeztermError on JSON parse failure or missing required fields.
    Returns an empty dict for an empty JSON array.
    """
    try:
        entries: list[dict[str, Any]] = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise WeztermError(f"Failed to parse wezterm JSON output: {exc}") from exc

    result: dict[str, PaneInfo] = {}
    for entry in entries:
        # Validate top-level required fields
        for field in _REQUIRED_FIELDS:
            if field not in entry:
                raise WeztermError(
                    f"wezterm list entry missing required field {field!r}: {entry!r}"
                )
        size = entry["size"]
        for field in _SIZE_FIELDS:
            if field not in size:
                raise WeztermError(
                    f"wezterm list entry 'size' missing field {field!r}: {entry!r}"
                )
        pane_id = str(entry["pane_id"])
        result[pane_id] = PaneInfo(
            pane_id=pane_id,
            left=entry["left_col"],
            top=entry["top_row"],
            width=size["cols"],
            height=size["rows"],
            active=bool(entry["is_active"]),
        )
    return result


# ---------------------------------------------------------------------------
# Pane-id validation
# ---------------------------------------------------------------------------

_PANE_ID_RE = re.compile(r"^\d+$")


def _validate_pane_id(pane_id: str) -> None:
    """Reject anything that isn't purely numeric to prevent injection."""
    if not _PANE_ID_RE.match(pane_id):
        raise ValueError(
            f"Invalid pane_id {pane_id!r}. Must be numeric digits only (e.g. '1', '42')."
        )


# ---------------------------------------------------------------------------
# WeztermClient
# ---------------------------------------------------------------------------

class WeztermClient:
    """Stateless WezTerm CLI client.

    Each public method shells out to `wezterm cli ...` fresh. There is no
    persistent subprocess — wezterm has no control-mode equivalent.

    Usage::

        with WeztermClient() as client:
            panes = client.get_panes()
            client.select_pane(panes[some_id].pane_id)
    """

    def __init__(self) -> None:
        pass  # no socket — wezterm uses its own IPC automatically

    # ------------------------------------------------------------------
    # Lifecycle (no-ops: no persistent process to manage)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """No-op: included for TmuxClientProto compliance."""

    def __enter__(self) -> "WeztermClient":
        return self

    def __exit__(self, *a: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API (implements TmuxClientProto)
    # ------------------------------------------------------------------

    def get_panes(self) -> dict[str, PaneInfo]:
        """Return geometry for all panes across all windows and tabs."""
        try:
            result = subprocess.run(
                ["wezterm", "cli", "list", "--format", "json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2,
            )
        except FileNotFoundError as exc:
            raise WeztermError("wezterm CLI not found in PATH") from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise WeztermError(
                f"wezterm cli list exited with code {result.returncode}"
                + (f": {stderr}" if stderr else "")
            )

        if not result.stdout.strip():
            raise WeztermError("wezterm cli list returned empty output")

        return parse_wezterm_list(result.stdout)

    def get_active_pane(self) -> str:
        """Return the pane_id of the currently focused pane.

        If multiple panes are active (one per window/tab), the first one
        found is returned. Raises WeztermError if no active pane exists.
        """
        panes = self.get_panes()
        for pane_id, info in panes.items():
            if info.active:
                return pane_id
        raise WeztermError("No active pane found in wezterm cli list output.")

    def select_pane(self, pane_id: str) -> None:
        """Focus the pane identified by *pane_id*.

        Raises ValueError for pane_ids that aren't purely numeric, to
        prevent command-injection via shell argument passing.
        """
        _validate_pane_id(pane_id)
        subprocess.run(
            ["wezterm", "cli", "activate-pane", "--pane-id", pane_id],
            check=False,
        )
