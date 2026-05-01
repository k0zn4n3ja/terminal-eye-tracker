"""tmux control-mode IPC client.

Design rationale (sol_arch.md §3 decision #4, §7):
  - Uses `tmux -C attach-session` as a long-lived subprocess rather than
    fork+exec'ing `tmux select-pane` per frame (~5-20 ms each).
  - Control-mode frames responses as %begin...%end blocks; notifications
    like %window-pane-changed are interleaved and discarded in Phase 1.

Known limitation: `tmux -C attach-session` requires an existing session.
  For the daemon this is always true (user is already running tmux).
  A robust implementation would catch the failure, run `tmux new-session -d`
  once, then retry — not implemented here to keep the POC minimal.
"""

from __future__ import annotations

import re
import subprocess
from subprocess import PIPE
from typing import Optional

from tmux_eyes.types import PaneInfo


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class TmuxError(RuntimeError):
    """Raised when tmux reports an error or the subprocess exits unexpectedly."""


# ---------------------------------------------------------------------------
# Pure parsing helper
# ---------------------------------------------------------------------------

# Format issued: list-panes -a -F '#{pane_id} #{pane_left} #{pane_top}
#                                    #{pane_width} #{pane_height} #{pane_active}'
# Example line:  %0 0 0 80 24 1
_PANE_LINE_RE = re.compile(
    r"^(%\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([01])\s*$"
)


def parse_list_panes(output: str) -> dict[str, PaneInfo]:
    """Parse the stdout of list-panes into a {pane_id: PaneInfo} mapping.

    Malformed lines (wrong field count, non-numeric geometry) raise ValueError.
    Blank / whitespace-only lines are silently skipped.
    """
    result: dict[str, PaneInfo] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _PANE_LINE_RE.match(line)
        if m is None:
            raise ValueError(f"Malformed list-panes line: {raw_line!r}")
        pane_id, left, top, width, height, active_flag = m.groups()
        result[pane_id] = PaneInfo(
            pane_id=pane_id,
            left=int(left),
            top=int(top),
            width=int(width),
            height=int(height),
            active=active_flag == "1",
        )
    return result


# ---------------------------------------------------------------------------
# Pane-id validation
# ---------------------------------------------------------------------------

_PANE_ID_RE = re.compile(r"^%\d+$")


def _validate_pane_id(pane_id: str) -> None:
    """Reject anything that isn't exactly `%<digits>` to prevent injection."""
    if not _PANE_ID_RE.match(pane_id):
        raise ValueError(
            f"Invalid pane_id {pane_id!r}. Must match %<digits> (e.g. %0, %12)."
        )


# ---------------------------------------------------------------------------
# TmuxClient
# ---------------------------------------------------------------------------

class TmuxClient:
    """Persistent tmux control-mode client.

    Opens one `tmux -C attach-session` subprocess and keeps it alive for the
    lifetime of the daemon.  Commands are sent on stdin; responses are read
    from stdout as %begin...%end blocks.

    Single-threaded: the caller must not share an instance across threads.

    Usage::

        with TmuxClient() as client:
            panes = client.get_panes()
            client.select_pane(panes[some_id].pane_id)
    """

    def __init__(self, socket: str = "") -> None:
        """
        Args:
            socket: Path to the tmux server socket (``-S`` flag).
                    Empty string means use the default socket.
        """
        self._socket = socket
        self._proc: Optional[subprocess.Popen[str]] = None
        self._start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        args = ["tmux"]
        if self._socket:
            args += ["-S", self._socket]
        args += ["-C", "attach-session"]
        self._proc = subprocess.Popen(
            args,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._drain_initial_handshake()

    def _drain_initial_handshake(self) -> None:
        """Consume the response to attach-session itself.

        When ``tmux -C attach-session`` starts it emits a %begin/%end block
        (or %begin/%error if the attach failed) as the response to the
        attach-session command — before we've sent anything. If we don't
        consume it here, the next ``_run_command`` will read this block as
        if it were the response to its own command, returning empty payload.
        """
        proc = self._proc
        assert proc is not None and proc.stdout is not None

        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if line.startswith("%end"):
                return
            if line.startswith("%error"):
                # Read whatever stderr captured for context, then raise.
                stderr_text = ""
                if proc.stderr is not None:
                    try:
                        stderr_text = proc.stderr.read()
                    except Exception:
                        pass
                raise TmuxError(
                    "tmux -C attach-session failed."
                    + (f" stderr: {stderr_text.strip()}" if stderr_text.strip() else "")
                )
            # Otherwise: %begin, notifications, etc. — ignore.
        # stdout closed without %end → tmux exited
        stderr_text = ""
        if proc.stderr is not None:
            try:
                stderr_text = proc.stderr.read()
            except Exception:
                pass
        raise TmuxError(
            "tmux exited before completing control-mode handshake."
            + (f" stderr: {stderr_text.strip()}" if stderr_text.strip() else "")
        )

    def close(self) -> None:
        """Shut down the control-mode subprocess cleanly.

        Closes stdin (signals EOF to tmux), waits up to 2 s, then kills.
        Does NOT issue `kill-session` — that would destroy the user's session.
        """
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        finally:
            self._proc = None

    def __enter__(self) -> "TmuxClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal command dispatch
    # ------------------------------------------------------------------

    def _assert_alive(self) -> subprocess.Popen[str]:
        """Return the live Popen object or raise TmuxError."""
        if self._proc is None:
            raise TmuxError("TmuxClient is closed.")
        ret = self._proc.poll()
        if ret is not None:
            raise TmuxError(
                f"tmux control-mode subprocess exited with code {ret}."
            )
        return self._proc

    def _run_command(self, cmd: str) -> list[str]:
        """Send *cmd* to tmux and return the response payload lines.

        Protocol:
          1. Write ``cmd\\n`` to stdin.
          2. Read stdout lines, discarding notification lines (those that
             start with ``%`` but are not ``%begin``/``%end``/``%error``).
          3. Capture lines between a matching ``%begin`` and ``%end``.
          4. Raise TmuxError if ``%error`` is seen instead of ``%end``.

        Note: the %begin/%end tokens include a sequence number and timestamp
        that we do not try to correlate across concurrent commands — there is
        only ever one in-flight command (single-threaded design).
        """
        proc = self._assert_alive()
        assert proc.stdin is not None, "stdin should be PIPE"
        assert proc.stdout is not None, "stdout should be PIPE"

        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

        payload: list[str] = []
        in_block = False

        for raw in proc.stdout:
            line = raw.rstrip("\n")

            if line.startswith("%begin"):
                in_block = True
                payload = []
                continue

            if line.startswith("%end"):
                if in_block:
                    return payload
                # stray %end — ignore
                continue

            if line.startswith("%error"):
                raise TmuxError(f"tmux returned an error for command {cmd!r}")

            if in_block:
                payload.append(line)
            # else: notification line (e.g. %window-pane-changed) — discard

        # stdout closed before we saw %end
        raise TmuxError(
            "tmux stdout closed unexpectedly while waiting for %end."
        )

    # ------------------------------------------------------------------
    # Public API (implements TmuxClientProto)
    # ------------------------------------------------------------------

    def get_panes(self) -> dict[str, PaneInfo]:
        """Return geometry for all panes across all sessions."""
        lines = self._run_command(
            "list-panes -a -F"
            " '#{pane_id} #{pane_left} #{pane_top}"
            " #{pane_width} #{pane_height} #{pane_active}'"
        )
        return parse_list_panes("\n".join(lines))

    def get_active_pane(self) -> str:
        """Return the pane_id of the currently focused pane."""
        lines = self._run_command("display-message -p '#{pane_id}'")
        if not lines:
            raise TmuxError("display-message returned no output.")
        return lines[0].strip()

    def select_pane(self, pane_id: str) -> None:
        """Focus *pane_id*.

        Raises ValueError for pane_ids that don't match ``%<digits>`` to
        prevent command-injection.
        """
        _validate_pane_id(pane_id)
        self._run_command(f"select-pane -t {pane_id}")
