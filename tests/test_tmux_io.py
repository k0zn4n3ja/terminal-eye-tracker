"""Tests for tmux_eyes/tmux_io.py.

All TmuxClient tests mock subprocess.Popen so no real tmux process is needed.
parse_list_panes tests are pure-function unit tests with no external deps.

Malformed-line policy: parse_list_panes raises ValueError on any line that
doesn't match the expected format (documented in tmux_io.parse_list_panes).
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

from tmux_eyes.types import PaneInfo
from tmux_eyes.tmux_io import (
    TmuxClient,
    TmuxError,
    parse_list_panes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stdout(*lines: str) -> io.StringIO:
    """Return a StringIO that yields the given lines (each terminated by \\n)."""
    return io.StringIO("".join(line + "\n" for line in lines))


# Initial handshake block emitted by `tmux -C attach-session` itself —
# `_drain_initial_handshake` must consume this before any user command.
_HANDSHAKE_LINES = ["%begin 1 0 0", "%end 1 0 0"]


def _make_mock_proc(stdout_lines: list[str]) -> MagicMock:
    """Build a Popen mock with a synthetic attach-session handshake prepended."""
    proc = MagicMock()
    proc.poll.return_value = None  # process is alive
    proc.stdin = MagicMock()
    all_lines = _HANDSHAKE_LINES + stdout_lines
    proc.stdout = iter(line + "\n" for line in all_lines)
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""
    return proc


# ---------------------------------------------------------------------------
# parse_list_panes — pure function tests
# ---------------------------------------------------------------------------

CANONICAL_OUTPUT = "%0 0 0 80 24 1\n%1 80 0 80 24 0\n"


class TestParseListPanes(unittest.TestCase):

    # Test 1: canonical two-pane output
    def test_canonical_two_panes(self) -> None:
        result = parse_list_panes(CANONICAL_OUTPUT)
        self.assertEqual(len(result), 2)

        p0 = result["%0"]
        self.assertIsInstance(p0, PaneInfo)
        self.assertEqual(p0.pane_id, "%0")
        self.assertEqual(p0.left, 0)
        self.assertEqual(p0.top, 0)
        self.assertEqual(p0.width, 80)
        self.assertEqual(p0.height, 24)
        self.assertTrue(p0.active)

        p1 = result["%1"]
        self.assertEqual(p1.pane_id, "%1")
        self.assertEqual(p1.left, 80)
        self.assertEqual(p1.top, 0)
        self.assertEqual(p1.width, 80)
        self.assertEqual(p1.height, 24)
        self.assertFalse(p1.active)

    # Test 2: extra whitespace and trailing newlines
    def test_extra_whitespace_and_trailing_newlines(self) -> None:
        messy = "  %0   0   0   80   24   1  \n\n  %1  80  0  80  24  0  \n\n\n"
        result = parse_list_panes(messy)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["%0"].active)
        self.assertFalse(result["%1"].active)

    # Test 3: empty string → empty dict
    def test_empty_string(self) -> None:
        result = parse_list_panes("")
        self.assertEqual(result, {})

    # Test 3b: whitespace-only string → empty dict
    def test_whitespace_only(self) -> None:
        result = parse_list_panes("   \n\n  \n")
        self.assertEqual(result, {})

    # Test 4: malformed line raises ValueError
    def test_malformed_line_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_list_panes("not a pane line at all\n")

    def test_malformed_line_too_few_fields(self) -> None:
        with self.assertRaises(ValueError):
            parse_list_panes("%0 0 0 80\n")

    def test_malformed_line_non_numeric_geometry(self) -> None:
        with self.assertRaises(ValueError):
            parse_list_panes("%0 0 0 eighty 24 1\n")

    def test_malformed_line_bad_pane_id(self) -> None:
        # pane_id must start with %
        with self.assertRaises(ValueError):
            parse_list_panes("0 0 0 80 24 1\n")


# ---------------------------------------------------------------------------
# select_pane — injection prevention + valid ids
# ---------------------------------------------------------------------------

class TestSelectPane(unittest.TestCase):

    def _make_client_with_mock_run(self) -> tuple[TmuxClient, MagicMock]:
        """Return a TmuxClient whose _run_command is replaced by a Mock."""
        with patch("tmux_eyes.tmux_io.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_mock_proc([])
            client = TmuxClient()
        mock_run = MagicMock(return_value=[])
        client._run_command = mock_run  # type: ignore[method-assign]
        return client, mock_run

    # Test 5a: valid id calls _run_command with correct argument
    def test_select_pane_valid_calls_run_command(self) -> None:
        client, mock_run = self._make_client_with_mock_run()
        client.select_pane("%1")
        mock_run.assert_called_once_with("select-pane -t %1")

    # Test 5b: injection attempt raises ValueError, no _run_command call
    def test_select_pane_injection_raises(self) -> None:
        client, mock_run = self._make_client_with_mock_run()
        with self.assertRaises(ValueError):
            client.select_pane("%1; rm -rf /")
        mock_run.assert_not_called()

    def test_select_pane_empty_string_raises(self) -> None:
        client, mock_run = self._make_client_with_mock_run()
        with self.assertRaises(ValueError):
            client.select_pane("")
        mock_run.assert_not_called()

    def test_select_pane_no_percent_raises(self) -> None:
        client, mock_run = self._make_client_with_mock_run()
        with self.assertRaises(ValueError):
            client.select_pane("1")
        mock_run.assert_not_called()

    # Test 6: valid pane ids %0, %1, %999
    def test_select_pane_accepts_percent_zero(self) -> None:
        client, mock_run = self._make_client_with_mock_run()
        client.select_pane("%0")
        mock_run.assert_called_once_with("select-pane -t %0")

    def test_select_pane_accepts_percent_large(self) -> None:
        client, mock_run = self._make_client_with_mock_run()
        client.select_pane("%999")
        mock_run.assert_called_once_with("select-pane -t %999")


# ---------------------------------------------------------------------------
# _run_command — control-mode parsing tests
# ---------------------------------------------------------------------------

class TestRunCommand(unittest.TestCase):

    def _client_with_stdout(self, stdout_lines: list[str]) -> TmuxClient:
        """Construct a TmuxClient whose subprocess stdout is pre-baked."""
        with patch("tmux_eyes.tmux_io.subprocess.Popen") as mock_popen:
            proc = _make_mock_proc(stdout_lines)
            mock_popen.return_value = proc
            client = TmuxClient()
        # Replace _proc with our mock so subsequent _run_command calls use it
        client._proc = proc  # type: ignore[assignment]
        return client

    # Test 7: happy path — payload between %begin and %end is returned
    def test_run_command_happy_path(self) -> None:
        client = self._client_with_stdout([
            "%begin 1234 1 0",
            "%1",
            "%end 1234 1 0",
        ])
        result = client._run_command("display-message -p '#{pane_id}'")
        self.assertEqual(result, ["%1"])

    # Test 7b: multiple payload lines
    def test_run_command_multiple_payload_lines(self) -> None:
        client = self._client_with_stdout([
            "%begin 1234 1 0",
            "%0 0 0 80 24 1",
            "%1 80 0 80 24 0",
            "%end 1234 1 0",
        ])
        result = client._run_command("list-panes -a -F '...'")
        self.assertEqual(result, ["%0 0 0 80 24 1", "%1 80 0 80 24 0"])

    # Test 8: %error raises TmuxError
    def test_run_command_error_raises(self) -> None:
        client = self._client_with_stdout([
            "%begin 1234 1 0",
            "%error 1234 1 0",
        ])
        with self.assertRaises(TmuxError):
            client._run_command("display-message -p '#{pane_id}'")

    # Test 9: notifications before %begin are discarded
    def test_run_command_discards_notifications(self) -> None:
        client = self._client_with_stdout([
            "%window-pane-changed @1 %3",
            "%session-changed $1 mysession",
            "%begin 1234 1 0",
            "%2",
            "%end 1234 1 0",
        ])
        result = client._run_command("display-message -p '#{pane_id}'")
        self.assertEqual(result, ["%2"])

    # Test 9b: notification interleaved inside block is kept as payload
    # (tmux won't do this, but our parser should handle it gracefully —
    # anything between %begin and %end is treated as payload, including lines
    # that look like notifications)
    def test_run_command_empty_payload(self) -> None:
        client = self._client_with_stdout([
            "%begin 1234 1 0",
            "%end 1234 1 0",
        ])
        result = client._run_command("select-pane -t %0")
        self.assertEqual(result, [])

    # Test: stdout closes before %end → TmuxError
    def test_run_command_stdout_closed_early(self) -> None:
        client = self._client_with_stdout([
            "%begin 1234 1 0",
            "partial payload",
            # no %end — stdout ends here
        ])
        with self.assertRaises(TmuxError):
            client._run_command("list-panes -a -F '...'")

    # Test: _assert_alive raises TmuxError when process has exited
    def test_run_command_dead_process_raises(self) -> None:
        # Build the client with a live mock (handshake succeeds),
        # then simulate the process dying afterwards.
        with patch("tmux_eyes.tmux_io.subprocess.Popen") as mock_popen:
            proc = _make_mock_proc([])
            mock_popen.return_value = proc
            client = TmuxClient()
        proc.poll.return_value = 1  # now mark it dead
        with self.assertRaises(TmuxError):
            client._run_command("anything")


# ---------------------------------------------------------------------------
# Integration: get_panes / get_active_pane round-trip
# ---------------------------------------------------------------------------

class TestHighLevelCommands(unittest.TestCase):

    def _client_with_stdout(self, stdout_lines: list[str]) -> TmuxClient:
        with patch("tmux_eyes.tmux_io.subprocess.Popen") as mock_popen:
            proc = _make_mock_proc(stdout_lines)
            mock_popen.return_value = proc
            client = TmuxClient()
        client._proc = proc  # type: ignore[assignment]
        return client

    def test_get_panes_returns_pane_info(self) -> None:
        client = self._client_with_stdout([
            "%begin 1234 1 0",
            "%0 0 0 80 24 1",
            "%1 80 0 80 24 0",
            "%end 1234 1 0",
        ])
        panes = client.get_panes()
        self.assertIn("%0", panes)
        self.assertIn("%1", panes)
        self.assertTrue(panes["%0"].active)
        self.assertFalse(panes["%1"].active)

    def test_get_active_pane_returns_stripped_id(self) -> None:
        client = self._client_with_stdout([
            "%begin 1234 1 0",
            "%3",
            "%end 1234 1 0",
        ])
        active = client.get_active_pane()
        self.assertEqual(active, "%3")

    def test_get_active_pane_empty_response_raises(self) -> None:
        client = self._client_with_stdout([
            "%begin 1234 1 0",
            "%end 1234 1 0",
        ])
        with self.assertRaises(TmuxError):
            client.get_active_pane()


if __name__ == "__main__":
    unittest.main()
