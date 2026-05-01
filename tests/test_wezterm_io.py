"""Tests for tmux_eyes/wezterm_io.py.

All WeztermClient tests mock subprocess.run — no real wezterm process needed.
parse_wezterm_list tests are pure-function unit tests with no external deps.
"""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from tmux_eyes.types import PaneInfo
from tmux_eyes.wezterm_io import (
    WeztermClient,
    WeztermError,
    parse_wezterm_list,
)


# ---------------------------------------------------------------------------
# Canonical JSON fixture
# ---------------------------------------------------------------------------

CANONICAL_JSON = (
    '['
    '{"window_id":0,"tab_id":0,"pane_id":1,'
    '"size":{"rows":47,"cols":80},"left_col":0,"top_row":0,"is_active":true},'
    '{"window_id":0,"tab_id":0,"pane_id":2,'
    '"size":{"rows":47,"cols":80},"left_col":80,"top_row":0,"is_active":false}'
    ']'
)


def _make_completed_process(
    stdout: str = CANONICAL_JSON,
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Build a CompletedProcess stub for subprocess.run mocking."""
    cp: subprocess.CompletedProcess[str] = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.stderr = ""
    cp.returncode = returncode
    return cp


# ---------------------------------------------------------------------------
# Test 1-4: parse_wezterm_list — pure function tests
# ---------------------------------------------------------------------------

class TestParseWeztermList(unittest.TestCase):

    # Test 1: canonical two-pane JSON
    def test_canonical_two_panes(self) -> None:
        result = parse_wezterm_list(CANONICAL_JSON)
        self.assertEqual(len(result), 2)

        p1 = result["1"]
        self.assertIsInstance(p1, PaneInfo)
        self.assertEqual(p1.pane_id, "1")
        self.assertEqual(p1.left, 0)
        self.assertEqual(p1.top, 0)
        self.assertEqual(p1.width, 80)
        self.assertEqual(p1.height, 47)
        self.assertTrue(p1.active)

        p2 = result["2"]
        self.assertIsInstance(p2, PaneInfo)
        self.assertEqual(p2.pane_id, "2")
        self.assertEqual(p2.left, 80)
        self.assertEqual(p2.top, 0)
        self.assertEqual(p2.width, 80)
        self.assertEqual(p2.height, 47)
        self.assertFalse(p2.active)

    # Test 2: empty array → empty dict
    def test_empty_array(self) -> None:
        result = parse_wezterm_list("[]")
        self.assertEqual(result, {})

    # Test 3: malformed JSON → raises WeztermError
    def test_malformed_json_raises(self) -> None:
        with self.assertRaises(WeztermError):
            parse_wezterm_list("not valid json {[}")

    # Test 4: entry missing required field → raises WeztermError
    def test_missing_required_field_raises(self) -> None:
        # Missing 'is_active'
        json_text = (
            '[{"window_id":0,"tab_id":0,"pane_id":1,'
            '"size":{"rows":47,"cols":80},"left_col":0,"top_row":0}]'
        )
        with self.assertRaises(WeztermError):
            parse_wezterm_list(json_text)

    def test_missing_size_field_raises(self) -> None:
        # size dict missing 'cols'
        json_text = (
            '[{"window_id":0,"tab_id":0,"pane_id":1,'
            '"size":{"rows":47},"left_col":0,"top_row":0,"is_active":true}]'
        )
        with self.assertRaises(WeztermError):
            parse_wezterm_list(json_text)


# ---------------------------------------------------------------------------
# Test 5-8: select_pane — injection prevention + valid ids
# ---------------------------------------------------------------------------

class TestSelectPane(unittest.TestCase):

    # Test 5: valid numeric pane_id calls subprocess correctly
    def test_select_pane_valid_calls_subprocess(self) -> None:
        with patch("tmux_eyes.wezterm_io.subprocess.run") as mock_run:
            client = WeztermClient()
            client.select_pane("42")
            mock_run.assert_called_once_with(
                ["wezterm", "cli", "activate-pane", "--pane-id", "42"],
                check=False,
            )

    # Test 6: injection attempt raises ValueError, no subprocess call
    def test_select_pane_injection_raises(self) -> None:
        with patch("tmux_eyes.wezterm_io.subprocess.run") as mock_run:
            client = WeztermClient()
            with self.assertRaises(ValueError):
                client.select_pane("1; rm -rf /")
            mock_run.assert_not_called()

    # Test 7: % prefix (tmux format) raises ValueError
    def test_select_pane_percent_prefix_raises(self) -> None:
        with patch("tmux_eyes.wezterm_io.subprocess.run") as mock_run:
            client = WeztermClient()
            with self.assertRaises(ValueError):
                client.select_pane("%1")
            mock_run.assert_not_called()

    # Test 8: empty string raises ValueError
    def test_select_pane_empty_string_raises(self) -> None:
        with patch("tmux_eyes.wezterm_io.subprocess.run") as mock_run:
            client = WeztermClient()
            with self.assertRaises(ValueError):
                client.select_pane("")
            mock_run.assert_not_called()

    def test_select_pane_accepts_zero(self) -> None:
        with patch("tmux_eyes.wezterm_io.subprocess.run") as mock_run:
            client = WeztermClient()
            client.select_pane("0")
            mock_run.assert_called_once_with(
                ["wezterm", "cli", "activate-pane", "--pane-id", "0"],
                check=False,
            )


# ---------------------------------------------------------------------------
# Test 9-11: get_panes
# ---------------------------------------------------------------------------

class TestGetPanes(unittest.TestCase):

    # Test 9: mock subprocess.run returns canonical JSON, verify panes parsed
    def test_get_panes_happy_path(self) -> None:
        cp = _make_completed_process(stdout=CANONICAL_JSON, returncode=0)
        with patch("tmux_eyes.wezterm_io.subprocess.run", return_value=cp):
            client = WeztermClient()
            panes = client.get_panes()
        self.assertIn("1", panes)
        self.assertIn("2", panes)
        self.assertTrue(panes["1"].active)
        self.assertFalse(panes["2"].active)
        self.assertEqual(panes["1"].width, 80)
        self.assertEqual(panes["1"].height, 47)

    # Test 10: non-zero exit code → raises WeztermError
    def test_get_panes_nonzero_exit_raises(self) -> None:
        cp = _make_completed_process(stdout="", returncode=1)
        cp.stderr = "some error"
        with patch("tmux_eyes.wezterm_io.subprocess.run", return_value=cp):
            client = WeztermClient()
            with self.assertRaises(WeztermError):
                client.get_panes()

    # Test 11: wezterm binary not found → raises WeztermError with helpful message
    def test_get_panes_binary_not_found_raises(self) -> None:
        with patch(
            "tmux_eyes.wezterm_io.subprocess.run",
            side_effect=FileNotFoundError("wezterm not found"),
        ):
            client = WeztermClient()
            with self.assertRaises(WeztermError) as ctx:
                client.get_panes()
        self.assertIn("PATH", str(ctx.exception))

    def test_get_panes_empty_output_raises(self) -> None:
        cp = _make_completed_process(stdout="   ", returncode=0)
        with patch("tmux_eyes.wezterm_io.subprocess.run", return_value=cp):
            client = WeztermClient()
            with self.assertRaises(WeztermError):
                client.get_panes()


# ---------------------------------------------------------------------------
# Test 12-13: get_active_pane
# ---------------------------------------------------------------------------

class TestGetActivePane(unittest.TestCase):

    # Test 12: returns id of the active pane
    def test_get_active_pane_returns_active_id(self) -> None:
        cp = _make_completed_process(stdout=CANONICAL_JSON, returncode=0)
        with patch("tmux_eyes.wezterm_io.subprocess.run", return_value=cp):
            client = WeztermClient()
            active = client.get_active_pane()
        # pane_id 1 has is_active=true in CANONICAL_JSON
        self.assertEqual(active, "1")

    # Test 13: no active pane → raises WeztermError
    def test_get_active_pane_none_active_raises(self) -> None:
        json_text = (
            '[{"window_id":0,"tab_id":0,"pane_id":1,'
            '"size":{"rows":47,"cols":80},"left_col":0,"top_row":0,"is_active":false},'
            '{"window_id":0,"tab_id":0,"pane_id":2,'
            '"size":{"rows":47,"cols":80},"left_col":80,"top_row":0,"is_active":false}]'
        )
        cp = _make_completed_process(stdout=json_text, returncode=0)
        with patch("tmux_eyes.wezterm_io.subprocess.run", return_value=cp):
            client = WeztermClient()
            with self.assertRaises(WeztermError):
                client.get_active_pane()


# ---------------------------------------------------------------------------
# Context manager compliance
# ---------------------------------------------------------------------------

class TestContextManager(unittest.TestCase):

    def test_context_manager_returns_self(self) -> None:
        client = WeztermClient()
        with client as c:
            self.assertIs(c, client)

    def test_close_is_noop(self) -> None:
        client = WeztermClient()
        client.close()  # should not raise
        client.close()  # idempotent


if __name__ == "__main__":
    unittest.main()
