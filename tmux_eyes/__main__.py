"""tmux-eyes daemon entry point.

Wires camera → vision → classifier → tmux_io into a single loop.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from .backend import BackendError, select_backend
from .camera import Camera, CameraError
from .classifier import Classifier
from .config import Config
from .tmux_io import TmuxError
from .types import MultiplexerClientProto, PaneInfo, SwitchDecision
from .vision import FaceTracker

logger = logging.getLogger("tmux_eyes")

PANE_REFRESH_INTERVAL_MS = 1000  # how often to re-list panes (geometry rarely changes)
ACTIVE_REFRESH_INTERVAL_MS = 100  # how often to query the active pane


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tmux-eyes",
        description="Eye-tracking pane switcher for tmux (head-pose driven).",
    )
    p.add_argument(
        "--log-level",
        default=None,
        help="Override TMUX_EYES_LOG_LEVEL (DEBUG, INFO, WARNING, ERROR).",
    )
    p.add_argument(
        "--show-classification",
        action="store_true",
        help="Log every frame's smoothed classification (verbose).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run vision + classifier but do not actually switch panes.",
    )
    return p.parse_args(argv)


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class _Daemon:
    def __init__(self, config: Config, *, dry_run: bool, verbose_class: bool) -> None:
        self._cfg = config
        self._dry_run = dry_run
        self._verbose_class = verbose_class
        self._stop = False

        # Cached tmux state.
        self._panes: dict[str, PaneInfo] = {}
        self._active_pane: str = ""
        self._last_pane_refresh_ms: int = 0
        self._last_active_refresh_ms: int = 0

    def stop(self) -> None:
        self._stop = True

    def run(self) -> int:
        logger.info("starting tmux-eyes (dry_run=%s)", self._dry_run)

        mux = select_backend(self._cfg)
        with Camera(
            device=self._cfg.camera_device,
            width=self._cfg.camera_width,
            height=self._cfg.camera_height,
            fps=self._cfg.camera_fps,
        ) as cam:
            tracker = FaceTracker(model_path=self._cfg.face_model_path)
            classifier = Classifier(self._cfg)

            try:
                self._refresh_tmux_state(mux, force=True)
                if not self._panes:
                    logger.error("no panes found — is your multiplexer running?")
                    return 2

                logger.info(
                    "ready: %d panes, active=%s",
                    len(self._panes),
                    self._active_pane,
                )

                for frame in cam.frames():
                    if self._stop:
                        break

                    signal_ = tracker.process(frame)
                    self._refresh_tmux_state(mux)

                    decision = classifier.update(
                        signal_, self._active_pane, self._panes
                    )

                    if self._verbose_class:
                        logger.debug(
                            "yaw=%+6.1f° iris=%s detected=%s active=%s decision=%s",
                            signal_.yaw_deg,
                            f"{signal_.iris_ratio:.2f}" if signal_.iris_ratio is not None else "—",
                            signal_.detected,
                            self._active_pane,
                            decision.target_pane_id if decision else "—",
                        )

                    if decision is not None:
                        self._handle_decision(mux, decision)

            finally:
                tracker.close()
                mux.close()

        logger.info("shutdown clean")
        return 0

    def _refresh_tmux_state(self, tmux: MultiplexerClientProto, *, force: bool = False) -> None:
        now_ms = int(time.monotonic() * 1000)

        if force or now_ms - self._last_pane_refresh_ms >= PANE_REFRESH_INTERVAL_MS:
            try:
                self._panes = tmux.get_panes()
            except RuntimeError:  # TmuxError / WeztermError both subclass RuntimeError
                if force:
                    raise  # surface the real error during startup
                logger.warning("get_panes failed (will retry next cycle)")
            self._last_pane_refresh_ms = now_ms

        if force or now_ms - self._last_active_refresh_ms >= ACTIVE_REFRESH_INTERVAL_MS:
            try:
                self._active_pane = tmux.get_active_pane()
            except RuntimeError:
                if force:
                    raise
                logger.warning("get_active_pane failed (will retry next cycle)")
            self._last_active_refresh_ms = now_ms

    def _handle_decision(self, tmux: MultiplexerClientProto, decision: SwitchDecision) -> None:
        logger.info(
            "switch %s → %s  (%s)",
            self._active_pane,
            decision.target_pane_id,
            decision.reason,
        )
        if self._dry_run:
            return
        try:
            tmux.select_pane(decision.target_pane_id)
            # Optimistic local update — the next refresh will reconcile.
            self._active_pane = decision.target_pane_id
        except (RuntimeError, ValueError) as exc:
            logger.error("select_pane failed: %s", exc)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = Config.from_env()
    log_level = args.log_level or config.log_level
    setup_logging(log_level)

    daemon = _Daemon(
        config,
        dry_run=args.dry_run,
        verbose_class=args.show_classification,
    )

    def _on_signal(signum: int, _frame: object) -> None:
        logger.info("received signal %d, shutting down", signum)
        daemon.stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        return daemon.run()
    except CameraError as exc:
        logger.error("camera error: %s", exc)
        return 3
    except TmuxError as exc:
        logger.error("tmux error: %s", exc)
        return 4
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
