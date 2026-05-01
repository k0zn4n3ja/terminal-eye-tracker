# tmux-eyes

Webcam-driven pane switcher for tmux. Look at the other pane and focus follows.

> **Status:** Phase 1 POC. Two-pane left/right switching using head-pose yaw.
> Iris-gaze confirmation and per-user calibration are Phase 2; multi-pane
> geometry is Phase 3. See [`sol_arch.md`](./sol_arch.md) for the full design.

## How it works

```
webcam → MediaPipe FaceLandmarker → head-pose yaw (solvePnP)
       → EMA smoothing + 750ms dwell + cool-down hysteresis
       → tmux -C select-pane -t <pane>
```

Head pose is the primary signal (more robust through glasses, immune to
reading saccades). Iris gaze is wired in as an opt-in confirmation signal
(`TMUX_EYES_USE_IRIS_CONFIRMATION=1`).

## Requirements

- Linux, X11 (Wayland not yet validated — Phase 3).
- A webcam reachable at `/dev/video0` (override via `TMUX_EYES_CAMERA_DEVICE`).
- tmux running with at least one session.
- Python 3.11 (3.13 not yet supported by MediaPipe).

On NixOS, the project ships a `shell.nix` that pulls in `uv`, `python311`,
`tmux`, and the runtime libraries MediaPipe / OpenCV need at load time
(via `NIX_LD_LIBRARY_PATH`). `programs.nix-ld.enable = true;` must be on
in your system config (it usually is if you run any prebuilt binaries).

## Quickstart

```bash
cd /home/nixdt1/Repos/vectorise/tmux_eye_tracking
nix-shell                       # creates .venv and runs `uv sync` on first entry
pytest                          # run the unit tests
python -m tmux_eyes             # start the daemon
```

To watch the classification stream while debugging:

```bash
python -m tmux_eyes --show-classification --log-level DEBUG
```

To validate the pipeline end-to-end without actually switching panes:

```bash
python -m tmux_eyes --dry-run
```

## Configuration

All knobs are environment variables prefixed `TMUX_EYES_`. Defaults live in
`tmux_eyes/config.py` and come from the design doc.

| Variable | Default | Purpose |
|---|---|---|
| `TMUX_EYES_CAMERA_DEVICE` | `0` | V4L2 device index |
| `TMUX_EYES_CAMERA_WIDTH` | `640` | capture width |
| `TMUX_EYES_CAMERA_HEIGHT` | `480` | capture height |
| `TMUX_EYES_CAMERA_FPS` | `30` | requested FPS |
| `TMUX_EYES_YAW_LEFT_DEG` | `-12.0` | yaw threshold for LEFT |
| `TMUX_EYES_YAW_RIGHT_DEG` | `12.0` | yaw threshold for RIGHT |
| `TMUX_EYES_EMA_ALPHA` | `0.35` | smoothing factor (higher = more responsive) |
| `TMUX_EYES_DWELL_MS` | `750` | hold duration before switching |
| `TMUX_EYES_COOLDOWN_MS` | `250` | min gap between switches |
| `TMUX_EYES_USE_IRIS_CONFIRMATION` | `0` | set to `1` to require iris+head-pose agreement |
| `TMUX_EYES_FACE_MODEL_PATH` | `""` | path to a `.task` model; empty = legacy FaceMesh auto-download |
| `TMUX_EYES_LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |

## Layout

```
tmux_eyes/
├── __main__.py    # daemon glue (camera → vision → classifier → tmux_io)
├── camera.py      # cv2 V4L2 capture
├── vision.py      # MediaPipe + head-pose math
├── classifier.py  # EMA + dwell state machine (pure logic, no I/O)
├── tmux_io.py     # persistent tmux -C control-mode client
├── config.py      # env-driven thresholds
└── types.py       # shared dataclasses + protocols

tests/
├── test_vision.py     # math helpers (no MediaPipe required)
├── test_classifier.py # state machine, deterministic timestamps
└── test_tmux_io.py    # control-mode parser + injection guards
```

## Tuning

If the daemon switches too aggressively while you read:
- raise `TMUX_EYES_DWELL_MS` to 900–1000
- raise `TMUX_EYES_YAW_LEFT_DEG`/`YAW_RIGHT_DEG` magnitudes (e.g. ±18°)
- enable `TMUX_EYES_USE_IRIS_CONFIRMATION=1`

If the daemon feels sluggish:
- lower `TMUX_EYES_DWELL_MS` to 500–600
- raise `TMUX_EYES_EMA_ALPHA` to 0.5

## Roadmap

- **Phase 2** — visual dwell feedback (`select-pane -P bg=…`), 2-point per-user
  calibration, iris confirmation as default-on.
- **Phase 3** — N-pane support via pixel→cell mapping (`xdotool`/`swaymsg`/
  `hyprctl`), multi-monitor.

## Notes

- Voice / Wispr-Flow integration is intentionally out of scope. The eye
  tracker only ensures the right pane is focused; whatever types into stdin
  is downstream and orthogonal.
- `select-pane` is hardened against injection (pane-id regex `^%\d+$`).
- The control-mode client doesn't react to `%window-pane-changed`
  notifications yet — geometry is refreshed on a 1 s cadence instead. That's
  noted as a Phase 2 nicety in `sol_arch.md` §7.1.
