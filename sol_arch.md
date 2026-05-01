# tmux Eye-Tracking — Solution Architecture (POC)

> **Status:** Draft v0.1 · 2026-05-01
> **Scope:** Proof-of-concept that switches the active tmux pane (left ↔ right split) based on where the user is looking, using only a standard laptop webcam on Linux.
> **Out of scope (for now):** Wispr Flow / dictation integration — voice input is orthogonal and the user will own that layer separately. The eye-tracker only needs to make sure the correct pane is focused; whatever types into stdin (keyboard, dictation, etc.) is downstream.

---

## 1. Goal

Build the smallest credible daemon that:

1. Reads a webcam frame.
2. Decides "is the user looking at the LEFT pane or the RIGHT pane?"
3. Switches the focused tmux pane to match — fast enough that it feels like the cursor follows the eyes, not like a delayed reaction.

Success criterion for the POC: in a side-by-side two-pane tmux layout, glancing at the other pane causes focus to switch within ~1 second of fixating, with no false switches while reading either side.

---

## 2. Non-Goals (POC)

- Continuous (x, y) gaze coordinates on the screen — we only need a region classifier.
- Multi-monitor or multi-window tracking.
- N-pane layouts beyond left/right (deferred to Phase 3).
- Per-user calibration as a hard requirement.
- Any voice / dictation / Wispr integration.
- Cross-platform support — Linux first (X11 and Wayland both targeted; Wayland is bonus).

---

## 3. Headline Decisions (the "why" matters)

| # | Decision | Why |
|---|---|---|
| 1 | **MediaPipe FaceLandmarker (Tasks API)** for vision | 30 FPS on CPU, ships iris landmarks, Apache-2.0, actively maintained on PyPI (0.10.35, Apr 2026). Beats GazeTracking on robustness and L2CS-Net on latency. |
| 2 | **Head-pose yaw as the primary L/R signal**, not iris gaze | More robust to glasses, lighting, and small saccades. A user "switching attention" naturally rotates their head a few degrees — we exploit that. Iris-only gaze is twitchy and fails on people who eye-flick without head movement. |
| 3 | **Iris-ratio gaze as a *confirmation* signal**, not a trigger | Reduces false positives near the threshold by requiring agreement between two independent signals. |
| 4 | **tmux control mode (`tmux -C`) over a persistent socket**, not subprocess `tmux select-pane` per frame | Eliminates ~5–20 ms fork+exec overhead per command and gives us a free push-stream of `%window-pane-changed` events so we know the current state without polling. |
| 5 | **Dwell-based activation, 700–800 ms**, with hysteresis | Empirically validated sweet spot from gaze-UX literature. Anything &lt; 500 ms triggers on saccades during reading; &gt; 1 s feels sluggish. |
| 6 | **Calibration-free MVP, optional 2-point calibration as a refinement** | Geometric iris-corner ratio is normalized and works across most users at the L/R-only granularity we need. We add calibration only if false-trigger rate is unacceptable. |
| 7 | **Python**, single process, no microservices | A POC is one file. Vision libs are Python-native. The whole daemon should fit in ~300 lines. |

---

## 4. Stack

```
Language:        Python 3.11 (3.9–3.12 supported by MediaPipe; 3.13 not yet)
Vision:          mediapipe == 0.10.35      # FaceLandmarker (Tasks API), iris + 478 landmarks
Camera I/O:      opencv-python-headless    # cv2.VideoCapture(..., cv2.CAP_V4L2)
Math:            numpy                     # solvePnP for head pose
tmux IPC:        libtmux (optional) OR raw subprocess to a persistent `tmux -C` pipe
Window geom:     xdotool (X11) | swaymsg (Sway/Wayland) | hyprctl (Hyprland)
Packaging:       venv or nix-shell         # MediaPipe ships its own .so files,
                                           # no system OpenCV / dlib version conflicts
```

Why no PyTorch / L2CS-Net: ResNet-50 CPU inference is 5–8 FPS — too slow for sub-200 ms feel. Held in reserve for Phase 3 if continuous gaze coordinates are needed.

Why no GazeTracking (`antoinelame/GazeTracking`): API is friendly (`is_left()`/`is_right()`) but the dlib HOG face detector is ~2× slower than MediaPipe and significantly less robust to glasses / dim light.

Why no OpenFace, WebGazer, Pupil Core: install pain, latency, hardware coupling — see the research notes for full reasons.

---

## 5. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  tmux_eyes daemon  (single Python process)                         │
│                                                                    │
│   [ webcam ]                                                       │
│       │  V4L2, 30 fps, 640x480                                    │
│       ▼                                                            │
│   ┌────────────────────┐                                           │
│   │ MediaPipe Tasks    │  LIVE_STREAM mode → async callback        │
│   │ FaceLandmarker     │  (lowest-latency path, no per-frame wait) │
│   └─────────┬──────────┘                                           │
│             │ 478 landmarks + iris (468–477) per frame             │
│             ▼                                                      │
│   ┌────────────────────┐    ┌────────────────────┐                 │
│   │ HEAD-POSE          │    │ IRIS GAZE          │                 │
│   │ solvePnP →         │    │ x-ratio per eye →  │                 │
│   │ yaw (deg)          │    │ avg ratio          │                 │
│   └─────────┬──────────┘    └──────────┬─────────┘                 │
│             │   primary                │  confirmation             │
│             ▼                          ▼                           │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │ FUSION + EMA SMOOTHING  (α=0.35)                           │   │
│   │ classify → {LEFT, CENTER, RIGHT, UNKNOWN}                  │   │
│   └─────────────────────────┬──────────────────────────────────┘   │
│                             ▼                                      │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │ DWELL STATE MACHINE                                        │   │
│   │   candidate ≠ active && dwell ≥ 750ms  →  emit switch      │   │
│   │   any class change  →  reset dwell timer                   │   │
│   │   hysteresis: don't switch back inside 250ms cool-down     │   │
│   └─────────────────────────┬──────────────────────────────────┘   │
│                             ▼                                      │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │ tmux CONTROL-MODE CLIENT  (persistent `tmux -C` pipe)      │   │
│   │   cmd:    select-pane -t <pane_id>                         │   │
│   │   event:  %window-pane-changed → updates local state cache │   │
│   └────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### 5.1 Module layout (proposed)

```
tmux_eye_tracking/
├── sol_arch.md                  ← this file
├── pyproject.toml
├── tmux_eyes/
│   ├── __main__.py              # CLI: `python -m tmux_eyes`
│   ├── camera.py                # cv2 capture, frame pacing
│   ├── vision.py                # MediaPipe wrapper, head-pose, iris ratio
│   ├── classifier.py            # fusion + EMA + dwell state machine
│   ├── tmux_io.py               # control-mode client, pane geometry cache
│   ├── calibration.py           # optional 2-point calibration (Phase 2)
│   └── config.py                # thresholds, paths
└── tests/
    └── ...
```

---

## 6. Signal Pipeline — Detail

### 6.1 Head-pose yaw (primary)

Using 6 stable face landmarks (nose tip, chin, eye outer corners, mouth corners), pass them with a generic 3D face model into `cv2.solvePnP`. Extract Euler yaw.

Decision rule (initial, tunable):

```
yaw < -12°  →  LEFT
yaw >  12°  →  RIGHT
otherwise   →  CENTER
```

Why head pose first: works through glasses, isn't fooled by reading saccades, runs at full FPS, and matches the natural "I'm looking over there" gesture. Commercial head-mouse tools (eViacam) ship at this fidelity already.

### 6.2 Iris-ratio gaze (confirmation)

For each eye, compute:

```
ratio = (iris_center_x - inner_corner_x) / (outer_corner_x - inner_corner_x)
# average left and right eye ratios
```

Decision rule:

```
ratio < 0.42  →  LEFT
ratio > 0.58  →  RIGHT
otherwise     →  CENTER
```

Landmark indices: iris centers `468` (left eye) and `473` (right eye); eye corners `33/133` and `362/263`.

### 6.3 Fusion

```
final = head_pose          # primary
if head_pose != iris_gaze and iris_gaze != CENTER:
    final = CENTER         # disagreement → suppress, require alignment
```

Then EMA-smooth a numeric form (LEFT=-1, CENTER=0, RIGHT=+1) with α=0.35 and re-discretize.

### 6.4 Dwell state machine

```
state:
  active_pane:    %1
  candidate:      None
  candidate_since: t0

each frame:
  cls = fusion()
  target = LEFT_PANE_ID if cls == LEFT else RIGHT_PANE_ID if cls == RIGHT else None

  if target is None or target == active_pane:
    candidate = None; continue

  if candidate != target:
    candidate = target; candidate_since = now; continue

  if (now - candidate_since) >= DWELL_MS:
    if (now - last_switch) >= COOLDOWN_MS:
      tmux.select_pane(target)
      active_pane = target
      last_switch = now
      candidate = None
```

Defaults: `DWELL_MS=750`, `COOLDOWN_MS=250`.

---

## 7. tmux Integration

### 7.1 Control mode

Open one persistent `tmux -C attach-session -t <session>` subprocess. Feed it commands on stdin, parse notifications from stdout.

Commands the daemon issues:
```
select-pane -t %3
display-message -p '#{pane_id}'                   # initial sync
list-panes -F '#{pane_id} #{pane_left} #{pane_top} #{pane_width} #{pane_height} #{pane_active}'
```

Notifications the daemon consumes:
```
%window-pane-changed @<window> %<pane>            # someone (incl. us) switched panes
%layout-change                                     # geometry invalidated → re-list-panes
```

### 7.2 Identifying "left" vs "right" pane

For the two-pane MVP we don't need pixel-to-cell mapping. We just need to know *which pane is on the left* and *which is on the right* in the active window. The pane with the smaller `pane_left` is LEFT; the other is RIGHT. Cache the mapping; refresh on `%layout-change`.

For Phase 3 (N panes), we extend this to: gaze direction (continuous angle) → screen pixel → terminal cell → pane. That requires the window-geometry tools (xdotool/swaymsg/hyprctl) listed in §4.

### 7.3 Why not poll `tmux display-message` 30×/sec

Each shell-out costs 5–20 ms on a busy laptop. At 30 FPS that's up to 600 ms/sec of wasted CPU and adds variable latency to the control loop. Control mode is one persistent connection with push notifications.

---

## 8. Phased POC Plan

### Phase 1 — "Glance switching" (target: 1 day)

- [ ] `camera.py`: open V4L2 webcam, yield frames at 30 FPS.
- [ ] `vision.py`: MediaPipe FaceLandmarker (LIVE_STREAM mode), extract head-pose yaw.
- [ ] `classifier.py`: yaw threshold + EMA + dwell state machine. Hardcode `DWELL_MS=750`.
- [ ] `tmux_io.py`: persistent `tmux -C` client, `select-pane -t %N`, listen for `%window-pane-changed`.
- [ ] `__main__.py`: glue, env-driven config, signal handlers for clean shutdown.
- [ ] **Manual test:** open two panes, look left → focus left pane, look right → focus right pane. Read a paragraph in the right pane without triggering a switch. Measure end-to-end latency with a stopwatch.

**Done when:** the demo works on the user's laptop, false-positive rate during reading is &lt; 1 switch/min.

### Phase 2 — Robustness

- [ ] Add iris-ratio confirmation signal and the disagreement-suppression rule from §6.3.
- [ ] Optional 2-point calibration: 5-second "look left" / "look right" capture that fits per-user yaw thresholds.
- [ ] Visual feedback during dwell (`tmux select-pane -P 'bg=...'` on the candidate pane, restore on switch or cancel).
- [ ] Logging + a small TUI status line (current class, dwell %, last switch) for debugging.

### Phase 3 — Beyond two panes

- [ ] Pixel-to-cell mapping (xdotool / swaymsg / hyprctl).
- [ ] Continuous gaze coords via L2CS-MobileNet (CPU-friendly variant) when N&gt;2 panes.
- [ ] Multi-monitor support (per-display offset).

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Webcam latency / driver jitter on Linux | M | M | Force `cv2.CAP_V4L2`; measure end-to-end latency with timestamped frames; use lowest-resolution mode that gives good landmarks (640×480). |
| False switches during reading saccades | H (without dwell) | H | Dwell state machine + hysteresis cool-down. Tune `DWELL_MS`. |
| Glasses / lighting break iris detection | M | L (head pose still works) | Head-pose primary, iris secondary — system degrades gracefully. |
| User has eye-only attention (doesn't move head) | M | M | Phase 2 weights iris ratio more heavily; or expose a per-user "eye-dominant" config flag. |
| Wayland vs X11 window geometry differences | L (Phase 1) / M (Phase 3) | L | Phase 1 doesn't need it. Phase 3 adds a thin compositor-detection adapter. |
| MediaPipe Python 3.13 incompatibility | L | L | Pin to 3.11 in `pyproject.toml`. |
| NixOS dynamic-linker friction (MediaPipe `.so` bundles) | M | M | Use `nix-shell` with `python311` + `pip install` inside a venv; or `buildFHSUserEnv` if needed. |
| `select-pane` command bursts during oscillation | L | L | Cool-down already enforces ≥250 ms between switches. |
| User on remote tmux (SSH session) | L | M | Control-mode client must connect to the *local* tmux server where rendering happens. Document as a known limitation. |

---

## 10. Reference Implementations Consulted

- **`andoshin11/eye-tracker`** — closest existing prior art: MediaPipe → 9-point calibration → EMA → dwell → `tmux select-pane`. Worth reading before writing Phase 1.
- **Talon Voice** — gold-standard hands-free coding system. Architectural lesson: cleanly separate the *tracker layer* (raw stream) from the *action layer* (region → command).
- **MediaPipe FaceLandmarker docs** — Tasks API migration, `LIVE_STREAM` async callback pattern.
- **tmux Control-Mode wiki** — `%window-pane-changed`, `refresh-client -B` subscriptions.
- **ScienceDirect 2021** — dwell-time usability study confirming 600–800 ms as the validated range.

---

## 11. Immediate Next Steps

1. Confirm this architecture matches your intent — particularly the **head-pose-as-primary** decision (some users expect iris-only gaze; this is a deliberate simpler-and-more-robust choice).
2. Decide: do we want Phase 1 implemented in this repo right now, or do you want to adjust the design first?
3. Decide: NixOS install path — venv inside a `nix-shell`, or a `flake.nix` for the project? (Recommend venv-in-nix-shell for the POC; flake later if it sticks.)

Once those are settled, Phase 1 is roughly:

```
pip install mediapipe opencv-python-headless numpy libtmux
# implement camera.py, vision.py, classifier.py, tmux_io.py, __main__.py
python -m tmux_eyes
```

— ~300 LOC, one afternoon if the webcam cooperates.
