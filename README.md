# aiterm — AI-first terminal emulator

**Status:** Pre-alpha, M0/M1 scaffolding

aiterm is an AI-first terminal for Linux/macOS. It combines a fast GPU-accelerated terminal renderer with integrated AI capabilities: voice input via Whisper, gaze tracking for block-targeted prompting, automatic prompt-cache and token optimization at the model boundary, and a blocks-first architecture so every command/output pair is addressable by the AI.

## Build Prerequisites

You'll need:
- **Rust:** rustc 1.78 or later
- **Linux (v0):** X11 development libraries

On Ubuntu/Debian, install build dependencies:

```bash
apt install libwayland-dev libxkbcommon-dev libfontconfig1-dev libfreetype6-dev \
            libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libx11-xcb-dev
```

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run
```

## Test Coverage

Run coverage locally inside the nix-shell:

```bash
./scripts/coverage.sh
```

This generates an HTML report at `target/llvm-cov/html/index.html`. No overall coverage gate is enforced: pure-logic modules (currently `config.rs`, later `parser`/`grid`/`block`/`llm`) are tested aggressively; GPU/window-binding code in `render.rs` and `app.rs` is dogfood-tested until a headless-GPU fixture is worth the maintenance cost (likely M3+).

## Project Roadmap

See [DESIGN.md](DESIGN.md) for detailed milestones and acceptance criteria, from the current scaffold (M0) through frontier features (M8+).

## Architecture

For a deep dive into the layered terminal architecture, dependency choices, and module map, see [ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT — see LICENSE file.
