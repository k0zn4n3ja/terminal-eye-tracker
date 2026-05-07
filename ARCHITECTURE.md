# aiterm Architecture

## The Terminal Emulator Layers

A terminal emulator is fundamentally a stack of layers, each handling a distinct concern. aiterm follows a classical 6-layer model, extended with two additional layers for AI integration.

### Layer 1: PTY (Pseudoterminal)

The PTY is the kernel interface that bridges user-space terminal emulation with the OS shell. It presents itself as a file descriptor pair — one side acts as a virtual keyboard/screen for the shell, the other as a window into the shell's I/O. We use **portable-pty 0.9** to handle platform-specific PTY spawning and management (Linux, macOS, Windows compatibility).

### Layer 2: VT/xterm Parser

Raw PTY output is a stream of bytes — some printable, most not. The parser interprets control sequences (escape codes) that tell the terminal what to do: move the cursor, change colors, clear the screen, etc. We use **vte 0.13**, a battle-tested finite-state machine that decodes the xterm standard. The parser is event-driven: on each byte, it emits events like `Print(char)`, `CursorMove(row, col)`, or `SGR(color)`.

### Layer 3: Grid Model

The grid is the in-memory representation of what's on screen. Each cell holds a character, foreground color, background color, and attribute flags (bold, underline, etc.). As the parser emits events, the grid state mutates. Scrollback is stored as a ring buffer of historical lines. The grid also handles wide-character Unicode (e.g., emoji, CJK) where one logical character occupies multiple columns.

### Layer 4: Renderer

The renderer converts the grid into pixels for display. It rasterizes text using a font engine, applies colors, handles diacritics and ligatures, and uploads everything to the GPU. We use **cosmic-text 0.12** for text shaping and **wgpu 23** for GPU compute. Rendering runs at ~60fps on modern hardware via GPU-accelerated glyph caching and batch geometry.

### Layer 5: Window & Input

The window is the OS-level surface where rendered pixels appear. Input (keyboard, mouse, resize events) flows back from the window. We use **winit 0.30** to abstract over X11, Wayland, macOS, and Windows windowing APIs. Input events are routed to the grid (for selection, scrolling, mouse-driven actions) or to the command prompt (if one is open).

### Layer 6: UI Chrome

UI chrome is everything above the raw terminal: the tab bar, the pane splitter drag handles, the inline command prompt, status bar. It's rendered on top of the terminal grid and handles its own event routing.

### Layer 7: Block Model

The block model extends the grid to tag command/output pairs. When the shell sends an OSC 133 sequence (from shell integration scripts), we mark the start and end of a command and its output as discrete "blocks" — addressable objects with IDs. Each block can be queried, analyzed, and prompted on independently. This is the foundation for AI-aware terminal operation.

### Layer 8: Agent Integration

The agent layer calls the Anthropic Claude API to run LLM operations on blocks: summarizing output, fixing a command, translating a question into a shell incantation, or streaming a full response inline. It includes prompt assembly with context budgeting, cache-aware header construction, and streaming response rendering.

## Dependency Choices & Rationale

### Graphics & Rendering
- **wgpu 23:** Rust abstraction over Vulkan/Metal/D3D12. Gives us portability and GPU compute without vendor lock-in. Permissive license (MIT/Apache-2.0).
- **cosmic-text 0.12:** Text shaping in pure Rust — handles CJK, ligatures, diacritics. Used by the Cosmic DE. MIT license.
- **swash 0.2:** Glyph outline rasterization. Permissive (Apache-2.0).
- **fontdb 0.23:** Font database and matching. Permissive (MIT/Apache-2.0).

### Windowing & Input
- **winit 0.30:** Cross-platform window creation and event loop. Rust-native, well-maintained, MIT/Apache-2.0.

### Terminal Protocol & PTY
- **vte 0.13:** VT escape code parser. Lightweight, no_std compatible, MIT. Standard reference for xterm parsing.
- **portable-pty 0.9:** PTY abstraction over Unix/Windows. Used by wezterm. Permissive (MIT).

### Async Runtime & HTTP
- **tokio 1 (full features):** Async runtime. Industry standard. MIT/Apache-2.0.
- **reqwest 0.12 (rustls-tls):** HTTP client for LLM API calls. Explicitly configured to use rustls (permissive TLS, no OpenSSL GPL entanglement). Apache-2.0/MIT.

### Serialization & Configuration
- **serde 1:** Serialization framework (derive macros for JSON/TOML config). MIT/Apache-2.0.
- **serde_json 1, toml 0.8:** JSON and TOML parsing. All permissive.

### Error Handling & Logging
- **anyhow 1:** Ergonomic error chaining. MIT/Apache-2.0.
- **thiserror 2:** Error macros. MIT/Apache-2.0.
- **tracing 0.1, tracing-subscriber 0.3:** Structured logging. MIT/Apache-2.0.

### CLI & System Integration
- **clap 4 (derive):** Command-line parsing. MIT/Apache-2.0.
- **dirs 5:** Platform-specific config/data directories. MIT/Apache-2.0.

### License Position

All dependencies are permissively licensed (MIT, Apache-2.0, BSD-3-Clause). We avoid GPL to prevent license contagion — the terminal itself is MIT, and we keep it that way. Dependencies like rustls (instead of OpenSSL) and explicit avoidance of kitty/iTerm2 code (which are GPL or proprietary) maintain this stance.

## Module Map

The codebase is organized by layer:

- **src/main.rs:** Entry point, app setup, event loop orchestration.
- **src/app.rs:** Application state, window management, high-level control flow.
- **src/render.rs:** GPU rendering pipeline (wgpu, cosmic-text integration, glyph cache).
- **src/pty.rs:** PTY spawning, process management (portable-pty wrapper).
- **src/parser.rs:** VT escape sequence parsing (vte integration, event dispatch).
- **src/grid.rs:** In-memory terminal grid, scrollback, cell state.
- **src/block.rs:** Block model — command/output pairing, OSC 133 parsing, block addressability.
- **src/llm.rs:** LLM integration — API client (reqwest), prompt assembly, streaming response handling.
- **src/config.rs:** Configuration loading (TOML, serde, CLI parsing via clap).

## Reference Designs

We studied and drew inspiration from:

- **wezterm:** Mux for splits, tab management, and Rust async patterns. MIT license.
- **alacritty:** Grid + parser integration, GPU rendering philosophy. BSD-3-Clause.
- **Zellij:** Layout DSL and pane management UX. MIT.
- **Ghostty:** Terminal design patterns and renderer optimization strategies. MIT.

We explicitly do *not* borrow code from kitty or iTerm2, which are GPL or proprietary licensed. We read their design decisions for inspiration only.
