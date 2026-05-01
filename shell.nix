{ pkgs ? import <nixpkgs> { } }:

# Project shell for tmux-eyes.
#
# Why this exists: NixOS doesn't ship glibc/libstdc++/libGL at FHS paths,
# so MediaPipe's bundled .so files (and uv's downloaded Python interpreters)
# can't find their dynamic linker dependencies. nix-ld is enabled system-wide
# (configuration.nix line 17) but with no library set — we provide one here
# scoped to this project.

pkgs.mkShell {
  packages = with pkgs; [
    uv
    python311
    tmux
    xdotool # X11 window geometry — needed in Phase 3, harmless now

    # Rust toolchain (for the rs/ port)
    rustc
    cargo
    pkg-config

    # bindgen needs libclang + kernel headers (nokhwa's v4l2-sys-mit transitive)
    llvmPackages.clang
    linuxHeaders

    # System libs that nokhwa / ort link against on Linux
    v4l-utils
    udev
    openssl
  ];

  # bindgen needs to find libclang at build time
  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
  BINDGEN_EXTRA_CLANG_ARGS =
    "-isystem ${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.llvmPackages.libclang.version}/include "
    + "-isystem ${pkgs.linuxHeaders}/include "
    + "-isystem ${pkgs.glibc.dev}/include";

  # Make these libraries discoverable to dynamically-linked binaries
  # (MediaPipe, OpenCV, NumPy wheels, etc.).
  #   - NIX_LD_LIBRARY_PATH covers binaries that go through nix-ld's FHS shim
  #     (e.g. uv-downloaded Python interpreters).
  #   - LD_LIBRARY_PATH covers .so files loaded by the Nix-provided Python
  #     interpreter at import time (e.g. numpy's _multiarray_umath.so linking
  #     against libstdc++.so.6).
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
    libGL
    libglvnd
    # OpenCV manylinux wheels (even -headless) link against these:
    xorg.libxcb
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    # MediaPipe + ORT sometimes want:
    libdrm
    # Nokhwa (Rust webcam) needs v4l + udev at runtime on Linux:
    udev
    v4l-utils
    openssl
  ]);
  NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
    libGL
    libglvnd
  ]);

  shellHook = ''
    # Anchor to the project root (where this shell.nix lives), not whatever
    # CWD the user invoked nix-shell from — otherwise `cd rs && nix-shell ..`
    # tries to source rs/.venv and prints a confusing "No such file" error.
    PROJECT_ROOT="${toString ./.}"
    export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"

    # Only set up the Python venv if there's a pyproject.toml here. Skip
    # silently if not (e.g. someone using shell.nix purely for Rust deps).
    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
      if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        echo "[tmux-eyes] First run: creating venv and syncing deps via uv..."
        ( cd "$PROJECT_ROOT" && uv sync )
      fi
      if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.venv/bin/activate"
      fi
    fi

    echo "[tmux-eyes] env ready. Python: python -m tmux_eyes  |  Rust: cd rs && cargo run"
  '';
}
