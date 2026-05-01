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
  ];

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
    # MediaPipe sometimes wants:
    libdrm
  ]);
  NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
    libGL
    libglvnd
  ]);

  shellHook = ''
    export UV_PROJECT_ENVIRONMENT=.venv

    if [ ! -d .venv ]; then
      echo "[tmux-eyes] First run: creating venv and syncing deps via uv..."
      uv sync
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "[tmux-eyes] env ready. Run: python -m tmux_eyes  (or: pytest)"
  '';
}
