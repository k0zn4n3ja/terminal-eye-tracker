{ pkgs ? import <nixpkgs> {} }:

let
  # Runtime libs winit + wgpu + cosmic-text dlopen at startup. NixOS doesn't
  # expose these via the system loader path by default, so we have to set
  # LD_LIBRARY_PATH for the binary to find them.
  runtimeLibs = with pkgs; [
    libxkbcommon
    libGL
    vulkan-loader

    # Wayland backend
    wayland

    # X11 backend
    xorg.libX11
    xorg.libXcursor
    xorg.libXi
    xorg.libXrandr
    xorg.libxcb

    # cosmic-text uses fontconfig to discover system fonts
    fontconfig
    freetype
  ];
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust toolchain (use rustup outside the shell if you prefer pinned versions)
    rustc
    cargo
    rustfmt
    clippy

    pkg-config
  ] ++ runtimeLibs;

  # Both for `cargo run` (loader needs to find these) and for the standalone
  # binary in `target/release/aiterm` when run inside the shell.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

  shellHook = ''
    echo "aiterm dev shell — winit/wgpu runtime libs on LD_LIBRARY_PATH"
  '';
}
