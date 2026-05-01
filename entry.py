"""PyInstaller entry point.

PyInstaller compiles this file as a top-level script. Importing the package
this way preserves relative imports inside ``tmux_eyes/__main__.py``.
"""

from tmux_eyes.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
