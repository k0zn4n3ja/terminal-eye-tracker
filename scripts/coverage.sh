#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo llvm-cov --workspace --html --open
