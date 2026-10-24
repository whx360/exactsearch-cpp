"""Locate CMake executables on single- and multi-config generators."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def executable(name):
    build = Path(os.environ.get("EXACTSEARCH_BUILD_DIR", ROOT / "build"))
    if not build.is_absolute():
        build = ROOT / build
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [build / (name + suffix), build / "Release" / (name + suffix),
                  build / "Debug" / (name + suffix)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Build {name} first; searched: {candidates}")
