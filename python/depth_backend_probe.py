#!/usr/bin/env python3
"""Print JSON diagnostics for depth backends (used by Electron IPC)."""
import json
import sys
from pathlib import Path

# Allow `python depth_backend_probe.py` from repo root or python/
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from depth_extractor import depth_backend_info  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(depth_backend_info()))
