# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
Electron desktop app + Python backend for generating ControlNet training datasets from 360°/180° equirectangular media. No database, no server, no Docker — Electron spawns Python as a child process.

### Running the app
- `npm start` — production mode
- `npm run dev` — development mode (`NODE_ENV=development`)
- The app requires a display (Electron GUI); `$DISPLAY` is already set to `:1` in cloud VMs.
- D-Bus and GPU errors in console are harmless in headless/VM environments and can be ignored.

### Python backend (CLI)
The Python script can be tested standalone without Electron:
```
python3 python/equirect_dataset_generator.py <input> -o <output_dir> [options]
```

### Dependencies
- **Node/npm**: `npm install` (see `package.json`)
- **Python**: `pip3 install py360convert opencv-python-headless numpy Pillow` (installs to user site-packages)
- **System**: `ffmpeg` (pre-installed; only required for video file processing, images work without it)

### No test framework
This project has no automated tests or linting configuration. Verification is done by launching the app and running the Python CLI.

### Gotchas
- Python packages install to `~/.local/` (user site-packages) since system site-packages is not writable. The `python3` on `PATH` picks these up automatically.
- The Electron main process searches for `python3` at several hardcoded paths (see `findPython()` in `src/main.js`); `/usr/bin/python3` is found first in cloud VMs.
- `titleBarStyle: "hiddenInset"` is macOS-specific but renders fine on Linux — the window just uses default Linux window decorations.
