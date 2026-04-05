# Stereoscopic 180 SBS LoRA Trainer

Desktop app + Python tools to build **FLUX.1-dev LoRA** training data from **360° / 180° / 180° side-by-side stereo** media, then train a LoRA that generates **stereoscopic 180° half-equirect SBS** images from text.

Built by Light Sail VR.

---

## What it does

1. **Dataset builder** (`python/equirect_dataset_generator.py`): extracts frames, resizes to **2:1** (width = 2× height), writes `frames/<id>.png`, caption sidecars `frames/<id>.txt`, optional **disparity maps** in `depth/` (180 SBS only), and `dataset_manifest.json`.
2. **Prepare** (`train/prepare_dataset.py`): symlinks into `train_ready/images/`, optional `train_ready/depth/`, writes `metadata.jsonl` for training.
3. **Train** (`train/train_lora.py` + `train/train_lora_flux.py`): runs **Hugging Face diffusers** Flux LoRA training (`accelerate launch`) with your JSONL manifest.

**Base model:** `black-forest-labs/FLUX.1-dev` (configurable in `train/configs/presets.py` for a future FLUX.2 swap).

**Inference:** text prompt → stereo 180 SBS image; no input stereo pair required. Optional Phase 2: stack **FLUX.1-Depth-dev-lora** with your LoRA for depth-guided generation (not wired in this repo UI yet).

---

## Two-machine workflow (recommended)

| Step | Where | What |
|------|--------|------|
| **Dataset + depth** | **PC with NVIDIA GPU** (Windows / Linux / WSL2) | Run the Electron app or CLI with **FoundationStereo** for high-quality stereo disparity (`depth/`). See [FoundationStereo depth (PC)](#foundationstereo-depth-pc) and [Windows PC](#windows-pc-install-and-run) below. |
| **Training** | **Mac Studio (or any machine)** | Copy the finished `dataset/` folder. Install training deps and run `train/prepare_dataset.py` + `train/train_lora.py` (PyTorch **MPS** on Apple Silicon is supported). |

Depth extraction uses **CUDA** and the official **NVlabs/FoundationStereo** stack; LoRA training only needs the exported PNGs + JSONL and does **not** require FoundationStereo at train time.

---

## Prerequisites

### Node.js (Electron)

```bash
brew install node   # or https://nodejs.org
```

### Python 3 + dataset packages

```bash
python3 --version
pip3 install opencv-python-headless numpy Pillow
```

### Training stack (separate venv recommended)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r train/requirements.txt        # Mac / CPU / MPS training
# On a CUDA training box you may use train/requirements_nvidia.txt instead
```

### ffmpeg (video only)

```bash
brew install ffmpeg
```

---

## FoundationStereo depth (PC)

For production disparity maps (instead of noisy OpenCV SGBM):

1. Run **`scripts/setup_depth_env.sh`** on an **NVIDIA** machine. It clones [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) (default: `third_party/FoundationStereo`), creates the `foundation_stereo` conda env, and installs **flash-attn**.
2. Download pretrained weights (e.g. folder **`23-51-11`**) from the [FoundationStereo readme](https://github.com/NVlabs/FoundationStereo/blob/master/readme.md) into:
   - `third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth`
   - (same folder must include **`cfg.yaml`** from the download)
3. Before building the dataset:
   ```bash
   export FOUNDATION_STEREO_ROOT="/path/to/FoundationStereo"   # e.g. repo/third_party/FoundationStereo
   conda activate foundation_stereo
   ```
4. Use **`--depth-backend foundation_stereo`** (or **Auto** in the app if CUDA + weights are detected).

Reference pip pins (conda is preferred): [`python/requirements_depth_pc.txt`](python/requirements_depth_pc.txt).

**Tips:** For wide frames (~1024+), consider `--foundation-stereo-hiera 1` or `--foundation-stereo-scale 0.5` (see `equirect_dataset_generator.py --help`).

---

## Windows PC: install and run

These steps assume **Windows 10 or 11** (64-bit). Use an **NVIDIA GPU** if you want **FoundationStereo** depth; the app still runs on CPU-only machines using **OpenCV SGBM** or depth off.

### 1. Install system tools

- **Node.js (LTS):** [https://nodejs.org](https://nodejs.org) — installer includes `npm`. Confirm in **Command Prompt** or **PowerShell**: `node -v` and `npm -v`.
- **Python 3.10+ (64-bit):** [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/) — in the installer, enable **“Add python.exe to PATH”**. Confirm: `python --version` (or `py --version`).
- **Git:** [https://git-scm.com/download/win](https://git-scm.com/download/win) — needed to clone this repo (and optional FoundationStereo).
- **ffmpeg** (required for video extraction): e.g. [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/) (“ffmpeg-release-essentials.zip”), unpack, and add the `bin` folder to your user **PATH**. Or with **winget**: `winget install --id Gyan.FFmpeg`. Confirm: `ffmpeg -version`.

### 2. Clone and start the Electron app

In **Command Prompt** or **PowerShell** (replace the path if you use another folder):

```bat
cd %USERPROFILE%\Projects
git clone https://github.com/lightsailvr/controlnet-dataset-generator.git
cd controlnet-dataset-generator
npm install
npm start
```

The dataset builder calls **`python`** (and tries **`python3`** first on some setups). If the app says Python is missing, install Python as above or use **“Python Launcher”** so `python` works in a new terminal.

### 3. Python packages for dataset generation

From the repo root:

```bat
python -m pip install --upgrade pip
python -m pip install opencv-python-headless numpy Pillow
```

If you use a **virtual environment** for a clean install, create it in the repo root and **activate it in the same terminal** before `npm start` so child processes inherit `PATH` and pick up that `python`:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install opencv-python-headless numpy Pillow
npm start
```

### 4. FoundationStereo on Windows (depth quality)

[FoundationStereo](https://nvlabs.github.io/FoundationStereo/) is tested on **Linux + NVIDIA CUDA**; building **flash-attn** and PyTorch CUDA stacks on **native Windows** is often painful.

**Recommended:** use **WSL2** (Ubuntu) with an NVIDIA driver on Windows, install CUDA inside WSL per NVIDIA’s docs, then run **`scripts/setup_depth_env.sh`** and the **CLI** dataset builder inside WSL, writing datasets to a path Windows can read (e.g. under `/mnt/c/...`).

**Alternative:** install **Miniconda** on Windows and follow the [official FoundationStereo `readme`](https://github.com/NVlabs/FoundationStereo/blob/master/readme.md) (`environment.yml`, then `pip install flash-attn`). Set a user environment variable **`FOUNDATION_STEREO_ROOT`** to your clone path (e.g. `C:\Users\you\FoundationStereo`).

### 5. Training on Windows (optional)

You can train on the same PC if you have a suitable GPU and CUDA PyTorch:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r train\requirements_nvidia.txt
```

Then use the **Train LoRA** flow in the app or the CLI (use `python` instead of `python3`):

```bat
python train\prepare_dataset.py C:\path\to\dataset
python train\train_lora.py --detect-hardware
python train\train_lora.py --dataset C:\path\to\dataset --preset …   # see train/configs/presets.py
```

**Symlinks:** `train/prepare_dataset.py` creates **symbolic links** under `train_ready/`. On Windows, enable **Developer Mode** (Settings → Privacy & security → For developers → Developer Mode) or run the terminal as Administrator so symlink creation succeeds. If that is not an option, prepare the dataset on **macOS/Linux/WSL** and copy `train_ready/` over.

### 6. CLI examples (Windows paths)

```bat
python python\equirect_dataset_generator.py C:\media\clip.mov -o C:\data\my_dataset -t 180sbs -r 512
set FOUNDATION_STEREO_ROOT=C:\path\to\FoundationStereo
python python\equirect_dataset_generator.py C:\media\clip.mov -o C:\data\my_dataset -t 180sbs --depth-backend foundation_stereo
```

---

## Installation

```bash
cd controlnet-dataset-generator
npm install
npm start
```

---

## Usage

### App

1. **Select** video/image files or a folder.
2. Set **Source type** (default **180° SBS**), **frame interval**, **frame height** (output width = 2× height).
3. Edit **caption prefix** (trigger tokens for the LoRA).
4. Toggle **disparity depth maps** (SBS only) and choose **depth engine** (Auto / FoundationStereo / OpenCV SGBM).
5. **Build LoRA dataset** → review → **Train LoRA** (prepares `train_ready/` then runs training).

### CLI — dataset

```bash
python3 python/equirect_dataset_generator.py footage.mov -o ./dataset -t 180sbs -r 512
python3 python/equirect_dataset_generator.py --no-depth ./stills/ -o ./dataset -t 180sbs
# PC + CUDA + FoundationStereo:
export FOUNDATION_STEREO_ROOT=~/FoundationStereo
python3 python/equirect_dataset_generator.py ./footage.mov -o ./dataset -t 180sbs \
  --depth-backend foundation_stereo --foundation-stereo-hiera 1
```

### CLI — prepare + train

```bash
python3 train/prepare_dataset.py ./dataset
python3 train/train_lora.py --dataset ./dataset --preset macbook_m4_max
python3 train/train_lora.py --detect-hardware
```

---

## Output layout

```
dataset/
├── frames/                 # PNG + .txt captions
├── depth/                  # optional disparity (FoundationStereo or SGBM)
├── source_equirects/       # extracted video frames
├── dataset_manifest.json   # master list
├── pairs.json              # legacy summary for the UI
├── train_ready/
│   ├── images/             # symlinks + captions
│   ├── depth/              # optional
│   └── metadata.jsonl      # image + text (+ optional depth_image)
└── training_output/<ts>/   # LoRA weights, checkpoints, validation_samples/
```

---

## License

MIT — see [LICENSE](LICENSE).

Credits: Light Sail VR.
