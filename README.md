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
| **Dataset + depth** | **PC with NVIDIA GPU** (Windows / Linux / WSL2) | Run the Electron app or CLI with **FoundationStereo** for high-quality stereo disparity (`depth/`). See [Windows PC guide](#windows-pc-full-setup-guide) or [FoundationStereo depth (PC)](#foundationstereo-depth-pc) below. |
| **Training** | **Mac Studio (or any machine)** | Copy the finished `dataset/` folder. Install training deps and run `train/prepare_dataset.py` + `train/train_lora.py` (PyTorch **MPS** on Apple Silicon is supported). |

Depth extraction uses **CUDA** and the official **NVlabs/FoundationStereo** stack; LoRA training only needs the exported PNGs + JSONL and does **not** require FoundationStereo at train time.

---

## Prerequisites (macOS / Linux)

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

## Windows PC: full setup guide

This is the complete step-by-step process for building datasets **with FoundationStereo depth** on a Windows PC, then optionally training on the same machine or copying the dataset to a Mac.

**Requirements:** Windows 10 (build 19041+) or Windows 11, an NVIDIA GPU (RTX 3090 / 4090 / A100 recommended; 8 GB VRAM minimum), and an internet connection.

---

### Part A — Electron app (runs natively on Windows)

These tools let you run the desktop UI and generate datasets. Without FoundationStereo (Part B), depth maps will use OpenCV SGBM (low quality preview) or be off.

#### A1. Install Node.js

Download the **LTS** installer from [https://nodejs.org](https://nodejs.org). Run it with defaults (includes `npm`). Open a **new** Command Prompt or PowerShell and verify:

```bat
node -v
npm -v
```

#### A2. Install Python 3.10+

Download from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/). **Important:** check **"Add python.exe to PATH"** in the installer. Verify in a new terminal:

```bat
python --version
```

If `python` opens the Microsoft Store instead, go to Settings → Apps → Advanced app settings → App execution aliases and turn **off** the aliases for `python.exe` and `python3.exe`.

#### A3. Install Git

Download from [https://git-scm.com/download/win](https://git-scm.com/download/win) and install with defaults. Verify:

```bat
git --version
```

#### A4. Install ffmpeg (needed for video input)

Easiest with winget:

```bat
winget install --id Gyan.FFmpeg
```

Or manually: download "ffmpeg-release-essentials.zip" from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/), extract it, and add the `bin` folder inside it to your system `PATH`. Verify in a **new** terminal:

```bat
ffmpeg -version
```

#### A5. Clone repo, install deps, start app

```bat
cd %USERPROFILE%\Projects
git clone https://github.com/lightsailvr/controlnet-dataset-generator.git
cd controlnet-dataset-generator

python -m pip install --upgrade pip
python -m pip install opencv-python-headless numpy Pillow

npm install
npm start
```

The Electron app should launch. You can select media, configure the dataset, and build. At this point depth will use **OpenCV SGBM** (low quality) or be off. Part B adds FoundationStereo for production-quality depth maps.

---

### Part B — FoundationStereo depth via WSL2 (NVIDIA GPU required)

FoundationStereo requires **Linux + CUDA + flash-attn + xformers**. The cleanest path on Windows is **WSL2** with Ubuntu. You will run the dataset builder CLI inside WSL, writing output to a shared path that Windows and the Electron app can read.

#### B1. Enable WSL2 and install Ubuntu

Open **PowerShell as Administrator**:

```powershell
wsl --update
wsl --set-default-version 2
wsl --install -d Ubuntu-24.04
```

Restart your PC if prompted. After reboot, Ubuntu will launch and ask you to create a username and password. Then inside the Ubuntu terminal:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl
```

#### B2. Install NVIDIA CUDA toolkit inside WSL2

Your **Windows** NVIDIA driver already exposes the GPU to WSL2 — you do **not** install a Linux GPU driver. You only need the CUDA **toolkit** inside Ubuntu.

First, make sure your Windows NVIDIA driver is up to date: use GeForce Experience, the NVIDIA App, or download from [nvidia.com/drivers](https://www.nvidia.com/drivers).

Then inside the Ubuntu terminal:

```bash
# Remove any stale GPG key
sudo apt-key del 7fa2af80 2>/dev/null || true

# Add NVIDIA's WSL-Ubuntu CUDA repository (version 12.8 — check nvidia.com for latest)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

Add CUDA to your shell path — append these to `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify everything works:

```bash
nvidia-smi          # should show your Windows GPU and driver version
nvcc --version      # should show CUDA 12.8.x
```

If `nvidia-smi` fails, your Windows NVIDIA driver may be too old — update it on the Windows side and restart WSL (`wsl --shutdown` in PowerShell, then reopen Ubuntu).

#### B3. Install Miniconda inside WSL2

FoundationStereo uses a conda environment. Inside the Ubuntu terminal:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Accept defaults and say **yes** to initialize conda. Then restart the terminal or run:

```bash
source ~/.bashrc
conda --version     # should print conda 24.x or later
```

#### B4. Clone this repo inside WSL2

You have two path strategies:

| Strategy | Path | Pros | Cons |
|----------|------|------|------|
| **Shared Windows drive** | `/mnt/c/Users/<you>/Projects/` | Windows Electron app + WSL CLI share the same folder; no copy step | Slower filesystem I/O |
| **WSL-native home** | `~/` | Fastest I/O for FoundationStereo processing | Must copy the finished dataset out to Windows or Mac |

**Shared (recommended for simplicity):**

```bash
cd /mnt/c/Users/$USER/Projects
git clone https://github.com/lightsailvr/controlnet-dataset-generator.git
cd controlnet-dataset-generator
```

**WSL-native (recommended for speed):**

```bash
cd ~
git clone https://github.com/lightsailvr/controlnet-dataset-generator.git
cd controlnet-dataset-generator
```

#### B5. Run the FoundationStereo setup script

This clones FoundationStereo into `third_party/FoundationStereo`, creates the `foundation_stereo` conda environment, and installs all dependencies including **flash-attn** (compiles from source with CUDA — takes 5–15 minutes):

```bash
bash scripts/setup_depth_env.sh
```

If `flash-attn` fails to build:
- Confirm `nvcc --version` works.
- Confirm your NVIDIA driver version is compatible with CUDA 12.8 (`nvidia-smi` shows "CUDA Version: 12.x" in the top-right).
- See [FoundationStereo issue #20](https://github.com/NVlabs/FoundationStereo/issues/20) for known fixes.

#### B6. Download pretrained weights

The weights are on Google Drive:
**[https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf)**

Download the **`23-51-11`** folder (ViT-Large, best quality). It must contain exactly two files:
- `cfg.yaml` — model configuration
- `model_best_bp2.pth` — weights (~1.3 GB)

**Option 1 — browser download (easiest):**

1. Open the Google Drive link on your Windows browser.
2. Enter the **`23-51-11`** folder. Download `cfg.yaml` and `model_best_bp2.pth` individually (downloading the whole folder sometimes produces a broken zip — see [issue #89](https://github.com/NVlabs/FoundationStereo/issues/89)).
3. Copy them into the WSL path:

```bash
mkdir -p third_party/FoundationStereo/pretrained_models/23-51-11
cp /mnt/c/Users/$USER/Downloads/cfg.yaml third_party/FoundationStereo/pretrained_models/23-51-11/
cp /mnt/c/Users/$USER/Downloads/model_best_bp2.pth third_party/FoundationStereo/pretrained_models/23-51-11/
```

**Option 2 — gdown (command line):**

```bash
conda activate foundation_stereo
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf -O third_party/FoundationStereo/pretrained_models/
```

**Alternate mirror** if Google Drive quota is exceeded:
[https://huggingface.co/datasets/steve-redefine/FoundationStereoWeights](https://huggingface.co/datasets/steve-redefine/FoundationStereoWeights)

Verify the files are in place:

```bash
ls -lh third_party/FoundationStereo/pretrained_models/23-51-11/
# Expected:
#   cfg.yaml            (~1 KB)
#   model_best_bp2.pth  (~1.3 GB)
```

#### B7. Sanity check — run FoundationStereo demo

```bash
conda activate foundation_stereo
cd third_party/FoundationStereo

python scripts/run_demo.py \
  --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth \
  --out_dir ./test_outputs/

cd -   # back to repo root
```

You should see `Output saved to ./test_outputs/` and find `test_outputs/vis.png` containing a left-image + turbo-colored disparity side-by-side. If this works, FoundationStereo is ready.

#### B8. Build a dataset with FoundationStereo depth

From the repo root inside WSL:

```bash
conda activate foundation_stereo
export FOUNDATION_STEREO_ROOT="$(pwd)/third_party/FoundationStereo"

# Point at your media. If it's on your Windows C: drive:
python python/equirect_dataset_generator.py \
  "/mnt/c/Users/$USER/Media/my_vr_footage.mov" \
  -o ./my_dataset \
  -t 180sbs \
  -r 512 \
  --depth-backend foundation_stereo
```

Useful flags:
- `--foundation-stereo-hiera 1` — hierarchical mode for high-res input (>1024 px per eye). Better quality, slower.
- `--foundation-stereo-scale 0.5` — process at half resolution for speed. Slight quality loss.
- `--foundation-stereo-iters 16` — fewer refinement passes (default 32). Faster, slightly lower accuracy.

The output folder (`my_dataset/`) now contains:
- `frames/` — training PNGs + `.txt` caption sidecars
- `depth/` — high-quality FoundationStereo disparity maps
- `dataset_manifest.json` — master manifest

#### B9. Get the dataset to the Electron app or Mac for training

**If you used the shared `/mnt/c/...` path:**
The dataset is already on your Windows drive. In the Electron app on Windows, click **"Review Existing Dataset"** and browse to `C:\Users\<you>\Projects\controlnet-dataset-generator\my_dataset`.

**If you used the WSL-native `~/` path:**
Copy the dataset to your Windows drive:

```bash
cp -r ~/controlnet-dataset-generator/my_dataset /mnt/c/Users/$USER/Desktop/my_dataset
```

**To transfer to a Mac for training:**
Copy the dataset folder to your Mac via network share, USB drive, AirDrop, or cloud storage. You only need these from the dataset:
- `frames/` — the training images + `.txt` captions
- `depth/` — the disparity maps
- `dataset_manifest.json`

You can skip `source_equirects/` (raw extracted frames) to save transfer time.

---

### Part C — Training on Windows (optional)

You can train on the same Windows PC if you have a CUDA GPU. Training runs natively on Windows (no WSL needed).

#### C1. Create a training venv

In **Command Prompt** or **PowerShell**, from the repo root:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r train\requirements_nvidia.txt
```

#### C2. Enable symlinks

`train/prepare_dataset.py` creates symbolic links under `train_ready/`. On Windows you must enable one of:

- **Developer Mode** (recommended): Settings → System → For developers → Developer Mode → **On**.
- **Or** run your terminal as **Administrator**.

Without this, symlink creation will fail and training will not find images.

#### C3. Prepare and train

```bat
python train\prepare_dataset.py C:\Users\you\Desktop\my_dataset
python train\train_lora.py --detect-hardware
python train\train_lora.py --dataset C:\Users\you\Desktop\my_dataset --preset …
```

Check `train/configs/presets.py` for available presets, or use the **Train LoRA** button in the Electron app.

---

## Quick start (macOS / Linux)

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
