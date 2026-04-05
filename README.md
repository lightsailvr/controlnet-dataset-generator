# Stereoscopic 180 SBS LoRA Trainer

Desktop app + Python tools to build **FLUX.1-dev LoRA** training data from **360° / 180° / 180° side-by-side stereo** media, then train a LoRA that generates **stereoscopic 180° half-equirect SBS** images from text.

Built by Light Sail VR.

---

## What it does

1. **Dataset builder** (`python/equirect_dataset_generator.py`): extracts frames, resizes to **2:1** (width = 2× height), writes `frames/<id>.png`, caption sidecars `frames/<id>.txt`, optional **disparity maps** in `depth/` (OpenCV stereo, 180 SBS only), and `dataset_manifest.json`.
2. **Prepare** (`train/prepare_dataset.py`): symlinks into `train_ready/images/`, optional `train_ready/depth/`, writes `metadata.jsonl` for training.
3. **Train** (`train/train_lora.py` + `train/train_lora_flux.py`): runs **Hugging Face diffusers** Flux LoRA training (`accelerate launch`) with your JSONL manifest.

**Base model:** `black-forest-labs/FLUX.1-dev` (configurable in `train/configs/presets.py` for a future FLUX.2 swap).

**Inference:** text prompt → stereo 180 SBS image; no input stereo pair required. Optional Phase 2: stack **FLUX.1-Depth-dev-lora** with your LoRA for depth-guided generation (not wired in this repo UI yet).

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
pip install -r train/requirements_mac.txt   # or requirements_nvidia.txt on CUDA
```

### ffmpeg (video only)

```bash
brew install ffmpeg
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
4. Toggle **disparity depth maps** (SBS only).
5. **Build LoRA dataset** → review → **Train LoRA** (prepares `train_ready/` then runs training).

### CLI — dataset

```bash
python3 python/equirect_dataset_generator.py footage.mov -o ./dataset -t 180sbs -r 512
python3 python/equirect_dataset_generator.py --no-depth ./stills/ -o ./dataset -t 180sbs
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
├── depth/                  # optional disparity (180 SBS)
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
