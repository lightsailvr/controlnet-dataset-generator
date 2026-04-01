# ControlNet Dataset Generator

A desktop application for generating ControlNet training datasets from 360° and 180° immersive footage. Point it at your equirectangular video files or images, configure extraction parameters, and generate paired rectilinear → equirect training data ready for ControlNet fine-tuning.

Built by Light Sail VR.

---

## What It Does

For each source equirectangular frame, the tool:

1. Extracts frames from video at your specified interval (or loads images directly)
2. Generates multiple randomized rectilinear perspective crops (simulating "normal" camera views)
3. Pairs each crop with the corresponding full equirectangular frame
4. Outputs everything in a structured dataset ready for ControlNet training

The result is a dataset of **conditioning images** (rectilinear crops) and **target images** (equirect frames) that teaches a ControlNet the mapping from flat perspective views to 360° panoramic output.

---

## Prerequisites

You need three things installed on your Mac before running the app:

### 1. Node.js (for Electron)

```bash
# Check if you have it
node --version

# If not, install via Homebrew
brew install node

# Or download from https://nodejs.org (LTS version recommended)
```

### 2. Python 3 + packages

```bash
# Check if you have Python 3
python3 --version

# If not, install via Homebrew
brew install python@3.11

# Install required Python packages
pip3 install py360convert opencv-python-headless numpy Pillow
```

**If you use conda/miniconda:**
```bash
conda activate your_env
pip install py360convert opencv-python-headless numpy Pillow
```

**Note:** The app auto-detects Python at these paths:
- `python3` (system PATH)
- `/opt/homebrew/bin/python3` (Apple Silicon Homebrew)
- `/usr/local/bin/python3` (Intel Homebrew)
- `~/miniconda3/bin/python3`
- `~/anaconda3/bin/python3`

### 3. ffmpeg (for video file support)

```bash
# Check if you have it
ffmpeg -version

# If not, install via Homebrew
brew install ffmpeg
```

Images work without ffmpeg — it's only needed if you're processing video files.

---

## Installation

```bash
# Clone or copy the project folder, then:
cd controlnet-dataset-generator

# Install Electron
npm install

# That's it — run the app
npm start
```

---

## Usage

### Desktop App (Recommended)

```bash
npm start
```

1. The app checks your dependencies on launch (green dot = ready)
2. Click **Select Files** or **Select Folder** to add media
3. Configure extraction settings:
   - **Source Type**: 360° equirect, 180° equirect, or 180° SBS stereo
   - **Frame Interval**: Extract every Nth frame from video (lower = more frames)
   - **Crops per Frame**: How many rectilinear perspectives to generate per source frame
   - **Training/Conditioning Resolution**: Output image sizes
   - **FOV Range**: Simulated lens field of view (60°–110° covers wide-to-normal)
   - **Horizon Bias**: What % of crops cluster near the horizon (where the action usually is)
4. Choose an output directory
5. Click **Generate Dataset**

### CLI Mode

The Python script also works standalone:

```bash
# Single video file
python3 python/equirect_dataset_generator.py my_360_footage.mov -o ./dataset

# Entire folder of media
python3 python/equirect_dataset_generator.py /path/to/footage/ -o ./dataset

# 180° SBS stereo, denser extraction
python3 python/equirect_dataset_generator.py my_180sbs.mov \
    --source-type 180sbs \
    --frame-interval 15 \
    --crops-per-frame 15 \
    --training-res 1024

# Full options
python3 python/equirect_dataset_generator.py input.mov \
    -o ./dataset \
    -t 360 \
    -i 30 \
    -c 10 \
    -r 512 \
    --conditioning-res 512 \
    --fov-min 60 \
    --fov-max 110 \
    --horizon-bias 0.7 \
    --seed 42
```

---

## Output Structure

```
dataset/
├── source_equirects/    ← Original extracted frames (full resolution)
│   ├── my_video/        ← Subfolder per video file
│   │   ├── frame_000001.png
│   │   └── ...
│   └── panorama.png     ← Image files stored directly
├── conditioning/        ← Rectilinear perspective crops (conditioning inputs)
│   ├── my_video_frame_000001_y180_p-5_f90.png
│   └── ...
├── target/              ← Equirect frames resized to training resolution
│   ├── my_video_frame_000001.png
│   └── ...
├── metadata/            ← Per-crop JSON with extraction parameters
│   ├── my_video_frame_000001_y180_p-5_f90.json
│   └── ...
├── pairs.json           ← Master manifest linking all pairs
└── generation_results.json  ← Summary stats
```

### pairs.json Format

```json
{
  "generator": "equirect_dataset_generator.py",
  "total_pairs": 15000,
  "total_frames": 1500,
  "config": { ... },
  "pairs": [
    {
      "conditioning": "conditioning/my_video_frame_000001_y180_p-5_f90.png",
      "target": "target/my_video_frame_000001.png",
      "metadata": "metadata/my_video_frame_000001_y180_p-5_f90.json",
      "yaw": 180.0,
      "pitch": -5.23,
      "fov_deg": 90.41,
      "source_frame": "my_video_frame_000001",
      "source_file": "my_video.mov"
    }
  ]
}
```

---

## Supported Formats

### Video (requires ffmpeg)
MOV, MP4, MKV, AVI, MXF, WebM, M4V, MPG, MPEG, TS, MTS, M2TS, WMV, FLV, 3GP, OGV, R3D, BRAW

### Image
JPG/JPEG, PNG, TIFF/TIF, EXR, HDR, BMP, WebP, DPX

---

## Training Tips

- **Minimum dataset size**: Aim for 5,000+ pairs. 50,000+ is ideal for production quality.
- **Resolution**: 512px is standard for SDXL ControlNet. Use 1024px for Flux-based models.
- **Diversity**: Use footage from many different scenes, lighting conditions, and environments.
- **Horizon bias**: 70% is a good default for eye-level content. Reduce for architectural or aerial footage.
- **FOV range**: 60–110° covers the range from telephoto to ultrawide. If your inference input will always be ~90° FOV, narrow the training range to 80–100°.

### Using the Dataset for Training

The output is structured for use with:

- **diffusers** ControlNet training scripts
- **kohya_ss** (sd-scripts) with custom dataset config
- **SimpleTuner** with paired image datasets

Point the training config at `conditioning/` for input images and `target/` for ground truth. The `pairs.json` manifest provides the mapping.

---

## Troubleshooting

**"Python 3 not found"**
Make sure `python3` is on your PATH. On Apple Silicon Macs with Homebrew:
```bash
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**"Missing Python packages"**
```bash
pip3 install py360convert opencv-python-headless numpy Pillow
```
If you have multiple Python installs, make sure you're installing to the right one:
```bash
python3 -m pip install py360convert opencv-python-headless numpy Pillow
```

**"ffmpeg not found"**
```bash
brew install ffmpeg
```

**EXR/HDR files look wrong**
The tool tonemaps HDR to 8-bit LDR for training. If your source footage is in linear light EXR, the automatic tonemapping may not be ideal. Consider pre-converting to PNG with your preferred tonemap curve in DaVinci Resolve or similar.

**Large datasets running slowly**
The bottleneck is usually disk I/O for writing PNGs. Consider:
- Using an SSD for the output directory
- Reducing conditioning resolution if you don't need 1024px
- Processing on a machine with fast storage

---

## License

MIT
