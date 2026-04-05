#!/usr/bin/env bash
# Set up NVlabs FoundationStereo for stereo depth on an NVIDIA PC (Linux/WSL).
# Training can still run on Mac; only dataset depth extraction needs this stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FS_ROOT="${FOUNDATION_STEREO_ROOT:-$PROJECT_ROOT/third_party/FoundationStereo}"
FS_URL="${FOUNDATION_STEREO_GIT_URL:-https://github.com/NVlabs/FoundationStereo.git}"

echo "Project root:  $PROJECT_ROOT"
echo "FoundationStereo dir: $FS_ROOT"
echo ""

if [[ ! -d "$FS_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$FS_ROOT")"
  echo "Cloning FoundationStereo..."
  git clone "$FS_URL" "$FS_ROOT"
else
  echo "FoundationStereo repo already present."
fi

echo ""
echo "Creating/updating conda env from FoundationStereo environment.yml (name: foundation_stereo)..."
set +e
conda env create -f "$FS_ROOT/environment.yml"
CREATE_EC=$?
set -e
if [[ "$CREATE_EC" -ne 0 ]]; then
  echo "Env may already exist — updating from environment.yml..."
  conda env update -f "$FS_ROOT/environment.yml" --prune
fi

echo ""
echo "Installing flash-attn inside conda env (CUDA build; may take several minutes)..."
conda run -n foundation_stereo pip install flash-attn --no-build-isolation || {
  echo "flash-attn install failed — see https://github.com/NVlabs/FoundationStereo/issues/20"
  exit 1
}

echo ""
echo "=== Weights (manual) ==="
echo "Download a pretrained folder (e.g. 23-51-11) from the FoundationStereo readme and place it at:"
echo "  $FS_ROOT/pretrained_models/23-51-11/"
echo "so that this file exists:"
echo "  $FS_ROOT/pretrained_models/23-51-11/model_best_bp2.pth"
echo ""
echo "Export before running the dataset builder:"
echo "  export FOUNDATION_STEREO_ROOT=\"$FS_ROOT\""
echo ""
echo "Run inference sanity check (after weights are in place):"
echo "  conda activate foundation_stereo"
echo "  cd \"$FS_ROOT\" && python scripts/run_demo.py --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/"
