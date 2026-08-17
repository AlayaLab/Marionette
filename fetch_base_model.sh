#!/bin/bash
# Put the Wan2.2-Fun-5B-Control backbone in place.
#
# This project does not redistribute it. It is a third-party model released by Alibaba-PAI
# under the Apache License 2.0, and the VAE, the umT5-xxl text encoder and the CLIP image
# encoder our pipeline loads all ship inside that same repository. Downloading it from the
# source keeps provenance and licence attached to it.
#
#   bash fetch_base_model.sh                    # download from HuggingFace (~23 GB)
#   bash fetch_base_model.sh --link /path/to/Wan2.2-Fun-5B-Control   # reuse a local copy
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=$ROOT/weights/Wan2.2-Fun-5B-Control
REPO=alibaba-pai/Wan2.2-Fun-5B-Control

if [ "${1:-}" = "--link" ]; then
  SRC=${2:?usage: fetch_base_model.sh --link /path/to/Wan2.2-Fun-5B-Control}
  [ -d "$SRC" ] || { echo "not a directory: $SRC"; exit 2; }
  mkdir -p "$ROOT/weights"
  ln -sfn "$SRC" "$DEST"
  echo "linked $DEST -> $SRC"
else
  command -v huggingface-cli >/dev/null || { echo "huggingface-cli not found: pip install huggingface_hub"; exit 2; }
  mkdir -p "$DEST"
  huggingface-cli download "$REPO" --local-dir "$DEST" || { echo "download failed"; exit 3; }
fi

# The three files this pipeline actually loads. The Fun-5B-Control release carries no CLIP
# image encoder — the wider Wan2.2 family does, and checking for it here reports a healthy
# download as broken. Checking by name still turns a partial download into an error now rather
# than a confusing failure thirty seconds into a GPU job.
missing=0
for f in diffusion_pytorch_model.safetensors Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth; do
  [ -e "$DEST/$f" ] || { echo "  MISSING $f"; missing=1; }
done
[ "$missing" = 0 ] && echo "base model OK at $DEST" || { echo "base model incomplete"; exit 4; }
