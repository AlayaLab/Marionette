#!/bin/bash
# STAGE 1 — dynamics + bridge.  Seed pose -> generated 276D state -> pose-control video.
#
# Runs in the RENDER environment (moderngl + EGL, system python3.12). It does not need the
# Wan environment and it does not import torch's video stack, so keep it separate: the two
# environments are mutually exclusive in practice (see README, "Two environments").
#
#   bash run_stage1_render.sh                 # defaults: seed segment 133
#   SEG=134 HORIZON=600 bash run_stage1_render.sh
#
# Output: $OUT_DIR/pose.mp4 plus the first-frame appearance reference the Wan stage consumes.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MARIONETTE_ROOT=$ROOT

SEG=${SEG:-53}
SEED_FRAMES=${SEED_FRAMES:-32}
HORIZON=${HORIZON:-340}
TEMP=${TEMP:-0.9}
# The action stream is sampled (temperature > 0 goes through torch.multinomial), so without a
# fixed seed two runs of the same command produce different motion. Upstream leaves this at -1
# (unseeded); a released artifact should be reproducible by default, so pin it here. Set
# TORCH_SEED=-1 to get the unseeded behaviour back.
TORCH_SEED=${TORCH_SEED:-43}
OUT_DIR=${OUT_DIR:-$ROOT/samples/seg${SEG}_seed${SEED_FRAMES}}
REF_RGB=${REF_RGB:-$ROOT/data/first_frame_ref.mp4}
PY=${PY:-python3.12}

export POSE_DATA_DIR=$ROOT/data
export POSE_TERRAIN_DIR=$ROOT/data/terrain
export POSE_FILTER_DIR=$ROOT/data/filter
# The render image has no nvenc ffmpeg; libx264 goes through the bundled imageio_ffmpeg
# binary. Leaving the upstream 'nvenc' default here fails at the encode step, after the whole
# rollout has already been computed.
export POSE_ENCODER=${POSE_ENCODER:-libx264}

mkdir -p "$OUT_DIR"
echo "[stage1] seg=$SEG seed_frames=$SEED_FRAMES horizon=$HORIZON temp=$TEMP -> $OUT_DIR/pose.mp4"
"$PY" "$ROOT/bridge/gen_pose_video.py" \
  --combined_dir "$ROOT/data/seeds" \
  --terrain_pose_ckpt "$ROOT/weights/dynamics/pose_gpt.pt" \
  --terrain_action_ckpt "$ROOT/weights/dynamics/action_gpt.pt" \
  --stage 101 --segment "$SEG" --seed_frames "$SEED_FRAMES" --horizon "$HORIZON" \
  --temperature "$TEMP" --torch_seed "$TORCH_SEED" \
  --cull_occluders --collide --fps 30 --gen_fps 20 \
  --out "$OUT_DIR/pose.mp4" || { echo "STAGE1 FAILED"; exit 3; }

# The observation model is conditioned on a first frame for appearance. Stage 2 looks for
# rgb.mp4 next to pose.mp4 and reads only its first frame.
if [ -e "$REF_RGB" ]; then
  cp -f "$REF_RGB" "$OUT_DIR/rgb.mp4"
else
  echo "WARN: no appearance reference at $REF_RGB; stage 2 needs $OUT_DIR/rgb.mp4"
fi
echo "STAGE1 DONE: $OUT_DIR/pose.mp4"
