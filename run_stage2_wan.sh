#!/bin/bash
# STAGE 2 — observation model.  pose-control video + first frame -> RGB, chunk-relay rollout.
#
# Runs in the WAN environment (torch 2.8+cu128, decord, diffusers). Consumes stage 1's
# $OUT_DIR/{pose.mp4, rgb.mp4}.
#
#   bash run_stage2_wan.sh
#   SEG=134 N_CHUNKS=8 bash run_stage2_wan.sh
#
# CKPT and PROMPT are one pair and must move together — see the note in
# observation/examples/wan2.2_fun/predict_mh_pose_ar_baseline.py.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MARIONETTE_ROOT=$ROOT

SEG=${SEG:-133}
SEED_FRAMES=${SEED_FRAMES:-32}
OUT_DIR=${OUT_DIR:-$ROOT/samples/seg${SEG}_seed${SEED_FRAMES}}
N_CHUNKS=${N_CHUNKS:-6}
SAVE_DIR=${SAVE_DIR:-$ROOT/samples_out}
PY=${PY:-python}

[ -s "$OUT_DIR/pose.mp4" ] || { echo "no $OUT_DIR/pose.mp4 -- run stage 1 first"; exit 3; }
[ -s "$OUT_DIR/rgb.mp4" ]  || { echo "no $OUT_DIR/rgb.mp4 (appearance reference)"; exit 3; }

cd "$ROOT/observation"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WAN2_2_CONFIG_PATH=config/wan2.2/wan_civitai_5b.yaml
export WAN2_2_MODEL_PATH=${WAN2_2_MODEL_PATH:-$ROOT/weights/Wan2.2-Fun-5B-Control}
export WAN2_2_TRANSFORMER_PATH=${CKPT:-$ROOT/weights/observation/diffusion_pytorch_model.safetensors}
export WAN2_2_PROMPT=${PROMPT:-"Monster Hunter Wilds video game gameplay, stage 101, hunter appearance id 9 wielding weapon type 4, fighting monster id 19."}
export MH_DATASET_BASE=$(dirname "$OUT_DIR")
export MH_SAMPLES=$(basename "$OUT_DIR")
export MH_START_FRAMES=0
export WAN2_2_SAMPLE_H=704 WAN2_2_SAMPLE_W=1280
export MH_N_CHUNKS=$N_CHUNKS MH_CHUNK_LEN=81
export WAN2_2_FPS=30 WAN2_2_NUM_INFERENCE_STEPS=40 WAN2_2_GUIDANCE_SCALE=6.0
export WAN2_2_SEED=${WAN2_2_SEED:-43}
export WAN2_2_SAVE_PATH=$SAVE_DIR

echo "[stage2] $MH_SAMPLES n_chunks=$N_CHUNKS seed=$WAN2_2_SEED"
echo "         ckpt=$WAN2_2_TRANSFORMER_PATH"
"$PY" -u examples/wan2.2_fun/predict_mh_pose_ar_baseline.py || { echo "STAGE2 FAILED"; exit 4; }
echo "STAGE2 DONE -> $SAVE_DIR/${MH_SAMPLES}_s0_n${N_CHUNKS}/rollout.mp4"
