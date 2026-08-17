#!/bin/bash
# One command, one video.
#
#   bash run_demo.sh                      # the shipped aligned window -> RGB   (default)
#   MODE=generate bash run_demo.sh        # generate new motion first, then render it
#   N_CHUNKS=6 bash run_demo.sh           # longer (6 x 81 frames = 16.2 s)
#
# Two modes, because the appearance reference is not a free variable.
#
# The observation model is conditioned on a first frame, and it was trained with that frame
# being *the pose video's own frame 0* (control_ref_image=first_frame). Honouring that means
# having ground-truth video for the same moment the pose starts at -- which the bridge does
# automatically when it seeds from a recorded clip, and which is impossible when it seeds from
# a bare state segment, because the release ships seed states without their footage.
#
#   MODE=aligned (default)  stage 1 already run for you: a pose video and the matching
#                           reference frame, shipped together in the assets repo. This is the
#                           protocol every published result uses. Needs only the diffusion
#                           environment. ~5 min per chunk on one H100.
#
#   MODE=generate           run the dynamics model here and now, from a seed segment, so the
#                           motion is new. The reference frame is then a stand-in from another
#                           moment, the model has to reconcile two disagreeing conditions, and
#                           identity drifts at chunk boundaries. Good for watching the dynamics
#                           model work; not what to judge image quality on.
#
# Needs one GPU with ~40 GB for the observation stage.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MARIONETTE_ROOT=$ROOT

MODE=${MODE:-aligned}
SEG=${SEG:-53}
SEED_FRAMES=${SEED_FRAMES:-32}
N_CHUNKS=${N_CHUNKS:-2}
SAVE_DIR=${SAVE_DIR:-$ROOT/samples_out}
PY1=${PY1:-python3.12}
PY2=${PY2:-python}

case "$MODE" in
  aligned)  NAME=aligned_w1 ;;
  generate) NAME=seg${SEG}_seed${SEED_FRAMES} ;;
  *) echo "MODE must be 'aligned' or 'generate' (got '$MODE')"; exit 2 ;;
esac
OUT_DIR=$ROOT/samples/$NAME

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- preflight
say "[0/3] checking what is in place  (MODE=$MODE)"
missing=0
need="weights/observation/diffusion_pytorch_model.safetensors"
[ "$MODE" = generate ] && need="$need weights/dynamics/pose_gpt.pt \
  weights/dynamics/action_gpt.pt data/seeds/metadata.npz data/first_frame_ref.mp4"
[ "$MODE" = aligned ]  && need="$need data/demo/aligned_pose.mp4 data/demo/aligned_ref.mp4"
for f in $need; do
  [ -s "$ROOT/$f" ] || { echo "  MISSING $f"; missing=1; }
done
[ -e "$ROOT/weights/Wan2.2-Fun-5B-Control/diffusion_pytorch_model.safetensors" ] \
  || { echo "  MISSING weights/Wan2.2-Fun-5B-Control/ (the base model)"; missing=1; }
if [ "$missing" != 0 ]; then
  echo
  echo "Fetch what is missing first:"
  echo "  bash fetch_weights.sh        # our weights + assets, ~11 GB"
  echo "  bash fetch_base_model.sh     # third-party base model, ~23 GB"
  exit 3
fi
echo "  weights, assets and base model present"

# Probe by importing, not by looking for an interpreter: having python3.12 on PATH says
# nothing about whether moderngl can make an EGL context.
CAN1=0; CAN2=0
have "$PY1" && "$PY1" -c "import moderngl, torch, numpy" >/dev/null 2>&1 && CAN1=1
have "$PY2" && "$PY2" -c "import torch, diffusers, decord" >/dev/null 2>&1 && CAN2=1
echo "  stage 1 (dynamics + bridge, $PY1): $([ $CAN1 = 1 ] && echo available || echo 'not available here')"
echo "  stage 2 (observation, $PY2):       $([ $CAN2 = 1 ] && echo available || echo 'not available here')"

if [ "$CAN2" != 1 ]; then
  echo
  echo "Stage 2 cannot run in this environment, and it is the stage that produces the video."
  echo "It needs torch>=2.8 (cu128), diffusers and decord. See README, 'Two environments'."
  exit 4
fi

# ---------------------------------------------------------------- stage 1
mkdir -p "$OUT_DIR"
if [ "$MODE" = aligned ]; then
  say "[1/3] using the shipped pose video and its matching reference frame"
  cp -f "$ROOT/data/demo/aligned_pose.mp4" "$OUT_DIR/pose.mp4"
  cp -f "$ROOT/data/demo/aligned_ref.mp4"  "$OUT_DIR/rgb.mp4"
  echo "  pose: dynamics-model output, seeded from a recorded window"
  echo "  ref : ground-truth video for the same moment, so pose[0] and the reference agree"
elif [ -s "$OUT_DIR/pose.mp4" ]; then
  say "[1/3] pose video already present, reusing $OUT_DIR/pose.mp4"
else
  if [ "$CAN1" != 1 ]; then
    echo
    echo "MODE=generate needs the render environment (moderngl + EGL under $PY1),"
    echo "which is not available here. Use the default MODE=aligned instead."
    exit 5
  fi
  say "[1/3] stage 1 -- $SEED_FRAMES seed frames -> 340 frames of predicted state -> pose video"
  SEG=$SEG SEED_FRAMES=$SEED_FRAMES OUT_DIR=$OUT_DIR bash "$ROOT/run_stage1_render.sh" || exit 5
  echo
  echo "  NOTE: the reference frame here is data/first_frame_ref.mp4, a frame from a different"
  echo "  moment. It disagrees with pose frame 0, and the rollout will show that -- most"
  echo "  visibly as the monster's appearance shifting at a chunk boundary. Judge the motion"
  echo "  here and the image quality in MODE=aligned."
fi

# ---------------------------------------------------------------- stage 2
say "[2/3] stage 2 -- $N_CHUNKS chunks x 81 frames at 704x1280, chunk-relay"
echo "  roughly 5 min per chunk on one H100; nothing is printed per frame"
OUT_DIR=$OUT_DIR N_CHUNKS=$N_CHUNKS SAVE_DIR=$SAVE_DIR PY=$PY2 \
  bash "$ROOT/run_stage2_wan.sh" || exit 7

# ---------------------------------------------------------------- report
OUT=$SAVE_DIR/${NAME}_s0_n${N_CHUNKS}/rollout.mp4
say "[3/3] done"
if [ -s "$OUT" ]; then
  echo "  RGB rollout : $OUT  ($(du -h "$OUT" | cut -f1))"
  echo "  pose control: $OUT_DIR/pose.mp4"
  if [ "$MODE" = aligned ]; then
    echo "  reference   : data/demo/aligned_rollout_n6.mp4  (our 6-chunk run of the same input)"
    echo "                Its first $((N_CHUNKS * 81)) frames are the same content as yours and"
    echo "                should look the same -- but not pixel for pixel. Measured across two"
    echo "                machines: max channel difference 36/255, mean 0.31. Bit-identical"
    echo "                output only holds within one machine; GPUs differ in floating-point"
    echo "                evaluation order and 40 diffusion steps amplify it. Compare by eye."
  fi
  echo
  echo "  pose.mp4 is what the model was told, rollout.mp4 is what it painted, frame for frame."
  echo "  Play them side by side -- that comparison is the whole claim."
  echo
  echo "  Controllability toys, pose-only and much cheaper: demos/"
else
  echo "  stage 2 reported success but $OUT is not there" >&2
  exit 8
fi
