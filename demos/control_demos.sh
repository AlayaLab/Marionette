#!/bin/bash
# Controllability demos. These are the recipes behind the control clips on the project page,
# reduced to a single script and run pose-only, so each takes about ninety seconds on one GPU
# instead of an hour of diffusion.
#
#   bash demos/control_demos.sh              # all four
#   ONLY=action bash demos/control_demos.sh
#
# Output: demos/out/<name>/*.mp4 -- pose-control videos, the same signal the observation model
# consumes. Turning them into RGB is a separate, much more expensive step; see demos/README.md.
#
# Why pose-only. The claim is that control applied to the *state* is obeyed by the *dynamics*
# model. That is settled in the pose video: if the character does not dodge there, no amount of
# rendering will make it dodge.
#
# Runs in the RENDER environment (moderngl + EGL, python3.12) -- the same one as
# run_stage1_render.sh. See README, "Two environments".
#
# ---------------------------------------------------------------------------------------------
# THE PARAMETERS ARE NOT ARBITRARY. Each demo below took several rounds to find, and the
# settings that look like taste are mostly load-bearing. In particular:
#
#   The controlled entity is the HUNTER, driven through --force_an. The monster is left free
#   (--force_am is not passed) because a world that keeps evolving while the player is commanded
#   is the point, and because pinning it makes the clip look staged.
#
#   The seed segments are chosen, not defaults. Segment 53 is a calm, flat window: forcing IDLE
#   from a combat state makes the hunter drift metres out of a close-up before the first
#   commanded action lands. Segment 408 was picked for the movement demo the same way.
#
#   Action ids are indices into THIS checkpoint's vocabulary. They are not portable: the same
#   integer in a model trained on a different corpus is a different animation, or nothing.
# ---------------------------------------------------------------------------------------------
set -uo pipefail
ROOT=$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)
export MARIONETTE_ROOT=$ROOT

SEED_FRAMES=${SEED_FRAMES:-32}
TEMP=${TEMP:-0.9}
OUT=${OUT:-$ROOT/demos/out}
PY=${PY:-python3.12}
ONLY=${ONLY:-}

export POSE_DATA_DIR=$ROOT/data
export POSE_TERRAIN_DIR=$ROOT/data/terrain
export POSE_FILTER_DIR=$ROOT/data/filter
export POSE_ENCODER=${POSE_ENCODER:-libx264}

for f in weights/dynamics/pose_gpt.pt weights/dynamics/action_gpt.pt \
         data/seeds/metadata.npz; do
  [ -s "$ROOT/$f" ] || { echo "MISSING $f -- run: bash fetch_weights.sh"; exit 3; }
done

# Hunter action ids in the released checkpoint's vocabulary.
IDLE=1; ATTACK=493; DODGE=407; HEAVY=499; RUN=379

gen() {   # $1=subdir $2=label $3=seg $4=horizon $5=torch_seed $6...=extra args
  local sub=$1 d=$OUT/$1 lab=$2 seg=$3 hz=$4 sd=$5; shift 5
  local cull=--cull_occluders
  [ "${CULL:-1}" = "0" ] && cull=""
  mkdir -p "$d"
  # $sub, not $1: the positional args have been shifted away by now, and printing $1 here
  # reported the first --force flag as the output directory.
  echo "  -> $sub/$lab.mp4  (seg $seg, horizon $hz = $((hz / 20)) s, seed $sd)"
  "$PY" "$ROOT/bridge/gen_pose_video.py" \
    --combined_dir "$ROOT/data/seeds" \
    --terrain_pose_ckpt "$ROOT/weights/dynamics/pose_gpt.pt" \
    --terrain_action_ckpt "$ROOT/weights/dynamics/action_gpt.pt" \
    --stage 101 --segment "$seg" --seed_frames "$SEED_FRAMES" --horizon "$hz" \
    --temperature "$TEMP" --torch_seed "$sd" \
    $cull --collide --fps 30 --gen_fps 20 \
    "$@" --out "$d/$lab.mp4" >"$d/$lab.log" 2>&1 \
    || { echo "     FAILED (see $d/$lab.log)"; return 1; }
}

run() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

# ---------------------------------------------------------------------------------- 1
# Baseline: nothing forced, both characters act on their own. Same seed and camera as the
# action demo, so the two can be put side by side.
if run free; then
  echo "[1] free -- no control at all"
  gen 01_free free 53 400 1 --force_an free --flat_terrain --npc_lock_cam --npc_cam_dist 6
fi

# ---------------------------------------------------------------------------------- 2
# THE ACTION DEMO -- this is the project page's control clip, pose-only.
#
# One action id written into the hunter's stream every five seconds:
#   IDLE -> ATTACK -> DODGE -> HEAVY
#
# HEAVY is re-triggered, not held. Some ids in the vocabulary name states and some name events;
# holding an event id plays its animation once and then sits in the terminal frame. Re-triggering
# it (26 frames of the id, 5 of idle to reset the edge) makes it replay.
#
# A flat floor and a camera locked to the hunter: nothing but the commanded action changes the
# picture. The monster stays free and wanders in and out, which is honest -- it is not being
# controlled.
if run action; then
  echo "[2] action -- IDLE / ATTACK / DODGE / HEAVY, five seconds each"
  gen 02_action action 53 400 0 \
    --force_an "bseq:hold:$IDLE;hold:$ATTACK;hold:$DODGE;retrig:$HEAVY:26:5" \
    --flat_terrain --npc_lock_cam --npc_cam_dist 6
fi

# ---------------------------------------------------------------------------------- 3
# The counterfactual pair: identical for 15 s, then one id changes.
#
# Both runs hold IDLE for the first three quarters of the horizon. Then `stay` keeps holding it
# and `switch` moves to ATTACK. Same seed, seed window, RNG seed, camera and terrain, and the
# first 15 s are identical FRAME FOR FRAME -- which is checked, not asserted:
#
#     python3 demos/check_counterfactual.py
#
# Two runs that merely differ throughout prove nothing: the action stream is sampled, so two
# runs of the same command already differ. A shared prefix ending exactly at the commanded
# switch is what makes the difference attributable.
#
# The switch is at 3/4 of the horizon rather than 1/2 because --fixed_cam frames the shot from
# the entity's position at MID-horizon; with the switch at the midpoint each run's camera is
# computed from an already-diverged position and every frame differs, first one included.
if run counterfactual; then
  echo "[3] counterfactual -- identical for 15 s, then IDLE becomes ATTACK"
  gen 03_counterfactual stay   53 400 0 --force_an "hold:$IDLE" \
      --flat_terrain --fixed_cam --cam_focus npc
  gen 03_counterfactual switch 53 400 0 --force_an "bseqn:300,100:hold:$IDLE;hold:$ATTACK" \
      --flat_terrain --fixed_cam --cam_focus npc
fi

# ---------------------------------------------------------------------------------- 4
# THE MOVEMENT DEMO -- the page's other control clip.
#
# Two things are driven at once, and both are needed:
#   --script_npc overwrites the root translation and yaw (where the character goes)
#   --force_an ... hold:$RUN  commands a locomotion action  (what the legs do)
#
# Driving only the root is the mistake this demo exists to avoid: the body keeps whatever
# animation it was in and the character slides across the ground without taking a step. Two
# seconds of IDLE first, so the transition into running is visible rather than starting mid-gait.
#
# Four seconds spinning in place, then a new heading every two seconds at 2.5 m/s -- the
# schedule the project page shows. The horizon matches it; running longer would spend the clip
# on the model doing as it likes after the script ends.
if run move; then
  echo "[4] move -- scripted root translation plus a commanded gait"
  gen 04_move move 408 320 0 \
    --script_npc --script_spin_secs 4 --script_spin_turns 2 \
    --script_run_seg_secs 2 --script_run_speed 2.5 --script_run_dirs 0,90,180,270 \
    --force_an "bseqn:40:hold:$IDLE;hold:$RUN" \
    --npc_lock_cam --npc_cam_dist 5
fi

echo
echo "pose videos in $OUT"
find "$OUT" -name '*.mp4' | sort | sed 's|^|  |'
echo
echo "Next:"
echo "  python3 demos/check_counterfactual.py     # the prefix check -- this is the evidence"
echo "  python3 demos/make_control_sheet.py       # one PNG per demo"
