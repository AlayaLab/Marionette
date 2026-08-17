# Controllability demos

These are the recipes behind the control clips on the project page, reduced to one script and
run **pose-only** — no diffusion model, no 23 GB base model, about ninety seconds each on one
GPU instead of an hour.

```bash
bash demos/control_demos.sh                  # all four, into demos/out/
ONLY=action bash demos/control_demos.sh
python3 demos/check_counterfactual.py        # the prefix check -- this is the evidence
python3 demos/make_control_sheet.py          # one PNG per demo
```

Needs the render environment (moderngl + EGL under `python3.12`) and the dynamics weights from
`fetch_weights.sh`.

## Why pose-only

The claim is that control applied to the *state* is obeyed by the *dynamics* model. That claim
is settled in the pose video: if the character does not dodge there, no amount of rendering will
make it dodge. The published RGB versions exist to show the control surviving the whole
pipeline, which is a different and weaker claim.

## What is being controlled

**The hunter**, through `--force_an`. The monster is left free — it is not passed `--force_am`
at all — because a world that keeps evolving while the player is commanded is the point, and
because pinning both makes the clip look staged. In the output the monster wanders in and out
of frame. That is not a bug in the demo; it is the uncontrolled half.

Movement is controlled differently, and deliberately so: `--script_npc` overwrites the root
translation and yaw directly, touching no action token. Where the character *is* and what it is
*doing* are separate control surfaces.

## The action ids are properties of this checkpoint

| id | |
|---|---|
| 1 | IDLE |
| 493 | ATTACK |
| 407 | DODGE |
| 499 | HEAVY (an event — see below) |
| 379 | RUN |

These are indices into the vocabulary built from **this checkpoint's** training corpus. Under a
model trained on a different corpus the same integer is a different animation, or nothing —
between the two corpora used during this project, almost none of the hunter ids survive. If you
retrain, these numbers do not carry over, and nothing will warn you: the rollout will simply not
do what you asked.

## The four demos

### 1 · `free` — no control

Nothing forced; both characters act on their own. Same seed and camera as the action demo, so
they can be read side by side.

### 2 · `action` — a commanded action stream

The project page's control clip. One id written into the hunter's stream every five seconds:
IDLE → ATTACK → DODGE → HEAVY.

**HEAVY is re-triggered rather than held.** Some ids in the vocabulary name *states* and some
name *events*. Holding an event id plays its animation once and then sits in the terminal frame,
which reads exactly like the model ignoring the command. Re-triggering it — 26 frames of the id,
5 frames of idle to reset the edge, repeated — makes it replay. Driving an event id like a state
id is the most common way to conclude, wrongly, that control does not work.

A flat floor (`--flat_terrain`) and a camera locked to the hunter, so nothing but the commanded
action changes the picture.

**Obedience is partial and that is a property of the model,** not something the demo hides. It
is also *contextual*: what the model was doing when a command arrives gates whether the command
fires at all. In the published version, HEAVY would not trigger until the DODGE block before it
was shortened — the hunter was still rolling out of the dodge and could not enter the heavy
attack. The block lengths in this demo are a straight five-second grid, which is easier to read;
the page's version uses uneven blocks for exactly that reason.

### 3 · `counterfactual` — one id apart

Two runs, `stay` and `switch`. Both hold IDLE for the first fifteen seconds. Then `stay` keeps
holding it and `switch` moves to ATTACK. Same seed, seed window, RNG seed, camera and terrain.

So the first fifteen seconds are identical **frame for frame**:

```bash
python3 demos/check_counterfactual.py
```

**Two videos that differ throughout would prove nothing.** The action stream is sampled, so two
runs of the *same* command already differ. A shared prefix that ends exactly where the command
changes is what makes the difference attributable.

Two things had to be right, and both were wrong first:

- **Do not force from frame 0.** The first version did, and its two videos differed at frame 0 —
  more dramatic, evidence of nothing.
- **The switch sits at 3/4 of the horizon, not 1/2.** `--fixed_cam` frames the shot from the
  entity's position at *mid*-horizon, so with the switch at the midpoint each run's camera is
  computed from an already-diverged position and every frame differs, the first included.

### 4 · `move` — commanded movement

The page's other control clip. Four seconds spinning in place, then a new heading every two
seconds at 2.5 m/s.

Two things are driven at once and **both are necessary**:

- `--script_npc` overwrites the root translation and yaw — where the character goes;
- `--force_an ... hold:379` commands a locomotion action — what the legs do.

Driving only the root is the mistake this demo exists to avoid: the body keeps whatever
animation it was in and the character **slides across the ground without taking a step**. Two
seconds of IDLE come first so the transition into running is visible rather than starting
mid-gait.

## Reading the results

`demos/out/<name>/*.mp4` are pose-control videos: **R = world height band, G = entity/skeleton
id, B = inverse depth.** The character reads as a coloured skeleton against the terrain's height
field. Not pretty; unambiguous.

## Knobs

| env | default | |
|---|---|---|
| `SEED_FRAMES` | 32 | frames of real state before prediction starts |
| `ONLY` | | run one demo by name |
| `CULL` | 1 | `0` disables `--cull_occluders` |

Seed segment, horizon and RNG seed are **per demo** and set in the script, because each was
chosen for that demo. Segment 53 is a calm, level window — forcing IDLE from a combat state
makes the hunter drift out of a close-up before the first commanded action lands. Segment 408
was picked the same way for movement. If you change a schedule, change its horizon and the
column table in `make_control_sheet.py` with it.

The `--force_am` / `--force_an` grammar (monster / hunter) is documented in
`bridge/gen_pose_video.py`:

```
free | gt | shuf | hold_common | hold_alt | hold:<id> | retrig:<id>:<on>:<off>
bseq:<spec>;<spec>;...          equal blocks, each resolved by a sub-spec
bseqn:<n1>,<n2>,...:<spec>;...  explicit block lengths in frames
<a>,<b>,<c>,...                 explicit per-frame ids
```

⚠️ Inside a `bseq`, each block resolves its sub-spec against the *first* frames of the
ground-truth window, not its own slice — so `hold_common` in blocks of different lengths can
resolve to different ids. Use explicit `hold:<id>` when a block must be predictable.

## Getting RGB out of these

Any of these pose videos can go through the observation model:

```bash
OUT_DIR=demos/out/02_action bash run_stage2_wan.sh    # needs pose.mp4 + rgb.mp4 in OUT_DIR
```

**Read this first.** The observation model is conditioned on a first frame, and it was trained
with that frame being the pose video's own frame 0. These demos generate new motion from a seed
*state*, and the release ships seed states without their footage — so there is no ground-truth
frame that matches. Whatever reference you supply will disagree with pose frame 0 and identity
will drift at chunk boundaries.

For image quality use `bash run_demo.sh`, which renders a pose video shipped together with its
matching reference. For controllability use these. The two questions want different setups.
