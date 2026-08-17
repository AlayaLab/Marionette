---
# NOT apache-2.0. This repository contains the weights and nothing else, and the weights are
# research-only: the restriction is inherited from the WildWorld corpus they were trained on.
# Apache-2.0 here would tell the Hub's filters, and anyone reading the badge, that these are
# free for commercial use and redistribution -- the opposite of the terms below. The code is
# Apache-2.0, but the code lives in the GitHub repository, not in this one.
license: other
license_name: marionette-research-only
license_link: https://github.com/AlayaLab/Marionette/blob/main/LICENSE.assets
library_name: marionette
pipeline_tag: video-to-video
tags:
  - world-model
  - video-generation
  - game
  - pose-control
  - autoregressive
  - arxiv:2608.14530
---

# Marionette — weights

Checkpoints for **Marionette: Predicting World States, Rendering Geometry, Painting
Appearance** ([arXiv:2608.14530](https://arxiv.org/abs/2608.14530)). Code, runnable scripts and
a gallery: <https://github.com/AlayaLab/Marionette>. Project page:
<https://alayalab.github.io/Marionette/>.

<p align="center">
  <a href="https://youtu.be/bLLtwXVcqEc"><img src="https://raw.githubusercontent.com/AlayaLab/Marionette/main/assets/video_thumb.jpg" width="100%"></a>
  <br><a href="https://youtu.be/bLLtwXVcqEc"><b>&#9654; Watch the overview video</b></a>
</p>

Marionette factorises an interactive game world model into three stages. Only the first and
the third carry weights; the middle one is fixed geometry.

```
seed pose ──▶ dynamics ──▶ 276D world state ──▶ bridge ──▶ pose-control video ──▶ observation ──▶ RGB
              ActionGPT                          zero-parameter,                  Wan2.2-Fun-5B-Control
              + PoseGPT                          deterministic                    fine-tune, chunk-relay
```

## Files

| file | size | stage | what it is |
|---|---|---|---|
| `observation/diffusion_pytorch_model.safetensors` | 10.0 GB | observation | control-conditioned video diffusion transformer, fine-tuned from Wan2.2-Fun-5B-Control |
| `dynamics/pose_gpt.pt` | 402 MB | dynamics | PoseGPT — predicts the next articulated state |
| `dynamics/action_gpt.pt` | 61 MB | dynamics | ActionGPT — predicts the next action token |

**These are not standalone.** The observation stage loads the VAE and the umT5-xxl text encoder
from the third-party base release, which is not redistributed here:

```bash
bash fetch_base_model.sh          # alibaba-pai/Wan2.2-Fun-5B-Control, ~23 GB, Apache-2.0
```

Seeds, the scanned terrain and the appearance reference are in the code repository, not here —
they pack small enough that a separate download would buy nothing.

## Use

```bash
git clone https://github.com/AlayaLab/Marionette && cd Marionette
bash fetch_weights.sh                      # the weights in this repo
bash fetch_base_model.sh                   # the third-party base model
bash run_demo.sh                           # -> samples_out/.../rollout.mp4
```

Inference settings the released checkpoint was evaluated at: 704×1280, 40 steps, guidance 6.0,
81-frame chunks, 30 fps.

## The prompt is part of the checkpoint

The observation model was trained on captions carrying an explicit appearance id:

```
Monster Hunter Wilds video game gameplay, stage 101, hunter appearance id 9 wielding weapon type 4, fighting monster id 19.
```

Pairing this checkpoint with an older caption format — or an older checkpoint with this one —
puts the text encoder out of distribution, and the failure is not subtle: monsters drop out of
the rollout entirely. Change the checkpoint and the prompt together or neither.

## Training

**Observation.** Fine-tuned from `alibaba-pai/Wan2.2-Fun-5B-Control` in `control_ref` mode with
the first frame as the appearance reference, at 704×1280, 16×H200 FSDP. The released checkpoint
is step 17000. The last 2000 of those steps are on a corpus recorded with a revised capture
path in which objects absent from the pose-control signal are also absent from the RGB — mounts
in particular — which removes a class of targets the model previously had no way to predict.

Earlier stages of training span 26 monster species; the final segment is a single configuration
(one stage, one monster, one weapon class). The checkpoint is correspondingly strongest there,
and the multi-species montage on the project page is rendered from an earlier checkpoint for
exactly this reason.

**Dynamics.** ActionGPT and PoseGPT over the 276D articulated state at 20 fps, conditioned on a
scanned terrain height field, trained on a single monster species (em19) over 2241 segments.
Action vocabularies: 168 for the monster, 977 for the hunter.

**Action ids are indices into this checkpoint's vocabulary and are not portable.** The
vocabulary is built from the training corpus, so the same integer under a model trained on a
different corpus is a different animation or nothing at all — an earlier corpus here shares
almost none of this one's hunter ids. When the demos name `493` as ATTACK, that is a fact about
this checkpoint.

## Reproducibility

The action stream is sampled, so the dynamics stage is only reproducible with a pinned seed.
`run_stage1_render.sh` pins `TORCH_SEED=43`; upstream leaves it unset. With the seed pinned,
the repository's `verify_reproduction.sh` reproduces its recorded pose video byte for byte and
its RGB rollout pixel for pixel.

## Scope and limitations

- The dynamics model covers **one monster species**. It is not a general character-motion model.
- Rollout is chunk-relay: each 81-frame chunk is conditioned on the last frame of the previous
  one, so appearance error compounds with horizon.
- The bridge consumes a terrain height field scanned for specific stages. Novel geometry needs
  a new scan.
- Research artifact. Not a product, not a game, not a renderer for anything but this pipeline.

## Licence and attribution

- **Our code**: Apache-2.0 (see the code repository's `LICENSE`).
- **These weights**: non-commercial research use only, no redistribution — see `LICENSE.assets`.
  The restriction comes from the training corpus, not from the base model.
- **Base model** `alibaba-pai/Wan2.2-Fun-5B-Control`: Apache-2.0, not redistributed here.
- **VideoX-Fun** (`aigc-apps/VideoX-Fun`), which the observation code is vendored from: Apache-2.0.
- **Training corpus**: the WildWorld dataset — https://github.com/AlayaLab/WildWorld — released
  for non-commercial research use, redistribution not permitted.

Rights in the game content the corpus was recorded from remain with its publisher, and this
project grants no rights in that content.

**Two files in this project are unmodified recorded gameplay video**, not model output:
`data/first_frame_ref.mp4` and `data/demo/aligned_ref.mp4`. The observation model is conditioned
on a reference frame, and these supply it. They are short clips from the same recordings the
corpus was built from, and they are covered by the research-only terms above. Everything else
distributed here — seed states, scanned terrain, pose videos, rollouts and the gallery stills —
is numeric derivation or model output.

## Citation

```bibtex
@article{meng2026marionette,
  title   = {Marionette: Predicting World States, Rendering Geometry, Painting Appearance},
  author  = {Meng, Zian and Li, Zhen and Li, Chuanhao and Li, Qiang and Zhang, Kaipeng},
  journal = {arXiv preprint arXiv:2608.14530},
  year    = {2026}
}
```
