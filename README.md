# Marionette: Predicting World States, Rendering Geometry, Painting Appearance

<p align="center"><a href="https://alayalab.ai/"><b>Alaya Lab</b></a></p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-2563eb?style=for-the-badge"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-e5e7eb?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.14530"><img src="https://img.shields.io/badge/arXiv-2608.14530-b31b1b?logo=arxiv"></a>
  <a href="https://alayalab.github.io/Marionette/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="https://github.com/AlayaLab/Marionette"><img src="https://img.shields.io/badge/Code-Available-brightgreen?logo=github"></a>
  <a href="https://huggingface.co/AlayaLab/Marionette"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HuggingFace-yellow"></a>
  <a href="https://github.com/AlayaLab/WildWorld"><img src="https://img.shields.io/badge/Corpus-WildWorld-orange"></a>
  <img src="https://img.shields.io/badge/Code-Apache--2.0-blue">
  <img src="https://img.shields.io/badge/Weights%20%26%20Assets-Research%20Only-red">
</p>

<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

> A world model that models the world rather than its pixels: a dynamics model predicts an
> explicit articulated state, a zero-parameter bridge turns that state into geometry, and a
> video model paints appearance onto it.

<p align="center">
  <a href="https://youtu.be/bLLtwXVcqEc"><img src="assets/video_thumb.jpg" width="100%"></a>
  <br><a href="https://youtu.be/bLLtwXVcqEc"><b>&#9654; Watch the overview video</b></a>
</p>

---

## 📰 News

- **[2026-08-17]** Paper on arXiv — [2608.14530](https://arxiv.org/abs/2608.14530). Overview
  video: [YouTube](https://youtu.be/bLLtwXVcqEc).
- **[2026-08-13]** Inference code, runtime assets and the controllability demos released.
- **[2026-08-13]** Project page released.

## 🚀 Release Roadmap

- [x] Project page
- [x] Inference code — the full three-stage pipeline
- [x] Runtime assets — seeds, terrain and references
- [x] Controllability demos, with the assertion script that checks them
- [x] Paper — [arXiv:2608.14530](https://arxiv.org/abs/2608.14530)
- [ ] Pretrained weights — 🤗 [`AlayaLab/Marionette`](https://huggingface.co/AlayaLab/Marionette) *(uploading)*
- [ ] Training code

## ✨ What this is

Interactive world models are usually trained to autoregress appearance directly, in pixel or
latent space. Long-horizon consistency, controllability and persistence then exist only as
byproducts of a sequence model, and are correspondingly fragile.

Marionette separates the parts that must be exact from the part that must look right.

### 🎯 Explicit state
The dynamics model predicts a 276D articulated world state — joint rotations and root motion for
each entity — not pixels. Geometry, contact and occlusion are properties of that state, so they
are exact by construction rather than learned approximately.

### 📐 A bridge with no parameters
The state becomes a pose-control video by fixed geometric operations: forward kinematics, terrain
collision, rasterisation. Nothing here is learned, so nothing here can drift, hallucinate, or
need more data.

### 🎮 Control that survives the pipeline
Because control is applied to the state, it is obeyed by the dynamics model and then carried
through rendering unchanged. Overriding one action id changes the character's behaviour and
nothing else — `demos/` demonstrates this and `demos/check_counterfactual.py` asserts it, by
showing two runs identical frame-for-frame until the commanded moment.

### 🖼️ Appearance, and only appearance
The observation model is a control-conditioned video diffusion model. It is asked to paint a
geometry it is given, not to remember where things are.

## Layout

```
dynamics/     ActionGPT + PoseGPT and the rollout code
bridge/       state -> world geometry -> rasterised pose-control video (moderngl/EGL)
observation/  control-conditioned video diffusion (videox_fun) and the rollout script
weights/      fetched, not vendored (gitignored)
data/         seeds, scanned terrain, skeleton filters, appearance reference, demo inputs
gallery/      contact sheets of every published sample, and how each was produced
demos/        controllability toys: forced actions, scripted movement, counterfactual pairs
hf/           model cards and the weight-upload script
run_demo.sh   one command, one video
run_stage1_render.sh / run_stage2_wan.sh    the two stages, with the knobs exposed
check_closure.py   asserts every import in this tree resolves
requirements.txt / requirements-bridge.txt  one per environment, see below
```

`dynamics/`, `bridge/` and `observation/` are vendored with their original layout preserved, so
that checking this package against the tree it came from tests the packaging rather than a
rewrite. Four hardcoded paths were changed and nothing else:

| file | change |
|---|---|
| `bridge/gen_pose_video.py` | the two repo roots now derive from the file location |
| `bridge/render_pose_terrain_gl.py` | data root defaulted to a path that does not exist outside the machine it was written on |
| `observation/examples/.../predict_mh_pose_ar_baseline.py` | checkpoint, prompt, dataset and resolution defaults |
| `run_stage1_render.sh` | pins `--torch_seed` (see *Reproducibility*) |

Upstream provenance for `observation/` — the exact commit, what was modified, and what was
removed — is in [`observation/PROVENANCE.md`](observation/PROVENANCE.md).

## Two environments

The two stages **cannot share one environment**, and this is the single most common way to
lose an afternoon here:

- **Stage 1** needs `moderngl` with an EGL context and a `python3.12` interpreter. It has no
  GPU-decode dependency. Set `POSE_ENCODER=libx264` unless an nvenc-enabled ffmpeg is present.
- **Stage 2** needs `torch>=2.8` (cu128), `diffusers`, and `decord`. `decord` does not build
  for the python that the render stack wants, which is why these are two environments and two
  scripts rather than one pipeline process.

This is also why the assets repo ships a precomputed stage-1 pose video: without it, anyone with
one environment can run half a pipeline and see nothing.

## Weights

Ours, on HuggingFace — `fetch_weights.sh` puts them in `weights/`:

| file | size | what |
|---|---|---|
| `observation/diffusion_pytorch_model.safetensors` | 10.0 GB | the fine-tuned control transformer |
| `dynamics/pose_gpt.pt` | 402 MB | PoseGPT (animation) |
| `dynamics/action_gpt.pt` | 61 MB | ActionGPT (decision) |

The backbone is **not** ours to redistribute. It is a third-party release and is fetched from
its source:

```bash
bash fetch_base_model.sh                                  # ~23 GB from HuggingFace
bash fetch_base_model.sh --link /path/to/local/copy       # or reuse one you already have
```

## Run it

```bash
# stage 1, in the render environment
bash run_stage1_render.sh                       # seed segment 53 -> samples/seg53_seed32/pose.mp4

# stage 2, in the Wan environment
bash run_stage2_wan.sh                          # -> samples_out/seg53_seed32_s0_n6/rollout.mp4
```

Both read overrides from the environment: `SEG`, `SEED_FRAMES`, `HORIZON`, `TEMP`, `TORCH_SEED`
for stage 1; `N_CHUNKS`, `WAN2_2_SEED`, `CKPT`, `PROMPT` for stage 2.

**The checkpoint and the prompt are one pair.** The shipped model was trained on text carrying
`hunter appearance id N`; pairing it with the older prompt, or pairing an older checkpoint with
this prompt, puts the model out of distribution and monsters drop out of the rollout. Change
both or neither.

## Reproducibility

The action stream is sampled (`temperature > 0` goes through `torch.multinomial`), so stage 1
is only reproducible with a fixed seed. Upstream leaves the seed unset; `run_stage1_render.sh`
pins `TORCH_SEED=43` by default. Pass `TORCH_SEED=-1` for the unseeded behaviour.

With the seed pinned, this package was checked against the tree it was assembled from, both run
back to back in one job on one GPU so the only variable was which tree the code and weights came
from. Stage 1 came out byte-identical; stage 2 came out identical on all 162 decoded frames.
Stage 2 is compared on decoded pixels rather than file bytes, because a video encoder is not
obliged to be bit-stable even when the tensors going into it are.

One caveat worth stating, because it will otherwise look like a bug: **the digest is not
portable, only the equality is.** The same command with the same seed and the same weights
produces a different video on a different machine — floating-point evaluation order is enough to
move one `torch.multinomial` draw, and the motion diverges from there. Do not check your own run
against a published hash.

## Seeds and data

`data/seeds/` carries `metadata.npz` (normalisation statistics and the action vocabularies)
plus **two** seed segments, not the full corpus: **53** and **408**, the two the published
control demos are built on. They were chosen rather than sampled — 53 is a calm, level window,
which matters because forcing IDLE from a combat state makes the hunter drift metres out of a
close-up before the first commanded action lands; 408 was picked the same way for movement.
Point `--combined_dir` at a full copy to use others.

`metadata.npz` still indexes all 2241 segments of the corpus. That is deliberate: normalisation
and the action-token lookup are read from it, so truncating it would silently change them.
Asking for a segment that is not here fails on a missing file rather than with a wrong answer.

`data/terrain/` is the scanned height field the bridge and the dynamics model both consume,
produced by this work and not part of the corpus release. Only `stage_101` ships: both entry
points pass `--stage 101`, and the loader reads a single stage directory, so the other scanned
stages are unreachable from the seeds published here.

## Licences

This project carries **two** licences, because its code and its data-derived artifacts do not
grant the same rights.

| | licence | |
|---|---|---|
| Code | Apache License 2.0 | [`LICENSE`](LICENSE) |
| Model weights and runtime assets | non-commercial research only, no redistribution | [`LICENSE.assets`](LICENSE.assets) |

The weights and the assets — seed states, scanned terrain, appearance reference, demo videos and
gallery stills — derive from the **WildWorld** dataset, released for non-commercial research use
and not redistributable. Those terms flow through to anything derived from it, so they apply
here too.

- **WildWorld** (source corpus): <https://github.com/AlayaLab/WildWorld> ·
  <https://huggingface.co/datasets/AlayaLab/WildWorld>
- **Wan2.2-Fun-5B-Control** (base model: transformer, VAE, umT5-xxl text encoder): Apache-2.0,
  Alibaba-PAI, <https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control>. **Not redistributed
  here** — `fetch_base_model.sh` downloads it from the original source.
- **VideoX-Fun** (vendored under `observation/`): Apache-2.0, aigc-apps,
  <https://github.com/aigc-apps/VideoX-Fun>. Upstream commit, modifications and removals are
  recorded in [`observation/PROVENANCE.md`](observation/PROVENANCE.md).

Full component list: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md). Attribution and the
licensing rationale: [`NOTICE`](NOTICE).

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
