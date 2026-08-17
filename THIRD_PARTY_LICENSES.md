# Third-Party Components

Every third-party component this project introduces, how it is introduced, and under what
licence. Copyright notices are preserved in the source files; third-party components remain
under their original licences.

This project's own code is Apache-2.0 (`LICENSE`); its weights and data-derived assets are
research/non-commercial (`LICENSE.assets`). Neither upstream below conflicts with that: both
are Apache-2.0, and no strong-copyleft (GPL/LGPL/AGPL) code is present anywhere in the tree.

---

## 1. VideoX-Fun — diffusion pipeline and model code

| | |
|---|---|
| **Directory** | `observation/videox_fun/`, `observation/config/`, `observation/examples/` |
| **Source** | aigc-apps |
| **Repository** | https://github.com/aigc-apps/VideoX-Fun |
| **Licence** | Apache License 2.0 |
| **Introduced as** | **vendored** — copied into this repository, not a pip dependency |
| **Modified** | Yes — defaults only, in one file; see `observation/PROVENANCE.md` |
| **Used by** | the observation stage: `observation/examples/wan2.2_fun/predict_mh_pose_ar_baseline.py` and everything it imports |

The upstream `LICENSE` is kept at `observation/LICENSE` and `observation/PROVENANCE.md` records
the upstream commit, the copy date, and the exact modifications.

Four upstream subdirectories were **removed** from this release rather than shipped:
`video_caption/`, `ui/`, `api/`, `reward/`. None is on the inference path. `video_caption/`
additionally carried third-party sample video clips whose redistribution rights were not clear,
which is the reason it is removed rather than merely unused.

---

## 2. Wan2.2-Fun-5B-Control — base model

| | |
|---|---|
| **Source** | Alibaba-PAI |
| **Repository** | https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control |
| **Licence** | Apache License 2.0 |
| **Introduced as** | **runtime download** — not redistributed by this project |
| **Used by** | the observation stage loads its transformer (as the fine-tuning base), its VAE, and its umT5-xxl text encoder |

`fetch_base_model.sh` downloads it from the original source. It is not vendored and not
re-hosted, so its licence and provenance stay attached to it. Our released observation weights
are a **fine-tune of this model**; Apache-2.0 permits redistributing that derivative, and our
weights are additionally restricted by `LICENSE.assets` because of the training data, not
because of this base model.

---

## 3. Python dependencies (pip)

Ordinary pip dependencies, not vendored, installed by the user from PyPI under their own
licences. The significant ones:

| package | licence |
|---|---|
| torch, torchvision | BSD-3-Clause |
| diffusers, transformers, accelerate, huggingface_hub, safetensors | Apache-2.0 |
| numpy, scipy | BSD-3-Clause |
| opencv-python | Apache-2.0 (bundled FFmpeg components: LGPL-2.1+) |
| imageio, imageio-ffmpeg | BSD-2-Clause (bundled FFmpeg binary: LGPL-2.1+) |
| decord | Apache-2.0 |
| moderngl, glcontext | MIT |
| Pillow | MIT-CMU |
| einops | MIT |
| omegaconf | BSD-3-Clause |

`imageio-ffmpeg` and `opencv-python` ship FFmpeg builds under LGPL-2.1+. They are used as
separate executables / dynamically linked libraries and are not statically combined with this
project's code, so no copyleft obligation propagates to it. No package here is GPL.

All are released versions on PyPI; none is pinned to an unreleased development branch, so
`pip install -r requirements.txt` is reproducible without any out-of-band instructions.

---

## 4. Source corpus

| | |
|---|---|
| **Name** | WildWorld |
| **Source** | Alaya Lab (this project's own group) |
| **Repository** | https://github.com/AlayaLab/WildWorld · https://huggingface.co/datasets/AlayaLab/WildWorld |
| **Licence** | non-commercial research use; redistribution not permitted |
| **Introduced as** | the corpus the released weights were trained on, and the origin of the seed/terrain/reference assets |

This is why the weights and assets in this project carry `LICENSE.assets` rather than
Apache-2.0. See `NOTICE`.
