# Provenance — `observation/`

This directory is **vendored**: the code was copied into this repository rather than declared as
a dependency, because the observation pipeline is coupled to a specific upstream revision and a
pip release matching it does not exist.

| | |
|---|---|
| **Upstream project** | VideoX-Fun |
| **Upstream repository** | https://github.com/aigc-apps/VideoX-Fun |
| **Upstream commit** | `7671af8b16701319cf043358941277b9a5a1cb75` (2026-01-20) |
| **Upstream licence** | Apache License 2.0 — full text kept at `observation/LICENSE` |
| **Copied** | 2026-07-31, refreshed 2026-08-13 |
| **Used by** | the observation stage — `run_stage2_wan.sh` → `examples/wan2.2_fun/predict_mh_pose_ar_baseline.py` and its imports |

## What was changed

Upstream code is otherwise **unmodified**. Every difference from the commit above is listed here.

**Modified upstream file — 1:**

- `videox_fun/utils/utils.py` (+6 lines). Stop decoding a control video once enough frames have
  been read. Upstream decodes the whole file into memory; the clips this project uses run to
  tens of GB and exhausted RAM before the frames were sliced.

**Files added by us inside the upstream package — 2.** Both are training-time dataset classes,
not reached by the inference entry point; they are our own work under this project's Apache-2.0
licence, placed here to sit alongside the upstream dataset classes they parallel:

- `videox_fun/data/dataset_pose_camera.py`
- `videox_fun/data/dataset_chunked.py`

**Files added by us outside the upstream package — 1:**

- `examples/wan2.2_fun/predict_mh_pose_ar_baseline.py` — the chunk-relay rollout script, our own
  work, adapted from an upstream example. Its checkpoint, prompt, dataset and resolution
  defaults were additionally changed when preparing this release so the script runs against the
  files shipped here.

**Removed for this release — 4 directories**, none on the inference path:

| removed | why |
|---|---|
| `videox_fun/video_caption/` | a data-captioning tool; it also carried third-party sample video clips whose redistribution rights were not established, so it is removed rather than shipped with an unresolved licence status |
| `videox_fun/ui/` | Gradio interface, not part of the released pipeline |
| `videox_fun/api/` | service wrapper, not part of the released pipeline |
| `videox_fun/reward/` | training-time reward models, not part of the released pipeline |

Removal is by deletion from the release snapshot; nothing in the inference path imports them,
and `check_closure.py` verifies that every import in the shipped tree still resolves.

## Attribution

Upstream copyright notices are preserved in the source files. Our additions and modifications
are Copyright 2026 Alaya Lab, licensed under Apache-2.0 — the same
licence as upstream, so the combined work carries no licence conflict.
