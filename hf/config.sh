#!/bin/bash
# Where the released artifacts live on HuggingFace.
#
# Sourced by fetch_weights.sh and read (as environment) by hf/upload_to_hf.py, so the repo
# names are written down exactly once. Override any of them from the environment.
#
# One place for the org, read by both the upload side and the download side, so the repo names
# are written down exactly once.

MARIONETTE_HF_ORG="${MARIONETTE_HF_ORG:-AlayaLab}"

# Weights. Ours, but research-only -- the terms come from the corpus they were trained on, not
# from the base model. The Apache-2.0 in this project covers the CODE, which ships on GitHub.
MARIONETTE_HF_MODEL_REPO="${MARIONETTE_HF_MODEL_REPO:-$MARIONETTE_HF_ORG/Marionette}"


marionette_hf_check_org() {
  if [ -z "$MARIONETTE_HF_ORG" ] || [ -z "$MARIONETTE_HF_MODEL_REPO" ]; then
    echo "HuggingFace org is not set (MARIONETTE_HF_ORG)." >&2
    return 2
  fi
  return 0
}
