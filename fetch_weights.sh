#!/bin/bash
# Pull our released weights and runtime assets from HuggingFace.
#
#   bash fetch_weights.sh              # the model weights (~10.5 GB)
#
# Only the weights are fetched. The seeds, terrain and demo videos ship in this repository --
# they pack to about a hundred megabytes, which is small enough that making people fetch them
# separately would buy nothing and cost a download step.
#
# This does NOT fetch the third-party base model; that is fetch_base_model.sh, kept separate so
# its provenance and its Apache-2.0 licence stay attached to it.
#
# Downloads are resumable and cached: re-running skips anything already complete. The cache
# lives under $HF_HOME (default ~/.cache/huggingface) and holds a second copy of every blob.
# Set HF_HUB_ENABLE_HF_TRANSFER=1 for a faster (and much less patient) downloader.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$ROOT/hf/config.sh"
marionette_hf_check_org || exit 2

WANT=${1:---all}
PY=${PY:-python3}

"$PY" - "$WANT" "$ROOT" "$MARIONETTE_HF_MODEL_REPO" "" <<'PYEOF'
import os, sys
want, root, model_repo, assets_repo = sys.argv[1:5]
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub is not installed in this interpreter:\n"
             "  pip install huggingface_hub\n"
             "(or run with PY=/path/to/python that has it)")

jobs = []
if want in ("--all", "--weights"):  # only mode now; kept so old invocations still work
    jobs.append(dict(repo_id=model_repo, repo_type="model",
                     local_dir=os.path.join(root, "weights"),
                     allow_patterns=["observation/*", "dynamics/*"]))
if not jobs:
    sys.exit(f"unknown option {want!r} (expected --all or --weights)")

for j in jobs:
    print(f"-> {j['repo_id']} ({j['repo_type']}) into {j['local_dir']}")
    try:
        snapshot_download(**j)
    except Exception as e:
        sys.exit(f"download failed: {e}\n"
                 "If the repo is still private, authenticate first:  hf auth login")
PYEOF
[ $? -eq 0 ] || exit 3

# Report what is actually usable now, rather than trusting that the download said OK. The
# three weight files and the seed metadata are what the two runner scripts open by path.
need_all=1
for f in weights/observation/diffusion_pytorch_model.safetensors \
         weights/dynamics/pose_gpt.pt \
         weights/dynamics/action_gpt.pt \
         ; do
  if [ -s "$ROOT/$f" ]; then
    printf '  ok      %-58s %s\n' "$f" "$(du -h "$ROOT/$f" | cut -f1)"
  else
    printf '  MISSING %s\n' "$f"
    need_all=0
  fi
done
if [ "$need_all" = 1 ]; then
  echo "weights ready. Next: bash fetch_base_model.sh, then bash run_demo.sh"
else
  echo "some files are missing -- re-run, the download resumes" >&2
  exit 4
fi
