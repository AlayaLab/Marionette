#!/usr/bin/env python3
"""Assert the vendored tree is import-complete.

This exists because it wasn't, twice. The first packaging pass listed the dynamics files by
hand and missed models/gpt.py. The second pass used a resolver whose set of package prefixes
was ALSO hand-written, so it missed utils/rec_bin_utils.py. Both surfaced as a
ModuleNotFoundError minutes into a GPU job.

So this does not enumerate anything. It models what Python actually does: each entry point puts
a fixed list of directories on sys.path, and every import is resolved against those roots in
order. Whatever the code imports, this finds it or fails.

    python check_closure.py        # exit 0 if every intra-project import resolves
"""
import ast
import os
import sys

ROOT = os.environ.get("MARIONETTE_ROOT") or os.path.dirname(os.path.abspath(__file__))
_OBS = f"{ROOT}/observation/examples/wan2.2_fun"
# Mirrors the sys.path inserts the two entry points make: dynamics/ and bridge/ from
# gen_pose_video.py, and its own three ancestor directories from the observation script.
SYS_PATH = [f"{ROOT}/dynamics", f"{ROOT}/bridge",
            _OBS, f"{ROOT}/observation/examples", f"{ROOT}/observation"]
ENTRIES = [f"{ROOT}/bridge/gen_pose_video.py",
           f"{_OBS}/predict_mh_pose_ar_baseline.py"]


def module_file(name):
    """Resolve a dotted module name against SYS_PATH the way the interpreter would."""
    rel = name.replace(".", os.sep)
    for root in SYS_PATH:
        for cand in (os.path.join(root, rel) + ".py", os.path.join(root, rel, "__init__.py")):
            if os.path.exists(cand):
                return cand
    return None


def imports_of(path):
    out = set()
    for n in ast.walk(ast.parse(open(path, encoding="utf-8").read(), filename=path)):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module)
    return out


THIRD_PARTY_OK = {  # installed by pip, not vendored; anything else unresolved is an error
    "torch", "numpy", "np", "cv2", "imageio", "imageio_ffmpeg", "PIL", "yaml", "tqdm",
    "matplotlib", "mpl_toolkits", "moderngl", "scipy", "einops", "omegaconf", "transformers",
    "diffusers", "decord", "safetensors", "accelerate", "pandas", "sklearn", "tensorboard",
    "huggingface_hub", "requests", "psutil", "av", "pyrr", "OpenGL", "glcontext",
    # PAI-internal accelerator. videox_fun guards every use of it behind find_spec(), so its
    # absence is the normal case, not a broken vendor.
    "paifuser",
}


def main():
    seen, pending, missing = set(), list(ENTRIES), {}
    while pending:
        f = pending.pop()
        if f in seen:
            continue
        seen.add(f)
        for name in sorted(imports_of(f)):
            top = name.split(".")[0]
            if top in THIRD_PARTY_OK or top in sys.builtin_module_names:
                continue
            p = module_file(name)
            if p:
                pending.append(p)
            else:
                try:
                    __import__(top)          # stdlib or an installed package: fine
                except Exception:
                    missing.setdefault(name, []).append(os.path.relpath(f, ROOT))

    print(f"entry points     : {[os.path.relpath(e, ROOT) for e in ENTRIES]}")
    print(f"reachable modules: {len(seen)}")
    if missing:
        for name, users in sorted(missing.items()):
            print(f"  MISSING {name}  (imported by {', '.join(users)})")
        sys.exit(f"{len(missing)} imports do not resolve in the vendored tree")
    print("closure OK: every import resolves in the vendored tree or as an installed package")
    for f in sorted(seen):
        print("   ", os.path.relpath(f, ROOT))


if __name__ == "__main__":
    main()
