#!/usr/bin/env python3
"""Publish the Marionette weights and runtime assets to HuggingFace.

One repo: the weights. The runtime assets (seeds, terrain, references) ship in the git
repository instead -- they pack to about a hundred megabytes, and a second download for that
would buy nothing.

Usage
-----
    export MARIONETTE_HF_ORG=your-org            # or edit hf/config.sh
    python hf/upload_to_hf.py --dry-run          # no network: prints the exact remote layout
    hf auth login                                # a WRITE token
    python hf/upload_to_hf.py                    # create + upload (private by default)
    python hf/upload_to_hf.py --verify           # re-check remote checksums against local

    python hf/upload_to_hf.py --only model       # one repo at a time
    python hf/upload_to_hf.py --public           # or flip visibility later in the web UI

Uploading is resumable. It goes through `upload_large_folder`, which commits in batches and
skips anything already present, so re-running after a dropped connection continues rather
than restarting the 10 GB file.

Verification is on **sha256, not size**. The Hub reports the sha256 of every LFS blob, and a
truncated upload that happens to land on the right size is exactly the failure that a size
check waves through.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "hf")
STAGING = os.path.join(HERE, "_staging")
SHA_CACHE = os.path.join(HERE, "_sha256.json")

# remote path -> local path, relative to ROOT. Order is presentation order in --dry-run.
MODEL_FILES = {
    "observation/diffusion_pytorch_model.safetensors":
        "weights/observation/diffusion_pytorch_model.safetensors",
    "dynamics/pose_gpt.pt":   "weights/dynamics/pose_gpt.pt",
    "dynamics/action_gpt.pt": "weights/dynamics/action_gpt.pt",
}
# Whole directories, mirrored as-is.
ASSET_DIRS = {
    "seeds": "data/seeds",
    "terrain":    "data/terrain",
}
ASSET_FILES = {
    "first_frame_ref.mp4": "data/first_frame_ref.mp4",
    # run_demo.sh's default input. The pose and the reference belong together and must ship
    # together: the reference is the ground-truth frame for the moment the pose starts at, and
    # that agreement is the trained-for condition. Substituting either one breaks the pair.
    "demo/aligned_pose.mp4": "data/demo/aligned_pose.mp4",
    "demo/aligned_ref.mp4": "data/demo/aligned_ref.mp4",
    "demo/aligned_rollout_n6.mp4": "data/demo/aligned_rollout_n6.mp4",
}
# Optional: produced by a MODE=generate run. Shows what seeding from a bare state segment gives
# you, reference-frame mismatch included. Absence is not an error.
ASSET_OPTIONAL = {
    "demo/generated_pose_seg53.mp4": "samples/seg53_seed32/pose.mp4",
    "demo/generated_rollout_seg53_n2.mp4": "samples_out/seg53_seed32_s0_n2/rollout.mp4",
}


def size_str(n):
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n/div:.2f} {unit}"
    return f"{n} B"


def collect(kind):
    """-> [(remote_path, local_abs_path)], plus the list of things that are missing."""
    plan, missing = [], []

    def add(remote, rel, optional=False):
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            plan.append((remote, p))
        elif not optional:
            missing.append((remote, rel))

    if kind == "model":
        for remote, rel in MODEL_FILES.items():
            add(remote, rel)
    else:
        for remote_dir, rel_dir in ASSET_DIRS.items():
            d = os.path.join(ROOT, rel_dir)
            if not os.path.isdir(d):
                missing.append((remote_dir + "/", rel_dir))
                continue
            for dirpath, _, files in os.walk(d):
                for fn in sorted(files):
                    p = os.path.join(dirpath, fn)
                    add(f"{remote_dir}/{os.path.relpath(p, d)}", os.path.relpath(p, ROOT))
        for remote, rel in ASSET_FILES.items():
            add(remote, rel)
        for remote, rel in ASSET_OPTIONAL.items():
            add(remote, rel, optional=True)
    return plan, missing


def sha256(path, cache):
    """Cached by (path, size, mtime) -- rehashing 10 GB on every invocation is not free."""
    st = os.stat(path)
    key = os.path.relpath(path, ROOT)
    ent = cache.get(key)
    if ent and ent["size"] == st.st_size and ent["mtime"] == int(st.st_mtime):
        return ent["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    cache[key] = {"size": st.st_size, "mtime": int(st.st_mtime), "sha256": h.hexdigest()}
    return cache[key]["sha256"]


def load_cache():
    try:
        return json.load(open(SHA_CACHE))
    except Exception:
        return {}


def save_cache(cache):
    json.dump(cache, open(SHA_CACHE, "w"), indent=1, sort_keys=True)


def card(kind, repos):
    """Read the card and substitute the repo names, so the README on the Hub is self-consistent."""
    text = open(os.path.join(HERE, "model_card.md"), encoding="utf-8").read()
    return text.replace("{{MODEL_REPO}}", repos["model"])


def stage(kind, plan, repos):
    """Hardlink the plan into a folder shaped like the remote repo.

    Hardlinks, not copies: the observation checkpoint alone is 10 GB and it already exists on
    this filesystem. Falls back to a symlink across filesystems, and to a copy if neither
    works -- upload reads through both, but a real link keeps `du` honest.
    """
    d = os.path.join(STAGING, kind)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write(card(kind, repos))
    for remote, local in plan:
        dst = os.path.join(d, remote)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.link(local, dst)
        except OSError:
            try:
                os.symlink(local, dst)
            except OSError:
                shutil.copy2(local, dst)
    return d


def do_upload(api, kind, repo_id, plan, repos, private):
    repo_type = "model" if kind == "model" else "dataset"
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    print(f"  repo ready: {repo_id} ({repo_type}, {'private' if private else 'public'})")
    folder = stage(kind, plan, repos)
    total = sum(os.path.getsize(p) for _, p in plan)
    print(f"  uploading {len(plan)} files, {size_str(total)} -- resumable, safe to re-run")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(folder_path=folder, repo_id=repo_id, repo_type=repo_type)
    else:  # huggingface_hub < 0.26
        api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type=repo_type,
                          commit_message="marionette release")
    shutil.rmtree(folder, ignore_errors=True)
    print(f"  done: https://huggingface.co/{'' if repo_type=='model' else 'datasets/'}{repo_id}")


def do_verify(api, kind, repo_id, plan, cache):
    """Compare every remote LFS blob's sha256 against the local file. Returns #problems."""
    repo_type = "model" if kind == "model" else "dataset"
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    bad = 0
    for rp, local in plan:
        s = remote.get(rp)
        if s is None:
            print(f"  MISSING on hub: {rp}")
            bad += 1
            continue
        lfs = getattr(s, "lfs", None)
        rsha = None
        if lfs is not None:
            rsha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if rsha:
            if rsha != sha256(local, cache):
                print(f"  SHA MISMATCH: {rp}")
                bad += 1
        elif s.size is not None and s.size != os.path.getsize(local):
            # Small files are stored as git blobs, not LFS, so the Hub reports no sha256.
            print(f"  SIZE MISMATCH: {rp}  remote {s.size} local {os.path.getsize(local)}")
            bad += 1
    print(f"  {len(plan) - bad}/{len(plan)} files verified in {repo_id}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", help="HuggingFace org or user (default: $MARIONETTE_HF_ORG)")
    ap.add_argument("--only", choices=["model", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, touch no network")
    ap.add_argument("--verify", action="store_true", help="check remote checksums, upload nothing")
    ap.add_argument("--public", action="store_true", help="create the repos public (default private)")
    ap.add_argument("--hash", action="store_true", help="with --dry-run, also hash local files")
    args = ap.parse_args()

    org = args.org or os.environ.get("MARIONETTE_HF_ORG", "AlayaLab")
    repos = {"model": os.environ.get("MARIONETTE_HF_MODEL_REPO") or f"{org}/Marionette"}
    kinds = ["model"]   # assets ship in the git repository, not on the Hub


    cache = load_cache()
    fatal = 0
    for kind in kinds:
        plan, missing = collect(kind)
        total = sum(os.path.getsize(p) for _, p in plan)
        print(f"== {kind}: {repos[kind]}  ({len(plan)} files, {size_str(total)})")
        # Collapse a directory of many same-shaped files into one line. terrain/ alone is 164
        # chunk files of identical size, and printing all of them buries the three that matter.
        groups = {}
        for rp, local in plan:
            groups.setdefault(os.path.dirname(rp), []).append((rp, local))
        for d, items in groups.items():
            if len(items) > 8 and not args.hash:
                n = sum(os.path.getsize(p) for _, p in items)
                print(f"   {d + '/':<52} {size_str(n):>10}   ({len(items)} files)")
                continue
            for rp, local in items:
                line = f"   {rp:<52} {size_str(os.path.getsize(local)):>10}"
                if args.dry_run and args.hash:
                    line += "  " + sha256(local, cache)[:16]
                print(line)
        for rp, rel in missing:
            print(f"   MISSING  {rp:<52} (expected at {rel})")
            fatal += 1
        for rp, rel in ASSET_OPTIONAL.items() if kind == "assets" else []:
            if not os.path.isfile(os.path.join(ROOT, rel)):
                print(f"   skipped  {rp:<52} (no {rel}; run the demo once to produce it)")
        print()

    if fatal:
        sys.exit(f"{fatal} required file(s) missing -- nothing uploaded")
    if args.dry_run:
        save_cache(cache)
        print("dry run: nothing was uploaded")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub")
    api = HfApi()
    try:
        who = api.whoami()
        print(f"authenticated as {who['name']}\n")
    except Exception as e:
        sys.exit(f"not authenticated ({e}). Run: hf auth login   (needs a WRITE token)")

    bad = 0
    for kind in kinds:
        plan, _ = collect(kind)
        if args.verify:
            bad += do_verify(api, kind, repos[kind], plan, cache)
        else:
            do_upload(api, kind, repos[kind], plan, repos, private=not args.public)
    save_cache(cache)
    if args.verify and bad:
        sys.exit(f"{bad} file(s) do not match -- re-run the upload, it resumes")


if __name__ == "__main__":
    main()
