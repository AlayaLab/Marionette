#!/usr/bin/env python3
"""Build PoseWorldModel motion_data.npz from VideoX-Fun mhwd-v2 state.npz (single monster).

Reads the `state.npz` pose part of the unified mhwd-v2 dataset (the SAME data Wan2.2
v2-9000 was trained on) and emits a motion_data.npz in the exact 780D format that
src/train_v16.py::load_data expects, so V16.3 (ActionGPT + PoseGPT) can be retrained
on it without any model-code change.

Single-monster scope (quick-train): filter to one em_id (default 19), reuse the
existing em_19_pruned / npc_pruned / wp_4 skeleton topology -> 276D layout unchanged.

780D layout (see src/train_gpt_continuous.py::slice_to_pos276):
  Monster: [0:3] root_delta_local, [3:162] rel_pos_local(53x3), [162:486] rot6d(54x6)
  NPC:     [486:489] root_delta_local, [489:582] rel_pos_local(31x3), [582:774] rot6d(32x6)
  Weapon:  [774:780] rel_pos_local(2x3, NPC-local)
V16 only consumes the ROOT joint rot6d (monster [162:168], npc [582:588]); other
per-joint rot6d slots are left zero.

segment_{i} is stored NORMALIZED (z-scored); mean/std are stored in raw-780 space.

Self-check: round-trips one segment through reconstruct_world and compares to the
original world joint positions in state.npz.
"""
import argparse
import glob
import json
import os
import sys
from collections import deque

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class FilesDict(dict):
    """dict that also exposes .files (like np.load npz) so the state.npz code path
    (detect_schema / joint_xyz / build_segment / pick_weapon_joints) works on raw-rec
    dicts unchanged."""
    @property
    def files(self):
        return list(self.keys())


# ------------------------------------------------------------------
# Skeleton topology (BFS order, lowercase joint names)
# ------------------------------------------------------------------
def bfs_joint_names(json_path):
    with open(json_path) as f:
        tree = json.load(f)
    names = []
    q = deque([tree])
    while q:
        n = q.popleft()
        names.append(n["name"].lower())
        for c in (n.get("childs") or []):
            q.append(c)
    return names


# ------------------------------------------------------------------
# Schema-aware field access
# ------------------------------------------------------------------
def detect_schema(keys):
    if "quest_meta.em_id_int" in keys:
        return "NEW"
    if "em.em.1.id_int" in keys:
        return "HYBRID"
    raise ValueError("unknown schema: no em_id field")


def prefixes(schema):
    if schema == "NEW":
        return dict(
            monster="monster.list.1.joints.",
            npc="npc_joints.body.1.",
            weapon="npc_joints.weapon.1.",
            m_rot="monster.list.1.rot.",
            n_rot="npc.list.1.rot.",
            m_motion="monster.list.1.motion.",
            n_motion="npc.list.1.motion.",
            em="quest_meta.em_id_int",
        )
    # HYBRID
    return dict(
        monster="em.em.1.joints.",
        npc="npc_joints.npc.",
        weapon="npc_joints.weapon.1.",
        m_rot="em.em.1.rot.",
        n_rot="npc.list.1.rot.",
        m_motion="em.em.1.motion.",
        n_motion="npc.list.1.motion.",
        em="em.em.1.id_int",
    )


def joint_xyz(d, prefix, name):
    """Return (T,3) world pos for one joint, or None if missing."""
    kx = f"{prefix}{name}.pos.x"
    if kx not in d.files:
        return None
    return np.stack([np.asarray(d[f"{prefix}{name}.pos.{a}"], dtype=np.float64)
                     for a in "xyz"], axis=-1)


def quat_to_R(d, prefix):
    """(T,3,3) rotation matrices from quaternion fields prefix+{w,x,y,z}."""
    w = np.asarray(d[f"{prefix}w"], dtype=np.float64)
    x = np.asarray(d[f"{prefix}x"], dtype=np.float64)
    y = np.asarray(d[f"{prefix}y"], dtype=np.float64)
    z = np.asarray(d[f"{prefix}z"], dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    T = len(w)
    R = np.empty((T, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def R_to_rot6d(R):
    """(T,3,3) -> (T,6) = first two columns flattened (matches rot6d_to_matrix)."""
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)


# ------------------------------------------------------------------
# Per-segment 780D feature builder
# ------------------------------------------------------------------
# Named "business-end" joints, in preference order, across weapon types.
# base->tip approximates the weapon's physical length / striking reach.
TIP_PRIORITY = ["vfx_attack", "spear"]  # vfx_attack: most weapons; spear: lance/gunlance


def pick_weapon_joints(d, pf):
    """Return (w_pick=[base, tip], wj_all). Tip = a known attack-end joint if present,
    else the weapon joint geometrically farthest from base (= the weapon's reach end),
    matching the 'base->tip = weapon body length' intent."""
    keys = set(d.files)
    wj = sorted(set(k.split(pf["weapon"])[1].rsplit(".pos.", 1)[0]
                    for k in keys if pf["weapon"] in k and ".pos." in k))
    base = "base" if "base" in wj else ("root" if "root" in wj else (wj[0] if wj else None))
    cands = [w for w in wj if w not in (None, base, "root")]
    tip = next((w for w in TIP_PRIORITY if w in cands), None)
    if tip is None and cands and base is not None:
        bpos = joint_xyz(d, pf["weapon"], base)
        best, bestd = None, -1.0
        for w in cands:
            wp = joint_xyz(d, pf["weapon"], w)
            if wp is None:
                continue
            dmean = float(np.linalg.norm(wp - bpos, axis=-1).mean())
            if dmean > bestd:
                best, bestd = w, dmean
        tip = best
    return [base, tip], wj


def build_segment(d, m_names, n_names, w_pick):
    keys = set(d.files)
    schema = detect_schema(keys)
    pf = prefixes(schema)

    # ---- gather monster joints (root + 53 rel), world frame ----
    m_root_w = joint_xyz(d, pf["monster"], m_names[0])  # 'root'
    if m_root_w is None:
        return None
    T = len(m_root_w)
    m_rel_w = []
    for nm in m_names[1:]:
        j = joint_xyz(d, pf["monster"], nm)
        if j is None:
            return None
        m_rel_w.append(j)
    m_rel_w = np.stack(m_rel_w, axis=1)  # (T,53,3)

    n_root_w = joint_xyz(d, pf["npc"], n_names[0])
    if n_root_w is None:
        return None
    n_rel_w = []
    for nm in n_names[1:]:
        j = joint_xyz(d, pf["npc"], nm)
        if j is None:
            return None
        n_rel_w.append(j)
    n_rel_w = np.stack(n_rel_w, axis=1)  # (T,31,3)

    # weapon: 2 joints (base + tip); fall back to zeros where missing
    w_w = np.zeros((T, 2, 3), dtype=np.float64)
    for i, wn in enumerate(w_pick):
        j = joint_xyz(d, pf["weapon"], wn) if wn else None
        if j is not None:
            w_w[:, i] = j

    # ---- rotations ----
    Rm = quat_to_R(d, pf["m_rot"])  # (T,3,3)
    Rn = quat_to_R(d, pf["n_rot"])
    RmT = np.transpose(Rm, (0, 2, 1))
    RnT = np.transpose(Rn, (0, 2, 1))

    # ---- local-frame features ----
    # rel_pos_local = R^T @ (joint_world - root_world)
    m_rel_local = np.einsum("tij,tkj->tki", RmT, m_rel_w - m_root_w[:, None, :])  # (T,53,3)
    n_rel_local = np.einsum("tij,tkj->tki", RnT, n_rel_w - n_root_w[:, None, :])  # (T,31,3)
    w_rel_local = np.einsum("tij,tkj->tki", RnT, w_w - n_root_w[:, None, :])      # (T,2,3)

    # root_delta_local[t] = R[t]^T @ (root[t]-root[t-1]); [0]=0
    def root_delta(root_w, RT):
        dl = np.zeros_like(root_w)
        dw = root_w[1:] - root_w[:-1]
        dl[1:] = np.einsum("tij,tj->ti", RT[1:], dw)
        return dl
    m_root_delta = root_delta(m_root_w, RmT)  # (T,3)
    n_root_delta = root_delta(n_root_w, RnT)

    m_rot6d = R_to_rot6d(Rm)  # (T,6) root rot
    n_rot6d = R_to_rot6d(Rn)

    # ---- assemble 780D ----
    feat = np.zeros((T, 780), dtype=np.float64)
    feat[:, 0:3] = m_root_delta
    feat[:, 3:162] = m_rel_local.reshape(T, -1)
    feat[:, 162:168] = m_rot6d                      # only root rot6d used by 276D path
    feat[:, 486:489] = n_root_delta
    feat[:, 489:582] = n_rel_local.reshape(T, -1)
    feat[:, 582:588] = n_rot6d
    feat[:, 774:780] = w_rel_local.reshape(T, -1)

    # ---- actions + progress ----
    # Action id = (bank, motion) COMPOSITE. bankid and motionid TOGETHER determine the
    # animation: the same motionid in different banks is a different action — e.g. npc
    # (bank0,motion0)=stand-idle vs (bank10,motion0)=mining vs (bank50,motion0)=mounted-eat.
    # Encoding: comp = bank*1000 + motion + 1  (motionid max ~464 < 1000), so every real
    # (bank>=0, motion>=0) pair is a distinct class >=1; invalid frames (id or bank < 0)
    # -> 0, the padding/ignore sentinel. Crucially (0,0) is a REAL action (idle) -> class 1,
    # NEVER padding. (Legacy motionid-only encoding collapsed idle=motion0 into padding.)
    BANK_BASE = 1000
    def action_id(prefix):
        m = np.asarray(d[f"{prefix}id_int"], dtype=np.int64)
        bkey = f"{prefix}bank_int"
        b = np.asarray(d[bkey], dtype=np.int64) if bkey in d else np.zeros_like(m)
        comp = b * BANK_BASE + m + 1
        comp[(m < 0) | (b < 0)] = 0          # invalid frame -> padding sentinel
        return comp
    am_raw = action_id(pf["m_motion"])
    an_raw = action_id(pf["n_motion"])

    def progress(prefix):
        fr = np.asarray(d[f"{prefix}frame"], dtype=np.float64)
        en = np.asarray(d[f"{prefix}end_frame"], dtype=np.float64)
        return np.clip(fr / np.maximum(en, 1.0), 0.0, 1.0)
    prog = np.stack([progress(pf["m_motion"]), progress(pf["n_motion"])], axis=-1).astype(np.float32)  # (T,2)

    # ---- HP (%): hp/max_hp clipped to [0,1]; monster uses -1 sentinel for invalid frames ----
    def hp_pct(hp_key, max_key):
        if hp_key not in d or max_key not in d:           # old state path lacks HP -> all invalid
            return np.ones(T, np.float32), np.zeros(T, bool)
        hp = np.asarray(d[hp_key], np.float64); mx = np.asarray(d[max_key], np.float64)
        valid = (hp >= 0) & (mx > 0)
        pct = np.where(valid, np.clip(hp / np.maximum(mx, 1.0), 0.0, 1.0), 1.0)  # invalid -> full
        return pct.astype(np.float32), valid
    m_hp, m_v = hp_pct("monster.list.1.hp", "monster.list.1.max_hp")
    n_hp, n_v = hp_pct("npc.list.1.health.hp", "npc.list.1.health.max_hp")
    hp = np.stack([m_hp, n_hp], axis=-1).astype(np.float32)        # (T,2)
    hp_valid = (m_v & n_v).astype(np.float32)                      # (T,) both entities valid

    init = dict(monster=m_root_w.astype(np.float32),
                npc=n_root_w.astype(np.float32),
                weapon=n_root_w.astype(np.float32))
    # keep originals for self-check
    gt = dict(m_root_w=m_root_w, m_rel_w=m_rel_w, n_root_w=n_root_w, n_rel_w=n_rel_w, Rm=Rm, Rn=Rn)
    return dict(feat=feat.astype(np.float32), am_raw=am_raw, an_raw=an_raw, prog=prog,
                hp=hp, hp_valid=hp_valid, init=init, gt=gt)


# ------------------------------------------------------------------
# Raw rec bin reader (pose-only; skips the whole video pipeline)
# ------------------------------------------------------------------
MH_REPO = "/path/to/workspace/the bridge repository"
NEEDED_PREFIXES = (
    "monster.list.1.joints.", "npc_joints.body.1.", "npc_joints.weapon.1.",
    "monster.list.1.rot.", "npc.list.1.rot.",
    "monster.list.1.motion.", "npc.list.1.motion.",
)
NEEDED_EXACT = ("quest_meta.em_id_int", "npc.list.1.weapon_type_int", "recording_status.time",
                "monster.list.1.hp", "monster.list.1.max_hp",
                "npc.list.1.health.hp", "npc.list.1.health.max_hp")


def read_rec_columns(path):
    """Vectorized column read of a rec .bin -> dict{lowercased field -> (N,) array}, N.
    Reads only the fields we need (joints/rot/motion/meta). Keys lowercased so the
    state.npz code path (detect_schema/prefixes/build_segment) works unchanged."""
    sys.path.insert(0, MH_REPO)
    from utils.rec_bin_utils import RecBinReader, UUID_SIZE
    with RecBinReader(path) as r:
        fields = r._field_names
        N, row_len, hoff = r.num_rows, r._row_len, r._header_offset
        wanted = [(i, f) for i, f in enumerate(fields)
                  if f in NEEDED_EXACT or any(f.startswith(p) for p in NEEDED_PREFIXES)]
        r._file.seek(hoff)
        raw = np.frombuffer(r._file.read(N * row_len), dtype=np.uint8)
        if raw.size < N * row_len:
            N = raw.size // row_len
        raw = raw[:N * row_len].reshape(N, row_len)
        out = FilesDict()
        for i, f in wanted:
            off = UUID_SIZE + i * 8
            col = raw[:, off:off + 8].copy().view("<i8" if f.endswith("_int") else "<f8").ravel()
            out[f.lower()] = col.astype(np.float64) if not f.endswith("_int") else col.astype(np.int64)
    return out, N


def downsample_dict(d, n, native_fps, target_fps=30.0):
    if native_fps <= target_fps + 1e-3:
        return d, n
    idx = np.unique(np.round(np.arange(0, n, native_fps / target_fps)).astype(int))
    idx = idx[idx < n]
    return FilesDict({k: v[idx] for k, v in d.items()}), len(idx)


def build_from_raw(args):
    skel = os.path.join(PROJECT_ROOT, "data/skeleton")
    # lowercase joint names to match lowercased raw keys
    m_names = [n.lower() for n in bfs_joint_names(os.path.join(skel, "em_19_pruned.json"))]
    n_names = [n.lower() for n in bfs_joint_names(os.path.join(skel, "npc_pruned.json"))]
    bins = []
    for src in ["capture-a", "capture-b"]:
        bins += sorted(glob.glob(f"{args.raw_root}/{src}/rec/em{args.em_id}_*.bin"))
    print(f"em{args.em_id} raw bins: {len(bins)}")

    segments, acts_m, acts_n, progs, inits, wtypes = [], [], [], [], [], []
    one_check = None
    gross_frames = ds_frames = valid_frames = 0
    for bp in bins:
        try:
            d, N = read_rec_columns(bp)
        except Exception as e:
            print(f"  skip {os.path.basename(bp)}: read err {e}"); continue
        gross_frames += N
        # native fps from recording_status.time if present
        t = d.get("recording_status.time")
        native_fps = 59.0
        if t is not None and len(t) > 1 and (t[-1] - t[0]) > 0:
            native_fps = (len(t) - 1) / (t[-1] - t[0])
        d, n = downsample_dict(d, N, native_fps, args.target_fps)
        ds_frames += n
        if "quest_meta.em_id_int" not in d:
            print(f"  skip {os.path.basename(bp)}: not NEW schema"); continue
        pf = prefixes("NEW")
        # weapon_type (constant per bin)
        wt = int(d["npc.list.1.weapon_type_int"][d["npc.list.1.weapon_type_int"] >= 0][0]) \
            if (d.get("npc.list.1.weapon_type_int") is not None
                and (d["npc.list.1.weapon_type_int"] >= 0).any()) else -1
        if args.weapon_type >= 0 and wt != args.weapon_type:
            continue
        # per-frame validity: monster root + npc root + monster hip present (nonzero)
        def jnz(prefix, name):
            xs = [d.get(f"{prefix}{name}.pos.{a}") for a in "xyz"]
            if any(x is None for x in xs):
                return np.zeros(n, bool)
            return (np.abs(xs[0]) + np.abs(xs[1]) + np.abs(xs[2])) > 1e-6
        valid = jnz(pf["monster"], m_names[0]) & jnz(pf["monster"], "hip") & jnz(pf["npc"], n_names[0])
        # weapon picking needs a representative row dict; reuse pick_weapon_joints on full d
        w_pick, _ = pick_weapon_joints(d, pf)
        # split into contiguous valid runs >= min_frames
        runs = []
        i = 0
        while i < n:
            if valid[i]:
                j = i
                while j < n and valid[j]:
                    j += 1
                if j - i >= args.min_frames:
                    runs.append((i, j))
                i = j
            else:
                i += 1
        for (s, e) in runs:
            d_run = FilesDict({k: v[s:e] for k, v in d.items()})
            try:
                b = build_segment(d_run, m_names, n_names, w_pick)
            except Exception as ex:
                print(f"  {os.path.basename(bp)} run[{s}:{e}] build err {ex}"); continue
            if b is None:
                continue
            valid_frames += len(b["feat"])
            segments.append(b["feat"]); acts_m.append(b["am_raw"]); acts_n.append(b["an_raw"])
            progs.append(b["prog"]); inits.append(b["init"]); wtypes.append(wt)
            if one_check is None:
                one_check = b; one_check["_name"] = os.path.basename(bp)
        print(f"  {os.path.basename(bp)[:40]} N={N} ds={n}@{native_fps:.1f}fps wt={wt} "
              f"valid_runs={len(runs)} weapon={w_pick}")

    print(f"\n=== em{args.em_id} raw volume ===")
    print(f"  gross frames (native fps): {gross_frames}  ({gross_frames/native_fps/3600:.2f} h equiv)")
    print(f"  after downsample to {args.target_fps}fps: {ds_frames}  ({ds_frames/args.target_fps/3600:.2f} h)")
    print(f"  after validity filter (usable): {valid_frames}  ({valid_frames/args.target_fps/3600:.2f} h)")
    print(f"  segments: {len(segments)}")
    return segments, acts_m, acts_n, progs, inits, wtypes, one_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="processed", choices=["processed", "raw"],
                    help="processed: state.npz segments; raw: rec .bin (pose-only, skips video pipeline)")
    ap.add_argument("--dataset", default="/path/to/workspace/dataset/mhwd-v2/processed")
    ap.add_argument("--raw_root", default="/path/to/workspace/dataset/mhwd-v2/raw")
    ap.add_argument("--target_fps", type=float, default=30.0)
    ap.add_argument("--em_id", type=int, default=19)
    ap.add_argument("--weapon_type", type=int, default=-1, help="-1 = keep all weapon types")
    ap.add_argument("--schema", default="NEW", choices=["NEW", "HYBRID", "BOTH"])
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "data/processed_em19_v1"))
    ap.add_argument("--min_frames", type=int, default=200)
    args = ap.parse_args()

    skel = os.path.join(PROJECT_ROOT, "data/skeleton")
    m_names = bfs_joint_names(os.path.join(skel, "em_19_pruned.json"))
    n_names = bfs_joint_names(os.path.join(skel, "npc_pruned.json"))
    print(f"skeleton: monster={len(m_names)} npc={len(n_names)}")
    assert len(m_names) == 54 and len(n_names) == 32

    if args.source == "raw":
        segments, acts_m, acts_n, progs, inits, wtypes, one_check = build_from_raw(args)
        _finalize(segments, acts_m, acts_n, progs, inits, wtypes, one_check, args)
        return

    segdirs = sorted(d for d in glob.glob(args.dataset + "/MH-*")
                     if os.path.isfile(os.path.join(d, "state.npz")))
    print(f"scanning {len(segdirs)} segments for em{args.em_id} ({args.schema})...")

    segments, acts_m, acts_n, progs, inits, wtypes = [], [], [], [], [], []
    one_check = None
    for sd in segdirs:
        d = np.load(os.path.join(sd, "state.npz"), allow_pickle=True)
        keys = set(d.files)
        try:
            schema = detect_schema(keys)
        except ValueError:
            continue
        if args.schema != "BOTH" and schema != args.schema:
            continue
        pf = prefixes(schema)
        if pf["em"] not in keys:
            continue
        eid = int(np.asarray(d[pf["em"]]).ravel()[0])
        if eid != args.em_id:
            continue
        # weapon_type (NPC weapon); movesets + weapon joints vary by type
        wt = -1
        for wk in ["npc.list.1.weapon_type_int", "NPC Data.npc.weapontype_int", "quest_meta.weapon_type_int.1"]:
            if wk in keys:
                v = np.asarray(d[wk]); v = v[v >= 0]
                wt = int(v[0]) if v.size else -1
                break
        if args.weapon_type >= 0 and wt != args.weapon_type:
            continue
        # 2 weapon joints: handle (base) + striking end (named tip, else farthest-from-base)
        w_pick, _ = pick_weapon_joints(d, pf)
        try:
            b = build_segment(d, m_names, n_names, w_pick)
        except Exception as e:
            print(f"  skip {os.path.basename(sd)}: {e}")
            continue
        if b is None or len(b["feat"]) < args.min_frames:
            continue
        segments.append(b["feat"]); acts_m.append(b["am_raw"]); acts_n.append(b["an_raw"])
        progs.append(b["prog"]); inits.append(b["init"]); wtypes.append(wt)
        if one_check is None:
            one_check = b
            one_check["_name"] = os.path.basename(sd)
        print(f"  + {os.path.basename(sd)[:34]} T={len(b['feat'])} wtype={wt} weapon={w_pick}")

    _finalize(segments, acts_m, acts_n, progs, inits, wtypes, one_check, args)


def _finalize(segments, acts_m, acts_n, progs, inits, wtypes, one_check, args):
    n_seg = len(segments)
    assert n_seg > 0, "no segments matched"
    print(f"\nmatched {n_seg} segments, total frames={sum(len(s) for s in segments)}")

    # ---- contiguous action vocab; reserve 0 for padding (masked_ce ignore_index=0) ----
    all_m = np.concatenate(acts_m); all_n = np.concatenate(acts_n)
    m_uniq = sorted(set(all_m.tolist())); n_uniq = sorted(set(all_n.tolist()))
    m_map = {v: i + 1 for i, v in enumerate(m_uniq)}; n_map = {v: i + 1 for i, v in enumerate(n_uniq)}
    acts_m = [np.array([m_map[v] for v in a], dtype=np.int32) for a in acts_m]
    acts_n = [np.array([n_map[v] for v in a], dtype=np.int32) for a in acts_n]
    m_vocab_size = len(m_uniq) + 1; n_vocab_size = len(n_uniq) + 1  # +1 for padding slot 0
    print(f"action vocab (incl. padding slot 0): monster={m_vocab_size} npc={n_vocab_size}")

    # ---- mean/std in raw-780 space, then normalize segments ----
    allf = np.concatenate(segments, axis=0)
    mean = allf.mean(axis=0).astype(np.float32)
    std = allf.std(axis=0).astype(np.float32)
    segments_norm = [((s - mean) / (std + 1e-8)).astype(np.float32) for s in segments]

    # ---- save ----
    os.makedirs(args.out_dir, exist_ok=True)
    save = dict(num_segments=np.int64(n_seg), mean=mean, std=std,
                m_action_vocab_size=np.int64(m_vocab_size),
                n_action_vocab_size=np.int64(n_vocab_size))
    for i in range(n_seg):
        save[f"segment_{i}"] = segments_norm[i]
        save[f"action_m_{i}"] = acts_m[i]
        save[f"action_n_{i}"] = acts_n[i]
        save[f"action_progress_{i}"] = progs[i]
        save[f"init_pos_{i}_monster"] = inits[i]["monster"]
        save[f"init_pos_{i}_npc"] = inits[i]["npc"]
        save[f"init_pos_{i}_weapon"] = inits[i]["weapon"]
        save[f"weapon_type_{i}"] = np.int64(wtypes[i])
        save[f"em_id_{i}"] = np.int64(args.em_id)
    out = os.path.join(args.out_dir, "motion_data.npz")
    np.savez(out, **save)
    print(f"saved -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

    # ---- self-check: reconstruct_world round-trip on one segment ----
    print(f"\n=== self-check on {one_check['_name']} ===")
    from src.train_gpt_continuous import slice_to_pos276
    raw = one_check["feat"]  # raw 780, this segment (unnormalized)
    p276 = slice_to_pos276(raw)
    # reconstruct monster world joints from 276D + init root
    f = p276
    Tc = f.shape[0]
    m_root_delta = f[:, 0:3]; m_rel = f[:, 3:162].reshape(Tc, 53, 3); m_rot6d = f[:, 264:270]

    def rot6d_to_R(r):
        a1 = r[:, :3]; a2 = r[:, 3:6]
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
        b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)
    Rm = rot6d_to_R(m_rot6d)
    wdelta = np.einsum("tij,tj->ti", Rm, m_root_delta)
    traj = np.cumsum(wdelta, axis=0) + one_check["init"]["monster"][0]
    rec = traj[:, None, :] + np.einsum("tij,tkj->tki", Rm, m_rel)  # (T,53,3) rel joints world
    gt_rel = one_check["gt"]["m_rel_w"]
    err_root = np.linalg.norm(traj - one_check["gt"]["m_root_w"], axis=-1).mean()
    err_joints = np.linalg.norm(rec - gt_rel, axis=-1).mean()
    print(f"  monster root recon err: {err_root:.5f} m   joint recon err: {err_joints:.5f} m")
    print("  -> OK (should be ~0)" if err_joints < 1e-2 else "  -> MISMATCH: check quat/coord convention!")


if __name__ == "__main__":
    main()
