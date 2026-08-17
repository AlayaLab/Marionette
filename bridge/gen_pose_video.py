#!/usr/bin/env python3
"""BRIDGE: PoseWorldModel (em19) generated motion -> production pose.mp4.

Generate monster+npc+weapon motion with the existing the dynamics model ckpt, drive the
camera_tracker follow-cam from the generated roots, build a synthetic per-frame
rec "row" (joint + camera fields) and feed it through the SAME production render
functions (compute_view_proj / build_skel_lines / render_frame) used to make the
training pose.mp4 — so the output is in-distribution for Wan v2-9000.

Joints come from features276_to_positions (order = em_19_pruned / npc_pruned / wp_4
BFS order). Correct monster<->npc relative placement is restored from the seed
segment's init_pos (the 276D only carries per-frame deltas), then the whole scene
is re-anchored onto stage-101 (沙原) scanned terrain.
"""
import argparse, math, os, subprocess, sys, json
from collections import deque
import numpy as np

# Standalone layout: dynamics/ and bridge/ sit side by side under the project root, so both
# roots are derived from this file rather than hardcoded to the development checkout. This is
# the only code change made while vendoring; everything else is byte-identical to the source
# repo, which is what makes the reproduction check in README meaningful.
_ROOT = os.environ.get("MARIONETTE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSEWM = os.path.join(_ROOT, "dynamics")
MH = os.path.join(_ROOT, "bridge")
sys.path.insert(0, POSEWM); sys.path.insert(0, MH)

import torch
import tracker_sphere as TRK   # ported camera_tracker.lua (sphere + spring-arm avoidance)
from scripts.eval_v16 import load_action_gpt, load_pose_gpt, combined_rollout
from scripts.eval_continuous import denormalize_276, features276_to_positions
from src.train_gpt_continuous import slice_to_pos276
import render_pose_terrain_gl as R


def resolve_force(mode, gt_fut, vocab_size, seed=0):
    """Resolve a --force_am / --force_an mode string into a (H,) int64 action-id array
    (contiguous vocab ids, 1..vocab_size-1) or None (=free/autonomous).

    gt_fut: (H,) the seed segment's GT future action ids for this entity (contiguous).
    Modes:
      ""/"free"     -> None (model samples autonomously)
      "gt"          -> force the GT future stream (action-locked to ground truth)
      "shuf"        -> GT future, temporally shuffled (chaotic control / negative)
      "hold_common" -> hold the MOST-frequent action in the GT future window
      "hold_alt"    -> hold the 2nd-most-frequent (a distinct, visually different action)
      "hold:<id>"   -> hold a specific contiguous action id
      "switch:<a>,<b>[,c...]" -> split the horizon into equal blocks, holding each id
                       in turn (e.g. switch:93,2 = first half stationary, second half run)
      "a,b,c,..."   -> explicit per-frame contiguous ids (cycled/truncated to H)
    """
    H = len(gt_fut)
    m = (mode or "").strip()
    if m == "" or m == "free":
        return None
    if m == "gt":
        return gt_fut.astype(np.int64)
    if m == "shuf":
        rng = np.random.default_rng(seed); s = gt_fut.copy(); rng.shuffle(s)
        return s.astype(np.int64)
    if m in ("hold_common", "hold_alt"):
        vals, cnts = np.unique(gt_fut, return_counts=True)
        order = vals[np.argsort(-cnts)]
        k = 0 if m == "hold_common" else min(1, len(order) - 1)
        return np.full(H, int(order[k]), np.int64)
    if m.startswith("retrig:"):
        # re-TRIGGER a one-shot action: <on> frames of id, then <off> idle(1) frames to
        # reset the action edge, repeated over the horizon -> the swing replays each cycle
        # (a held one-shot only plays once then freezes; this makes it visibly repeat).
        p = m.split(":"); aid = int(p[1]); on = int(p[2]) if len(p) > 2 else 26
        off = int(p[3]) if len(p) > 3 else 5
        seq = []
        while len(seq) < H:
            seq += [aid] * on + [1] * off      # 1 = idle -> clean reset edge
        return np.array(seq[:H], np.int64)
    if m.startswith("bseq:") or m.startswith("bseqn:"):
        # block sequence: each block is resolved by a sub-spec
        # (e.g. "bseq:hold:1;hold:493;hold:407;retrig:499:26:5" = idle | attack | dodge | heavy-retrig).
        #   bseq:<spec>;<spec>;...            -> EQUAL blocks
        #   bseqn:<n1>,<n2>,...:<spec>;...    -> explicit block lengths in frames, last block
        #                                        takes the remainder of the horizon.
        # bseqn exists for beat-cut demos: the action changes have to land on musical beats,
        # which are not an integer number of frames apart, so the blocks are unequal.
        if m.startswith("bseqn:"):
            lens_s, specs_s = m[len("bseqn:"):].split(":", 1)
            lens = [int(x) for x in lens_s.split(",") if x != ""]
            specs = specs_s.split(";")
            sizes = [lens[i] if i < len(lens) else 0 for i in range(len(specs))]
            sizes[-1] = max(0, H - sum(sizes[:-1]))
        else:
            specs = m[len("bseq:"):].split(";"); nb = len(specs); per = H // nb
            sizes = [per] * nb; sizes[-1] = H - per * (nb - 1)
        out = []
        for sp, h in zip(specs, sizes):
            if h <= 0: continue
            sub = resolve_force(sp, gt_fut[:h], vocab_size, seed)
            if sub is None: sub = gt_fut[:h].astype(np.int64)
            if len(sub) < h: sub = np.resize(sub, h)
            out.append(sub[:h])
        return np.concatenate(out)[:H]
    if m == "switch_alt":
        # idle->action->idle in equal thirds, auto-picked from the GT window
        # (most-frequent, 2nd-most-frequent, most-frequent) so both ids are
        # data-supported. Visible mid-clip action change without hand-picked ids.
        vals, cnts = np.unique(gt_fut, return_counts=True)
        order = vals[np.argsort(-cnts)]
        a = int(order[0]); b = int(order[min(1, len(order) - 1)])
        per = H // 3
        return np.concatenate([np.full(per, a, np.int64),
                               np.full(per, b, np.int64),
                               np.full(H - 2 * per, a, np.int64)])
    if m.startswith("hold:"):
        return np.full(H, int(m.split(":", 1)[1]), np.int64)
    if m.startswith("switch:"):
        ids = [int(x) for x in m.split(":", 1)[1].split(",") if x != ""]
        per = H // len(ids)
        blocks = [np.full(per, i, np.int64) for i in ids[:-1]]
        blocks.append(np.full(H - per * (len(ids) - 1), ids[-1], np.int64))
        return np.concatenate(blocks)
    ids = np.array([int(x) for x in m.split(",") if x != ""], np.int64)
    if len(ids) == 0:
        return None
    return np.resize(ids, H).astype(np.int64)   # cycle/truncate to horizon


def load_clip_features(args):
    """ALIGNED protocol v2: build the 780D feature stream directly from a processed
    clip's state.npz (build_segment machinery), downsampled 30->20fps with the index
    map kept, so motion frame k maps to raw/rgb frame idx[k] EXACTLY. Returns
    (feat20, am_raw, an_raw, idx_map, m_root20, n_root20, state_npz)."""
    from src.build_motion_data_from_state import (build_segment, detect_schema,
                                                  prefixes, pick_weapon_joints,
                                                  bfs_joint_names)
    skel = f"{POSEWM}/data/skeleton"
    m_names = bfs_joint_names(f"{skel}/em_19_pruned.json")
    n_names = bfs_joint_names(f"{skel}/npc_pruned.json")
    zc = np.load(os.path.join(args.seed_from_clip, "state.npz"), allow_pickle=True)

    class _Adapted:
        """npz wrapper remapping legacy npc keys (npc_data.npc.rot.{1..4}=x,y,z,w by
        component statistics; npc_data.npc.motioninfo.*) onto the schema build_segment
        expects, leaving everything else untouched."""
        REMAP = {"npc.list.1.rot.x": "npc_data.npc.rot.1",
                 "npc.list.1.rot.y": "npc_data.npc.rot.2",
                 "npc.list.1.rot.z": "npc_data.npc.rot.3",
                 "npc.list.1.rot.w": "npc_data.npc.rot.4",
                 "npc.list.1.motion.id_int": "npc_data.npc.motioninfo.motionid_int",
                 "npc.list.1.motion.frame": "npc_data.npc.motioninfo.frame",
                 "npc.list.1.motion.end_frame": "npc_data.npc.motioninfo.endframe"}

        @staticmethod
        def _weapon_remap(files):
            """Processed clips store weapon joints as npc_joints.weapon.<name>.pos.*, but the
            HYBRID schema table expects the raw-rec prefix npc_joints.weapon.1.<name>. Without
            this the weapon lookup misses, build_segment substitutes zeros, and subtracting the
            hunter root then places the weapon ~1300m away at the world origin — i.e. no weapon
            in the hunter's hands. Expose the real keys under the expected prefix."""
            out = {}
            for k in files:
                if k.startswith("npc_joints.weapon.") and ".pos." in k:
                    tail = k[len("npc_joints.weapon."):]
                    if tail.split(".")[0] == "1":          # already the expected layout
                        return {}
                    out[f"npc_joints.weapon.1.{tail}"] = k
            return out

        def __init__(self, z):
            self._z = z
            self.REMAP = dict(self.REMAP, **self._weapon_remap(list(z.files)))
            self.files = list(z.files) + [k for k, v in self.REMAP.items()
                                          if k not in z.files and v in z.files]
        def __getitem__(self, k):
            if k not in self._z.files and k in self.REMAP:
                return self._z[self.REMAP[k]]
            return self._z[k]
        def __contains__(self, k):
            return k in self.files

    zc_b = _Adapted(zc)
    pf = prefixes(detect_schema(set(zc_b.files)))
    if "quest_meta.stage_int" in zc.files:
        st = int(np.asarray(zc["quest_meta.stage_int"]).ravel()[0])
        assert st == args.stage, f"clip stage {st} != --stage {args.stage}"
    w_pick, _ = pick_weapon_joints(zc_b, pf)
    b = build_segment(zc_b, m_names, n_names, w_pick)
    assert b is not None, f"build_segment failed for {args.seed_from_clip}"
    T = len(b["feat"])
    idx = np.unique(np.round(np.arange(0, T, 30.0 / 20.0)).astype(int))
    idx = idx[idx < T]
    print(f"[seed-from-clip] {os.path.basename(args.seed_from_clip)}: {T} raw frames "
          f"-> {len(idx)} @20fps")
    return (b["feat"][idx].astype(np.float32), b["am_raw"][idx], b["an_raw"][idx],
            idx, b["gt"]["m_root_w"][idx], b["gt"]["n_root_w"][idx], zc)


def extract_rgb_window(rgb_path, start_frame, n_frames, out_path):
    """Cut GT rgb frames [start_frame, start_frame+n_frames) into out_path (re-encode
    for frame-exactness) using the bundled imageio ffmpeg binary."""
    import subprocess
    cmd = [R.FFMPEG, "-y", "-loglevel", "error", "-i", rgb_path,
           "-vf", f"trim=start_frame={start_frame}:end_frame={start_frame + n_frames},setpts=PTS-STARTPTS",
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", out_path]
    subprocess.run(cmd, check=True)
    print(f"[aligned-ref] GT rgb window ({n_frames}f from {start_frame}) -> {out_path}")


def bfs_names(json_path):
    t = json.load(open(json_path)); names = []
    q = deque([t])
    while q:
        n = q.popleft(); names.append(n["name"])
        for c in (n.get("childs") or []): q.append(c)
    return names


def make_row(m_xyz, n_xyz, w_xyz, m_names, n_names, w_names, eye, focus, fov, aspect):
    row = {}
    for k, nm in enumerate(m_names):
        row[f"monster.list.1.joints.{nm}.pos.x"] = float(m_xyz[k, 0])
        row[f"monster.list.1.joints.{nm}.pos.y"] = float(m_xyz[k, 1])
        row[f"monster.list.1.joints.{nm}.pos.z"] = float(m_xyz[k, 2])
    for k, nm in enumerate(n_names):
        row[f"npc_joints.body.1.{nm}.pos.x"] = float(n_xyz[k, 0])
        row[f"npc_joints.body.1.{nm}.pos.y"] = float(n_xyz[k, 1])
        row[f"npc_joints.body.1.{nm}.pos.z"] = float(n_xyz[k, 2])
    for k, nm in enumerate(w_names):  # w_xyz rows: [root(npc), Base, VFX_Attack]
        row[f"npc_joints.weapon.1.{nm}.pos.x"] = float(w_xyz[k, 0])
        row[f"npc_joints.weapon.1.{nm}.pos.y"] = float(w_xyz[k, 1])
        row[f"npc_joints.weapon.1.{nm}.pos.z"] = float(w_xyz[k, 2])
    # rec convention: compute_view_proj does look=-fwd, so the stored "fwd" points
    # from focus back toward eye (eye-focus), NOT the look direction.
    fwd = np.asarray(eye) - np.asarray(focus); fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    row["camera.ours.eye.x"], row["camera.ours.eye.y"], row["camera.ours.eye.z"] = map(float, eye)
    row["camera.ours.fwd.x"], row["camera.ours.fwd.y"], row["camera.ours.fwd.z"] = map(float, fwd)
    row["camera.ours.up.x"], row["camera.ours.up.y"], row["camera.ours.up.z"] = 0.0, 1.0, 0.0
    row["camera.ours.fov_deg"] = float(fov); row["camera.ours.aspect"] = float(aspect)
    return row


def generate_legacy(args, dev, se, re):
    """Old path: existing the dynamics model ckpt + GT-progress (the progress leak). Returns
    (mon, npc, wpn, m_init, n_init) — origin-rooted world joints + seed init roots."""
    d = np.load(args.motion_data, allow_pickle=True)
    mean = slice_to_pos276(d["mean"]); std = slice_to_pos276(d["std"])
    mv, nv = int(d["m_action_vocab_size"]), int(d["n_action_vocab_size"])
    ag = load_action_gpt(args.action_ckpt, mv, nv, dev)
    pg = load_pose_gpt(args.pose_ckpt, mv, nv, dev)
    seg = slice_to_pos276(d[f"segment_{args.segment}"]).astype(np.float32)
    am = d[f"action_m_{args.segment}"].astype(np.int64); an = d[f"action_n_{args.segment}"].astype(np.int64)
    prog = d[f"action_progress_{args.segment}"].astype(np.float32)
    full, _, _ = combined_rollout(ag, pg, torch.from_numpy(seg[:se]).unsqueeze(0).to(dev),
        torch.from_numpy(am[:se]).unsqueeze(0).to(dev), torch.from_numpy(an[:se]).unsqueeze(0).to(dev),
        torch.from_numpy(prog[:se]).unsqueeze(0).to(dev), num_frames=args.horizon,
        gt_progress=torch.from_numpy(prog[se:re]).unsqueeze(0).to(dev), progress_mode="gt",
        temperature=args.temperature)
    feat = denormalize_276(full[0].cpu().numpy(), mean, std)
    mon, npc, wpn, _, _ = features276_to_positions(feat)
    return mon, npc, wpn, np.asarray(d[f"init_pos_{args.segment}_monster"])[0], np.asarray(d[f"init_pos_{args.segment}_npc"])[0]


def generate_cloze(args, dev, se, re):
    """Cloze path: retrained goal-conditioned ActionGPT (use_goal, no progress leak) +
    PoseGPT, seeded from the lazy combined dataset. Sparse future-action anchors
    (option A) are derived from the seed segment's GT transitions and fed as goals;
    the rollout is autonomous (timing/actions generated). Returns origin-rooted joints + init roots."""
    from src.train_v16 import compute_goal
    meta = np.load(os.path.join(args.combined_dir, "metadata.npz"), allow_pickle=True)
    mean780 = meta["mean"].astype(np.float32); std780 = meta["std"].astype(np.float32)
    mean276 = slice_to_pos276(mean780); std276 = slice_to_pos276(std780)
    m_ids = meta["m_ids"]; n_ids = meta["n_ids"]
    mv, nv = int(meta["m_action_vocab_size"]), int(meta["n_action_vocab_size"])
    base = str(meta["seg_files"][args.segment])[:-4]
    feat_full = np.load(os.path.join(args.combined_dir, base + ".npy")).astype(np.float32)  # raw 780, full seg
    z = np.load(os.path.join(args.combined_dir, base + ".aux.npz"))
    im_full = np.asarray(z["init_monster"]); in_full = np.asarray(z["init_npc"])            # (T,3) GT world roots

    # seed from a CLOSE-COMBAT window (monster & npc together), not the task-start transient
    # (NPC spawns far/aloft and approaches). Pick the seed-window with smallest mean m-n dist.
    L = len(feat_full); assert L >= re, f"segment too short ({L}<{re})"
    dist = np.linalg.norm(im_full - in_full, axis=-1)
    cand = [s for s in range(0, L - re + 1) if dist[s:s + se].max() < args.seed_dist_thresh]
    s0 = cand[0] if cand else int(np.argmin([dist[s:s + se].mean() for s in range(0, L - re + 1)]))
    print(f"[cloze] seg{args.segment} seed_start={s0} (seed m-n dist mean={dist[s0:s0+se].mean():.1f}m)")
    feat = feat_full[s0:s0 + re]
    seg276 = slice_to_pos276((feat - mean780) / (std780 + 1e-8)).astype(np.float32)          # normalized 276

    def lut(ids):
        a = np.zeros(int(max(ids)) + 1, np.int64) if len(ids) else np.zeros(1, np.int64)
        for i, v in enumerate(ids): a[int(v)] = i + 1
        return a
    mlut, nlut = lut(m_ids), lut(n_ids)
    am = mlut[np.clip(np.asarray(z["action_m"])[s0:s0 + re], 0, len(mlut) - 1)]   # raw->contiguous vocab
    an = nlut[np.clip(np.asarray(z["action_n"])[s0:s0 + re], 0, len(nlut) - 1)]

    ag = load_action_gpt(args.action_ckpt, mv, nv, dev)   # use_goal read from ckpt cfg
    pg = load_pose_gpt(args.pose_ckpt, mv, nv, dev)
    goal_m = compute_goal(torch.from_numpy(am).unsqueeze(0).to(dev))       # positioned anchors (GT transitions)
    goal_n = compute_goal(torch.from_numpy(an).unsqueeze(0).to(dev))
    # ---- CONTROLLABILITY: force the monster/npc action stream over the generated horizon ----
    # gt future = the seed segment's own upcoming actions (contiguous vocab ids), length=horizon.
    fam = resolve_force(args.force_am, am[se:re], mv, seed=args.segment)
    fan = resolve_force(args.force_an, an[se:re], nv, seed=args.segment)
    fam_t = torch.from_numpy(fam).unsqueeze(0).to(dev) if fam is not None else None
    fan_t = torch.from_numpy(fan).unsqueeze(0).to(dev) if fan is not None else None
    print(f"[force] am={args.force_am!r} -> {None if fam is None else np.unique(fam).tolist()[:8]} | "
          f"an={args.force_an!r} -> {None if fan is None else 'set'}")
    full, _, _ = combined_rollout(ag, pg, torch.from_numpy(seg276[:se]).unsqueeze(0).to(dev),
        torch.from_numpy(am[:se]).unsqueeze(0).to(dev), torch.from_numpy(an[:se]).unsqueeze(0).to(dev),
        torch.zeros(1, se, 2, device=dev), num_frames=args.horizon, temperature=args.temperature,
        goal_m_seq=goal_m, goal_n_seq=goal_n, force_am_seq=fam_t, force_an_seq=fan_t)
    feat_out = denormalize_276(full[0].cpu().numpy(), mean276, std276)
    mon, npc, wpn, _, _ = features276_to_positions(feat_out)
    return mon, npc, wpn, im_full[s0], in_full[s0]   # init = GT world roots at the close-combat seed frame


def generate_cloze_terrain(args, dev, se, re):
    """Terrain-aware path (B v3): goal-conditioned ActionGPT + TERRAIN PoseGPT rolled out with live
    egocentric-patch re-sampling from the model's own predicted root (scripts.eval_terrain). Returns
    the generated-portion world joints ALREADY anchored to the seed's TRUE stage-101 position (where
    the rollout sampled terrain) — so the rendered terrain matches the conditioning. Main skips the
    generic re-anchor for this path."""
    from scripts.eval_terrain import LiveTerrain, terrain_rollout, load_terrain_pose_gpt
    from src.train_v16 import compute_goal
    meta = np.load(os.path.join(args.combined_dir, "metadata.npz"), allow_pickle=True)
    mean780 = meta["mean"].astype(np.float32); std780 = meta["std"].astype(np.float32)
    # zero_center_root ckpts (v5+) were trained with mean=0 root-delta normalization; detect
    # from the ckpt config so inference normalization always matches training.
    try:
        _tc = torch.load(args.terrain_pose_ckpt, map_location="cpu", weights_only=False).get("config", {})
        if bool(_tc.get("data", {}).get("zero_center_root", False)):
            mean780 = mean780.copy(); mean780[0:3] = 0.0; mean780[486:489] = 0.0
            print("[cloze-terrain] zero_center_root ckpt: root-delta normalized with mean=0")
    except Exception as e:
        print(f"[cloze-terrain] WARN: could not read ckpt config ({e}); using dataset mean as-is")
    mean276 = slice_to_pos276(mean780); std276 = slice_to_pos276(std780)
    m_ids, n_ids = meta["m_ids"], meta["n_ids"]
    mv, nv = int(meta["m_action_vocab_size"]), int(meta["n_action_vocab_size"])
    base = str(meta["seg_files"][args.segment])[:-4]
    if args.seed_from_clip:
        # ---- ALIGNED protocol v2: features straight from the processed clip ----
        feat_full, am_raw_full, an_raw_full, idx_map, m_root20, n_root20, zc = load_clip_features(args)
        L = len(feat_full); assert L >= re, f"clip too short ({L}<{re})"
        dist = np.linalg.norm(m_root20 - n_root20, axis=-1)
        cand = [s for s in range(0, L - re + 1) if dist[s:s + se].max() < args.seed_dist_thresh]
        s0 = cand[0] if cand else 0
        if args.seed_start >= 0:
            s0 = args.seed_start
        anchor = m_root20[s0]
        t_raw = int(idx_map[s0 + se - 1])                    # raw/rgb frame of the seed end (exact)
        def _cam(name):
            return float(np.asarray(zc[f"camera.ours.{name}"])[t_raw])
        eye = np.array([_cam("eye.x"), _cam("eye.y"), _cam("eye.z")])
        fwd = np.array([_cam("fwd.x"), _cam("fwd.y"), _cam("fwd.z")])
        args._cam0 = (eye, eye - fwd * 8.0, _cam("fov_deg"))
        print(f"[seed-from-clip] s0={s0} seed-end motion frame {s0+se-1} -> raw/rgb frame {t_raw}")
        ref_out = os.path.join(os.path.dirname(os.path.abspath(args.out)), "rgb.mp4")
        extract_rgb_window(os.path.join(args.seed_from_clip, "rgb.mp4"), t_raw, args.ref_rgb_len, ref_out)
        return _finish_cloze_terrain(args, dev, se, re, feat_full, am_raw_full, an_raw_full,
                                     s0, anchor, mean780, std780, mean276, std276,
                                     m_ids, n_ids, mv, nv, aligned=True,
                                     npc_anchor=n_root20[s0 + se - 1])

    feat_full = np.load(os.path.join(args.combined_dir, base + ".npy")).astype(np.float32)
    z = np.load(os.path.join(args.combined_dir, base + ".aux.npz"))
    im = np.asarray(z["init_monster"]); im_r = im.reshape(-1, 3)
    L = len(feat_full); assert L >= re, f"segment too short ({L}<{re})"
    if im.shape == (54, 3):                 # OLD seg: only frame-0 world snapshot (no per-frame roots)
        # Pick an ENGAGED-COMBAT window: where monster & npc move TOGETHER (low relative drift over the
        # seed) rather than the warp-in/approach phase (one locomotes away -> drifts off terrain). The
        # relative offset traj (npc-monster) comes from the per-entity world-delta cumsum in the feat.
        from scripts.eval_continuous import _rot6d_to_matrix
        f276f = slice_to_pos276(feat_full.astype(np.float32))
        Rm = _rot6d_to_matrix(f276f[:, 264:270].astype(np.float64))
        Rn = _rot6d_to_matrix(f276f[:, 270:276].astype(np.float64))
        m_traj = np.cumsum(np.einsum('tij,tj->ti', Rm, f276f[:, 0:3]), 0)        # monster world traj
        n_traj = np.cumsum(np.einsum('tij,tj->ti', Rn, f276f[:, 162:165]), 0)    # npc world traj
        rel = (n_traj - m_traj)[:, [0, 2]]                                        # horizontal npc-mon offset
        best_s, best_d = 0, 1e9
        for s in range(0, L - re + 1, 16):
            w = rel[s:s + se]; d = float(np.linalg.norm(w.max(0) - w.min(0)))     # offset range over seed
            if d < best_d: best_d, best_s = d, s
        s0 = best_s
        anchor = im[0] + m_traj[s0]          # monster world root at s0 (frame-0 root + integrated displacement)
        print(f"[cloze-terrain] seg{args.segment} OLD engaged-combat window s0={s0} (rel-drift={best_d:.2f}m)")
    else:                                   # NEW seg: per-frame world roots; pick close-combat window
        inp = np.asarray(z["init_npc"]).reshape(-1, 3)
        dist = np.linalg.norm(im_r - inp, axis=-1)
        cand = [s for s in range(0, L - re + 1) if dist[s:s + se].max() < args.seed_dist_thresh]
        s0 = cand[0] if cand else 0
        if args.seed_start >= 0:
            s0 = args.seed_start          # explicit window override (calm-window demos)
        anchor = im_r[s0]
        print(f"[cloze-terrain] seg{args.segment} seed_start={s0}")
    feat = feat_full[s0:s0 + re]
    seg276 = slice_to_pos276((feat - mean780) / (std780 + 1e-8)).astype(np.float32)

    # ---- ALIGNED-REF: locate this window in the source clip's RAW 30fps stream, grab the
    # recorded camera at the seed-end instant, and extract the GT rgb window starting there.
    # Root-position matching (same source floats) maps motion frame -> raw/rgb frame exactly.
    if args.ref_clip:
        from scripts.eval_continuous import _rot6d_to_matrix as _r6m
        f276a = slice_to_pos276(feat_full.astype(np.float32))
        Rm_all = _r6m(f276a[:, 264:270].astype(np.float64))
        m_traj_all = np.cumsum(np.einsum('tij,tj->ti', Rm_all, f276a[:, 0:3]), 0)
        base_root = im[0] if im.shape == (54, 3) else im_r[0]
        t_mo = s0 + se - 1                                   # motion frame of the seed end
        r_tgt = np.asarray(base_root, np.float64) + m_traj_all[t_mo]
        zc = np.load(os.path.join(args.ref_clip, "state.npz"), allow_pickle=True)
        zk = set(zc.files)
        rk = next(k for k in ("em.em.1.joints.root.pos", "monster.list.1.joints.root.pos")
                  if f"{k}.x" in zk)                         # NEW / HYBRID schema
        rx = np.asarray(zc[f"{rk}.x"], np.float64)
        ry = np.asarray(zc[f"{rk}.y"], np.float64)
        rz = np.asarray(zc[f"{rk}.z"], np.float64)
        d2 = (rx - r_tgt[0])**2 + (ry - r_tgt[1])**2 + (rz - r_tgt[2])**2
        t_raw = int(np.argmin(d2)); err = float(np.sqrt(d2[t_raw]))
        assert err < 0.05, f"aligned-ref root match failed for {args.ref_clip} (err {err:.3f} m)"
        def _cam(name):
            return float(np.asarray(zc[f"camera.ours.{name}"])[t_raw])
        eye = np.array([_cam("eye.x"), _cam("eye.y"), _cam("eye.z")])
        fwd = np.array([_cam("fwd.x"), _cam("fwd.y"), _cam("fwd.z")])
        args._cam0 = (eye, eye - fwd * 8.0, _cam("fov_deg"))  # cam_fields: fwd = eye - focus
        print(f"[aligned-ref] motion frame {t_mo} -> raw/rgb frame {t_raw} (root err {err*100:.2f} cm)")
        ref_out = os.path.join(os.path.dirname(os.path.abspath(args.out)), "rgb.mp4")
        extract_rgb_window(os.path.join(args.ref_clip, "rgb.mp4"), t_raw, args.ref_rgb_len, ref_out)

    # true hunter world root at the first output frame (NEW segs have per-frame roots;
    # OLD segs' init snapshot is unreliable for npc -> keep the legacy single shift there)
    npc_anchor = None
    if im.shape != (54, 3):
        inp_full = np.asarray(z["init_npc"]).reshape(-1, 3)
        t_first = s0 + (se - 1 if args.ref_clip else se)
        npc_anchor = inp_full[min(t_first, len(inp_full) - 1)]
    return _finish_cloze_terrain(args, dev, se, re, feat_full,
                                 np.asarray(z["action_m"]), np.asarray(z["action_n"]),
                                 s0, anchor, mean780, std780, mean276, std276,
                                 m_ids, n_ids, mv, nv, aligned=bool(args.ref_clip),
                                 npc_anchor=npc_anchor)


def _finish_cloze_terrain(args, dev, se, re, feat_full, am_raw_full, an_raw_full,
                          s0, anchor, mean780, std780, mean276, std276,
                          m_ids, n_ids, mv, nv, aligned=False, npc_anchor=None):
    """Shared tail of the terrain path: normalize, (optionally) GT-render, roll out,
    reconstruct world joints. aligned=True keeps the GT seed-end frame as frame 0.

    npc_anchor = TRUE hunter world root at the first OUTPUT frame. features276_to_
    positions cumsums each entity from the origin, so a single monster-anchored shift
    collapses the hunter onto the monster's start (the historical placement bug); with
    npc_anchor the hunter (and weapon) are re-anchored at their true world position."""
    from scripts.eval_terrain import LiveTerrain, terrain_rollout, load_terrain_pose_gpt
    from src.train_v16 import compute_goal
    feat = feat_full[s0:s0 + re]
    seg276 = slice_to_pos276((feat - mean780) / (std780 + 1e-8)).astype(np.float32)
    k0 = se - 1 if aligned else se

    # ---- GT-RENDER (bridge ablation, EXP-3): render the ground-truth state of the SAME
    # window instead of a model rollout; frame ranges match the generated path.
    if getattr(args, "gt_render", False):
        den_gt = slice_to_pos276(feat).astype(np.float64)     # raw feat = denormalized
        mon, npc, wpn, _, _ = features276_to_positions(den_gt)
        sh = np.asarray(anchor, np.float64) - mon[0, 0]
        mon, npc, wpn = (mon + sh)[k0:], (npc + sh)[k0:], (wpn + sh)[k0:]
        if npc_anchor is not None:
            nsh = np.asarray(npc_anchor, np.float64) - npc[0, 0]
            npc, wpn = npc + nsh, wpn + nsh
            print(f"[cloze-terrain] hunter re-anchored ({np.linalg.norm(nsh):.1f} m correction)")
        print(f"[cloze-terrain] GT-RENDER: ground-truth state frames {s0 + k0}..{s0 + re}")
        return mon, npc, wpn

    def lut(ids):
        a = np.zeros(int(max(ids)) + 1, np.int64)
        for i, v in enumerate(ids): a[int(v)] = i + 1
        return a
    mlut, nlut = lut(m_ids), lut(n_ids)
    am = mlut[np.clip(am_raw_full[s0:s0 + re], 0, len(mlut) - 1)]
    an = nlut[np.clip(an_raw_full[s0:s0 + re], 0, len(nlut) - 1)]
    ag = load_action_gpt(args.terrain_action_ckpt, mv, nv, dev, combined_dir=args.combined_dir)
    pg = load_terrain_pose_gpt(args.terrain_pose_ckpt, mv, nv, dev, args.combined_dir)
    lt = LiveTerrain(args.combined_dir, args.stage, dev)
    if getattr(args, "flat_terrain", False):
        # flat platform: make the floor query return a constant height (the real floor
        # under the seed root) everywhere -> patch becomes flat, feet stay grounded on a plane.
        q = lt.floor.query(np.array([float(anchor[0])]), np.array([float(anchor[2])]),
                           np.array([float(anchor[1])]))
        plat_y = float(q[0]) if np.isfinite(q[0]) else float(anchor[1]) - 1.0
        lt.floor.query = (lambda x, z, ref, _p=plat_y: np.full(np.asarray(x).shape, _p, np.float64))
        args._flat_plat_y = plat_y
        print(f"[flat-terrain] rollout floor = constant platform y={plat_y:.2f}")
    gm = compute_goal(torch.from_numpy(am).unsqueeze(0).to(dev))
    gn = compute_goal(torch.from_numpy(an).unsqueeze(0).to(dev))
    # ---- CONTROLLABILITY: force the monster/npc action stream over the generated horizon ----
    fam = resolve_force(args.force_am, am[se:se + args.horizon], mv, seed=args.segment)
    fan = resolve_force(args.force_an, an[se:se + args.horizon], nv, seed=args.segment)
    fam_t = torch.from_numpy(fam).unsqueeze(0).to(dev) if fam is not None else None
    fan_t = torch.from_numpy(fan).unsqueeze(0).to(dev) if fan is not None else None
    print(f"[force] am={args.force_am!r} -> {None if fam is None else np.unique(fam).tolist()[:8]} | "
          f"an={args.force_an!r} -> {None if fan is None else 'set'}")
    full, _, _, seed_end_root = terrain_rollout(ag, pg, lt,
        torch.from_numpy(seg276[:se]).unsqueeze(0).to(dev),
        torch.from_numpy(am).unsqueeze(0).to(dev), torch.from_numpy(an).unsqueeze(0).to(dev),
        gm, gn, args.horizon, anchor, mean276, std276, args.temperature,
        force_am_seq=fam_t, force_an_seq=fan_t)
    den = denormalize_276(full[0, k0:].cpu().numpy(), mean276, std276)
    mon, npc, wpn, _, _ = features276_to_positions(den)                  # origin-rooted, consistent frame
    if aligned:
        sh = np.asarray(seed_end_root, np.float64) - mon[0, 0]           # frame 0 root == GT seed-end root
    else:
        sh = np.asarray(seed_end_root, np.float64)                       # place at true stage-101 pos
    mon, npc, wpn = mon + sh, npc + sh, wpn + sh
    if npc_anchor is not None:
        nsh = np.asarray(npc_anchor, np.float64) - npc[0, 0]
        npc, wpn = npc + nsh, wpn + nsh
        print(f"[cloze-terrain] hunter re-anchored ({np.linalg.norm(nsh):.1f} m correction)")
    return mon, npc, wpn


class TerrainHeightmap:
    """Fast ground-height query over a TerrainStage (TRN2 chunks, 100x100 cells = 1m each;
    world_x = cx*100+col, world_z = cy*100+row). Multi-layer: returns the valid layer
    height nearest the query ref_y (the surface the character stands on)."""
    def __init__(self, terrain):
        self.cells = {}  # (cx,cy) -> list per cell? store the layers array + valid heights lazily
        self.terrain = terrain

    def _cell_h(self, x, z, ref_y):
        """Nearest-to-ref_y valid layer height of the 1m cell containing (x,z)."""
        cx, cy = int(np.floor(x / 100.0)), int(np.floor(z / 100.0))
        ch = self.terrain.chunks.get((cx, cy))
        if ch is None:
            return None
        c = int(np.floor(x)) - cx * 100; r = int(np.floor(z)) - cy * 100
        if not (0 <= c < 100 and 0 <= r < 100):
            return None
        layers = ch["layers"][r * 100 + c]
        valid = (layers["flags"] & 0x01) == 1
        hs = layers["h"][valid]
        if len(hs) == 0:
            return None
        return float(hs[np.argmin(np.abs(hs - ref_y))])   # surface nearest ref_y (multi-level safe)

    def ground_at(self, x, z, ref_y):
        """Bilinear interpolation over the 4 surrounding cell centers, so ground height is
        continuous across cell boundaries (the old per-cell lookup stepped up to ~20cm at
        1m-cell edges, which snapped the clamped root -> visible jumps)."""
        gx, gz = x - 0.5, z - 0.5
        x0, z0 = np.floor(gx), np.floor(gz)
        fx, fz = gx - x0, gz - z0
        acc_h = acc_w = 0.0
        for xi, zi, w in ((x0, z0, (1 - fx) * (1 - fz)), (x0 + 1, z0, fx * (1 - fz)),
                          (x0, z0 + 1, (1 - fx) * fz), (x0 + 1, z0 + 1, fx * fz)):
            if w < 1e-6:
                continue
            h = self._cell_h(xi + 0.5, zi + 0.5, ref_y)
            if h is not None:
                acc_h += w * h; acc_w += w
        if acc_w < 1e-6:
            return None
        return acc_h / acc_w


def apply_collider(arr, weapon, hm, step=1.5, foot_clear=0.0, foot_idx=None, lift_cap=0.4):
    """Move-and-slide a generated entity's world joints (arr: (T,J,3), joint0=root) against
    the terrain heightmap: (a) wall-block — cancel a frame's horizontal root move if it would
    step up more than `step` m (a wall) while not airborne; (b) ground-clamp — shift the whole
    body vertically so its feet rest on the ground. Applies the SAME per-frame shift to
    `weapon` (npc's weapon). Returns corrected (arr, weapon).

    The ground-clamp reads the surface under EACH FOOT, not under the root. Reading it once
    under the root and lifting until the lowest joint of the whole body cleared that one
    height left 11.4% of frames with a key joint still under the terrain: the offending
    joints sit a median 1.57 m from the root, over ground a median 0.18 m higher than the
    root's, so a single root-side query cannot see them. It also let any low-hanging part,
    a tail or a dropped head, drive the correction.

    `lift_cap` bounds the per-frame lift. Without it a foot that strays over a ledge asks
    for metres of lift and the body levitates, which looks worse than the penetration it
    fixes; the measured request reaches 2.87 m at p90 when every key joint is allowed to
    vote.
    """
    T = len(arr)
    out = arr.copy(); wout = weapon.copy() if weapon is not None else None
    cpos = arr[0, 0].astype(np.float64).copy()   # corrected root xyz
    dy_prev = None
    DY_SLEW = 0.04   # max vertical clamp change per frame (m); kills snap-to-layer jumps
    n_capped = 0
    for t in range(T):
        if t > 0:
            d = arr[t, 0, [0, 2]] - arr[t - 1, 0, [0, 2]]           # generated horizontal delta
            cand_x, cand_z = cpos[0] + d[0], cpos[2] + d[1]
            g_cand = hm.ground_at(cand_x, cand_z, cpos[1])
            g_cur = hm.ground_at(cpos[0], cpos[2], cpos[1])
            airborne = (arr[t, 0, 1] - arr[t - 1, 0, 1]) > 0.3       # rising fast -> allow (jump)
            if g_cand is not None and g_cur is not None and (g_cand - g_cur) > step and not airborne:
                pass            # blocked by wall: keep cpos xz (slide stops into-wall motion)
            else:
                cpos[0], cpos[2] = cand_x, cand_z
            cpos[1] = arr[t, 0, 1]   # ref_y for ground queries = generated root height
        shift_xz = np.array([cpos[0] - arr[t, 0, 0], 0.0, cpos[2] - arr[t, 0, 2]])
        joints = arr[t] + shift_xz                                  # apply horizontal correction
        if foot_idx:
            # lift until no foot is below the surface directly under it
            lifts = []
            for j in foot_idx:
                gj = hm.ground_at(joints[j, 0], joints[j, 2], joints[j, 1])
                if gj is not None:
                    lifts.append(gj + foot_clear - joints[j, 1])
            dy = max(lifts) if lifts else 0.0
        else:
            g = hm.ground_at(cpos[0], cpos[2], arr[t, 0, 1])
            dy = (g + foot_clear - joints[:, 1].min()) if g is not None else 0.0
        if dy > lift_cap:
            dy = lift_cap; n_capped += 1
        # slew-limit the vertical clamp so residual layer-switch steps can't teleport the body
        if dy_prev is not None:
            dy = float(np.clip(dy, dy_prev - DY_SLEW, dy_prev + DY_SLEW))
        dy_prev = dy
        shift = shift_xz + np.array([0.0, dy, 0.0])
        out[t] = arr[t] + shift
        if wout is not None:
            wout[t] = weapon[t] + shift
        cpos = out[t, 0].astype(np.float64)
    if n_capped:
        print(f"[collide] lift capped at {lift_cap} m on {n_capped}/{T} frames")
    return out, wout


def apply_engagement(mon, npc, wpn, max_sep=8.0, max_step=None):
    """Keep the two entities engaged, by cancelling motion rather than teleporting bodies.

    Same shape of rule as the wall-block in apply_collider: when a frame's motion would
    violate the constraint, that frame's offending component is cancelled and everything
    else is left alone. Nothing is pulled toward a target, so no corrective offset can
    accumulate and grow into a rigid drag that makes the feet slide. The correction is a
    horizontal rigid shift of an entity's joints, so no pose is edited and no bone changes
    length.

      max_sep   cap on horizontal monster-hunter separation (m). When a frame's relative
                motion would exceed it, the outward component is removed, split evenly
                between the two so neither entity is privileged.
      max_step  optional cap on per-frame horizontal root displacement (m), per entity.
                None disables it.

    Returns corrected (mon, npc, wpn). Y is untouched: the terrain collider owns it, and
    this runs before it.
    """
    T = len(mon)
    mo, no = mon.copy(), npc.copy()
    wo = wpn.copy() if wpn is not None else None
    pm = mon[0, 0].astype(np.float64).copy()
    pn = npc[0, 0].astype(np.float64).copy()
    for t in range(1, T):
        dm = (mon[t, 0] - mon[t - 1, 0]).astype(np.float64)
        dn = (npc[t, 0] - npc[t - 1, 0]).astype(np.float64)
        if max_step is not None:
            for d in (dm, dn):
                h = float(np.hypot(d[0], d[2]))
                if h > max_step:
                    d[0] *= max_step / h
                    d[2] *= max_step / h
        cm = pm + dm
        cn = pn + dn
        rel = np.array([cn[0] - cm[0], cn[2] - cm[2]])
        dist = float(np.hypot(rel[0], rel[1]))
        if dist > max_sep and dist > 1e-6:
            pull = (dist - max_sep) / 2.0          # metres each, along the separation axis
            u = rel / dist
            cm[0] += u[0] * pull; cm[2] += u[1] * pull
            cn[0] -= u[0] * pull; cn[2] -= u[1] * pull
        pm, pn = cm, cn
        sm = pm - mon[t, 0].astype(np.float64); sm[1] = 0.0
        sn = pn - npc[t, 0].astype(np.float64); sn[1] = 0.0
        mo[t] = mon[t] + sm
        no[t] = npc[t] + sn
        if wo is not None:
            wo[t] = wpn[t] + sn
    return mo, no, wo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints_npz", default="",
                    help="skip generation: render world joints from an external npz "
                         "(m_pos (T,54,3), n_pos (T,32,3), w_pos (T,2,3) @gen_fps; "
                         "PoseWM-ardy rollout output contract)")
    ap.add_argument("--action_ckpt", default=f"{POSEWM}/output/dynamics/action_gpt.pt")
    ap.add_argument("--pose_ckpt", default=f"{POSEWM}/output/dynamics/pose_gpt.pt")
    ap.add_argument("--motion_data", default=f"{POSEWM}/data/processed/motion_data.npz")
    ap.add_argument("--segment", type=int, default=85)
    ap.add_argument("--seed_frames", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--stage", type=int, default=101)
    ap.add_argument("--anchor", type=float, nargs=3, default=[-942.2, -1.08, 974.1],
                    help="stage-101 on-terrain world pos to place the monster start at")
    ap.add_argument("--fps", type=int, default=30, help="output/render fps (Wan=30)")
    ap.add_argument("--gen_fps", type=float, default=20.0, help="PoseWorldModel native fps")
    ap.add_argument("--cull_occluders", action="store_true",
                    help="① drop terrain nearer than the character (minus --cull_margin) so it can't occlude")
    ap.add_argument("--cull_margin", type=float, default=2.5, help="keep this much terrain in front of the character")
    ap.add_argument("--max_sep", type=float, default=0.0,
                    help="engagement constraint: cap horizontal monster-hunter separation "
                         "(m). 0 disables. Cancels the outward component of a frame's "
                         "relative motion, like the collider's wall-block.")
    ap.add_argument("--max_step", type=float, default=0.0,
                    help="engagement constraint: cap per-frame horizontal root "
                         "displacement per entity (m). 0 disables. Only active with "
                         "--max_sep.")
    ap.add_argument("--collide", action="store_true",
                    help="② inference collider: clamp generated roots to the terrain surface (move-and-slide)")
    ap.add_argument("--save_state", default="",
                    help="if set: save raw (pre-resample/pre-collider) and final world ROOT "
                         "trajectories to this .npz for drift/jump diagnostics")
    ap.add_argument("--seed_start", type=int, default=-1,
                    help="if >=0: override the automatic seed-window selection with this start frame "
                         "(e.g. a calm window found offline for controllability demos)")
    ap.add_argument("--seed_dist_thresh", type=float, default=15.0,
                    help="cloze: seed from a window whose monster-npc dist stays under this (close combat)")
    ap.add_argument("--combined_dir", default="",
                    help="if set: cloze mode — seed from this lazy combined dataset, use goal-conditioned "
                         "(use_goal) ActionGPT ckpt + positioned anchors (option A), autonomous rollout")
    ap.add_argument("--terrain_pose_ckpt", default="",
                    help="if set: terrain-aware path (B) — terrain PoseGPT rolled out with live egocentric "
                         "patch re-sampling; renders at the seed's TRUE stage-101 position (no re-anchor)")
    ap.add_argument("--terrain_action_ckpt", default=f"{POSEWM}/output/the action model/action_gpt.pt",
                    help="ActionGPT for the terrain path (terrain-planner or plain cloze both work)")
    ap.add_argument("--gt_render", action="store_true",
                    help="render the GROUND-TRUTH state of the selected window (no model rollout); "
                         "bridge-ablation upper bound (EXP-3)")
    ap.add_argument("--seed_from_clip", default="",
                    help="ALIGNED protocol v2: seed directly from a processed clip dir "
                         "(state.npz + rgb.mp4). Features are built on the fly (build_segment, "
                         "30->20fps with an exact index map), the recorded camera and the GT rgb "
                         "window are taken at the seed-end instant, and pose[0] = GT seed-end "
                         "frame. Supersedes --ref_clip (no seg<->clip mapping needed).")
    ap.add_argument("--ref_clip", default="",
                    help="ALIGNED-REF mode: processed clip dir (state.npz + rgb.mp4, 1:1 frames) that "
                         "this OLD combined seg was built from. Pose frame 0 becomes the GT seed-end "
                         "frame, the camera starts at the RECORDED camera of that instant (blending to "
                         "the follow-cam), and the GT rgb window starting at the same instant is "
                         "extracted next to --out as rgb.mp4 — so ref and control agree at t=0, "
                         "matching the training contract (control_ref_image=first_frame).")
    ap.add_argument("--ref_rgb_len", type=int, default=510,
                    help="frames of GT rgb to extract for the ref clip (>= n_chunks*81)")
    ap.add_argument("--force_am", default="",
                    help="controllability: force MONSTER action stream. "
                         "free|gt|shuf|hold_common|hold_alt|hold:<id>|<csv of contiguous ids>")
    ap.add_argument("--force_an", default="",
                    help="controllability: force NPC/hunter action stream (same grammar as --force_am). "
                         "Set to 'gt' to hold the hunter fixed while varying --force_am.")
    ap.add_argument("--fixed_cam", action="store_true",
                    help="controllability demo: compute the follow-cam once at t=0 and hold it, so "
                         "cross-variant motion differences are entirely the characters', not the camera's")
    ap.add_argument("--hide_npc", action="store_true",
                    help="controllability demo: don't draw the hunter/weapon (rollout still generates "
                         "them); the rendered video shows only the monster + terrain")
    ap.add_argument("--torch_seed", type=int, default=-1,
                    help="if >=0: seed torch RNG so free-entity action sampling is reproducible "
                         "(cross-variant identical prefixes while the forced streams agree)")
    ap.add_argument("--cam_focus", choices=["pair", "npc", "monster"], default="pair",
                    help="fixed_cam framing target: 'pair' frames both entities; 'npc'/'monster' "
                         "frames that entity close-up (protagonist view for control demos)")
    ap.add_argument("--npc_lock_cam", action="store_true",
                    help="control-demo camera: glide WITH the npc at a fixed world offset "
                         "(camera translates with the player, no rotation, always centered "
                         "on it). A constant 3/4 side offset computed once at t=0.")
    ap.add_argument("--npc_cam_dist", type=float, default=6.0, help="npc_lock_cam side distance (m)")
    ap.add_argument("--flat_terrain", action="store_true",
                    help="action-control demo: replace the real terrain with a completely FLAT "
                         "platform (at the seed's floor height). Rollout is conditioned on flat "
                         "ground and the render draws a flat plane — isolates action from terrain.")
    ap.add_argument("--script_npc", action="store_true",
                    help="movement demo: DIRECTLY script the player's root translation + yaw "
                         "rotation (kinematic override, not action tokens). First half spins in "
                         "place, second half translates. Body animation still comes from the model.")
    ap.add_argument("--script_start_secs", type=float, default=0.0,
                    help="--script_npc: seconds of untouched model rollout before the script takes "
                         "over. Use with --seed_from_clip so the clip opens aligned to the "
                         "reference frame and the injection begins from a consistent state.")
    ap.add_argument("--script_spin_turns", type=float, default=2.0, help="--script_npc: full yaw turns in the spin phase")
    ap.add_argument("--script_spin_secs", type=float, default=4.0, help="--script_npc: spin phase duration (s)")
    ap.add_argument("--script_run_seg_secs", type=float, default=2.0, help="--script_npc: seconds per run heading")
    ap.add_argument("--script_run_speed", type=float, default=2.5, help="--script_npc: run speed (m/s)")
    ap.add_argument("--script_run_dirs", default="0,90,180,270",
                    help="--script_npc: run headings in degrees off the initial facing, one per segment")
    ap.add_argument("--script_move_dist", type=float, default=8.0, help="(legacy) unused by the schedule")
    ap.add_argument("--out", default=f"{MH}/output/_bridge_test/gen_pose_seg85.mp4")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.torch_seed >= 0:
        torch.manual_seed(args.torch_seed); torch.cuda.manual_seed_all(args.torch_seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. generate (terrain-aware B path, OR cloze goal-mode, OR legacy gt-progress) ----
    se = args.seed_frames; re = se + args.horizon
    if args.joints_npz:
        # external world joints (PoseWM-ardy block-AR rollout): already at the true
        # stage position; weapon npz has 2 joints [base, tip] -> prepend npc root to
        # match the 3-joint wp skeleton [root, Base, VFX_Attack]
        z = np.load(args.joints_npz)
        mon = np.asarray(z["m_pos"], np.float64)
        npc = np.asarray(z["n_pos"], np.float64)
        wpn = np.concatenate([npc[:, 0:1], np.asarray(z["w_pos"], np.float64)], axis=1)
        T = len(mon)
        print(f"[joints_npz] {args.joints_npz}: {T} frames @{args.gen_fps}fps (generation skipped)")
    elif args.terrain_pose_ckpt:
        # terrain path returns generated-portion world joints ALREADY at the true stage-101 position
        mon, npc, wpn = generate_cloze_terrain(args, dev, se, re)
        T = len(mon)
    else:
        if args.combined_dir:
            mon, npc, wpn, m_init, n_init = generate_cloze(args, dev, se, re)
        else:
            mon, npc, wpn, m_init, n_init = generate_legacy(args, dev, se, re)
        # ---- 2. restore relative placement from seed init_pos, then re-anchor to stage101 ----
        mon += m_init; npc += n_init; wpn += n_init       # correct monster<->npc relative geometry
        off = np.asarray(args.anchor) - mon[se, 0]        # place generated monster root on stage-101 terrain
        mon += off; npc += off; wpn += off
        mon, npc, wpn = mon[se:], npc[se:], wpn[se:]      # render only the generated portion
        T = len(mon)

    # ---- 1b. scripted player root/yaw (movement demo): direct kinematic control ----
    # The body pose still comes from the model; we overwrite WHERE the player is and WHICH WAY it
    # faces. Applied before the terrain collider, so the scripted path is grounded on real terrain.
    if getattr(args, "script_npc", False):
        T0 = len(npc); dt = 1.0 / max(1e-6, args.gen_fps)
        spin_f = int(round(args.script_spin_secs * args.gen_fps))     # spin phase length (frames)
        # Heading-change frames, each rounded from its own ABSOLUTE time rather than stepped
        # by a rounded interval: for a beat-cut demo the interval is not a whole number of
        # frames (a 116-frame beat is 77.33 frames at gen_fps 20), so stepping accumulates
        # error. Rounding the absolute time also stops a rounded spin_f from shifting every
        # later heading -- with spin 1.1333 s that alone cost a frame on heading 2.
        def run_bound(i):
            return int(round((args.script_spin_secs + i * args.script_run_seg_secs)
                             * args.gen_fps))
        root0 = np.asarray(npc[0, 0], np.float64).copy()
        orig_root = np.asarray(npc[:, 0], np.float64).copy()   # the model's own path (gives its facing)
        # ---- CAMERA-RELATIVE basis: screen-up == camera forward projected on XZ ----
        # Mirror npc_lock_cam's t=0 offset so headings match what the viewer sees:
        #   0 deg = into the screen, 90 = screen right, 180 = toward camera, 270 = screen left.
        a_fwd = np.asarray(mon[0, 0], np.float64) - root0; a_fwd[1] = 0.0
        nn = np.linalg.norm(a_fwd)
        a_fwd = a_fwd / nn if nn > 1e-6 else np.array([1.0, 0.0, 0.0])
        a_side = np.cross(a_fwd, [0.0, 1.0, 0.0])
        eye_off = a_side * args.npc_cam_dist - a_fwd * 1.5     # same as the npc_lock_cam offset
        cam_f = -eye_off.copy(); cam_f[1] = 0.0                # eye -> focus, flattened
        cam_f = cam_f / (np.linalg.norm(cam_f) + 1e-9)
        cam_r = np.cross(cam_f, [0.0, 1.0, 0.0])               # screen right
        turns_deg = [float(x) for x in str(args.script_run_dirs).split(",") if x != ""]
        run_bnds = [run_bound(i) for i in range(1, len(turns_deg))]
        def screen_dir(deg):
            a = np.deg2rad(deg)
            return cam_f * np.cos(a) + cam_r * np.sin(a)
        def yaw_onto(a, b):
            """signed yaw for Ry that maps unit XZ vector a onto b (matches Ry below)."""
            c = float(a[0] * b[0] + a[2] * b[2])
            s = float(a[2] * b[0] - a[0] * b[2])
            return float(np.arctan2(s, c))
        # phase 0: hold the model's OWN root for the first --script_start_secs, so the clip
        # opens as an ordinary aligned rollout and the injection starts from a state that
        # already agrees with the reference frame. Without it the script owns frame 0, and
        # while it happens to start at the model's own root with zero yaw, nothing downstream
        # guarantees that -- the lead-in makes the continuity explicit instead of incidental.
        start_f = int(round(getattr(args, "script_start_secs", 0.0) * args.gen_fps))
        start_f = max(0, min(start_f, T0 - 1))
        pos = np.asarray(npc[start_f, 0], np.float64).copy(); th = 0.0
        spin_end_th = 2 * np.pi * args.script_spin_turns
        last_dir = a_fwd.copy()
        for t in range(T0):
            if t < start_f:                                 # phase 0: untouched GT-aligned lead-in
                continue
            ts = t - start_f                                # script-local frame
            if ts < spin_f:                                 # phase 1: SPIN in place
                f = ts / max(1, spin_f - 1)
                th = spin_end_th * f
            else:                                           # phase 2: RUN, new heading every seg
                k = min(len(turns_deg) - 1,          # ts, not t: bounds are script-local
                        int(np.searchsorted(run_bnds, ts, side="right")))
                d = screen_dir(turns_deg[k])
                pos = pos + d * (args.script_run_speed * dt)
                # face the run direction: rotate the model's own heading onto d
                mv = orig_root[min(t + 1, T0 - 1)] - orig_root[t]; mv[1] = 0.0
                if np.linalg.norm(mv) > 1e-4:
                    last_dir = mv / np.linalg.norm(mv)
                th = yaw_onto(last_dir, d / (np.linalg.norm(d) + 1e-9))
            c, s = np.cos(th), np.sin(th)
            Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            r = np.asarray(npc[t, 0], np.float64).copy()
            npc[t] = (Ry @ (np.asarray(npc[t], np.float64) - r).T).T + pos
            wpn[t] = (Ry @ (np.asarray(wpn[t], np.float64) - r).T).T + pos
        print(f"[script-npc] {start_f} frame lead-in, then spin {args.script_spin_turns} turns over "
              f"{args.script_spin_secs}s, then run {args.script_run_speed} m/s, "
              f"heading {turns_deg} every {args.script_run_seg_secs}s")

    # diagnostics: keep the RAW generated world roots (native fps, before resample/collider)
    raw_mon_root = mon[:, 0].copy(); raw_npc_root = npc[:, 0].copy()

    # optional root dump for controllability analysis (env CTRL_DUMP_ROOT=1): per-frame
    # npc/monster world root + total path length, so we can measure whether a forced
    # action (e.g. idle) actually freezes the root without eyeballing the pose video.
    if os.environ.get("CTRL_DUMP_ROOT", "0") == "1":
        nd = float(np.linalg.norm(np.diff(raw_npc_root, axis=0), axis=1).sum())
        md = float(np.linalg.norm(np.diff(raw_mon_root, axis=0), axis=1).sum())
        np.savez(args.out.rsplit(".", 1)[0] + ".roots.npz",
                 npc_root=raw_npc_root, mon_root=raw_mon_root,
                 npc_path_len=nd, mon_path_len=md)
        print(f"[root-dump] npc_path_len={nd:.3f} mon_path_len={md:.3f} -> {args.out.rsplit('.',1)[0]}.roots.npz")

    # resample motion gen_fps -> render fps (e.g. 20 -> 30 to match Wan training fps).
    # linear interpolation of world joint positions (smooth motion upsamples cleanly).
    if abs(args.fps - args.gen_fps) > 1e-3:
        Tr = int(round(T * args.fps / args.gen_fps))
        src = np.linspace(0.0, T - 1, Tr)
        def resamp(a):  # (T,J,3) -> (Tr,J,3)
            lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, T - 1); fr = (src - lo)[:, None, None]
            return a[lo] * (1 - fr) + a[hi] * fr
        mon, npc, wpn = resamp(mon), resamp(npc), resamp(wpn)
        print(f"resampled {T}@{args.gen_fps}fps -> {Tr}@{args.fps}fps")
        T = Tr

    # ---- 3. skeleton names/edges (pruned topology; we only have pruned joints) ----
    skel = f"{POSEWM}/data/skeleton"
    m_names = bfs_names(f"{skel}/em_19_pruned.json")   # 54 (matches mon order)
    n_names = bfs_names(f"{skel}/npc_pruned.json")     # 32
    w_names = bfs_names(f"{skel}/wp_4.json")           # 3 [root, Base, VFX_Attack]
    from pathlib import Path
    m_edges = R.load_edges(Path(f"{skel}/em_19_pruned.json"))
    n_edges = R.load_edges(Path(f"{skel}/npc_pruned.json"))
    w_edges = R.load_edges(Path(f"{skel}/wp_4.json"))

    # ---- 4. renderer + terrain ----
    import moderngl
    terrain = R.TerrainStage(args.stage)
    if getattr(args, "flat_terrain", False):
        # flatten every chunk's floor layer to the platform height -> render draws a flat plane;
        # TerrainHeightmap/collider also read this, so the whole world is a flat platform.
        py = getattr(args, "_flat_plat_y", None)
        if py is None: py = float(terrain.height_min)
        for info in terrain.chunks.values():
            L = info["layers"]
            L["h"][:] = py; L["nx"][:] = 0.0; L["ny"][:] = 0.0; L["nz"][:] = 0.0
            L["flags"][:] = 0; L["hit_count"][:] = 0
            L["ny"][:, 0] = 1.0; L["flags"][:, 0] = 1; L["hit_count"][:, 0] = 2
        terrain.height_min = py; terrain.height_max = py
        print(f"[flat-terrain] flattened render terrain to y={py:.2f} ({len(terrain.chunks)} chunks)")
    print(f"[stage {args.stage}] {len(terrain.chunks)} chunks, h=[{terrain.height_min:.1f},{terrain.height_max:.1f}]")

    # ② inference collider (toggle): clamp generated roots to terrain surface + wall-block,
    # so the autonomous motion stops sinking into / walking through obstacles.
    # ①b engagement constraint (toggle): cap how far the two entities may separate, and
    # optionally how fast either may travel. Runs BEFORE the collider so the ground clamp
    # still has the last word on Y. A free rollout has no reason to stay in an encounter,
    # and once the pair separates the pair-framing camera pulls back until each body covers
    # almost none of the pose-control frame; this is the rule that says it may not.
    if args.max_sep > 0:
        mon, npc, wpn = apply_engagement(
            mon, npc, wpn, max_sep=args.max_sep,
            max_step=(args.max_step if args.max_step > 0 else None))
        d = np.linalg.norm(mon[:, 0, [0, 2]] - npc[:, 0, [0, 2]], axis=1)
        print(f"[engage] max_sep={args.max_sep} m max_step={args.max_step or 'off'} "
              f"-> separation now [{d.min():.1f}, {d.max():.1f}] m")

    if args.collide:
        hm = TerrainHeightmap(terrain)
        # Root-referenced clamp, measured better than the foot-referenced one and kept.
        # apply_collider still accepts foot_idx; passing it made penetration WORSE, 0.114
        # -> 0.238, because the penetration metric scores 11 "key joints" that are almost
        # all digit tips (R_RingF2, R_MiddleF2, ...) and share exactly one index with the
        # foot set. Clamping the foot segments does not lift the digits hanging below them,
        # while lifting until the lowest joint of the whole body clears the ground protects
        # them incidentally. Recorded state scores 0.082 on the same metric, so 0.114 is
        # already near the floor this measurement admits, and 0.000 would mean the generated
        # motion sits cleaner on the terrain than the motion capture does.
        mon, _ = apply_collider(mon, None, hm)
        npc, wpn = apply_collider(npc, wpn, hm)
        print("[collide] applied terrain collider to monster + npc(+weapon)")

    if args.save_state:
        # Stamp the generator's commit into the artefact. Rollouts from two versions of
        # this file were once compared in the same table and the difference was read as an
        # experimental effect; without this the confound is invisible on disk.
        try:
            _rev = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                   "rev-parse", "HEAD"], capture_output=True, text=True,
                                  timeout=10).stdout.strip() or "unknown"
        except Exception:
            _rev = "unknown"
        np.savez(args.save_state, gen_pose_video_commit=_rev,
                 raw_mon_root=raw_mon_root, raw_npc_root=raw_npc_root, gen_fps=args.gen_fps,
                 final_mon_root=mon[:, 0], final_npc_root=npc[:, 0], render_fps=args.fps,
                 collide=int(args.collide))
        print(f"[save_state] roots -> {args.save_state}")

    # optional world-geometry dump (env DUMP_JOINTS=1): the final metric world joints for both
    # entities plus the skeleton edges and a local terrain heightfield, i.e. exactly the geometry
    # this renderer is about to rasterize. Lets an external tool (the demo's PoseGPT act, which
    # draws the joint tree in its own style) work from the same numbers as the bridge instead of
    # re-deriving world space from state.npz, whose joints are root-local.
    if os.environ.get("DUMP_JOINTS", "0") == "1":
        base = args.out.rsplit(".", 1)[0]
        ghm = TerrainHeightmap(terrain)
        c = np.concatenate([mon[:, 0, [0, 2]], npc[:, 0, [0, 2]]], 0)   # XZ of both roots
        pad, step = 14.0, 0.5
        gx = np.arange(c[:, 0].min() - pad, c[:, 0].max() + pad + step, step)
        gz = np.arange(c[:, 1].min() - pad, c[:, 1].max() + pad + step, step)
        ref_y = float(np.median(np.concatenate([mon[:, 0, 1], npc[:, 0, 1]])))
        gh = np.full((len(gz), len(gx)), np.nan)
        for j, z in enumerate(gz):
            for i, x in enumerate(gx):
                h = ghm.ground_at(float(x), float(z), ref_y)
                if h is not None:
                    gh[j, i] = h
        np.savez_compressed(base + ".joints.npz",
                            mon=mon.astype(np.float32), npc=npc.astype(np.float32),
                            wpn=wpn.astype(np.float32), fps=args.fps,
                            mon_edges=np.array(m_edges), npc_edges=np.array(n_edges),
                            wpn_edges=np.array(w_edges),
                            mon_names=np.array(m_names), npc_names=np.array(n_names),
                            terr_x=gx.astype(np.float32), terr_z=gz.astype(np.float32),
                            terr_h=gh.astype(np.float32))
        print(f"[dump-joints] mon{mon.shape} npc{npc.shape} terrain{gh.shape} "
              f"({100*np.isfinite(gh).mean():.0f}% valid) -> {base}.joints.npz")

    renderer = R.GLRenderer(); renderer.bind_terrain(terrain)
    enc = R.start_ffmpeg_encoder(args.out, fps=args.fps)
    aspect = R.W / R.H

    def cam_fields(eye, focus, fov):
        fwd = np.asarray(eye) - np.asarray(focus); fwd = fwd / (np.linalg.norm(fwd) + 1e-8)  # RE: +Z away from target
        return {"camera.ours.eye.x": float(eye[0]), "camera.ours.eye.y": float(eye[1]), "camera.ours.eye.z": float(eye[2]),
                "camera.ours.fwd.x": float(fwd[0]), "camera.ours.fwd.y": float(fwd[1]), "camera.ours.fwd.z": float(fwd[2]),
                "camera.ours.up.x": 0.0, "camera.ours.up.y": 1.0, "camera.ours.up.z": 0.0,
                "camera.ours.fov_deg": float(fov), "camera.ours.aspect": float(aspect)}

    # ---- render-based terrain occlusion test (approach A) ----
    OW, OH = 320, 176
    occ_color = renderer.ctx.texture((OW, OH), 4, dtype="f2")
    occ_depth = renderer.ctx.depth_texture((OW, OH))
    occ_fbo = renderer.ctx.framebuffer(color_attachments=[occ_color], depth_attachment=occ_depth)

    def is_blocked(look_target, eye):
        """Render terrain-only depth for this candidate camera, project the NPC head,
        compare depths. Returns (blocked, clearance) — clearance higher = less occluded."""
        row = cam_fields(eye, look_target, TRK.CFG["FOV_DEG"])
        vp, _, _, _ = R.compute_view_proj(row)
        if vp is None:
            return True, -1.0
        hom = np.array([look_target[0], look_target[1], look_target[2], 1.0])
        clip = vp @ hom; w = float(clip[3])
        if w <= 0:
            return True, -1.0
        ndc = clip[:3] / w
        if abs(ndc[0]) > 1.0 or abs(ndc[1]) > 1.0:
            return True, -1.0
        head_winz = float(ndc[2]) * 0.5 + 0.5
        # render terrain depth
        occ_fbo.use(); occ_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        cam_y = float(eye[1])
        p = renderer.terrain_prog
        p["u_vp"].write(vp.T.astype("f4").tobytes())
        p["u_height_lo"].value = cam_y - 60.0; p["u_height_hi"].value = cam_y + 30.0
        p["u_depth_near"].value = 5.0; p["u_depth_far"].value = 80.0
        p["u_terrain_near_cull"].value = R.TERRAIN_NEAR_CULL
        if renderer._terrain_vao is not None:
            renderer._terrain_vao.render(mode=moderngl.TRIANGLES)
        renderer.fbo.use()  # restore main fbo
        px = int((ndc[0] * 0.5 + 0.5) * (OW - 1))
        gy = int((ndc[1] * 0.5 + 0.5) * (OH - 1))  # GL y is bottom-up = ndc y up
        x0, y0 = max(0, px - 2), max(0, gy - 2)
        x1, y1 = min(OW, px + 3), min(OH, gy + 3)
        data = occ_fbo.read(viewport=(x0, y0, x1 - x0, y1 - y0), components=1, attachment=-1, dtype="f4")
        terr = np.frombuffer(data, dtype="f4")
        terr_winz = float(terr.min()) if terr.size else 1.0  # nearest terrain in the patch
        blocked = terr_winz < head_winz - 1e-3   # terrain nearer than head -> occludes
        return blocked, terr_winz

    # ---- 5. per-frame: sphere camera tracker (+spring-arm) -> synthetic row -> render ----
    state = {}
    dt = 1.0 / args.fps
    cam0 = None
    cam_lock = None
    for t in range(T):
        if args.npc_lock_cam:
            # camera GLIDES with the npc: constant world-space offset (no rotation),
            # npc stays centered the whole clip. Offset = 3/4 side view fixed at t=0.
            if cam_lock is None:
                p_n, p_m = np.asarray(npc[0, 0], float), np.asarray(mon[0, 0], float)
                axis = p_m - p_n; axis[1] = 0.0
                if np.linalg.norm(axis) < 1e-6: axis = np.array([1.0, 0.0, 0.0])
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                side = np.cross(axis, [0.0, 1.0, 0.0])
                d = args.npc_cam_dist
                eye_off = side * d - axis * 1.5 + np.array([0.0, 0.45 * d, 0.0])
                foc_off = np.array([0.0, 1.0, 0.0])
                cam_lock = (eye_off, foc_off)
            eye = np.asarray(npc[t, 0], float) + cam_lock[0]
            focus = np.asarray(npc[t, 0], float) + cam_lock[1]
            fwd = None; fov = TRK.CFG["FOV_DEG"]
        elif args.fixed_cam:
            if cam0 is None:
                # geometric stage view: side-on to the hunter->monster axis at t=0. 'pair'
                # frames both entities (distance scales with separation); 'npc'/'monster'
                # is a close-up on that entity (protagonist view for player-control demos).
                p_n, p_m = np.asarray(npc[0, 0], float), np.asarray(mon[0, 0], float)
                axis = p_m - p_n; axis[1] = 0.0
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                side = np.cross(axis, [0.0, 1.0, 0.0])
                if args.cam_focus == "pair":
                    mid = (p_n + p_m) / 2
                    d = max(9.0, 1.7 * float(np.linalg.norm(p_m - p_n)))
                    eye = mid + side * d + np.array([0.0, 0.45 * d, 0.0])
                    focus = mid + np.array([0.0, 1.0, 0.0])
                else:
                    # protagonist framing that covers its whole start->switch drift path:
                    # target = midpoint of the entity's position at t=0 and at mid-horizon,
                    # distance scaled to that path. Forced-prefix variants share both
                    # endpoints, so their cameras coincide exactly.
                    tm = T // 2
                    ent = npc if args.cam_focus == "npc" else mon
                    q0, qm = np.asarray(ent[0, 0], float), np.asarray(ent[tm, 0], float)
                    q_n, q_m = np.asarray(npc[tm, 0], float), np.asarray(mon[tm, 0], float)
                    axis = q_m - q_n; axis[1] = 0.0
                    axis = axis / (np.linalg.norm(axis) + 1e-8)
                    side = np.cross(axis, [0.0, 1.0, 0.0])
                    tgt = (q0 + qm) / 2
                    base_d = 7.0 if args.cam_focus == "npc" else 14.0
                    d = max(base_d, 1.1 * float(np.linalg.norm(qm - q0)) + 3.0)
                    eye = tgt + side * d + np.array([0.0, 0.35 * d, 0.0])
                    focus = tgt + np.array([0.0, 1.0, 0.0])
                cam0 = (eye, focus, None, TRK.CFG["FOV_DEG"])
            eye, focus, fwd, fov = cam0
        else:
            eye, focus, fwd, fov = TRK.camera_update(npc[t, 0], mon[t, 0], state, dt, is_blocked=is_blocked)
        # aligned-ref: start at the RECORDED camera (so pose[0] matches the GT rgb ref
        # pixel-for-pixel), blending into the follow-cam over ~0.7 s.
        cam0_rec = getattr(args, "_cam0", None)
        if cam0_rec is not None:
            B = max(1, int(round(0.7 * args.fps)))
            a = min(1.0, t / B)
            eye = (1 - a) * np.asarray(cam0_rec[0]) + a * np.asarray(eye)
            focus = (1 - a) * np.asarray(cam0_rec[1]) + a * np.asarray(focus)
            fov = (1 - a) * cam0_rec[2] + a * fov
        row = make_row(mon[t], npc[t], wpn[t], m_names, n_names, w_names, eye, focus, fov, aspect)
        vp, cam_eye, _, _ = R.compute_view_proj(row)
        cam_y = float(cam_eye[1]); hl, hh = cam_y - 60.0, cam_y + 30.0
        ml = R.build_skel_lines(row, "monster.list.1.joints", m_edges, R.COLOR_MONSTER_BASE, R.COLOR_MONSTER_MAX)
        nl = R.build_skel_lines(row, "npc_joints.body.1", n_edges, R.COLOR_NPC_BASE, R.COLOR_NPC_MAX)
        wl = R.build_skel_lines(row, "npc_joints.weapon.1", w_edges, R.COLOR_WEAPON_BASE, R.COLOR_WEAPON_MAX)
        if args.hide_npc:
            nl = []; wl = []
        # ① occluder cull: drop ALL terrain nearer than the character (minus a body margin),
        # so terrain between camera and character can't occlude it. Dynamic per-frame near-plane.
        near_cull = R.TERRAIN_NEAR_CULL
        if args.cull_occluders:
            char_dist = min(float(np.linalg.norm(np.asarray(cam_eye) - mon[t, 0])),
                            float(np.linalg.norm(np.asarray(cam_eye) - npc[t, 0])))
            near_cull = max(R.TERRAIN_NEAR_CULL, char_dist - args.cull_margin)
        img = renderer.render_frame(vp, skel_lines=[("monster", ml), ("npc", nl), ("weapon", wl)],
                                    depth_near=5.0, depth_far=80.0, height_band=(hl, hh),
                                    npc_foot=None, monster_foot=None, terrain_near_cull=near_cull)
        enc.stdin.write(img.tobytes())
        if t % 30 == 0:
            print(f"  frame {t}/{T}")
    enc.stdin.close(); enc.wait()
    print(f"saved {args.out}  ({T} frames, {T/args.fps:.1f}s @ {args.fps}fps)")


if __name__ == "__main__":
    main()
