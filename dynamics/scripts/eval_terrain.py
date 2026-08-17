#!/usr/bin/env python3
"""Closed-loop terrain eval for terrain-aware PoseGPT (B). Autonomous rollout with per-step
egocentric-patch re-sampling from the model's OWN predicted world root, then penetration +
drift metrics on the generated world joints. Optionally compares vs a baseline (no-terrain) PoseGPT.

Metrics (SceneAdapt scorecard, on AUTONOMOUS rollout — drift is the failure mode):
  CFR  Collision-Frame Ratio  = frac frames with any key joint below terrain - margin
  MMP  Mean Max Penetration   = mean over frames of max_j max(0, -clearance_j)   [meters]
  JCR  Joint-Collision Ratio  = frac (frame,joint) below terrain - margin
  drift monster root path length over the horizon [m] + mean key-joint clearance
"""
import argparse, os, sys
import numpy as np, torch

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P)
MH = "/path/to/workspace/the bridge repository"
sys.path.insert(0, MH)
os.environ.setdefault("POSE_TERRAIN_DIR",
    "/path/to/workspace/dataset/mhwd-v2/raw/capture-a/rec/terrain")

import render_pose_terrain_gl as R_
from scripts.eval_v16 import load_action_gpt, load_pose_gpt, combined_rollout
from scripts.eval_continuous import features276_to_positions, denormalize_276, _rot6d_to_matrix
from src.train_gpt_continuous import slice_to_pos276
from src.train_v16 import (compute_goal, slice_root_18d, slice_body_258d, slice_weapon_6d)
from src.build_terrain_features import DenseFloor
from models.pose_gpt import PoseGPT


def load_terrain_pose_gpt(ckpt_path, mv, nv, dev, combined_dir):
    """Build a terrain-aware PoseGPT from a terrain ckpt + attach the loss/eval attrs."""
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt["config"]["model"]
    tm = np.load(f"{combined_dir}/metadata.terr.npz", allow_pickle=True)
    NN, K = int(tm["patch_dim"]), int(tm["n_key"])
    m = PoseGPT(mv, nv, action_emb_dim=cfg.get("action_emb_dim", 64), embed_dim=cfg["embed_dim"],
                block_size=cfg["block_size"], num_layers=cfg["num_layers"], n_head=cfg["n_head"],
                drop_out_rate=0.0, fc_rate=cfg.get("fc_rate", 4),
                root_input_dim=cfg.get("root_input_dim", 0), root_output_dim=cfg.get("root_output_dim", 0),
                terrain_patch_dim=NN, terrain_clear_dim=K, terrain_emb_dim=cfg.get("terrain_emb_dim", 64)).to(dev)
    sd = m.state_dict()
    if ckpt.get("ema"):
        shadow = ckpt["ema"]["shadow"]
        for k in sd:
            if k in shadow: sd[k] = shadow[k].to(dev)
        m.load_state_dict(sd); print(f"[terr pose] EMA weights from {os.path.basename(ckpt_path)}")
    else:
        m.load_state_dict(ckpt["model"]); print(f"[terr pose] raw weights from {os.path.basename(ckpt_path)}")
    m.eval()
    m.terr_NN, m.terr_K = NN, K
    # eval collision threshold = the margin the model was TRAINED with (config override > metadata)
    m.terrain_margin = float(cfg.get("terrain_margin", float(tm["margin"])))
    return m


class LiveTerrain:
    """Per-step egocentric patch + key-joint clearance sampler (matches build_terrain_features)."""
    def __init__(self, combined_dir, stage, dev):
        tm = np.load(f"{combined_dir}/metadata.terr.npz", allow_pickle=True)
        self.N = int(tm["N"]); self.NN = int(tm["patch_dim"]); self.K = int(tm["n_key"])
        self.key = np.asarray(tm["key_joints"], np.int64)
        cell = float(tm["cell_m"]); fb = float(tm["fwd_bias_m"]); fax = int(tm["fwd_axis"])
        half = (self.N - 1) / 2.0
        g = (np.arange(self.N) - half) * cell
        gx, gz = np.meshgrid(g, g, indexing="ij")
        local = np.zeros((self.NN, 3)); local[:, 0] = gx.ravel(); local[:, 2] = gz.ravel()
        local[:, fax] += fb
        self.local = local
        self.floor = DenseFloor(R_.TerrainStage(stage))
        self.dev = dev

    def sample(self, root_world, R, mon_world):
        """root_world (3,), R (3,3), mon_world (54,3) -> patch (NN,), clear (K,) torch on dev."""
        wq = root_world[None, :] + (self.local @ R.T)               # (NN,3) world query pts
        refp = np.full(self.NN, root_world[1])
        Hq = self.floor.query(wq[:, 0], wq[:, 2], refp)
        patch = np.nan_to_num((Hq - root_world[1]).astype(np.float32), nan=0.0).clip(-10, 10)
        kj = mon_world[self.key]
        Hk = self.floor.query(kj[:, 0], kj[:, 2], np.full(self.K, root_world[1]))
        clear = np.nan_to_num((kj[:, 1] - Hk).astype(np.float32), nan=0.0)
        return (torch.from_numpy(patch).to(self.dev), torch.from_numpy(clear).to(self.dev))


def recon_seq_world(feat276_denorm, anchor_root0):
    """-> mon_world (T,54,3), R (T,3,3), root_world (T,3). Matches features276_to_positions + anchor."""
    mon, _, _, _, _ = features276_to_positions(feat276_denorm)
    shift = np.asarray(anchor_root0, np.float64) - mon[0, 0]
    mon = mon + shift
    Rm = _rot6d_to_matrix(feat276_denorm[:, 264:270].astype(np.float64))
    return mon, Rm, mon[:, 0, :]


@torch.no_grad()
def terrain_rollout(ag, pg, lt, seed276, am, an, goal_m, goal_n, num_frames, anchor_root0,
                    mean276, std276, temperature=0.9, seed_hp=None,
                    force_am_seq=None, force_an_seq=None):
    """Autonomous rollout feeding live egocentric terrain to PoseGPT each step.

    HP (if ag.hp_dim>0): autoregressed — seeded from GT seed_hp (1,S,2), then the model's
    own hp_pred is fed forward each step (closed-loop world-state prediction).
    """
    dev = seed276.device; block = ag.block_size
    S = seed276.shape[1]
    full = seed276.clone(); am_s = am[:, :S].clone(); an_s = an[:, :S].clone()
    use_terr = pg.terrain_patch_dim > 0
    ag_hp = getattr(ag, "hp_dim", 0) > 0
    hp_buf = (seed_hp.clone() if seed_hp is not None
              else torch.ones(1, S, 2, device=dev)) if ag_hp else None

    # seed terrain buffer (patch+clear per seed frame) from reconstructed seed world
    seed_den = denormalize_276(seed276[0].cpu().numpy(), mean276, std276)
    mon0, R0, root0 = recon_seq_world(seed_den, anchor_root0)
    cur_root = root0[-1].copy()                                       # running world root
    tp_buf, tc_buf = [], []
    if use_terr:
        for t in range(S):
            p, c = lt.sample(root0[t], R0[t], mon0[t]); tp_buf.append(p); tc_buf.append(c)
    prog = torch.zeros(1, S, 2, device=dev)

    ag_terr = getattr(ag, "terrain_patch_dim", 0) > 0
    for t in range(num_frames):
        ctx = full[:, -block:]
        cl = full.shape[1]
        cmg = goal_m[:, max(0, cl - block):cl]; cng = goal_n[:, max(0, cl - block):cl]
        # terrain context buffer (shared by planner + pose); built causally from generated roots
        tp = tc = None
        if use_terr:
            tp = torch.stack(tp_buf[-block:], 0).unsqueeze(0)        # (1,ctx,NN)
            tc = torch.stack(tc_buf[-block:], 0).unsqueeze(0)
        a_tp = tp if ag_terr else None; a_tc = tc if ag_terr else None
        a_hp = hp_buf[:, -block:] if ag_hp else None
        pr, ml, nl, _, hp_pred, _ = ag(slice_root_18d(ctx), am_s[:, -block:], an_s[:, -block:], None,
                                       slice_weapon_6d(ctx), m_goal=cmg, n_goal=cng,
                                       terrain_patch=a_tp, terrain_clear=a_tc, hp=a_hp)
        if temperature > 0:
            na = torch.multinomial(torch.softmax(ml[0, -1] / temperature, -1), 1).view(1, 1)
            nn_ = torch.multinomial(torch.softmax(nl[0, -1] / temperature, -1), 1).view(1, 1)
        else:
            na = ml[:, -1:].argmax(-1); nn_ = nl[:, -1:].argmax(-1)
        # CONTROL injection: override sampled action with a forced action-id (controllability demo)
        if force_am_seq is not None:
            na = force_am_seq[:, t:t + 1].to(na.device)
        if force_an_seq is not None:
            nn_ = force_an_seq[:, t:t + 1].to(nn_.device)
        pose_actm = torch.cat([am_s[:, -block:][:, 1:], na], 1)
        pose_actn = torch.cat([an_s[:, -block:][:, 1:], nn_], 1)
        pose_pred, _ = pg(slice_body_258d(ctx), pose_actm, pose_actn, root=slice_root_18d(ctx),
                          terrain_patch=tp, terrain_clear=tc)
        nb = pose_pred[:, -1:, :258]; pro = pose_pred[:, -1:, 258:]
        n276 = torch.zeros(1, 1, 276, device=dev, dtype=full.dtype)
        n276[..., 3:162] = nb[..., 0:159]; n276[..., 165:258] = nb[..., 159:252]; n276[..., 258:264] = nb[..., 252:258]
        n276[..., 0:3] = pro[..., 0:3]; n276[..., 162:165] = pro[..., 3:6]
        n276[..., 264:270] = pro[..., 6:12]; n276[..., 270:276] = pro[..., 12:18]
        full = torch.cat([full, n276], 1); am_s = torch.cat([am_s, na], 1); an_s = torch.cat([an_s, nn_], 1)
        if ag_hp:
            next_hp = hp_pred[:, -1:, :].clamp(0.0, 1.0) if hp_pred is not None else hp_buf[:, -1:]
            hp_buf = torch.cat([hp_buf, next_hp], 1)   # feed predicted HP forward (autoregressive)
        if use_terr:
            den = denormalize_276(n276[0].cpu().numpy(), mean276, std276)   # (1,276)
            Rt = _rot6d_to_matrix(den[:, 264:270].astype(np.float64))[0]
            rd = den[0, 0:3]; cur_root = cur_root + Rt @ rd                 # integrate world root
            rel = den[0, 3:162].reshape(53, 3)
            mon_w = np.zeros((54, 3)); mon_w[0] = cur_root; mon_w[1:] = cur_root + (Rt @ rel.T).T
            p, c = lt.sample(cur_root, Rt, mon_w); tp_buf.append(p); tc_buf.append(c)
    # return the seed-END world root the rollout integrated from, so metrics reconstruct the
    # generated slice at the SAME world position the model's terrain conditioning used (HIGH-bug fix).
    return full, am_s, an_s, root0[-1].copy()


def world_from_gen(den_slice, seed_end_root):
    """Generated-slice world monster joints, anchored at the seed-END root the rollout continued from.
    features276_to_positions is origin-rooted (root[0]=R[0]@delta[0]); adding seed_end_root reproduces
    exactly the per-frame roots the rollout integrated (cur_root += R@delta from seed_end_root)."""
    mon, _, _, _, _ = features276_to_positions(den_slice)
    return mon + np.asarray(seed_end_root, np.float64)


def penetration_metrics(mon_world, lt, margin):
    """mon_world (T,54,3) -> CFR, MMP, JCR, mean clearance, drift."""
    root = mon_world[:, 0, :]; kj = mon_world[:, lt.key, :]          # (T,K,3)
    ref = np.broadcast_to(root[:, 1:2], (len(root), lt.K))
    Hk = lt.floor.query(kj[..., 0], kj[..., 2], ref)                # (T,K)
    clear = kj[..., 1] - Hk
    valid = np.isfinite(clear)
    # penetration depth = how far a joint is below (terrain - margin) = max(0, -clear - margin)
    pen = np.where(valid, np.maximum(0.0, -clear - margin), 0.0)
    frame_has = (pen > 0).any(axis=1)
    cfr = float(frame_has.mean())
    mmp = float(pen.max(axis=1).mean())
    jcr = float((pen[valid] > 0).mean()) if valid.any() else 0.0
    meanclear = float(clear[valid].mean()) if valid.any() else float("nan")
    drift = float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum())
    return dict(CFR=cfr, MMP=mmp, JCR=jcr, mean_clear=meanclear, drift=drift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_dir", default=f"{P}/data/the state corpus")
    ap.add_argument("--action_ckpt", default=f"{P}/output/dynamics/action_gpt.pt")
    ap.add_argument("--pose_ckpt", default=f"{P}/output/dynamics/pose_gpt.pt")
    ap.add_argument("--baseline_pose_ckpt", default="", help="no-terrain PoseGPT for head-to-head")
    ap.add_argument("--segments", type=int, nargs="+", default=[133, 200, 400, 600, 800])
    ap.add_argument("--seed_frames", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed_dist_thresh", type=float, default=15.0)
    ap.add_argument("--stage", type=int, default=101)
    ap.add_argument("--baseline_action_ckpt", default="", help="non-terrain ActionGPT for the baseline "
                    "rollout (required when --action_ckpt is a terrain-planner ActionGPT)")
    ap.add_argument("--seed", type=int, default=0, help="fix rollout sampling RNG for reproducible A/B")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)  # deterministic rollout sampling

    meta = np.load(f"{args.combined_dir}/metadata.npz", allow_pickle=True)
    mean780, std780 = meta["mean"].astype(np.float32), meta["std"].astype(np.float32)
    mean276, std276 = slice_to_pos276(mean780), slice_to_pos276(std780)
    m_ids, n_ids = meta["m_ids"], meta["n_ids"]
    mv, nv = int(meta["m_action_vocab_size"]), int(meta["n_action_vocab_size"])
    sf = meta["seg_files"]
    margin = float(np.load(f"{args.combined_dir}/metadata.terr.npz")["margin"])

    def lut(ids):
        a = np.zeros(int(max(ids)) + 1, np.int64)
        for i, v in enumerate(ids): a[int(v)] = i + 1
        return a
    mlut, nlut = lut(m_ids), lut(n_ids)

    ag = load_action_gpt(args.action_ckpt, mv, nv, dev, combined_dir=args.combined_dir)
    ag_terr = getattr(ag, "terrain_patch_dim", 0) > 0
    # baseline rollout's ActionGPT: a non-terrain one (combined_rollout feeds no terrain).
    if args.baseline_action_ckpt:
        ag_b = load_action_gpt(args.baseline_action_ckpt, mv, nv, dev)
    else:
        # ag_b is only used by the baseline rollout branch, which runs iff baseline_pose_ckpt is set.
        # A terrain-aware ag can't drive the (no-terrain) combined_rollout, so only require a
        # non-terrain baseline when we're actually doing the A/B comparison.
        assert not (ag_terr and args.baseline_pose_ckpt), \
            "--action_ckpt is terrain-aware; pass --baseline_action_ckpt (non-terrain) for the baseline A/B"
        ag_b = ag
    lt = LiveTerrain(args.combined_dir, args.stage, dev)
    pg_t = load_terrain_pose_gpt(args.pose_ckpt, mv, nv, dev, args.combined_dir)
    pg_b = load_pose_gpt(args.baseline_pose_ckpt, mv, nv, dev) if args.baseline_pose_ckpt else None

    se, re = args.seed_frames, args.seed_frames + args.horizon
    agg = {}
    for sid in args.segments:
        base = str(sf[sid])[:-4]
        feat_full = np.load(f"{args.combined_dir}/{base}.npy").astype(np.float32)
        z = np.load(f"{args.combined_dir}/{base}.aux.npz")
        if len(feat_full) < re:
            print(f"seg{sid}: too short, skip"); continue
        im = np.asarray(z["init_monster"]); inp = np.asarray(z["init_npc"])
        im_r = im.reshape(-1, 3); inp_r = inp.reshape(-1, 3)
        if im.shape == (54, 3):  # OLD: frame-0 snapshot -> approx per-frame dist via root only
            anchor0 = im[0]
            dist = np.full(len(feat_full), 0.0)  # old segs are close-combat throughout; seed at 0
            s0 = 0
        else:                    # NEW: per-frame world roots
            anchor0 = im_r[0]
            dist = np.linalg.norm(im_r - inp_r, axis=-1)
            cand = [s for s in range(0, len(feat_full) - re + 1) if dist[s:s + se].max() < args.seed_dist_thresh]
            s0 = cand[0] if cand else 0
        feat = feat_full[s0:s0 + re]
        seg276 = slice_to_pos276((feat - mean780) / (std780 + 1e-8)).astype(np.float32)
        am = mlut[np.clip(np.asarray(z["action_m"])[s0:s0 + re], 0, len(mlut) - 1)]
        an = nlut[np.clip(np.asarray(z["action_n"])[s0:s0 + re], 0, len(nlut) - 1)]
        gm = compute_goal(torch.from_numpy(am).unsqueeze(0).to(dev))
        gn = compute_goal(torch.from_numpy(an).unsqueeze(0).to(dev))
        s276 = torch.from_numpy(seg276[:se]).unsqueeze(0).to(dev)
        amt = torch.from_numpy(am).unsqueeze(0).to(dev); ant = torch.from_numpy(an).unsqueeze(0).to(dev)
        # anchor for the seed reconstruction = the seg's true world root at the seed frame
        anchor = im_r[s0] if im.shape != (54, 3) else anchor0
        margin = pg_t.terrain_margin   # use the TRAINED collision threshold

        # GT seed HP for autoregressive HP rollout (only if the planner uses HP).
        seed_hp_t = None
        if getattr(ag, "hp_dim", 0) > 0 and "hp" in z:
            seed_hp_t = torch.from_numpy(np.asarray(z["hp"], np.float32)[s0:s0 + se]).unsqueeze(0).to(dev)

        torch.manual_seed(args.seed + sid)   # same RNG luck for terrain & baseline branches
        full_t, _, _, seed_end_root = terrain_rollout(ag, pg_t, lt, s276, amt[:, :re], ant[:, :re],
                                       gm, gn, args.horizon, anchor, mean276, std276, args.temperature,
                                       seed_hp=seed_hp_t)
        den_t = denormalize_276(full_t[0, se:].cpu().numpy(), mean276, std276)
        mon_t = world_from_gen(den_t, seed_end_root)          # SAME world position the rollout queried
        mt = penetration_metrics(mon_t, lt, margin)
        line = f"seg{sid}: [terrain] CFR={mt['CFR']:.3f} MMP={mt['MMP']:.3f} JCR={mt['JCR']:.3f} drift={mt['drift']:.0f}m clear={mt['mean_clear']:.2f}"
        for k, v in mt.items(): agg.setdefault("t_" + k, []).append(v)

        if pg_b is not None:
            torch.manual_seed(args.seed + sid)   # identical RNG to the terrain branch
            full_b, _, _ = combined_rollout(ag_b, pg_b, s276, amt[:, :se], ant[:, :se],
                torch.zeros(1, se, 2, device=dev), num_frames=args.horizon,
                temperature=args.temperature, goal_m_seq=gm, goal_n_seq=gn)
            den_b = denormalize_276(full_b[0, se:].cpu().numpy(), mean276, std276)
            mon_b = world_from_gen(den_b, seed_end_root)      # same anchor as terrain branch -> comparable
            mb = penetration_metrics(mon_b, lt, margin)
            line += f"  | [base] CFR={mb['CFR']:.3f} MMP={mb['MMP']:.3f} JCR={mb['JCR']:.3f} drift={mb['drift']:.0f}m"
            for k, v in mb.items(): agg.setdefault("b_" + k, []).append(v)
        print(line)

    print(f"\n=== AGGREGATE ({len(agg.get('t_CFR', []))} segs, margin={margin}m) ===")
    for pfx, tag in [("t_", "TERRAIN"), ("b_", "BASELINE")]:
        if pfx + "CFR" in agg:
            print(f"  {tag:9s} CFR={np.mean(agg[pfx+'CFR']):.3f} MMP={np.mean(agg[pfx+'MMP']):.3f} "
                  f"JCR={np.mean(agg[pfx+'JCR']):.3f} drift={np.mean(agg[pfx+'drift']):.0f}m "
                  f"clear={np.mean(agg[pfx+'mean_clear']):.2f}")


if __name__ == "__main__":
    main()
