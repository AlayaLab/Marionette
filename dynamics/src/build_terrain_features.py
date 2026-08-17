#!/usr/bin/env python3
"""Precompute terrain-conditioning features for the em19 combined lazy dataset (B / terrain-aware).

For every segment, per frame, writes seg_*.terr.npz with:
  patch     (T, N*N)  f16  egocentric heightmap, root-yaw-oriented, forward-biased, ROOT-RELATIVE height
  clear     (T, K)    f16  GT vertical clearance of K key (foot/limb) joints  d_j = world_y_j - H(x_j,z_j)
  H_local   (T, K)    f16  terrain height under each key joint, ROOT-RELATIVE  (H_j - root_world_y)  [for loss]
  R_row1    (T, 3)    f16  world-up row of the body rotation; maps predicted rel_pos -> world y-offset [for loss]
  contact   (T, K)    u8   GT contact mask (|clearance| < contact_thresh)                              [for loss]
  valid     (T,)      u8   frame has terrain under the root (inside loaded chunks)

The loss (train_v16.pose_step) reconstructs predicted world y-offset = R_row1 . rel_pos_pred_k and applies
  L_pen = ReLU((H_local+margin) - y_off_pred)^2 ,  L_contact = contact * (y_off_pred - H_local)^2
so no heightmap is needed at train time; root path is GT (teacher-forced), gradient corrects local pose.

Conventions match scripts.eval_continuous.features276_to_positions exactly (validated, recon err ~0.12m).
metadata.terr.npz holds the shared params: key_joints, N, cell_m, fwd_bias_m, fwd_axis, margin, contact_thresh.
"""
import argparse, os, sys
import numpy as np

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P)
MH = "/path/to/workspace/the bridge repository"
sys.path.insert(0, MH)
os.environ.setdefault("POSE_TERRAIN_DIR",
    "/path/to/workspace/dataset/mhwd-v2/raw/capture-a/rec/terrain")

import render_pose_terrain_gl as R_
from scripts.eval_continuous import features276_to_positions, _rot6d_to_matrix
from src.train_gpt_continuous import slice_to_pos276


class DenseFloor:
    """Flatten a TerrainStage into a dense MULTI-LAYER grid (nR,nC,L) for vectorized query.
    world_x = cx*100 + col, world_z = cy*100 + row. query(x,z,ref_y) returns the valid-layer
    height NEAREST ref_y -- replicating TerrainHeightmap.ground_at exactly (validated to <1cm)."""
    def __init__(self, terrain):
        keys = list(terrain.chunks.keys())
        cxs = [k[0] for k in keys]; cys = [k[1] for k in keys]
        self.cx0, self.cy0 = min(cxs), min(cys)
        nC = (max(cxs) - self.cx0 + 1) * 100
        nR = (max(cys) - self.cy0 + 1) * 100
        L = max(info["max_layers"] for info in terrain.chunks.values())
        self.grid = np.full((nR, nC, L), np.nan, np.float32)   # [row(z), col(x), layer]
        for (cx, cy), info in terrain.chunks.items():
            layers = info["layers"]                            # (10000, max_layers)
            valid = (layers["flags"] & 0x01) == 1
            h = np.where(valid, layers["h"], np.nan).reshape(100, 100, -1)
            r0 = (cy - self.cy0) * 100; c0 = (cx - self.cx0) * 100
            self.grid[r0:r0 + 100, c0:c0 + 100, :h.shape[2]] = h
        self.x_min = self.cx0 * 100; self.z_min = self.cy0 * 100
        self.nR, self.nC, self.L = nR, nC, L

    def query(self, x, z, ref_y):
        """Vectorized: valid-layer height nearest ref_y at world (x,z). NaN where out-of-grid/hole.
        x,z,ref_y broadcast to a common shape S; returns (S,)."""
        x, z, ref_y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(z, np.float64),
                                          np.asarray(ref_y, np.float64))
        S = x.shape
        c = x.astype(np.int64) - self.x_min          # int() truncates toward zero, matching ground_at
        r = z.astype(np.int64) - self.z_min
        inb = (c >= 0) & (c < self.nC) & (r >= 0) & (r < self.nR)
        cc = np.clip(c, 0, self.nC - 1); rr = np.clip(r, 0, self.nR - 1)
        hs = self.grid[rr, cc]                                  # (S, L)
        d = np.abs(hs - ref_y[..., None])                       # (S, L)
        d = np.where(np.isfinite(hs), d, np.inf)
        li = np.argmin(d, axis=-1)                              # (S,)
        out = np.take_along_axis(hs, li[..., None], axis=-1)[..., 0]
        allnan = ~np.isfinite(hs).any(axis=-1)
        out[allnan | ~inb] = np.nan
        return out.astype(np.float32)


def recon_world(feat_raw780, anchor_root0):
    """Production reconstruction -> monster world joints (T,54,3) + R (T,3,3) + root world (T,3)."""
    f276 = slice_to_pos276(feat_raw780.astype(np.float32))
    mon, _, _, _, _ = features276_to_positions(f276)               # origin-rooted (T,54,3)
    shift = np.asarray(anchor_root0, np.float64) - mon[0, 0]
    mon = mon + shift                                               # world
    Rm = _rot6d_to_matrix(f276[:, 264:270].astype(np.float64))     # (T,3,3) monster root rot
    return mon, Rm, mon[:, 0, :]                                   # joints, R, root world


def pick_key_joints_and_fwd(meta, D, n_sample=40, K=12):
    """Key joints = lowest mean-y monster joints (ground contacts). Forward axis = dominant
    horizontal motion axis in the body-local frame (sign included)."""
    sf, src = meta["seg_files"], meta["seg_source"]
    old = np.where(src == "old")[0][:n_sample]
    ys = np.zeros(54); cnt = 0
    loc_v = np.zeros(3)
    for i in old:
        base = str(sf[i])[:-4]
        z = np.load(f"{D}/{base}.aux.npz"); m0 = np.asarray(z["init_monster"], np.float64)
        if m0.shape != (54, 3):
            continue
        feat = np.load(f"{D}/{base}.npy")[:300]
        mon, Rm, root = recon_world(feat, m0[0])
        ys += (mon[:, :, 1] - root[:, 1:2]).mean(0); cnt += 1
        v = np.diff(root, axis=0)                                  # world velocity
        lv = np.einsum('tij,tj->ti', np.transpose(Rm[1:], (0, 2, 1)), v)  # R^T v -> local
        loc_v += np.abs(lv).sum(0)
    ys /= max(cnt, 1)
    key = np.argsort(ys)[:K]                                       # lowest = feet/limb tips
    key = key[key != 0] if (key == 0).any() else key              # root has no rel_pos slot; drop if present
    fwd_axis = int(np.argmax([loc_v[0], 0, loc_v[2]]))            # 0=local x(b1) or 2=local z(b3)
    fwd_axis = 0 if loc_v[0] >= loc_v[2] else 2
    return key.astype(np.int64), fwd_axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_dir", default=f"{P}/data/the state corpus")
    ap.add_argument("--stage", type=int, default=101)
    ap.add_argument("--N", type=int, default=11, help="patch grid size")
    ap.add_argument("--cell_m", type=float, default=0.4, help="patch cell size (m) -> extent N*cell ~4.4m")
    ap.add_argument("--fwd_bias_m", type=float, default=1.0, help="shift patch center forward (m)")
    ap.add_argument("--K", type=int, default=12, help="number of key (foot) joints")
    ap.add_argument("--margin", type=float, default=0.15, help="penetration margin (absorbs ~0.12m recon noise)")
    ap.add_argument("--contact_thresh", type=float, default=0.25, help="|clearance|<thresh -> contact")
    ap.add_argument("--validate_only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process only first N segs (debug)")
    args = ap.parse_args()
    D = args.combined_dir

    meta = np.load(f"{D}/metadata.npz", allow_pickle=True)
    sf, slen = meta["seg_files"], meta["seg_len"]
    print("loading terrain + building dense floor grid ...")
    terr = R_.TerrainStage(args.stage)
    floor = DenseFloor(terr)
    print(f"  dense floor grid {floor.grid.shape}, valid={np.isfinite(floor.grid).mean()*100:.0f}%")

    key, fwd_axis = pick_key_joints_and_fwd(meta, D, K=args.K)
    print(f"  key joints (lowest-y): {key.tolist()}  fwd_axis={'b1(x)' if fwd_axis==0 else 'b3(z)'}")

    # local patch grid offsets (N x N), centered, forward-biased along fwd_axis
    half = (args.N - 1) / 2.0
    g = (np.arange(args.N) - half) * args.cell_m
    gx, gz = np.meshgrid(g, g, indexing="ij")                     # (N,N)
    local = np.zeros((args.N * args.N, 3))
    local[:, 0] = gx.ravel(); local[:, 2] = gz.ravel()
    local[:, fwd_axis] += args.fwd_bias_m                          # forward bias
    NN = args.N * args.N

    def process(feat, m0):
        mon, Rm, root = recon_world(feat, m0)                     # (T,54,3),(T,3,3),(T,3)
        T = len(mon)
        # patch: world query = root + R @ local   (rotate local grid into world)
        wq = root[:, None, :] + np.einsum('tij,nj->tni', Rm, local)   # (T,NN,3)
        refp = np.broadcast_to(root[:, 1:2], (T, NN))             # ref_y = root height
        Hq = floor.query(wq[..., 0], wq[..., 2], refp)            # (T,NN)
        patch = (Hq - root[:, 1:2]).astype(np.float32)            # root-relative
        patch = np.nan_to_num(patch, nan=0.0)
        np.clip(patch, -10, 10, out=patch)
        # key joints world clearance + H_local
        kj = mon[:, key, :]                                       # (T,K,3)
        refk = np.broadcast_to(root[:, 1:2], (T, len(key)))
        Hk = floor.query(kj[..., 0], kj[..., 2], refk)            # (T,K) terrain height under joint
        valid = np.isfinite(floor.query(root[:, 0], root[:, 2], root[:, 1]))  # (T,) root has terrain
        H_local = (Hk - root[:, 1:2]).astype(np.float32)          # (T,K) root-relative terrain
        clear = (kj[..., 1] - Hk).astype(np.float32)              # (T,K) world clearance
        H_local = np.nan_to_num(H_local, nan=0.0)
        clear_filled = np.nan_to_num(clear, nan=99.0)
        contact = (np.abs(clear_filled) < args.contact_thresh).astype(np.uint8)
        R_row1 = Rm[:, 1, :].astype(np.float32)                   # (T,3) world-up row
        return dict(patch=patch.astype(np.float16),
                    clear=np.nan_to_num(clear, nan=0.0).astype(np.float16),
                    H_local=H_local.astype(np.float16), R_row1=R_row1.astype(np.float16),
                    contact=contact, valid=valid.astype(np.uint8)), clear

    if args.validate_only:
        old = np.where(meta["seg_source"] == "old")[0][:15]
        allc = []
        for i in old:
            base = str(sf[i])[:-4]; z = np.load(f"{D}/{base}.aux.npz")
            m0 = np.asarray(z["init_monster"], np.float64)
            if m0.shape != (54, 3):
                continue
            feat = np.load(f"{D}/{base}.npy")[:600]
            _, clear = process(feat, m0[0])
            allc.append(clear[np.isfinite(clear)].ravel())
        c = np.concatenate(allc)
        print(f"\n[validate] key-joint clearance over {len(c)} samples: mean={c.mean():.3f} "
              f"median={np.median(c):.3f} std={c.std():.3f} p5={np.percentile(c,5):.2f} p95={np.percentile(c,95):.2f}")
        print("  (expect mean ~ -0.1m for feet joints; compare to gate -0.09m)")
        return

    # full build
    np.savez(f"{D}/metadata.terr.npz", key_joints=key, N=np.int64(args.N), cell_m=np.float32(args.cell_m),
             fwd_bias_m=np.float32(args.fwd_bias_m), fwd_axis=np.int64(fwd_axis), margin=np.float32(args.margin),
             contact_thresh=np.float32(args.contact_thresh), patch_dim=np.int64(NN), n_key=np.int64(len(key)),
             stage=np.int64(args.stage))
    n = len(sf) if args.limit == 0 else min(args.limit, len(sf))
    done = 0; root_valid_frac = []
    for i in range(n):
        base = str(sf[i])[:-4]
        outp = f"{D}/{base}.terr.npz"
        z = np.load(f"{D}/{base}.aux.npz")
        m0 = np.asarray(z["init_monster"], np.float64)
        anchor = m0[0] if m0.shape == (54, 3) else m0.reshape(-1, 3)[0]  # new: (T,3)->frame0 root
        feat = np.load(f"{D}/{base}.npy")
        out, _ = process(feat, anchor)
        np.savez(outp, **out)
        root_valid_frac.append(out["valid"].mean())
        done += 1
        if i % 100 == 0:
            print(f"  {i}/{n} segs (root-in-chunk frac so far {np.mean(root_valid_frac):.2f})")
    print(f"\ndone: {done} segs -> *.terr.npz ; mean root-in-chunk frac={np.mean(root_valid_frac):.2f}")


if __name__ == "__main__":
    main()
