#!/usr/bin/env python3
"""V16 training script: trains either ActionGPT or PoseGPT.

Two phases:
  - Phase 1 (0 .. warmup_iter): Teacher forcing
  - Phase 2 (warmup_iter .. total_iter): Pushforward 2-step (Brandstetter+ ICLR'22)

Usage:
  python src/train_v16.py configs/train_v16_action.yaml --model action
  python src/train_v16.py configs/train_v16_pose.yaml --model pose
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.action_gpt import ActionGPT
from models.pose_gpt import PoseGPT
from src.train_gpt_continuous import slice_to_pos276, EMA


# ============================================================
# Feature slicing: 276D -> root(18D) and body(258D)
# ============================================================

def slice_root_18d(data_276):
    """Extract 18D root tensor from 276D features.

    Layout returned: [M_root_delta(3), N_root_delta(3), M_rot6d(6), N_rot6d(6)]
    """
    if isinstance(data_276, torch.Tensor):
        return torch.cat([
            data_276[..., 0:3],     # M_root_delta
            data_276[..., 162:165], # N_root_delta
            data_276[..., 264:270], # M_rot6d
            data_276[..., 270:276], # N_rot6d
        ], dim=-1)
    else:
        return np.concatenate([
            data_276[..., 0:3],
            data_276[..., 162:165],
            data_276[..., 264:270],
            data_276[..., 270:276],
        ], axis=-1)


def slice_body_258d(data_276):
    """Extract 258D body tensor from 276D features.

    Layout returned: [M_rel_pos(159), N_rel_pos(93), weapon(6)]
    """
    if isinstance(data_276, torch.Tensor):
        return torch.cat([
            data_276[..., 3:162],   # M_rel_pos
            data_276[..., 165:258], # N_rel_pos
            data_276[..., 258:264], # weapon
        ], dim=-1)
    else:
        return np.concatenate([
            data_276[..., 3:162],
            data_276[..., 165:258],
            data_276[..., 258:264],
        ], axis=-1)


def slice_root_delta_6d(data_276):
    """Extract 6D root_delta tensor (M+N root_delta only, no rot6d)."""
    if isinstance(data_276, torch.Tensor):
        return torch.cat([data_276[..., 0:3], data_276[..., 162:165]], dim=-1)
    return np.concatenate([data_276[..., 0:3], data_276[..., 162:165]], axis=-1)


def slice_rot6d_12d(data_276):
    """Extract 12D rot6d tensor (M+N rot6d only, no root_delta)."""
    if isinstance(data_276, torch.Tensor):
        return torch.cat([data_276[..., 264:270], data_276[..., 270:276]], dim=-1)
    return np.concatenate([data_276[..., 264:270], data_276[..., 270:276]], axis=-1)


def slice_weapon_6d(data_276):
    """Extract 6D weapon tensor (used as ActionGPT input)."""
    return data_276[..., 258:264]


# ============================================================
# Dataset
# ============================================================

class V16Dataset(Dataset):
    """Sliding-window dataset for V16 training. Returns 276D features + actions + progress."""

    def __init__(self, segments_276, action_m, action_n, progress,
                 window_size=512, stride=256):
        self.window_size = window_size
        self.segments = segments_276
        self.action_m = action_m
        self.action_n = action_n
        self.progress = progress

        self.samples = []
        for i, seg in enumerate(segments_276):
            L = seg.shape[0]
            if L <= window_size:
                self.samples.append((i, 0, L))
            else:
                for start in range(0, L - window_size + 1, stride):
                    self.samples.append((i, start, window_size))
                if (L - window_size) % stride != 0:
                    self.samples.append((i, L - window_size, window_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg_idx, start, length = self.samples[idx]
        frames = self.segments[seg_idx][start:start + length].copy()
        am = self.action_m[seg_idx][start:start + length].copy()
        an = self.action_n[seg_idx][start:start + length].copy()
        pg = self.progress[seg_idx][start:start + length].copy()

        if length < self.window_size:
            pad_n = self.window_size - length
            frames = np.concatenate([frames, np.zeros((pad_n, frames.shape[-1]), dtype=np.float32)], axis=0)
            am = np.concatenate([am, np.zeros(pad_n, dtype=np.int32)])
            an = np.concatenate([an, np.zeros(pad_n, dtype=np.int32)])
            pg = np.concatenate([pg, np.zeros((pad_n, 2), dtype=np.float32)], axis=0)
            mask = np.zeros(self.window_size, dtype=np.float32)
            mask[:length] = 1.0
        else:
            mask = np.ones(self.window_size, dtype=np.float32)

        return (
            torch.from_numpy(frames),               # (T, 276)
            torch.from_numpy(am).long(),             # (T,)
            torch.from_numpy(an).long(),
            torch.from_numpy(pg),                    # (T, 2)
            torch.from_numpy(mask),                  # (T,)
        )


def load_data(config):
    # lazy per-segment dir (build_dataset_lazy.py) -> metadata.npz present
    if (Path(config["data"]["motion_dir"]) / "metadata.npz").exists():
        from src.dataset_lazy import load_data_lazy
        return load_data_lazy(config)
    motion_path = Path(config["data"]["motion_dir"]) / "motion_data.npz"
    data = np.load(motion_path, allow_pickle=True)
    num_seg = int(data["num_segments"])

    segments_276 = []
    for i in range(num_seg):
        seg_276 = slice_to_pos276(data[f"segment_{i}"]).astype(np.float32)
        segments_276.append(seg_276)

    has_action = f"action_m_0" in data
    if not has_action:
        raise RuntimeError("V16 requires action IDs in motion_data.npz")
    action_m = [data[f"action_m_{i}"].astype(np.int32) for i in range(num_seg)]
    action_n = [data[f"action_n_{i}"].astype(np.int32) for i in range(num_seg)]
    progress = [data[f"action_progress_{i}"].astype(np.float32) for i in range(num_seg)]

    m_vocab = int(data["m_action_vocab_size"])
    n_vocab = int(data["n_action_vocab_size"])

    # Train/val split
    indices = list(range(num_seg))
    rng = random.Random(config["train"].get("seed", 42))
    rng.shuffle(indices)
    val_ratio = config["data"]["val_ratio"]
    n_val = max(1, int(num_seg * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def sub(arrs, idxs):
        return [arrs[i] for i in idxs]

    window_size = config["data"]["window_size"]
    stride = config["data"]["window_stride"]

    train_ds = V16Dataset(
        sub(segments_276, train_idx), sub(action_m, train_idx),
        sub(action_n, train_idx), sub(progress, train_idx),
        window_size=window_size, stride=stride,
    )
    val_ds = V16Dataset(
        sub(segments_276, val_idx), sub(action_m, val_idx),
        sub(action_n, val_idx), sub(progress, val_idx),
        window_size=window_size, stride=stride,
    )

    return train_ds, val_ds, m_vocab, n_vocab


# ============================================================
# Loss helpers (mask-aware)
# ============================================================

def masked_smooth_l1(pred, target, mask):
    """Mean SmoothL1 over valid frames."""
    per_elem = F.smooth_l1_loss(pred, target, reduction='none')
    per_frame = per_elem.mean(dim=-1)  # (B, T)
    return (per_frame * mask).sum() / mask.sum().clamp(min=1)


def masked_ce(logits, target, mask, ignore_index=0):
    """Cross-entropy averaged over valid frames (mask=1).

    `ignore_index` is forwarded to F.cross_entropy (action ID 0 = padding).
    """
    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, V), target.reshape(-1),
        ignore_index=ignore_index, reduction='none'
    ).view(B, T)
    valid = mask * (target != ignore_index).float()
    return (ce * valid).sum() / valid.sum().clamp(min=1)


# ============================================================
# Training step: ActionGPT
# ============================================================

def compute_goal(a):
    """(B,T) long action ids -> (B,T) the 'goal' = action of the NEXT segment (the
    upcoming-transition action). For frames in the final segment (no future transition)
    goal = current action. Sparse future-action conditioning in goal mode.

    Vectorized (no Python frame loop): for each t, find the next segment-start index
    j>t via a reverse cummin over candidate start indices, then gather a[j]."""
    B, T = a.shape
    dev = a.device
    INF = T + 1
    start = torch.zeros_like(a, dtype=torch.bool)
    start[:, 1:] = a[:, 1:] != a[:, :-1]                       # segment starts
    idx = torch.arange(T, device=dev).expand(B, T)
    cand = torch.where(start, idx, torch.full_like(idx, INF))  # start idx else INF
    suffix_min = torch.flip(torch.cummin(torch.flip(cand, [1]), dim=1).values, [1])  # min(cand[t:])
    next_start = torch.full((B, T), INF, device=dev, dtype=torch.long)
    next_start[:, :-1] = suffix_min[:, 1:]                     # min(cand[t+1:])
    has = next_start < INF
    ns = torch.where(has, next_start, torch.full_like(next_start, T - 1))
    goal = torch.gather(a, 1, ns)
    return torch.where(has, goal, a)                           # no future start -> current


def action_step(model, batch, phase, weights):
    """Returns total loss + dict of components.

    phase: 'tf' (teacher forcing) or 'pf' (pushforward 2-step)
    Handles root_out_dim=18 (V16.0), 12 (V16.2), 0 (V16.3).
    """
    use_terr = getattr(model, 'terrain_patch_dim', 0) > 0
    use_hp = getattr(model, 'hp_dim', 0) > 0
    # Batch order: frames, am, an, progress, mask [, terr] [, hp]. terr precedes hp.
    items = list(batch)
    frames_276, action_m, action_n, progress, mask = items[:5]
    idx = 5
    terr = None
    if use_terr:
        terr = items[idx]; idx += 1
    hp = None
    if use_hp:
        hp = items[idx]; idx += 1  # (B, T, 3) = [monster_hp%, npc_hp%, valid]
    B, T, _ = frames_276.shape

    root = slice_root_18d(frames_276)
    weapon = slice_weapon_6d(frames_276)
    rod = model.root_out_dim  # 0, 12, or 18
    use_goal = getattr(model, 'use_goal', False)

    in_root = root[:, :-1]
    in_actm = action_m[:, :-1]
    in_actn = action_n[:, :-1]
    in_prog = progress[:, :-1]
    in_weap = weapon[:, :-1]

    # Planner terrain (causal: patch around the current frame to steer the next action/root)
    in_tp = in_tc = None
    if use_terr:
        NN, K = model.terrain_patch_dim_meta, model.terrain_n_key
        patch, clear, _, _, _, _ = split_terrain(terr, NN, K)
        in_tp = patch[:, :-1]; in_tc = clear[:, :-1]

    # HP conditioning (causal: current-frame HP in, predict next-frame HP).
    in_hp = tgt_hp = hp_valid = None
    if use_hp:
        in_hp = hp[:, :-1, :2]      # (B, T-1, 2) feed [monster%, npc%]
        tgt_hp = hp[:, 1:, :2]      # next-frame HP target
        hp_valid = hp[:, 1:, 2]     # (B, T-1) valid flag (monster -1 sentinel frames excluded)

    tgt_actm = action_m[:, 1:]
    tgt_actn = action_n[:, 1:]
    pred_mask = mask[:, 1:]

    # Goal mode: sparse future-action conditioning (the action at the next transition).
    # Derived from the GT action sequence; progress becomes a prediction target, not input.
    in_mgoal = in_ngoal = None
    if use_goal:
        in_mgoal = compute_goal(action_m)[:, :-1]
        in_ngoal = compute_goal(action_n)[:, :-1]

    # Build root target based on root_out_dim
    if rod == 18:
        tgt_root = root[:, 1:]
    elif rod == 12:
        tgt_root = slice_rot6d_12d(frames_276[:, 1:])
    else:
        tgt_root = None

    pf_k = weights.get('_pf_k', 2)

    if phase == 'tf':
        pred_root, m_logits, n_logits, prog_pred, hp_pred, _ = model(
            in_root, in_actm, in_actn, in_prog, in_weap, m_goal=in_mgoal, n_goal=in_ngoal,
            terrain_patch=in_tp, terrain_clear=in_tc, hp=in_hp)
    else:
        # K-step pushforward
        cur_root = in_root
        cur_actm = in_actm
        cur_actn = in_actn
        for step in range(pf_k - 1):
            with torch.no_grad():
                p_root, p_mlog, p_nlog, _, _, _ = model(
                    cur_root, cur_actm, cur_actn, in_prog, in_weap, m_goal=in_mgoal, n_goal=in_ngoal,
                    terrain_patch=in_tp, terrain_clear=in_tc, hp=in_hp)
                p_actm = p_mlog.argmax(dim=-1)
                p_actn = p_nlog.argmax(dim=-1)
            cur_actm = torch.cat([in_actm[:, :1], p_actm[:, :-1]], dim=1)
            cur_actn = torch.cat([in_actn[:, :1], p_actn[:, :-1]], dim=1)
            if rod > 0 and p_root is not None:
                cur_root = torch.cat([in_root[:, :1], p_root[:, :-1]], dim=1)
            # else cur_root stays as GT (pure discrete mode)
        pred_root, m_logits, n_logits, prog_pred, hp_pred, _ = model(
            cur_root, cur_actm, cur_actn, in_prog, in_weap, m_goal=in_mgoal, n_goal=in_ngoal,
            terrain_patch=in_tp, terrain_clear=in_tc, hp=in_hp)

    loss_dict = {}
    total = torch.tensor(0.0, device=frames_276.device)
    if pred_root is not None and tgt_root is not None:
        loss_root = masked_smooth_l1(pred_root, tgt_root, pred_mask)
        total = total + weights.get('root', 10.0) * loss_root
        loss_dict['root'] = loss_root.item()

    loss_actm = masked_ce(m_logits, tgt_actm, pred_mask)
    loss_actn = masked_ce(n_logits, tgt_actn, pred_mask)
    total = total + weights['action_m'] * loss_actm + weights['action_n'] * loss_actn
    loss_dict['action_m'] = loss_actm.item()
    loss_dict['action_n'] = loss_actn.item()

    # Goal mode: predict progress (no longer an input leak) — masked MSE on input frames.
    if use_goal and prog_pred is not None:
        in_mask = mask[:, :-1]
        per = ((prog_pred - in_prog) ** 2).mean(dim=-1)
        loss_prog = (per * in_mask).sum() / in_mask.sum().clamp(min=1)
        total = total + weights.get('progress', 1.0) * loss_prog
        loss_dict['progress'] = loss_prog.item()

    # HP prediction: next-frame [monster_hp%, npc_hp%], masked Smooth-L1.
    # hp_valid drops monster -1 sentinel frames from the loss.
    if use_hp and hp_pred is not None and tgt_hp is not None:
        loss_hp = masked_smooth_l1(hp_pred, tgt_hp, pred_mask * hp_valid)
        total = total + weights.get('hp', 0.5) * loss_hp
        loss_dict['hp'] = loss_hp.item()

    loss_dict['total'] = total.item()
    return total, loss_dict


# ============================================================
# Training step: PoseGPT
# ============================================================

def split_terrain(terr, NN, K):
    """Split the bundled terrain tensor (B,T,terr_dim) into its parts."""
    o = 0
    patch = terr[..., o:o + NN]; o += NN
    clear = terr[..., o:o + K]; o += K
    H_local = terr[..., o:o + K]; o += K
    R_row1 = terr[..., o:o + 3]; o += 3
    contact = terr[..., o:o + K]; o += K
    valid = terr[..., o:o + 1]
    return patch, clear, H_local, R_row1, contact, valid


def pose_step(model, batch, phase, weights):
    """PoseGPT training step. Handles root_input_dim/root_output_dim for B/C variants
    and optional terrain conditioning + penetration/contact loss (B / terrain-aware)."""
    use_terr = getattr(model, 'terrain_patch_dim', 0) > 0
    # Batch order: frames, am, an, progress, mask [, terr] [, hp]. PoseGPT ignores hp.
    items = list(batch)
    frames_276, action_m, action_n, progress, mask = items[:5]
    terr = items[5] if use_terr else None
    B, T, _ = frames_276.shape

    body = slice_body_258d(frames_276)  # (B, T, 258)
    rid = model.root_input_dim
    rod = model.root_output_dim

    # Build root input/target based on mode
    if rid == 6:  # V16.2 (Plan B): root_delta only
        root_in = slice_root_delta_6d(frames_276)
    elif rid == 18:  # V16.3 (Plan C): full root
        root_in = slice_root_18d(frames_276)
    else:
        root_in = None

    if rod == 6:
        root_tgt = slice_root_delta_6d(frames_276[:, 1:])
    elif rod == 18:
        root_tgt = slice_root_18d(frames_276[:, 1:])
    else:
        root_tgt = None

    # PoseGPT convention: input action is for the frame being predicted (+1 shift)
    in_body = body[:, :-1]
    in_actm = action_m[:, 1:]
    in_actn = action_n[:, 1:]
    in_root = root_in[:, :-1] if root_in is not None else None
    tgt_body = body[:, 1:]
    pred_mask = mask[:, 1:]

    # Terrain: INPUT patch/clear are CAUSAL (frame t, around the current position) to predict
    # frame t+1 — mirrors closed-loop inference (sample patch from latest known root). The
    # penetration LOSS uses the PREDICTED frame's (t+1) terrain height + body orientation.
    in_tp = in_tc = None
    if use_terr:
        NN, K = model.terrain_patch_dim_meta, model.terrain_n_key
        patch, clear, H_local, R_row1, contact, valid = split_terrain(terr, NN, K)
        in_tp = patch[:, :-1]; in_tc = clear[:, :-1]
        tgt_H = H_local[:, 1:]; tgt_R1 = R_row1[:, 1:]
        tgt_contact = contact[:, 1:]; tgt_valid = valid[:, 1:, 0]

    pf_k = weights.get('_pf_k', 2)  # pushforward steps (default 2 for backward compat)

    if phase == 'tf':
        pred, _ = model(in_body, in_actm, in_actn, root=in_root,
                        terrain_patch=in_tp, terrain_clear=in_tc)
    else:
        # K-step pushforward: K-1 detached passes, then 1 pass with gradient
        cur_body = in_body
        cur_root = in_root
        for step in range(pf_k - 1):
            with torch.no_grad():
                p_pred, _ = model(cur_body, in_actm, in_actn, root=cur_root,
                                  terrain_patch=in_tp, terrain_clear=in_tc)
            cur_body = torch.cat([in_body[:, :1], p_pred[:, :-1, :258]], dim=1)
            if cur_root is not None and rod > 0:
                cur_root = torch.cat([in_root[:, :1], p_pred[:, :-1, 258:]], dim=1)
        pred, _ = model(cur_body, in_actm, in_actn, root=cur_root,
                        terrain_patch=in_tp, terrain_clear=in_tc)

    # Split prediction into body and root parts
    pred_body = pred[..., :258]
    loss_body = masked_smooth_l1(pred_body, tgt_body, pred_mask)

    loss_dict = {'body': loss_body.item()}
    total = weights['body'] * loss_body

    if rod > 0 and root_tgt is not None:
        pred_root = pred[..., 258:]
        loss_root = masked_smooth_l1(pred_root, root_tgt, pred_mask)
        total = total + weights.get('root', 10.0) * loss_root
        loss_dict['root'] = loss_root.item()

    # ---- Terrain penetration + contact loss (B) ----
    # world_y_offset of monster joint k = R_row1 . rel_pos_k (root-path stop-grad; root teacher-forced).
    # L_pen = ReLU((H_local + margin) - y_off)^2  (one-sided hinge, fires only below terrain+margin)
    # L_contact = contact * (y_off - H_local)^2   (pin GT-contact joints to the surface)
    if use_terr:
        kr = model.terrain_key_rel                              # (K,) rel_pos indices
        rel_pred = pred_body[..., :159].reshape(B, T - 1, 53, 3)[:, :, kr, :]  # (B,T-1,K,3) NORMALIZED
        rel_pred = rel_pred * model.terrain_relpos_std + model.terrain_relpos_mean  # -> METERS
        y_off = torch.einsum('btc,btkc->btk', tgt_R1, rel_pred)              # world y-offset of key joints
        m = (pred_mask * tgt_valid).unsqueeze(-1)                            # (B,T-1,1) valid+real frames
        floor = tgt_H + model.terrain_margin
        pen = torch.relu(floor - y_off) ** 2
        loss_pen = (pen * m).sum() / m.sum().clamp(min=1) / max(kr.numel(), 1)
        cm = tgt_contact * m
        loss_contact = ((y_off - tgt_H) ** 2 * cm).sum() / cm.sum().clamp(min=1)
        total = total + weights.get('pen', 0.0) * loss_pen + weights.get('contact', 0.0) * loss_contact
        loss_dict['pen'] = loss_pen.item()
        loss_dict['contact'] = loss_contact.item()

        # v3 anti-sink: penalize accumulated WORLD-Y root drift vs GT over the window. The deep
        # penetrations are AR root-sink (cumulative downward drift); world_dy = R_row1 . root_delta
        # (same R the precompute used). cumsum(pred dy) must track cumsum(GT dy) -> monster stays on
        # terrain like GT instead of sinking. Targets the residual seg900-type sink.
        w_as = weights.get('antisink', 0.0)
        if w_as > 0 and rod == 18:
            rdm, rds = model.terrain_rd_mean, model.terrain_rd_std
            pred_rd = pred[..., 258:261] * rds + rdm                # M_root_delta (B,T-1,3) meters
            gt_rd = root_tgt[..., 0:3] * rds + rdm
            mf = pred_mask                                          # (B,T-1) valid frames
            pred_dy = torch.einsum('btc,btc->bt', tgt_R1, pred_rd) * mf
            gt_dy = torch.einsum('btc,btc->bt', tgt_R1, gt_rd) * mf
            drift = torch.cumsum(pred_dy - gt_dy, dim=1)            # accumulated world-Y divergence (m)
            # normalize by accumulated-frame count -> MEAN per-frame drift (O(m^2), scale-stable vs T)
            denom = torch.arange(1, drift.shape[1] + 1, device=drift.device).float().unsqueeze(0)
            drift = drift / denom
            loss_antisink = ((drift ** 2) * mf).sum() / mf.sum().clamp(min=1)
            total = total + w_as * loss_antisink
            loss_dict['antisink'] = loss_antisink.item()

    loss_dict['total'] = total.item()
    return total, loss_dict


# ============================================================
# Main training loop
# ============================================================

def build_model(model_kind, config, m_vocab, n_vocab, device):
    mc = config['model']
    if model_kind == 'action':
        # Planner terrain (B v2): patch/clear dims from metadata.terr.npz (same precompute as PoseGPT)
        a_tp = a_tc = 0
        if config['data'].get('terrain', False) and mc.get('terrain_planner', False):
            tm = np.load(Path(config['data']['motion_dir']) / 'metadata.terr.npz', allow_pickle=True)
            a_tp = int(tm['patch_dim']); a_tc = int(tm['n_key'])
        # HP conditioning: 2D input ([monster%, npc%]) + 2D prediction head, gated by data.hp.
        hp_on = config['data'].get('hp', False)
        a_hp_in = mc.get('hp_dim', 2) if hp_on else 0
        a_hp_out = mc.get('hp_out_dim', 2) if hp_on else 0
        model = ActionGPT(
            m_vocab_size=m_vocab,
            n_vocab_size=n_vocab,
            action_emb_dim=mc.get('action_emb_dim', 64),
            progress_dim=2,
            weapon_dim=6,
            distance_dim=mc.get('distance_dim', 0),
            root_out_dim=mc.get('root_out_dim', 18),
            embed_dim=mc['embed_dim'],
            block_size=mc['block_size'],
            num_layers=mc['num_layers'],
            n_head=mc['n_head'],
            drop_out_rate=mc.get('drop_out_rate', 0.1),
            fc_rate=mc.get('fc_rate', 4),
            use_goal=mc.get('use_goal', False),
            terrain_patch_dim=a_tp,
            terrain_clear_dim=a_tc,
            terrain_emb_dim=mc.get('terrain_emb_dim', 32),
            hp_dim=a_hp_in,
            hp_out_dim=a_hp_out,
        ).to(device)
        if a_tp > 0:
            model.terrain_patch_dim_meta = a_tp; model.terrain_n_key = a_tc
    elif model_kind == 'pose':
        # Terrain conditioning (B): derive patch/clear dims + loss params from metadata.terr.npz
        # so model width and the precompute can never silently disagree.
        tp_dim = tc_dim = 0
        if config['data'].get('terrain', False):
            tm = np.load(Path(config['data']['motion_dir']) / 'metadata.terr.npz', allow_pickle=True)
            tp_dim = int(tm['patch_dim']); tc_dim = int(tm['n_key'])
        model = PoseGPT(
            m_vocab_size=m_vocab,
            n_vocab_size=n_vocab,
            action_emb_dim=mc.get('action_emb_dim', 64),
            embed_dim=mc['embed_dim'],
            block_size=mc['block_size'],
            num_layers=mc['num_layers'],
            n_head=mc['n_head'],
            drop_out_rate=mc.get('drop_out_rate', 0.1),
            fc_rate=mc.get('fc_rate', 4),
            root_input_dim=mc.get('root_input_dim', 0),
            root_output_dim=mc.get('root_output_dim', 0),
            terrain_patch_dim=tp_dim,
            terrain_clear_dim=tc_dim,
            terrain_emb_dim=mc.get('terrain_emb_dim', 64),
        ).to(device)
        if config['data'].get('terrain', False):
            # rel_pos index = monster-joint index - 1 (rel_pos excludes root joint 0); used by the loss
            key = np.asarray(tm['key_joints'], np.int64)
            model.terrain_key_rel = torch.as_tensor(key - 1, dtype=torch.long, device=device)
            # margin is config-overridable (tuned across iterations) — precompute stores raw H_local
            model.terrain_margin = float(mc.get('terrain_margin', float(tm['margin'])))
            model.terrain_patch_dim_meta = tp_dim
            model.terrain_n_key = tc_dim
            # The model predicts NORMALIZED body; the precomputed terrain (clear/H_local/R_row1) is in
            # METERS. Stash per-key-joint rel_pos mean/std (276 layout [3:162]) to denormalize in the loss.
            md = np.load(Path(config['data']['motion_dir']) / 'metadata.npz', allow_pickle=True)
            mean780 = md['mean'].astype(np.float32)
            if bool(config['data'].get('zero_center_root', False)):
                # must match dataset_lazy's zero_center_root normalization exactly
                mean780 = mean780.copy(); mean780[0:3] = 0.0; mean780[486:489] = 0.0
            mean276 = slice_to_pos276(mean780)
            std276 = slice_to_pos276(md['std'].astype(np.float32))
            mrel_mean = mean276[3:162].reshape(53, 3)[key - 1]    # (K,3)
            mrel_std = std276[3:162].reshape(53, 3)[key - 1]
            model.terrain_relpos_mean = torch.as_tensor(mrel_mean, dtype=torch.float32, device=device)
            model.terrain_relpos_std = torch.as_tensor(mrel_std, dtype=torch.float32, device=device)
            # M_root_delta (276 layout [0:3]) mean/std — for the v3 anti-sink Y-consistency loss
            model.terrain_rd_mean = torch.as_tensor(mean276[0:3], dtype=torch.float32, device=device)
            model.terrain_rd_std = torch.as_tensor(std276[0:3], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    return model


def train(config, model_kind, resume=None):
    seed = config['train'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[V16/{model_kind}] device={device}")

    train_ds, val_ds, m_vocab, n_vocab = load_data(config)
    print(f"[V16/{model_kind}] train_samples={len(train_ds)}, val_samples={len(val_ds)}, "
          f"m_vocab={m_vocab}, n_vocab={n_vocab}")

    tc = config['train']
    # persistent_workers: keep per-worker seg caches (feat mmap + aux) alive ACROSS
    # epochs — otherwise workers are recreated each epoch and pay the ~119ms/seg
    # networked-FS first-touch cost every epoch. prefetch_factor hides remaining I/O.
    train_loader = DataLoader(train_ds, batch_size=tc['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=tc['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    model = build_model(model_kind, config, m_vocab, n_vocab, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[V16/{model_kind}] params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc['lr']),
        betas=tuple(tc.get('betas', [0.9, 0.99])),
        weight_decay=float(tc.get('weight_decay', 1e-5)),
    )

    warmup_iter = int(tc['warmup_iter'])
    total_iter = int(tc['total_iter'])
    pf_start_iter = int(tc.get('pf_start_iter', warmup_iter))  # when pushforward kicks in

    def lr_lambda(it):
        if it < warmup_iter:
            return it / max(1, warmup_iter)
        progress = (it - warmup_iter) / max(1, total_iter - warmup_iter)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = None
    ema_decay = float(tc.get('ema_decay', 0))
    if ema_decay > 0:
        ema = EMA(model, decay=ema_decay,
                  update_every=int(tc.get('ema_update_every', 10)))
        print(f"[V16/{model_kind}] EMA enabled: decay={ema_decay}")

    start_iter = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if ema is not None and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        start_iter = ckpt.get('iter', 0)
        print(f"[V16/{model_kind}] resumed from {resume} at iter {start_iter}")

    output_dir = Path(tc['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / 'tb_logs')

    weights = dict(tc['loss_weights'])  # copy so we can inject _pf_k
    step_fn = action_step if model_kind == 'action' else pose_step

    # Pushforward K curriculum: list of (start_iter, K) pairs
    # e.g., pf_k_schedule: [[20000, 2], [40000, 3], [60000, 4]]
    pf_k_schedule = tc.get('pf_k_schedule', None)
    base_pf_k = int(tc.get('pf_k', 2))

    train_iter = iter(train_loader)
    model.train()

    for it in tqdm(range(start_iter + 1, total_iter + 1),
                   desc=f"V16/{model_kind}"):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = tuple(b.to(device, non_blocking=True) for b in batch)

        phase = 'pf' if it >= pf_start_iter else 'tf'

        # Determine current K from curriculum or base
        if pf_k_schedule and phase == 'pf':
            cur_k = base_pf_k
            for sched_iter, sched_k in pf_k_schedule:
                if it >= sched_iter:
                    cur_k = sched_k
            weights['_pf_k'] = cur_k
        else:
            weights['_pf_k'] = base_pf_k

        loss, loss_dict = step_fn(model, batch, phase, weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        if it % tc['log_iter'] == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/phase_pf', 1.0 if phase == 'pf' else 0.0, it)

        if it % tc['eval_iter'] == 0:
            model.eval()
            if ema is not None:
                ema.apply_shadow(model)
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = tuple(b.to(device, non_blocking=True) for b in vbatch)
                    # Validation always uses TF (clean comparison)
                    _, vd = step_fn(model, vbatch, 'tf', weights)
                    val_losses.append(vd)
            agg = {k: float(np.mean([d[k] for d in val_losses])) for k in val_losses[0]}
            for k, v in agg.items():
                writer.add_scalar(f'val/{k}', v, it)
            print(f"[V16/{model_kind}] iter {it} val: " + ", ".join(f"{k}={v:.6f}" for k, v in agg.items()))
            if ema is not None:
                ema.restore(model)
            model.train()

        if it % tc['save_iter'] == 0 or it == total_iter:
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iter': it,
                'config': config,
            }
            if ema is not None:
                ckpt['ema'] = ema.state_dict()
            torch.save(ckpt, output_dir / f'v16_{model_kind}_iter{it}.pt')

    writer.close()
    print(f"[V16/{model_kind}] training complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('--model', choices=['action', 'pose'], required=True)
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.model, resume=args.resume)
