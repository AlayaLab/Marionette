#!/usr/bin/env python3
"""Training script for Continuous Frame GPT: next-frame prediction on 276D features.

Supports model types:
  - standard: single-stage ContinuousMotionGPT
  - adaln: AdaLN-Zero conditioning
  - twostage: two-stage root/body decomposition (V10, Kimodo-inspired)
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.gpt_continuous import ContinuousMotionGPT


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMA:
    """Exponential Moving Average of model parameters for stable inference."""

    def __init__(self, model, decay=0.995, update_every=10):
        self.decay = decay
        self.update_every = update_every
        self.step_count = 0
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Swap model params with EMA params (for eval)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original params after eval."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'step_count': self.step_count}

    def load_state_dict(self, sd):
        self.shadow = sd['shadow']
        self.step_count = sd['step_count']


# ============================================================
# Two-stage loss
# ============================================================

def compute_twostage_loss(root_pred, body_pred, target_276, pred_mask, weights,
                          body_loss_type='smooth_l1', monster_body_weight=1.0):
    """Compute component-weighted loss for two-stage model.

    Args:
        root_pred: (B, T, 12) from Stage 1 [m_rd(3), n_rd(3), m_rot(6)]
        body_pred: (B, T, 258) from Stage 2 [m_rp(159), n_rp(93), weapon(6)]
        target_276: (B, T, 276) GT target
        pred_mask: (B, T) valid frame mask
        weights: dict with keys root_delta, root_rot, body_pos, weapon, velocity
        body_loss_type: 'smooth_l1' or 'mse' for body components
        monster_body_weight: extra weight on Monster body loss (default 1.0 = equal)

    Returns:
        total_loss, loss_dict
    """
    # Extract targets
    t_m_rd = target_276[..., 0:3]
    t_n_rd = target_276[..., 162:165]
    t_m_rot = target_276[..., 264:270]
    t_m_rp = target_276[..., 3:162]
    t_n_rp = target_276[..., 165:258]
    t_wpn = target_276[..., 258:264]

    # Root predictions
    p_m_rd = root_pred[..., 0:3]
    p_n_rd = root_pred[..., 3:6]
    p_m_rot = root_pred[..., 6:12]

    # Body predictions
    p_m_rp = body_pred[..., 0:159]
    p_n_rp = body_pred[..., 159:252]
    p_wpn = body_pred[..., 252:258]

    def masked_smooth_l1(pred, target):
        per_elem = F.smooth_l1_loss(pred, target, reduction='none')
        per_frame = per_elem.mean(dim=-1)  # (B, T)
        return (per_frame * pred_mask).sum() / pred_mask.sum().clamp(min=1)

    def masked_mse(pred, target):
        per_elem = F.mse_loss(pred, target, reduction='none')
        per_frame = per_elem.mean(dim=-1)  # (B, T)
        return (per_frame * pred_mask).sum() / pred_mask.sum().clamp(min=1)

    body_loss_fn = masked_mse if body_loss_type == 'mse' else masked_smooth_l1

    # Root: always Smooth L1
    l_m_rd = masked_smooth_l1(p_m_rd, t_m_rd)
    l_n_rd = masked_smooth_l1(p_n_rd, t_n_rd)
    l_m_rot = masked_smooth_l1(p_m_rot, t_m_rot)

    # Body: configurable loss type
    l_m_rp = body_loss_fn(p_m_rp, t_m_rp)
    l_n_rp = body_loss_fn(p_n_rp, t_n_rp)
    l_wpn = body_loss_fn(p_wpn, t_wpn)

    l_root_delta = l_m_rd + l_n_rd
    l_body_pos = monster_body_weight * l_m_rp + l_n_rp

    # Velocity loss on all predicted dims (270D, excluding NPC rot6d)
    full_pred = torch.cat([p_m_rd, p_m_rp, p_n_rd, p_n_rp, p_wpn, p_m_rot], dim=-1)
    full_tgt = torch.cat([t_m_rd, t_m_rp, t_n_rd, t_n_rp, t_wpn, t_m_rot], dim=-1)
    vel_pred = full_pred[:, 1:] - full_pred[:, :-1]
    vel_tgt = full_tgt[:, 1:] - full_tgt[:, :-1]
    vel_mask = pred_mask[:, 1:] * pred_mask[:, :-1]
    vel_per_frame = F.smooth_l1_loss(vel_pred, vel_tgt, reduction='none').mean(dim=-1)
    l_vel = (vel_per_frame * vel_mask).sum() / vel_mask.sum().clamp(min=1)

    w = weights
    total = (w['root_delta'] * l_root_delta +
             w['root_rot'] * l_m_rot +
             w['body_pos'] * l_body_pos +
             w['weapon'] * l_wpn +
             w['velocity'] * l_vel)

    loss_dict = {
        'root_delta': l_root_delta.item(),
        'root_rot': l_m_rot.item(),
        'body_pos': l_body_pos.item(),
        'weapon': l_wpn.item(),
        'velocity': l_vel.item(),
        'total': total.item(),
    }

    return total, loss_dict


# ============================================================
# Feature slicing: 780D -> 264D position-only
# ============================================================

def slice_to_pos276(data_780):
    """Slice 780D features to 276D position + root rotation.

    Input layout (780D):
      Monster: [0:3] root_delta_local, [3:162] rel_pos_local(53×3), [162:486] rot6d(54×6)
      NPC:     [486:489] root_delta_local, [489:582] rel_pos_local(31×3), [582:774] rot6d(32×6)
      Weapon:  [774:780] NPC-rel_pos_local(2×3)

    Output layout (276D):
      [0:162]   Monster root_delta(3) + rel_pos(53×3)   = 162D
      [162:258] NPC root_delta(3) + rel_pos(31×3)       = 96D
      [258:264] Weapon rel_pos(2×3)                     = 6D
      [264:270] Monster root rot6d                      = 6D  (from 780D [162:168])
      [270:276] NPC root rot6d                          = 6D  (from 780D [582:588])
    """
    monster_pos = data_780[..., :162]           # root_delta(3) + rel_pos(53×3)
    npc_pos = data_780[..., 486:582]            # root_delta(3) + rel_pos(31×3)
    weapon_pos = data_780[..., 774:780]         # rel_pos(2×3)
    monster_root_rot = data_780[..., 162:168]   # root joint rot6d = world rotation
    npc_root_rot = data_780[..., 582:588]       # root joint rot6d = world rotation
    return np.concatenate([monster_pos, npc_pos, weapon_pos,
                           monster_root_rot, npc_root_rot], axis=-1)


# ============================================================
# Dataset
# ============================================================

class ContinuousMotionDataset(Dataset):
    """Sliding window dataset over continuous motion segments."""

    def __init__(self, segments, window_size=256, stride=128,
                 action_m=None, action_n=None, progress=None,
                 npc_swap_prob=0.0, npc_swap_min_run=10):
        self.window_size = window_size
        self.samples = []
        self.has_action = action_m is not None
        self.has_progress = progress is not None

        for i, seg in enumerate(segments):
            L = seg.shape[0]
            if L <= window_size:
                self.samples.append((i, 0, L))
            else:
                for start in range(0, L - window_size + 1, stride):
                    self.samples.append((i, start, window_size))
                # Include tail if not already covered
                if (L - window_size) % stride != 0:
                    self.samples.append((i, L - window_size, window_size))

        self.segments = segments
        self.action_m = action_m  # list of (L,) int32 arrays, or None
        self.action_n = action_n
        self.progress = progress  # list of (L, 2) float32 arrays, or None

        # NPC action segment swapping augmentation
        self.npc_swap_prob = npc_swap_prob
        if npc_swap_prob > 0 and action_n is not None:
            self.npc_action_lib = self._build_npc_action_library(npc_swap_min_run)
            self.npc_action_ids = [k for k, v in self.npc_action_lib.items()
                                   if len(v) > 0]
            total_runs = sum(len(v) for v in self.npc_action_lib.values())
            print(f"NPC swap library: {len(self.npc_action_ids)} actions, "
                  f"{total_runs} runs (min_run={npc_swap_min_run})")
        else:
            self.npc_action_lib = {}
            self.npc_action_ids = []

    def _build_npc_action_library(self, min_run_length=10):
        """Build index: npc_action_id -> list of (seg_idx, start, end).

        Only includes runs that start at a real action transition (start > 0,
        preceded by a different action) to ensure frame=0 of the action.
        Runs must be >= min_run_length frames.
        """
        from collections import defaultdict
        library = defaultdict(list)
        for seg_idx, act_n in enumerate(self.action_n):
            if len(act_n) < 2:
                continue
            cur = act_n[0]
            run_start = 0
            for t in range(1, len(act_n)):
                if act_n[t] != cur:
                    # New run starts at t — this is frame=0 of the new action
                    new_action = int(act_n[t])
                    # Find end of new run
                    run_end = t + 1
                    while run_end < len(act_n) and act_n[run_end] == act_n[t]:
                        run_end += 1
                    run_len = run_end - t
                    if run_len >= min_run_length and new_action > 0:
                        library[new_action].append((seg_idx, t, run_end))
                    cur = act_n[t]
                    run_start = t
        return library

    def _apply_npc_swap(self, frames, act_m, act_n, prog, length):
        """Replace NPC features + action from a random library entry.

        The donor always starts at frame=0 of its action run (real transition).
        NPC features swapped: [162:264] (pos+weapon) + [270:276] (rot6d) = 108D.
        Monster features [0:162] + [264:270] are unchanged.
        """
        if not self.npc_action_ids:
            return frames, act_m, act_n, prog

        # Pick swap insertion point in window (leave at least 10 frames before)
        min_prefix = 10
        if length <= min_prefix:
            return frames, act_m, act_n, prog
        swap_start = random.randint(min_prefix, length - 1)
        remaining = length - swap_start

        # Pick random donor action and entry
        target_aid = random.choice(self.npc_action_ids)
        entries = self.npc_action_lib[target_aid]
        donor_seg_idx, donor_start, donor_end = random.choice(entries)

        # Start from frame=0 of the donor action run, but extend beyond
        # the single run into the donor segment's natural continuation
        donor_seg = self.segments[donor_seg_idx]
        donor_seg_len = donor_seg.shape[0]
        avail_from_donor = donor_seg_len - donor_start  # frames from run start to seg end
        copy_len = min(remaining, avail_from_donor)

        # Copy NPC features from donor
        src_s = donor_start
        src_e = donor_start + copy_len
        frames[swap_start:swap_start + copy_len, 162:264] = \
            donor_seg[src_s:src_e, 162:264]
        frames[swap_start:swap_start + copy_len, 270:276] = \
            donor_seg[src_s:src_e, 270:276]

        # Replace NPC action IDs
        donor_act_n = self.action_n[donor_seg_idx][src_s:src_e]
        act_n[swap_start:swap_start + copy_len] = donor_act_n

        # Replace NPC progress (column 1) if available
        if prog is not None and self.progress is not None:
            donor_prog = self.progress[donor_seg_idx][src_s:src_e]
            prog[swap_start:swap_start + copy_len, 1] = donor_prog[:, 1]

        return frames, act_m, act_n, prog

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg_idx, start, length = self.samples[idx]
        frames = self.segments[seg_idx][start:start + length].copy()

        # Load action / progress arrays (before potential swap)
        act_m = act_n = prog = None
        if self.has_action:
            act_m = self.action_m[seg_idx][start:start + length].copy()
            act_n = self.action_n[seg_idx][start:start + length].copy()
        if self.has_progress:
            prog = self.progress[seg_idx][start:start + length].copy()

        # NPC action segment swap augmentation
        if (self.npc_swap_prob > 0 and act_n is not None
                and random.random() < self.npc_swap_prob):
            frames, act_m, act_n, prog = self._apply_npc_swap(
                frames, act_m, act_n, prog, length)

        # Pad short segments to window_size
        if length < self.window_size:
            pad = np.zeros((self.window_size - length, frames.shape[-1]), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
            mask = np.zeros(self.window_size, dtype=np.float32)
            mask[:length] = 1.0
        else:
            mask = np.ones(self.window_size, dtype=np.float32)

        result = [torch.from_numpy(frames), torch.from_numpy(mask)]

        if self.has_action:
            if length < self.window_size:
                act_m = np.concatenate([act_m, np.zeros(self.window_size - length, dtype=np.int32)])
                act_n = np.concatenate([act_n, np.zeros(self.window_size - length, dtype=np.int32)])
            result.append(torch.from_numpy(act_m).long())
            result.append(torch.from_numpy(act_n).long())

        if self.has_progress:
            if length < self.window_size:
                prog = np.concatenate([prog, np.zeros((self.window_size - length, 2), dtype=np.float32)])
            result.append(torch.from_numpy(prog))

        return tuple(result)


def load_data(config):
    """Load motion data, slice to 276D, normalize, create datasets.

    If action IDs are present in the npz, loads them too.
    """
    motion_path = Path(config["data"]["motion_dir"]) / "motion_data.npz"
    data = np.load(motion_path, allow_pickle=True)

    mean_780 = data["mean"]
    std_780 = data["std"]

    num_seg = int(data["num_segments"])
    segments_780 = [data[f"segment_{i}"] for i in range(num_seg)]

    # Slice norm stats to 276D (same indices)
    mean_276 = slice_to_pos276(mean_780)
    std_276 = slice_to_pos276(std_780)

    # Slice and keep normalized (data is already normalized in npz)
    segments_276 = []
    for seg in segments_780:
        seg_276 = slice_to_pos276(seg)
        segments_276.append(seg_276.astype(np.float32))

    # Load action IDs if available
    has_action = f"action_m_0" in data
    action_m_all = None
    action_n_all = None
    progress_all = None
    action_info = {}
    if has_action:
        action_m_all = [data[f"action_m_{i}"] for i in range(num_seg)]
        action_n_all = [data[f"action_n_{i}"] for i in range(num_seg)]
        action_info = {
            'm_vocab_size': int(data['m_action_vocab_size']),
            'n_vocab_size': int(data['n_action_vocab_size']),
            'vocab_monster': data['action_vocab_monster'],
            'vocab_npc': data['action_vocab_npc'],
        }
        # Load progress if available
        if f"action_progress_0" in data:
            progress_all = [data[f"action_progress_{i}"] for i in range(num_seg)]
            print(f"Loaded action IDs + progress: Monster vocab={action_info['m_vocab_size']}, "
                  f"NPC vocab={action_info['n_vocab_size']}")
        else:
            print(f"Loaded action IDs: Monster vocab={action_info['m_vocab_size']}, "
                  f"NPC vocab={action_info['n_vocab_size']} (no progress)")

    # Train/val split (shuffle indices to keep action arrays aligned)
    indices = list(range(len(segments_276)))
    random.shuffle(indices)
    val_ratio = config["data"]["val_ratio"]
    n_val = max(1, int(len(segments_276) * val_ratio))

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    val_segments = [segments_276[i] for i in val_idx]
    train_segments = [segments_276[i] for i in train_idx]

    train_act_m = [action_m_all[i] for i in train_idx] if has_action else None
    train_act_n = [action_n_all[i] for i in train_idx] if has_action else None
    val_act_m = [action_m_all[i] for i in val_idx] if has_action else None
    val_act_n = [action_n_all[i] for i in val_idx] if has_action else None
    train_progress = [progress_all[i] for i in train_idx] if progress_all else None
    val_progress = [progress_all[i] for i in val_idx] if progress_all else None

    window_size = config["data"]["window_size"]
    stride = config["data"]["window_stride"]

    npc_swap_prob = float(config["train"].get("npc_swap_prob", 0.0))
    npc_swap_min_run = int(config["train"].get("npc_swap_min_run", 10))

    train_dataset = ContinuousMotionDataset(
        train_segments, window_size, stride,
        action_m=train_act_m, action_n=train_act_n, progress=train_progress,
        npc_swap_prob=npc_swap_prob, npc_swap_min_run=npc_swap_min_run)
    val_dataset = ContinuousMotionDataset(
        val_segments, window_size, stride,
        action_m=val_act_m, action_n=val_act_n, progress=val_progress)

    return train_dataset, val_dataset, mean_276, std_276, action_info


# ============================================================
# Training
# ============================================================

def train(config, resume_ckpt=None):
    seed = config["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_dataset, val_dataset, mean_276, std_276, action_info = load_data(config)
    has_action = bool(action_info)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    mc = config["model"]
    model_type = mc.get("type", "standard")

    if model_type == "adaln":
        from models.gpt_continuous_adaln import ContinuousMotionGPT_AdaLN
        model = ContinuousMotionGPT_AdaLN(
            feat_dim=mc["feat_dim"],
            embed_dim=mc["embed_dim"],
            block_size=mc["block_size"],
            num_layers=mc["num_layers"],
            n_head=mc["n_head"],
            drop_out_rate=mc["drop_out_rate"],
            fc_rate=mc["fc_rate"],
            cond_dim=mc.get("cond_dim", 6),
            cond_hidden=mc.get("cond_hidden", 256),
        ).to(device)
    elif model_type == "twostage":
        from models.gpt_continuous_twostage import TwoStageMotionGPT
        root_cfg = dict(mc["root"])
        root_cfg["block_size"] = mc["block_size"]
        body_cfg = dict(mc["body"])
        body_cfg["block_size"] = mc["block_size"]
        if "rope_base" in mc:
            root_cfg["rope_base"] = mc["rope_base"]
            body_cfg["rope_base"] = mc["rope_base"]
        # Action config from data vocab + model config
        action_cfg = None
        if has_action and mc.get("action"):
            action_cfg = dict(mc["action"])
            action_cfg["m_vocab_size"] = action_info["m_vocab_size"]
            action_cfg["n_vocab_size"] = action_info["n_vocab_size"]
        model = TwoStageMotionGPT(root_cfg, body_cfg, action_cfg=action_cfg).to(device)
    else:
        model = ContinuousMotionGPT(
            feat_dim=mc["feat_dim"],
            embed_dim=mc["embed_dim"],
            block_size=mc["block_size"],
            num_layers=mc["num_layers"],
            n_head=mc["n_head"],
            drop_out_rate=mc["drop_out_rate"],
            fc_rate=mc["fc_rate"],
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    if model_type == "twostage":
        root_params = sum(p.numel() for p in model.root_gpt.parameters())
        body_params = sum(p.numel() for p in model.body_gpt.parameters())
        print(f"  RootGPT: {root_params:,}, BodyGPT: {body_params:,}")

    # Optimizer
    tc = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["lr"]),
        betas=tuple(tc["betas"]),
        weight_decay=float(tc.get("weight_decay", 1e-5)),
    )

    # Cosine annealing with warmup
    warmup_iter = tc["warmup_iter"]
    total_iter = tc["total_iter"]

    def lr_lambda(it):
        if it < warmup_iter:
            return it / warmup_iter
        progress = (it - warmup_iter) / (total_iter - warmup_iter)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema = None
    ema_decay = float(tc.get("ema_decay", 0))
    if ema_decay > 0:
        ema = EMA(model, decay=ema_decay,
                  update_every=int(tc.get("ema_update_every", 10)))
        print(f"EMA enabled: decay={ema_decay}, update_every={tc.get('ema_update_every', 10)}")

    # Resume or load pretrained
    start_iter = 0
    pretrained_path = tc.get("pretrained", None)
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_iter = ckpt.get("iter", 0)
        print(f"Resumed from {resume_ckpt} at iter {start_iter}")
    elif pretrained_path:
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        # Load matching weights only (allows architecture changes)
        pretrained_sd = ckpt["model"]
        model_sd = model.state_dict()
        loaded = []
        for k, v in pretrained_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded.append(k)
        model.load_state_dict(model_sd)
        print(f"Loaded {len(loaded)}/{len(model_sd)} pretrained weights from {pretrained_path}")
        # Also init EMA from pretrained
        if ema is not None and "ema" in ckpt:
            for name in loaded:
                if name in ckpt["ema"]["shadow"]:
                    ema.shadow[name] = ckpt["ema"]["shadow"][name].to(device)
            print(f"  Initialized EMA from pretrained")

    # Output
    output_dir = Path(tc["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "tb_logs")

    vel_loss_weight = float(tc.get("vel_loss_weight", 0.0))
    rot_loss_weight = float(tc.get("rot_loss_weight", 0.0))
    rot_dim_start = int(tc.get("rot_dim_start", 264))  # where rot6d dims begin
    loss_dim_end = int(tc.get("loss_dim_end", 0))  # if >0, only compute loss on [:loss_dim_end]

    # Scheduled sampling: gradually replace GT input with model's own predictions
    ss_start = int(tc.get("ss_start_iter", 0))       # iter to start scheduled sampling
    ss_end = int(tc.get("ss_end_iter", 0))            # iter to reach max ss_prob
    ss_max_prob = float(tc.get("ss_max_prob", 0.0))   # max probability of using model prediction

    # Training loop
    train_iter = iter(train_loader)
    model.train()

    action_loss_weight = float(tc.get("action_loss_weight", 1.0))

    # Transition-aware action scheduled sampling
    action_ss_start = int(tc.get("action_ss_start_iter", 0))
    action_ss_end = int(tc.get("action_ss_end_iter", 0))
    action_ss_max = float(tc.get("action_ss_max_prob", 0.0))

    def apply_transition_ss(action_seq, miss_prob):
        """Apply transition-aware scheduled sampling.

        At each action transition in the GT sequence, with probability miss_prob,
        skip the transition (extend previous action). Simulates autoregressive
        prediction failing to detect transitions.

        Args:
            action_seq: (B, T) int tensor, GT action IDs
            miss_prob: float in [0, 1]
        Returns:
            (B, T) modified action sequence (transitions randomly removed)
        """
        if miss_prob <= 0:
            return action_seq
        B, T = action_seq.shape
        modified = action_seq.clone()
        orig = action_seq.cpu().numpy()
        mod = modified.cpu().numpy()
        for b in range(B):
            cur = orig[b, 0]
            for t in range(1, T):
                if orig[b, t] != orig[b, t - 1]:
                    # Transition point — miss with probability miss_prob
                    if random.random() < miss_prob:
                        mod[b, t] = cur  # extend previous action
                    else:
                        cur = orig[b, t]  # keep transition
                        mod[b, t] = cur
                else:
                    mod[b, t] = cur
        return torch.from_numpy(mod).to(action_seq.device)

    for it in tqdm(range(start_iter + 1, total_iter + 1), desc="Training Continuous GPT"):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        frames, mask = batch[0].to(device), batch[1].to(device)  # (B, T, D), (B, T)
        act_m = batch[2].to(device) if len(batch) > 2 else None  # (B, T)
        act_n = batch[3].to(device) if len(batch) > 3 else None
        prog = batch[4].to(device) if len(batch) > 4 else None  # (B, T, 2)

        # Compute scheduled sampling probability for this iteration
        if ss_max_prob > 0 and ss_end > ss_start and it >= ss_start:
            ss_prob = ss_max_prob * min(1.0, (it - ss_start) / (ss_end - ss_start))
        else:
            ss_prob = 0.0

        # Two-stage loss path
        if model_type == "twostage":
            detach_root = bool(tc.get("detach_root", True))
            loss_weights = tc["loss_weights"]

            # Action: input uses +1 shift, target uses +2 shift
            if act_m is not None:
                # Transition-aware scheduled sampling (before shift)
                if action_ss_max > 0 and action_ss_end > action_ss_start and it >= action_ss_start:
                    action_ss_prob = action_ss_max * min(1.0, (it - action_ss_start) / (action_ss_end - action_ss_start))
                else:
                    action_ss_prob = 0.0
                if action_ss_prob > 0:
                    act_m_ss = apply_transition_ss(act_m, action_ss_prob)
                    act_n_ss = apply_transition_ss(act_n, action_ss_prob)
                else:
                    act_m_ss = act_m
                    act_n_ss = act_n
                # Input action: shifted +1 from pose input (frames[:, :-1])
                action_inp_m = act_m_ss[:, 1:]   # (B, T-1)
                action_inp_n = act_n_ss[:, 1:]
            else:
                action_inp_m = None
                action_inp_n = None
                action_ss_prob = 0.0
            progress_inp = prog[:, 1:] if prog is not None else None

            action_drop = float(tc.get("action_drop_prob", 0.0))
            root_pred, body_pred, full_pred, action_logits = model(
                frames[:, :-1], detach_root=detach_root,
                action_m=action_inp_m, action_n=action_inp_n,
                progress=progress_inp,
                action_drop_prob=action_drop)
            target = frames[:, 1:]
            pred_mask = mask[:, 1:]
            body_loss_type = tc.get("body_loss_type", "smooth_l1")
            monster_body_weight = tc.get("monster_body_weight", 1.0)
            loss, loss_dict = compute_twostage_loss(
                root_pred, body_pred, target, pred_mask, loss_weights,
                body_loss_type=body_loss_type,
                monster_body_weight=monster_body_weight)

            # Action CE loss: target is +2 shift from pose input
            if action_logits is not None:
                m_logits, n_logits = action_logits  # (B, T-1, vocab)
                # Target: act[2:] padded, last frame masked
                act_tgt_m = F.pad(act_m[:, 2:], (0, 1), value=0)  # (B, T-1)
                act_tgt_n = F.pad(act_n[:, 2:], (0, 1), value=0)
                # Mask: same as pred_mask but last position zeroed
                action_mask = pred_mask.clone()
                action_mask[:, -1] = 0

                m_ce = F.cross_entropy(
                    m_logits.reshape(-1, m_logits.size(-1)),
                    act_tgt_m.reshape(-1),
                    ignore_index=0, reduction='none')
                n_ce = F.cross_entropy(
                    n_logits.reshape(-1, n_logits.size(-1)),
                    act_tgt_n.reshape(-1),
                    ignore_index=0, reduction='none')
                m_ce = (m_ce.view_as(action_mask) * action_mask).sum() / action_mask.sum().clamp(min=1)
                n_ce = (n_ce.view_as(action_mask) * action_mask).sum() / action_mask.sum().clamp(min=1)
                action_loss = m_ce + n_ce
                loss = loss + action_loss_weight * action_loss
                loss_dict['action_ce'] = action_loss.item()
                loss_dict['action_m_ce'] = m_ce.item()
                loss_dict['action_n_ce'] = n_ce.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            if it % tc["log_iter"] == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v, it)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)
                if action_ss_prob > 0:
                    writer.add_scalar("train/action_ss_prob", action_ss_prob, it)

        else:
            # Standard / AdaLN loss path (unchanged)
            # Build input sequence with optional scheduled sampling
            if ss_prob > 0:
                B, T, D = frames.shape
                inp = frames[:, :-1].clone()
                with torch.no_grad():
                    pred_all = model(inp)
                ss_mask = torch.rand(B, T - 2, device=device) < ss_prob
                for t in range(1, T - 1):
                    replace = ss_mask[:, t - 1].unsqueeze(-1)
                    inp[:, t] = torch.where(replace, pred_all[:, t - 1], inp[:, t])
                pred = model(inp)
            else:
                pred = model(frames[:, :-1])

            target = frames[:, 1:]
            pred_mask = mask[:, 1:]

            ld_end = loss_dim_end if loss_dim_end > 0 else pred.shape[-1]
            pos_end = min(rot_dim_start, ld_end)
            loss_type = tc.get("loss_type", "mse")

            def _elem_loss(p, t):
                if loss_type == "smooth_l1":
                    return F.smooth_l1_loss(p, t, reduction='none')
                return (p - t) ** 2

            pos_mse = _elem_loss(pred[..., :pos_end], target[..., :pos_end]).mean(dim=-1)
            pos_mse = (pos_mse * pred_mask).sum() / pred_mask.sum().clamp(min=1)

            if rot_loss_weight > 0 and ld_end > rot_dim_start:
                rot_mse = _elem_loss(pred[..., rot_dim_start:ld_end], target[..., rot_dim_start:ld_end]).mean(dim=-1)
                rot_mse = (rot_mse * pred_mask).sum() / pred_mask.sum().clamp(min=1)
            else:
                rot_mse = torch.tensor(0.0, device=device)

            mse = pos_mse + rot_loss_weight * rot_mse

            if vel_loss_weight > 0:
                pred_vel = pred[:, 1:] - pred[:, :-1]
                target_vel = target[:, 1:] - target[:, :-1]
                vel_mask = pred_mask[:, 1:] * pred_mask[:, :-1]
                vel_mse = _elem_loss(pred_vel, target_vel).mean(dim=-1)
                vel_mse = (vel_mse * vel_mask).sum() / vel_mask.sum().clamp(min=1)
                loss = mse + vel_loss_weight * vel_mse
            else:
                vel_mse = torch.tensor(0.0, device=device)
                loss = mse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            if it % tc["log_iter"] == 0:
                writer.add_scalar("train/loss", loss.item(), it)
                writer.add_scalar("train/mse", mse.item(), it)
                writer.add_scalar("train/pos_mse", pos_mse.item(), it)
                writer.add_scalar("train/rot_mse", rot_mse.item(), it)
                writer.add_scalar("train/vel_mse", vel_mse.item(), it)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)
                if ss_max_prob > 0:
                    writer.add_scalar("train/ss_prob", ss_prob, it)

        # Evaluation
        if it % tc["eval_iter"] == 0:
            model.eval()
            if ema is not None:
                ema.apply_shadow(model)

            if model_type == "twostage":
                val_loss_list = []
                loss_weights = tc["loss_weights"]
                with torch.no_grad():
                    for vbatch in val_loader:
                        vframes = vbatch[0].to(device)
                        vmask = vbatch[1].to(device)
                        v_act_m = vbatch[2].to(device) if len(vbatch) > 2 else None
                        v_act_n = vbatch[3].to(device) if len(vbatch) > 3 else None
                        v_prog = vbatch[4].to(device) if len(vbatch) > 4 else None

                        v_action_inp_m = v_act_m[:, 1:] if v_act_m is not None else None
                        v_action_inp_n = v_act_n[:, 1:] if v_act_n is not None else None
                        v_progress_inp = v_prog[:, 1:] if v_prog is not None else None

                        vroot, vbody, vfull, v_action_logits = model(
                            vframes[:, :-1],
                            action_m=v_action_inp_m, action_n=v_action_inp_n,
                            progress=v_progress_inp)
                        vtarget = vframes[:, 1:]
                        vpred_mask = vmask[:, 1:]
                        vloss, vloss_dict = compute_twostage_loss(
                            vroot, vbody, vtarget, vpred_mask, loss_weights,
                            body_loss_type=tc.get("body_loss_type", "smooth_l1"),
                            monster_body_weight=tc.get("monster_body_weight", 1.0))

                        if v_action_logits is not None:
                            vm_logits, vn_logits = v_action_logits
                            v_act_tgt_m = F.pad(v_act_m[:, 2:], (0, 1), value=0)
                            v_act_tgt_n = F.pad(v_act_n[:, 2:], (0, 1), value=0)
                            v_action_mask = vpred_mask.clone()
                            v_action_mask[:, -1] = 0
                            vm_ce = F.cross_entropy(
                                vm_logits.reshape(-1, vm_logits.size(-1)),
                                v_act_tgt_m.reshape(-1),
                                ignore_index=0, reduction='none')
                            vn_ce = F.cross_entropy(
                                vn_logits.reshape(-1, vn_logits.size(-1)),
                                v_act_tgt_n.reshape(-1),
                                ignore_index=0, reduction='none')
                            vm_ce = (vm_ce.view_as(v_action_mask) * v_action_mask).sum() / v_action_mask.sum().clamp(min=1)
                            vn_ce = (vn_ce.view_as(v_action_mask) * v_action_mask).sum() / v_action_mask.sum().clamp(min=1)
                            vloss_dict['action_ce'] = (vm_ce + vn_ce).item()

                        val_loss_list.append(vloss_dict)

                avg_dict = {}
                for k in val_loss_list[0]:
                    avg_dict[k] = np.mean([d[k] for d in val_loss_list])
                for k, v in avg_dict.items():
                    writer.add_scalar(f"val/{k}", v, it)
                msg = f"[Iter {it}] val_total={avg_dict['total']:.6f}"
                msg += f", root_d={avg_dict['root_delta']:.6f}"
                msg += f", body={avg_dict['body_pos']:.6f}"
                msg += f", vel={avg_dict['velocity']:.6f}"
                if 'action_ce' in avg_dict:
                    msg += f", action_ce={avg_dict['action_ce']:.4f}"
                tqdm.write(msg)
            else:
                val_mse_list, val_vel_list, val_rot_list = [], [], []
                val_loss_type = tc.get("loss_type", "mse")
                def _val_elem_loss(p, t):
                    if val_loss_type == "smooth_l1":
                        return F.smooth_l1_loss(p, t, reduction='none')
                    return (p - t) ** 2
                with torch.no_grad():
                    for vframes, vmask in val_loader:
                        vframes, vmask = vframes.to(device), vmask.to(device)
                        vpred = model(vframes[:, :-1])
                        vtarget = vframes[:, 1:]
                        vpred_mask = vmask[:, 1:]

                        vmse = _val_elem_loss(vpred, vtarget).mean(dim=-1)
                        vmse = (vmse * vpred_mask).sum() / vpred_mask.sum().clamp(min=1)
                        val_mse_list.append(vmse.item())

                        if vpred.shape[-1] > rot_dim_start:
                            vrot = _val_elem_loss(vpred[..., rot_dim_start:], vtarget[..., rot_dim_start:]).mean(dim=-1)
                            vrot = (vrot * vpred_mask).sum() / vpred_mask.sum().clamp(min=1)
                            val_rot_list.append(vrot.item())

                        if vel_loss_weight > 0:
                            vpv = vpred[:, 1:] - vpred[:, :-1]
                            vtv = vtarget[:, 1:] - vtarget[:, :-1]
                            vvm = vpred_mask[:, 1:] * vpred_mask[:, :-1]
                            vvel = _val_elem_loss(vpv, vtv).mean(dim=-1)
                            vvel = (vvel * vvm).sum() / vvm.sum().clamp(min=1)
                            val_vel_list.append(vvel.item())

                avg_mse = np.mean(val_mse_list)
                writer.add_scalar("val/mse", avg_mse, it)
                msg = f"[Iter {it}] val_mse={avg_mse:.6f}"
                if val_rot_list:
                    avg_rot = np.mean(val_rot_list)
                    writer.add_scalar("val/rot_mse", avg_rot, it)
                    msg += f", val_rot={avg_rot:.6f}"
                if val_vel_list:
                    avg_vel = np.mean(val_vel_list)
                    writer.add_scalar("val/vel_mse", avg_vel, it)
                    msg += f", val_vel_mse={avg_vel:.6f}"
                tqdm.write(msg)

            if ema is not None:
                ema.restore(model)
            model.train()

        # Save
        if it % tc["save_iter"] == 0:
            ckpt_path = output_dir / f"gpt_cont_iter{it}.pt"
            save_dict = {
                "iter": it,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "mean": mean_276,
                "std": std_276,
            }
            if ema is not None:
                save_dict["ema"] = ema.state_dict()
            if action_info:
                save_dict["action_info"] = action_info
            torch.save(save_dict, ckpt_path)
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

    # Final
    final_dict = {
        "iter": total_iter,
        "model": model.state_dict(),
        "config": config,
        "mean": mean_276,
        "std": std_276,
    }
    if ema is not None:
        final_dict["ema"] = ema.state_dict()
    if action_info:
        final_dict["action_info"] = action_info
    torch.save(final_dict, output_dir / "gpt_cont_final.pt")
    writer.close()
    print("Training complete!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "configs" / "train_gpt_continuous.yaml"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total_iter", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--motion_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.total_iter:
        config["train"]["total_iter"] = args.total_iter
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.motion_dir:
        config["data"]["motion_dir"] = args.motion_dir
    train(config, resume_ckpt=args.resume)


if __name__ == "__main__":
    main()
