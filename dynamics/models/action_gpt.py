"""ActionGPT (V16): Tactical decision model.

Predicts next-frame root motion (root_delta + rot6d for both Monster and NPC)
plus next-frame action IDs (Monster + NPC) from history of:
  - root motion (M+N root_delta and rot6d) -- 18D
  - action embeddings (M + N) -- 128D (configurable)
  - animation progress (M + N) -- 2D
  - weapon position -- 6D
  - mutual distance (optional, default off in V16.0; needs preprocessing) -- 0 or 1D

Total input per frame: 163D (V16.0 baseline, no distance) or 164D (with distance)
Output per frame:
  - root regression (18D): M_root_delta(3) + N_root_delta(3) + M_rot6d(6) + N_rot6d(6)
  - M action logits (m_vocab_size)
  - N action logits (n_vocab_size)

Lightweight by design: ~0.5M params at 2L/128D/4H baseline.

NOTE on distance: requires per-frame absolute M-N world distance, which needs
preprocessing from BIN files (motion_data.npz lacks absolute positions). For V16.0,
distance_dim=0. Add as V16.x ablation by setting distance_dim=1 and providing a
distance_data.npz alongside motion_data.npz.
"""

import torch
import torch.nn as nn

from models.gpt import (
    SinusoidalPositionEncoding,
    TransformerBlock,
)


# Indices within the 18D root tensor
ROOT_LAYOUT = {
    'm_root_delta': (0, 3),
    'n_root_delta': (3, 6),
    'm_rot6d': (6, 12),
    'n_rot6d': (12, 18),
}
ROOT_DIM = 18


class ActionGPT(nn.Module):
    """Tactician model: predict next root motion + next action IDs."""

    def __init__(
        self,
        m_vocab_size,
        n_vocab_size,
        action_emb_dim=64,
        progress_dim=2,
        weapon_dim=6,
        distance_dim=0,  # V16.0 baseline: no distance (would need preprocessing)
        root_out_dim=18,  # 18=V16.0/V16.1, 12=V16.2 (rot6d only), 0=V16.3 (pure discrete)
        embed_dim=128,
        block_size=512,
        num_layers=2,
        n_head=4,
        drop_out_rate=0.1,
        fc_rate=4,
        use_goal=False,   # cloze mode: drop progress INPUT, feed sparse future-action
                          # goal embeddings instead, and predict progress via a head.
        terrain_patch_dim=0,   # planner terrain (B v2): egocentric heightmap patch (reuses the
        terrain_clear_dim=0,   # same precomputed .terr.npz as PoseGPT — no separate coarse rebuild).
        terrain_emb_dim=32,
        hp_dim=0,        # HP conditioning INPUT: 2 = [monster_hp%, npc_hp%] in [0,1], concat raw.
        hp_out_dim=0,    # HP prediction HEAD: 2 = predict next-frame [monster_hp%, npc_hp%].
    ):
        super().__init__()
        self.m_vocab_size = m_vocab_size
        self.n_vocab_size = n_vocab_size
        self.action_emb_dim = action_emb_dim
        self.progress_dim = progress_dim
        self.weapon_dim = weapon_dim
        self.distance_dim = distance_dim
        self.block_size = block_size
        self.use_goal = use_goal
        self.terrain_patch_dim = terrain_patch_dim
        self.terrain_clear_dim = terrain_clear_dim
        self.terrain_emb_dim = terrain_emb_dim if terrain_patch_dim > 0 else 0
        self.hp_dim = hp_dim
        self.hp_out_dim = hp_out_dim

        # Action embeddings (shared by current action + goal/anchor action in goal mode)
        self.m_action_emb = nn.Embedding(m_vocab_size, action_emb_dim)
        self.n_action_emb = nn.Embedding(n_vocab_size, action_emb_dim)

        # Planner terrain encoder: the patch lets ActionGPT steer the root/trajectory away from
        # obstacles BEFORE PoseGPT renders the pose (design doc #3 — avoidance is a planning behavior).
        if terrain_patch_dim > 0:
            self.terrain_enc = nn.Sequential(
                nn.Linear(terrain_patch_dim, terrain_emb_dim), nn.SiLU(),
                nn.Linear(terrain_emb_dim, terrain_emb_dim), nn.LayerNorm(terrain_emb_dim),
            )

        # Total per-frame input dim. Goal mode replaces the 2D progress input with
        # m_goal + n_goal action embeddings (the upcoming-transition action).
        ctrl_dim = (2 * action_emb_dim) if use_goal else progress_dim
        feat_dim = (
            ROOT_DIM
            + 2 * action_emb_dim
            + ctrl_dim
            + weapon_dim
            + distance_dim
            + self.terrain_emb_dim
            + terrain_clear_dim
            + hp_dim  # HP input concatenated raw (no encoder)
        )
        self.feat_dim = feat_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pos_enc = SinusoidalPositionEncoding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output heads
        # root_out_dim: 18=V16.0/V16.1 (full root), 12=V16.2 (rot6d only), 0=V16.3 (pure discrete)
        self.root_out_dim = root_out_dim
        if root_out_dim > 0:
            self.root_head = nn.Linear(embed_dim, root_out_dim)
        self.m_action_head = nn.Linear(embed_dim, m_vocab_size)
        self.n_action_head = nn.Linear(embed_dim, n_vocab_size)
        # Goal mode: progress is no longer an input leak — predict it instead.
        if use_goal:
            self.progress_head = nn.Linear(embed_dim, progress_dim)
        # HP prediction head: next-frame [monster_hp%, npc_hp%] (world-state forward model).
        if hp_out_dim > 0:
            self.hp_head = nn.Linear(embed_dim, hp_out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def assemble_features(self, root, action_m, action_n, progress, weapon, distance=None,
                          m_goal=None, n_goal=None, terrain_patch=None, terrain_clear=None,
                          hp=None):
        """Concatenate input features for one or many frames.

        Default mode: parts = [root, m_emb, n_emb, progress(2), weapon].
        Goal mode (use_goal): parts = [root, m_emb, n_emb, m_goal_emb, n_goal_emb, weapon]
            — progress is dropped from the input (predicted via a head instead) and
            the upcoming-transition action (goal) is fed as conditioning.
        Terrain (optional): + encoded heightmap patch + per-joint clearances appended.
        HP (optional): + [monster_hp%, npc_hp%] (2D, in [0,1]) appended raw.
        """
        m_emb = self.m_action_emb(action_m)
        n_emb = self.n_action_emb(action_n)
        if self.use_goal:
            assert m_goal is not None and n_goal is not None, "goal mode needs m_goal/n_goal"
            parts = [root, m_emb, n_emb, self.m_action_emb(m_goal), self.n_action_emb(n_goal), weapon]
        else:
            parts = [root, m_emb, n_emb, progress, weapon]
        if self.distance_dim > 0:
            assert distance is not None, "distance required when distance_dim>0"
            parts.append(distance)
        if self.terrain_patch_dim > 0:
            assert terrain_patch is not None, "terrain_patch required when terrain_patch_dim>0"
            parts.append(self.terrain_enc(terrain_patch))
        if self.terrain_clear_dim > 0:
            assert terrain_clear is not None, "terrain_clear required when terrain_clear_dim>0"
            parts.append(terrain_clear)
        if self.hp_dim > 0:
            assert hp is not None, "hp required when hp_dim>0"
            parts.append(hp)
        return torch.cat(parts, dim=-1)

    def forward(self, root, action_m, action_n, progress, weapon, distance=None,
                m_goal=None, n_goal=None, terrain_patch=None, terrain_clear=None,
                hp=None):
        """Single forward pass.

        Returns (root_pred, m_logits, n_logits, progress_pred, hp_pred, hidden).
          progress_pred: (B, T, 2) in goal mode (predicted), else None.
          hp_pred:       (B, T, hp_out_dim) if hp_out_dim>0, else None.
        """
        B, T = action_m.shape
        assert T <= self.block_size, f"T={T} exceeds block_size={self.block_size}"

        x = self.assemble_features(root, action_m, action_n, progress, weapon, distance,
                                   m_goal=m_goal, n_goal=n_goal,
                                   terrain_patch=terrain_patch, terrain_clear=terrain_clear,
                                   hp=hp)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.drop(h)
        for block in self.blocks:
            h, _ = block(h)
        h = self.ln_f(h)

        root_pred = self.root_head(h) if self.root_out_dim > 0 else None
        m_logits = self.m_action_head(h)
        n_logits = self.n_action_head(h)
        progress_pred = self.progress_head(h) if self.use_goal else None
        hp_pred = self.hp_head(h) if self.hp_out_dim > 0 else None
        return root_pred, m_logits, n_logits, progress_pred, hp_pred, h


def compute_distance(root):
    """Compute mutual M-N distance from root tensor (denormalized assumed for meaning,
    but works on normalized values too — model just learns the mapping).

    Args:
        root: (B, T, 18) with M_root_delta(0:3), N_root_delta(3:6) at start.
              We use root_delta as a proxy — note this is per-frame delta, not absolute pos.
              For meaningful distance the caller should pass accumulated root positions.
    Returns:
        (B, T, 1)
    """
    m_pos = root[..., 0:3]
    n_pos = root[..., 3:6]
    dist = (m_pos - n_pos).norm(dim=-1, keepdim=True)
    return dist
