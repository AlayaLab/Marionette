"""PoseGPT (V16): Animation player.

Conditioned on action ID, predicts next-frame body pose (and optionally root motion).

Modes (controlled by root_input_dim / root_output_dim):
  V16.0/V16.1: root_input_dim=0, root_output_dim=0 — body only (258D in/out)
  V16.2 (Plan B): root_input_dim=6, root_output_dim=6 — body + root_delta (264D out)
  V16.3 (Plan C): root_input_dim=18, root_output_dim=18 — body + full root (276D out)

Default size: 8 layers, 512 dim, 8 heads (~25M params).
"""

import torch
import torch.nn as nn

from models.gpt import (
    SinusoidalPositionEncoding,
    TransformerBlock,
)


# Indices within the 258D body tensor
BODY_LAYOUT = {
    'm_rel_pos': (0, 159),     # Monster rel_pos (53 joints * 3)
    'n_rel_pos': (159, 252),   # NPC rel_pos (31 joints * 3)
    'weapon':    (252, 258),   # weapon rel_pos (2 joints * 3)
}
BODY_DIM = 258


class PoseGPT(nn.Module):
    """Animation player: predict next body pose given current body history + action."""

    def __init__(
        self,
        m_vocab_size,
        n_vocab_size,
        action_emb_dim=64,
        embed_dim=512,
        block_size=512,
        num_layers=8,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
        root_input_dim=0,
        root_output_dim=0,
        terrain_patch_dim=0,
        terrain_clear_dim=0,
        terrain_emb_dim=64,
    ):
        super().__init__()
        self.m_vocab_size = m_vocab_size
        self.n_vocab_size = n_vocab_size
        self.action_emb_dim = action_emb_dim
        self.block_size = block_size
        self.root_input_dim = root_input_dim
        self.root_output_dim = root_output_dim
        # Terrain conditioning (B / terrain-aware). 0 => disabled (backward-compatible).
        self.terrain_patch_dim = terrain_patch_dim
        self.terrain_clear_dim = terrain_clear_dim
        self.terrain_emb_dim = terrain_emb_dim if terrain_patch_dim > 0 else 0

        # Action embeddings (separate from ActionGPT's — they don't share)
        self.m_action_emb = nn.Embedding(m_vocab_size, action_emb_dim)
        self.n_action_emb = nn.Embedding(n_vocab_size, action_emb_dim)

        # Egocentric heightmap-patch encoder: flatten NxN patch -> 2-layer MLP -> compact embedding
        # (concat fusion, the lightweight/physics-RL convention: PFNN/legged_gym/PARC/SCENIC).
        if terrain_patch_dim > 0:
            self.terrain_enc = nn.Sequential(
                nn.Linear(terrain_patch_dim, terrain_emb_dim),
                nn.SiLU(),
                nn.Linear(terrain_emb_dim, terrain_emb_dim),
                nn.LayerNorm(terrain_emb_dim),
            )

        # Total per-frame input dim (+ encoded terrain patch + raw per-joint clearances)
        feat_dim = (BODY_DIM + root_input_dim + 2 * action_emb_dim
                    + self.terrain_emb_dim + terrain_clear_dim)
        self.feat_dim = feat_dim

        # Output dim
        out_dim = BODY_DIM + root_output_dim
        self.out_dim = out_dim

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
        self.body_head = nn.Linear(embed_dim, out_dim)

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

    def assemble_features(self, body, action_m, action_n, root=None,
                          terrain_patch=None, terrain_clear=None):
        """Concatenate body (+ optional root) + action embeddings (+ optional terrain).

        Args:
            body: (B, T, 258)
            action_m, action_n: (B, T) long
            root: (B, T, root_input_dim) or None
            terrain_patch: (B, T, terrain_patch_dim) egocentric heightmap, or None
            terrain_clear: (B, T, terrain_clear_dim) per-joint clearances, or None
        Returns:
            (B, T, feat_dim)
        """
        m_emb = self.m_action_emb(action_m)
        n_emb = self.n_action_emb(action_n)
        parts = [body]
        if self.root_input_dim > 0:
            assert root is not None, "root required when root_input_dim>0"
            parts.append(root)
        parts.extend([m_emb, n_emb])
        if self.terrain_patch_dim > 0:
            assert terrain_patch is not None, "terrain_patch required when terrain_patch_dim>0"
            parts.append(self.terrain_enc(terrain_patch))
        if self.terrain_clear_dim > 0:
            assert terrain_clear is not None, "terrain_clear required when terrain_clear_dim>0"
            parts.append(terrain_clear)
        return torch.cat(parts, dim=-1)

    def forward(self, body, action_m, action_n, root=None,
                terrain_patch=None, terrain_clear=None):
        """Single forward pass.

        Args:
            body: (B, T, 258) body pose history
            action_m, action_n: (B, T) action IDs
            root: (B, T, root_input_dim) optional root features
            terrain_patch: (B, T, terrain_patch_dim) optional egocentric heightmap
            terrain_clear: (B, T, terrain_clear_dim) optional per-joint clearances

        Returns:
            pred: (B, T, out_dim) predicted next body (+ root if root_output_dim>0)
                  layout: [body(258D), root(root_output_dim)]
            hidden: (B, T, embed_dim)
        """
        B, T = action_m.shape
        assert T <= self.block_size, f"T={T} exceeds block_size={self.block_size}"

        x = self.assemble_features(body, action_m, action_n, root,
                                   terrain_patch, terrain_clear)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.drop(h)
        for block in self.blocks:
            h, _ = block(h)
        h = self.ln_f(h)
        pred = self.body_head(h)
        return pred, h
