"""Continuous Frame GPT: autoregressive next-frame prediction on raw motion features.

No VQ-VAE quantization — directly models 264D position-only features.
Reuses CausalSelfAttention, TransformerBlock, SinusoidalPositionEncoding from gpt.py.
"""

import torch
import torch.nn as nn

from models.gpt import CausalSelfAttention, TransformerBlock, SinusoidalPositionEncoding


class ContinuousMotionGPT(nn.Module):
    """Autoregressive GPT for continuous motion frame prediction.

    Input:  (B, T, feat_dim) continuous motion features
    Output: (B, T, feat_dim) next-frame predictions
    """

    def __init__(self, feat_dim=264, embed_dim=512, block_size=256,
                 num_layers=8, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.block_size = block_size

        # Input projection (replaces token embedding)
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

        # Output head (replaces logit head)
        self.output_head = nn.Linear(embed_dim, feat_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """Forward pass. x: (B, T, feat_dim). Returns predictions (B, T, feat_dim)."""
        B, T, D = x.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.drop(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_f(x)
        pred = self.output_head(x)
        return pred

    @torch.no_grad()
    def generate(self, seed, num_frames, clamp_std=None):
        """Autoregressive generation.

        Args:
            seed: (1, S, feat_dim) seed sequence
            num_frames: number of frames to generate after seed
            clamp_std: if set, clamp predictions to [-clamp_std, clamp_std] (in normalized space)
        Returns:
            (1, S + num_frames, feat_dim) full sequence including seed
        """
        self.eval()
        seq = seed.clone()

        for _ in range(num_frames):
            # Use last block_size frames as context
            ctx = seq[:, -self.block_size:]
            pred = self.forward(ctx)
            next_frame = pred[:, -1:, :]  # (1, 1, feat_dim)

            if clamp_std is not None:
                next_frame = next_frame.clamp(-clamp_std, clamp_std)

            seq = torch.cat([seq, next_frame], dim=1)

        return seq

    @torch.no_grad()
    def generate_with_npc_rot(self, seed, gt_frames, npc_rot_start=270, npc_rot_end=276):
        """Autoregressive generation with GT NPC rotation injected each frame.

        Model predicts all dims, but NPC rot6d [270:276] is replaced with GT.
        Monster rot6d [264:270] is kept from model prediction.
        This simulates joystick-controlled NPC heading.

        Args:
            seed: (1, S, feat_dim) seed sequence (normalized)
            gt_frames: (1, G, feat_dim) GT frames after seed (normalized)
            npc_rot_start: start dim of NPC rot6d (default 270)
            npc_rot_end: end dim of NPC rot6d (default 276)
        Returns:
            (1, S + G, feat_dim) full sequence
        """
        self.eval()
        num_frames = gt_frames.shape[1]
        seq = seed.clone()

        for i in range(num_frames):
            ctx = seq[:, -self.block_size:]
            pred = self.forward(ctx)
            next_frame = pred[:, -1:, :].clone()

            # Replace ONLY NPC rot6d with GT
            next_frame[:, :, npc_rot_start:npc_rot_end] = gt_frames[:, i:i+1, npc_rot_start:npc_rot_end]

            seq = torch.cat([seq, next_frame], dim=1)

        return seq

    @torch.no_grad()
    def generate_with_gt_rot(self, seed, gt_frames, rot_dim_start=264):
        """Autoregressive generation with GT rotation injected each frame.

        Model predicts all 276D, but the rot6d dims [264:276] are replaced
        with ground truth values before being fed back as context. This tests
        whether correct heading info fixes long-rollout drift.

        Args:
            seed: (1, S, feat_dim) seed sequence (normalized)
            gt_frames: (1, G, feat_dim) GT frames AFTER seed (normalized)
                       G >= num frames to generate
            rot_dim_start: where rot6d dims begin (default 264)
        Returns:
            (1, S + G, feat_dim) full sequence including seed
        """
        self.eval()
        num_frames = gt_frames.shape[1]
        seq = seed.clone()

        for i in range(num_frames):
            ctx = seq[:, -self.block_size:]
            pred = self.forward(ctx)
            next_frame = pred[:, -1:, :].clone()  # (1, 1, feat_dim)

            # Replace predicted rot6d with GT rot6d
            next_frame[:, :, rot_dim_start:] = gt_frames[:, i:i+1, rot_dim_start:]

            seq = torch.cat([seq, next_frame], dim=1)

        return seq
