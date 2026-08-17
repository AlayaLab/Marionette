"""Continuous Frame GPT with AdaLN-Zero conditioning.

NPC rotation (6D) is injected via Adaptive Layer Normalization into every
transformer block, giving the low-dimensional control signal strong influence
over the model's behavior at every layer.

Architecture:
  Input (B, T, 276) → split → motion (B, T, 270) + cond (B, T, 6)
  motion → Linear(270, hidden) → + PosEnc
  cond → MLP(6, hidden) → per-layer (scale, shift, gate) × 2
  N × AdaLNBlock(causal attention, conditioned LayerNorm)
  → LayerNorm → Linear(hidden, 270) → Output (B, T, 270)

The model predicts only [0:270] (position + monster rot6d).
NPC rot6d [270:276] is the condition signal, not predicted.
"""

import torch
import torch.nn as nn

from models.gpt import CausalSelfAttention, SinusoidalPositionEncoding


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning.

    Two AdaLN modulations per block (one for attn, one for MLP).
    Each produces (scale, shift, gate) from the condition embedding.
    Gate is initialized to zero (AdaLN-Zero) so the block starts as identity.
    """

    def __init__(self, embed_dim=512, block_size=512, n_head=8,
                 drop_out_rate=0.1, fc_rate=4, cond_dim=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = CausalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

        # AdaLN projections: cond_dim → 6 * embed_dim
        # (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * embed_dim),
        )
        # Initialize gate weights to zero (AdaLN-Zero)
        nn.init.zeros_(self.adaln_proj[-1].weight)
        nn.init.zeros_(self.adaln_proj[-1].bias)

    def forward(self, x, cond):
        """
        Args:
            x: (B, T, embed_dim) hidden states
            cond: (B, T, cond_dim) condition embedding (per-frame)
        """
        # Project condition to 6 modulation vectors
        adaln_params = self.adaln_proj(cond)  # (B, T, 6*D)
        s1, sh1, g1, s2, sh2, g2 = adaln_params.chunk(6, dim=-1)

        # Attention with AdaLN
        h = self.ln1(x)
        h = (1 + s1) * h + sh1  # modulate
        h = self.attn(h)
        x = x + g1 * h  # gated residual

        # MLP with AdaLN
        h = self.ln2(x)
        h = (1 + s2) * h + sh2
        h = self.mlp(h)
        x = x + g2 * h

        return x


class ContinuousMotionGPT_AdaLN(nn.Module):
    """Autoregressive GPT with AdaLN-Zero NPC rotation conditioning.

    Input: (B, T, 276) full features
    - [0:270] = motion features (position + monster rot6d) → predicted
    - [270:276] = NPC rot6d → condition signal (not predicted)

    Output: (B, T, 270) predicted motion features
    """

    def __init__(self, feat_dim=276, embed_dim=512, block_size=512,
                 num_layers=8, n_head=8, drop_out_rate=0.1, fc_rate=4,
                 cond_dim=6, cond_hidden=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.block_size = block_size
        self.motion_dim = feat_dim - cond_dim  # 270
        self.cond_dim = cond_dim               # 6 (NPC rot6d)
        self.cond_start = feat_dim - cond_dim  # 270

        # Motion input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.motion_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pos_enc = SinusoidalPositionEncoding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)

        # Condition encoder: 6D NPC rot6d → cond_hidden per frame
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_hidden),
            nn.SiLU(),
            nn.Linear(cond_hidden, cond_hidden),
        )

        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(embed_dim, block_size, n_head,
                                 drop_out_rate, fc_rate, cond_hidden)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, self.motion_dim)  # predict 270D

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

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, T, 276) full features including NPC rot6d at [270:276]

        Returns:
            pred: (B, T, 276) predictions. [0:270] are model outputs,
                  [270:276] is copied from input (condition passthrough).
        """
        B, T, D = x.size()
        assert T <= self.block_size

        # Split motion and condition
        motion = x[..., :self.cond_start]   # (B, T, 270)
        cond_raw = x[..., self.cond_start:] # (B, T, 6)

        # Encode condition
        cond = self.cond_encoder(cond_raw)  # (B, T, cond_hidden)

        # Project motion input
        h = self.input_proj(motion)
        h = self.pos_enc(h)
        h = self.drop(h)

        # Transformer blocks with AdaLN conditioning
        for block in self.blocks:
            h = block(h, cond)

        h = self.ln_f(h)
        pred_motion = self.output_head(h)  # (B, T, 270)

        # Concat condition passthrough to match input shape
        pred = torch.cat([pred_motion, cond_raw], dim=-1)  # (B, T, 276)

        return pred

    @torch.no_grad()
    def generate(self, seed, num_frames, clamp_std=None):
        """Autoregressive generation. NPC rot6d must be provided externally."""
        raise NotImplementedError(
            "AdaLN model requires NPC rot6d condition. Use generate_with_npc_rot().")

    @torch.no_grad()
    def generate_with_npc_rot(self, seed, gt_frames, npc_rot_start=270, npc_rot_end=276):
        """Autoregressive generation with GT NPC rotation as condition.

        Args:
            seed: (1, S, 276) seed sequence (normalized, includes NPC rot6d)
            gt_frames: (1, G, 276) GT frames after seed (need NPC rot6d from here)
        Returns:
            (1, S + G, 276) full sequence
        """
        self.eval()
        num_frames = gt_frames.shape[1]
        seq = seed.clone()

        for i in range(num_frames):
            ctx = seq[:, -self.block_size:]
            pred = self.forward(ctx)
            next_frame = pred[:, -1:, :].clone()  # (1, 1, 276)

            # Replace NPC rot6d with GT (condition for NEXT frame)
            next_frame[:, :, npc_rot_start:npc_rot_end] = \
                gt_frames[:, i:i+1, npc_rot_start:npc_rot_end]

            seq = torch.cat([seq, next_frame], dim=1)

        return seq

    @torch.no_grad()
    def generate_free(self, seed, num_frames, clamp_std=None):
        """Free generation (no external condition). Uses model's own NPC rot6d prediction.
        For comparison/ablation only — model not trained for this."""
        self.eval()
        seq = seed.clone()
        for _ in range(num_frames):
            ctx = seq[:, -self.block_size:]
            pred = self.forward(ctx)
            next_frame = pred[:, -1:, :]
            if clamp_std is not None:
                next_frame = next_frame.clamp(-clamp_std, clamp_std)
            seq = torch.cat([seq, next_frame], dim=1)
        return seq
