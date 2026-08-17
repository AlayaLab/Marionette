"""Two-Stage Continuous Frame GPT: Root/Body Decomposition (Kimodo-inspired).

Stage 1 (RootGPT): Lightweight trajectory planner — predicts root motion (12D)
Stage 2 (BodyGPT): Body animator — predicts body poses given planned root (258D)

Stage 1 output is detached before feeding to Stage 2, so each stage
optimizes independently (Kimodo's key insight for reducing foot skate).

276D Feature Layout:
  Stage 1 predicts:
    [0:3]     Monster root_delta_local   3D
    [162:165] NPC root_delta_local       3D
    [264:270] Monster root rot6d         6D
  Stage 2 predicts:
    [3:162]   Monster rel_pos_local    159D
    [165:258] NPC rel_pos_local         93D
    [258:264] Weapon rel_pos_local       6D
  GT condition (not predicted):
    [270:276] NPC root rot6d             6D

Root prediction (12D) layout:
    [0:3]   Monster root_delta_local
    [3:6]   NPC root_delta_local
    [6:12]  Monster root rot6d

Body prediction (258D) layout:
    [0:159]   Monster rel_pos_local (53×3)
    [159:252] NPC rel_pos_local (31×3)
    [252:258] Weapon rel_pos_local (2×3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gpt import (
    CausalSelfAttention, TransformerBlock, SinusoidalPositionEncoding,
    TransformerBlockRoPE,
)


# Indices within 276D features
ROOT_INDICES = {
    'm_root_delta': (0, 3),
    'n_root_delta': (162, 165),
    'm_root_rot': (264, 270),
}
BODY_INDICES = {
    'm_rel_pos': (3, 162),
    'n_rel_pos': (165, 258),
    'weapon': (258, 264),
}
NPC_ROT = (270, 276)  # GT condition


class RootGPT(nn.Module):
    """Stage 1: Trajectory planner.

    Sees full 276D context, predicts 12D root motion per frame.
    """

    def __init__(self, feat_dim=276, root_out_dim=12, embed_dim=256,
                 block_size=512, num_layers=4, n_head=8,
                 drop_out_rate=0.1, fc_rate=4, use_rope=False, rope_base=10000.0):
        super().__init__()
        self.block_size = block_size
        self.use_rope = use_rope

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        if not use_rope:
            self.pos_enc = SinusoidalPositionEncoding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)

        if use_rope:
            self.blocks = nn.ModuleList([
                TransformerBlockRoPE(embed_dim, block_size, n_head, drop_out_rate, fc_rate, rope_base=rope_base)
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers)
            ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, root_out_dim)
        self.num_layers = num_layers

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

    def forward(self, x, past_kvs=None):
        """
        Args:
            x: (B, T, feat_dim) full motion features (276 or 276+action_emb)
            past_kvs: optional list of per-layer KV caches
        Returns:
            output: (B, T, 12) root predictions
            new_kvs: list of per-layer KV caches
            hidden: (B, T, embed_dim) last hidden state (for action heads)
        """
        B, T, D = x.size()
        assert T <= self.block_size
        h = self.input_proj(x)
        pos_offset = 0
        if self.use_rope:
            if past_kvs is not None:
                pos_offset = past_kvs[0].size(3)
        else:
            h = self.pos_enc(h) if past_kvs is None else h + self.pos_enc.pe[:, past_kvs[0].size(3):past_kvs[0].size(3) + T]
        h = self.drop(h)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if self.use_rope:
                h, new_kv = block(h, past_kv=past_kv, pos_offset=pos_offset)
            else:
                h, new_kv = block(h, past_kv=past_kv)
            new_kvs.append(new_kv)
        h = self.ln_f(h)
        return self.output_head(h), new_kvs, h


class BodyGPT(nn.Module):
    """Stage 2: Body animator.

    Sees 276D context with root dims replaced by Stage 1 predictions,
    predicts 258D body features.
    """

    def __init__(self, feat_dim=276, body_out_dim=258, embed_dim=512,
                 block_size=512, num_layers=8, n_head=8,
                 drop_out_rate=0.1, fc_rate=4, use_rope=False, rope_base=10000.0):
        super().__init__()
        self.block_size = block_size
        self.use_rope = use_rope

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        if not use_rope:
            self.pos_enc = SinusoidalPositionEncoding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)

        if use_rope:
            self.blocks = nn.ModuleList([
                TransformerBlockRoPE(embed_dim, block_size, n_head, drop_out_rate, fc_rate, rope_base=rope_base)
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers)
            ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, body_out_dim)
        self.num_layers = num_layers

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

    def forward(self, x, past_kvs=None):
        """
        Args:
            x: (B, T, feat_dim) features with root dims from Stage 1
            past_kvs: optional list of per-layer KV caches
        Returns:
            output: (B, T, 258) body predictions
            new_kvs: list of per-layer KV caches
            hidden: (B, T, embed_dim) last hidden state
        """
        B, T, D = x.size()
        assert T <= self.block_size
        h = self.input_proj(x)
        pos_offset = 0
        if self.use_rope:
            if past_kvs is not None:
                pos_offset = past_kvs[0].size(3)
        else:
            h = self.pos_enc(h) if past_kvs is None else h + self.pos_enc.pe[:, past_kvs[0].size(3):past_kvs[0].size(3) + T]
        h = self.drop(h)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if self.use_rope:
                h, new_kv = block(h, past_kv=past_kv, pos_offset=pos_offset)
            else:
                h, new_kv = block(h, past_kv=past_kv)
            new_kvs.append(new_kv)
        h = self.ln_f(h)
        return self.output_head(h), new_kvs, h


class TwoStageMotionGPT(nn.Module):
    """Two-stage autoregressive motion prediction.

    Stage 1 (RootGPT): full context → 12D root motion + action prediction
    Stage 2 (BodyGPT): context with detached root → 258D body poses

    NPC rot6d [270:276] is always GT condition (never predicted).
    Optional action_cfg enables discrete action ID embedding + prediction.
    """

    def __init__(self, root_cfg, body_cfg, action_cfg=None):
        super().__init__()
        self.has_action = action_cfg is not None

        if self.has_action:
            ac = action_cfg
            emb_dim = ac.get('emb_dim', 64)
            self.action_emb_dim = emb_dim
            self.m_action_emb = nn.Embedding(ac['m_vocab_size'], emb_dim)
            self.n_action_emb = nn.Embedding(ac['n_vocab_size'], emb_dim)
            self.m_action_head = nn.Linear(
                root_cfg.get('embed_dim', 256), ac['m_vocab_size'])
            self.n_action_head = nn.Linear(
                root_cfg.get('embed_dim', 256), ac['n_vocab_size'])
            # Optional progress input (2D continuous: monster_progress, npc_progress)
            self.use_progress = ac.get('use_progress', False)
            # Echo gate: learned bias from action input to action output
            self.use_echo = ac.get('use_echo', False)
            if self.use_echo:
                root_emb = root_cfg.get('embed_dim', 256)
                self.m_echo_gate = nn.Sequential(
                    nn.Linear(root_emb, 1), nn.Sigmoid())
                self.n_echo_gate = nn.Sequential(
                    nn.Linear(root_emb, 1), nn.Sigmoid())
                self.echo_scale = ac.get('echo_scale', 10.0)
            extra_dim = 2 * emb_dim + (2 if self.use_progress else 0)
            # Expand feat_dim for both stages to include action embeddings (+progress)
            root_cfg = dict(root_cfg)
            root_cfg['feat_dim'] = root_cfg.get('feat_dim', 276) + extra_dim
            body_cfg = dict(body_cfg)
            body_cfg['feat_dim'] = body_cfg.get('feat_dim', 276) + extra_dim

        self.root_gpt = RootGPT(**root_cfg)
        self.body_gpt = BodyGPT(**body_cfg)
        self.block_size = root_cfg['block_size']

    def _concat_action_emb(self, x, action_m, action_n, progress=None):
        """Concat action embeddings (and optional progress) to pose features."""
        m_emb = self.m_action_emb(action_m)  # (B, T, emb_dim)
        n_emb = self.n_action_emb(action_n)  # (B, T, emb_dim)
        parts = [x, m_emb, n_emb]
        if self.use_progress:
            if progress is not None:
                parts.append(progress)
            else:
                parts.append(torch.zeros(x.shape[0], x.shape[1], 2,
                                         device=x.device, dtype=x.dtype))
        return torch.cat(parts, dim=-1)

    def _apply_echo(self, m_logits, n_logits, hidden, action_m, action_n):
        """Apply echo gate bias to action logits (if enabled)."""
        if not self.use_echo:
            return m_logits, n_logits
        m_gate = self.m_echo_gate(hidden)
        n_gate = self.n_echo_gate(hidden)
        m_onehot = F.one_hot(action_m, m_logits.size(-1)).float()
        n_onehot = F.one_hot(action_n, n_logits.size(-1)).float()
        return (m_logits + m_gate * self.echo_scale * m_onehot,
                n_logits + n_gate * self.echo_scale * n_onehot)

    def _concat_zero_action(self, x):
        """Concat zero action embeddings (unconditional mode for CFG)."""
        B, T = x.shape[:2]
        zero_dim = self.action_emb_dim * 2
        if self.use_progress:
            zero_dim += 2
        zeros = torch.zeros(B, T, zero_dim, device=x.device, dtype=x.dtype)
        return torch.cat([x, zeros], dim=-1)

    def _inject_root(self, x, root_pred):
        """Replace root dims in x with root_pred values.
        Works on both 276D and extended (276+action) inputs.
        """
        out = x.clone()
        out[..., 0:3] = root_pred[..., 0:3]       # m_root_delta
        out[..., 162:165] = root_pred[..., 3:6]    # n_root_delta
        out[..., 264:270] = root_pred[..., 6:12]   # m_root_rot6d
        return out

    def _assemble_full(self, root_pred, body_pred, npc_rot):
        """Assemble full 276D prediction from root + body + NPC rot6d."""
        B, T = root_pred.shape[:2]
        full = torch.zeros(B, T, 276, device=root_pred.device, dtype=root_pred.dtype)
        # Root
        full[..., 0:3] = root_pred[..., 0:3]       # m_root_delta
        full[..., 162:165] = root_pred[..., 3:6]    # n_root_delta
        full[..., 264:270] = root_pred[..., 6:12]   # m_root_rot6d
        # Body
        full[..., 3:162] = body_pred[..., 0:159]    # m_rel_pos
        full[..., 165:258] = body_pred[..., 159:252] # n_rel_pos
        full[..., 258:264] = body_pred[..., 252:258] # weapon
        # NPC rot6d passthrough
        full[..., 270:276] = npc_rot
        return full

    def forward(self, x, detach_root=True, action_m=None, action_n=None,
                progress=None, action_drop_prob=0.0):
        """Forward pass (training — no cache).

        Args:
            x: (B, T, 276) input features
            detach_root: if True, Stage 1 output is detached before Stage 2
            action_m: (B, T) int monster action IDs (optional)
            action_n: (B, T) int NPC action IDs (optional)
            progress: (B, T, 2) float animation progress (optional)
            action_drop_prob: probability of dropping action (unconditional, for CFG)

        Returns:
            root_pred, body_pred, full_pred, action_logits
        """
        # Optionally concat action embeddings (with per-sample dropout for CFG)
        if self.has_action and action_m is not None:
            if self.training and action_drop_prob > 0 and torch.rand(1).item() < action_drop_prob:
                x_in = self._concat_zero_action(x)
            else:
                x_in = self._concat_action_emb(x, action_m, action_n, progress)
        else:
            x_in = x

        # Stage 1: predict root motion
        root_pred, _, root_hidden = self.root_gpt(x_in)  # (B, T, 12)

        # Action prediction from Stage 1 hidden state (with optional echo gate)
        action_logits = None
        if self.has_action:
            m_logits = self.m_action_head(root_hidden)
            n_logits = self.n_action_head(root_hidden)
            if self.use_echo and action_m is not None:
                m_gate = self.m_echo_gate(root_hidden)  # (B, T, 1)
                n_gate = self.n_echo_gate(root_hidden)
                # Use the actual action input (not the shifted one from outside)
                m_onehot = F.one_hot(action_m, m_logits.size(-1)).float()
                n_onehot = F.one_hot(action_n, n_logits.size(-1)).float()
                m_logits = m_logits + m_gate * self.echo_scale * m_onehot
                n_logits = n_logits + n_gate * self.echo_scale * n_onehot
            action_logits = (m_logits, n_logits)

        # Build Stage 2 input: replace root dims with Stage 1 predictions
        root_for_s2 = root_pred.detach() if detach_root else root_pred
        x_s2 = self._inject_root(x_in, root_for_s2)

        # Stage 2: predict body poses
        body_pred, _, _ = self.body_gpt(x_s2)  # (B, T, 258)

        # Assemble full prediction
        npc_rot = x[..., 270:276]  # GT passthrough (from original 276D)
        full_pred = self._assemble_full(root_pred, body_pred, npc_rot)

        return root_pred, body_pred, full_pred, action_logits

    @torch.no_grad()
    def generate_with_npc_rot(self, seed, gt_frames,
                               npc_rot_start=270, npc_rot_end=276,
                               seed_action_m=None, seed_action_n=None,
                               gt_action_m=None, gt_action_n=None,
                               seed_progress=None, gt_progress=None):
        """Autoregressive generation with GT NPC rotation.

        Args:
            seed: (1, S, 276) seed sequence
            gt_frames: (1, G, 276) GT frames after seed (for NPC rot6d)
            seed_action_m/n: (1, S) seed action IDs (optional, for action model)
            gt_action_m/n: (1, G) GT action IDs after seed (optional)
            seed_progress: (1, S, 2) seed progress (optional)
            gt_progress: (1, G, 2) GT progress after seed (optional)
        Returns:
            (1, S + G, 276) full generated sequence
        """
        self.eval()
        num_frames = gt_frames.shape[1]
        seq = seed.clone()

        # Build action sequence for autoregressive feeding
        if self.has_action and seed_action_m is not None:
            action_m_seq = seed_action_m.clone()  # (1, S)
            action_n_seq = seed_action_n.clone()
            progress_seq = seed_progress.clone() if seed_progress is not None else None
        else:
            action_m_seq = None
            progress_seq = None

        for i in range(num_frames):
            ctx = seq[:, -self.block_size:]

            if action_m_seq is not None:
                act_m_ctx = action_m_seq[:, -self.block_size:]
                act_n_ctx = action_n_seq[:, -self.block_size:]
                prog_ctx = progress_seq[:, -self.block_size:] if progress_seq is not None else None
                ctx_in = self._concat_action_emb(ctx, act_m_ctx, act_n_ctx, prog_ctx)
            else:
                ctx_in = ctx

            root_pred, _, root_hidden = self.root_gpt(ctx_in)
            ctx_s2 = self._inject_root(ctx_in, root_pred)
            body_pred, _, _ = self.body_gpt(ctx_s2)

            root_next = root_pred[:, -1:, :]
            body_next = body_pred[:, -1:, :]

            npc_rot = gt_frames[:, i:i+1, npc_rot_start:npc_rot_end]
            next_frame = self._assemble_full(root_next, body_next, npc_rot)
            seq = torch.cat([seq, next_frame], dim=1)

            # Action: use GT if available, otherwise predict
            if action_m_seq is not None:
                if gt_action_m is not None and i < gt_action_m.shape[1]:
                    next_act_m = gt_action_m[:, i:i+1]
                    next_act_n = gt_action_n[:, i:i+1]
                else:
                    m_logits = self.m_action_head(root_hidden[:, -1:, :])
                    n_logits = self.n_action_head(root_hidden[:, -1:, :])
                    next_act_m = m_logits.argmax(dim=-1)
                    next_act_n = n_logits.argmax(dim=-1)
                action_m_seq = torch.cat([action_m_seq, next_act_m], dim=1)
                action_n_seq = torch.cat([action_n_seq, next_act_n], dim=1)
                # Progress: use GT if available, otherwise zeros
                if progress_seq is not None:
                    if gt_progress is not None and i < gt_progress.shape[1]:
                        next_prog = gt_progress[:, i:i+1]
                    else:
                        next_prog = torch.zeros(1, 1, 2, device=seed.device)
                    progress_seq = torch.cat([progress_seq, next_prog], dim=1)

        return seq

    @torch.no_grad()
    def generate_hybrid(self, seed, gt_frames,
                        seed_action_m=None, seed_action_n=None,
                        npc_control=None,
                        npc_rot_start=270, npc_rot_end=276):
        """Hybrid generation: Monster auto, NPC user-controlled at sparse frames.

        Args:
            seed: (1, S, 276) seed sequence
            gt_frames: (1, G, 276) GT frames (for NPC rot6d)
            seed_action_m/n: (1, S) seed action IDs
            npc_control: (G,) int tensor — NPC action ID at controlled frames,
                         0 = autoregressive. Only injected at transition frames.
        Returns:
            (1, S + G, 276) full generated sequence
        """
        self.eval()
        device = seed.device
        num_frames = gt_frames.shape[1]
        seq = seed.clone()

        if not self.has_action or seed_action_m is None:
            return self.generate_with_npc_rot(seed, gt_frames)

        action_m_seq = seed_action_m.clone()
        action_n_seq = seed_action_n.clone()

        for i in range(num_frames):
            ctx = seq[:, -self.block_size:]
            act_m_ctx = action_m_seq[:, -self.block_size:]
            act_n_ctx = action_n_seq[:, -self.block_size:]
            ctx_in = self._concat_action_emb(ctx, act_m_ctx, act_n_ctx)

            root_pred, _, root_hidden = self.root_gpt(ctx_in)
            ctx_s2 = self._inject_root(ctx_in, root_pred)
            body_pred, _, _ = self.body_gpt(ctx_s2)

            root_next = root_pred[:, -1:, :]
            body_next = body_pred[:, -1:, :]
            npc_rot = gt_frames[:, i:i+1, npc_rot_start:npc_rot_end]
            next_frame = self._assemble_full(root_next, body_next, npc_rot)
            seq = torch.cat([seq, next_frame], dim=1)

            # Monster: always autoregressive
            m_logits = self.m_action_head(root_hidden[:, -1:, :])
            next_act_m = m_logits.argmax(dim=-1)

            # NPC: use control signal if provided, else autoregressive
            if npc_control is not None and i < len(npc_control) and npc_control[i] > 0:
                next_act_n = torch.tensor([[npc_control[i]]],
                                          device=device, dtype=torch.long)
            else:
                n_logits = self.n_action_head(root_hidden[:, -1:, :])
                next_act_n = n_logits.argmax(dim=-1)

            action_m_seq = torch.cat([action_m_seq, next_act_m], dim=1)
            action_n_seq = torch.cat([action_n_seq, next_act_n], dim=1)

        return seq

    @torch.no_grad()
    def generate_with_cfg(self, seed, gt_frames, seed_action_m, seed_action_n,
                          npc_control=None, guidance_weight=2.0,
                          cfg_decay=0.0,
                          npc_rot_start=270, npc_rot_end=276):
        """Generate with selective classifier-free guidance.

        CFG is only applied at frames where npc_control injects an action.
        Other frames use unconditional mode (w=0). This prevents CFG from
        amplifying self-predicted action errors.

        Optional exponential decay: after injection, w decays as w*exp(-t/decay)
        over subsequent frames (decay=0 means instant cutoff).

        Args:
            guidance_weight: CFG weight w at injection frames
            cfg_decay: decay half-life in frames (0=no decay, instant w→0)
            npc_control: (G,) int — NPC action at controlled frames, 0=auto
        Returns:
            seq, pred_action_m (G,), pred_action_n (G,)
        """
        self.eval()
        device = seed.device
        W = guidance_weight
        num_frames = gt_frames.shape[1]
        seq = seed.clone()

        action_m_seq = seed_action_m.clone()
        action_n_seq = seed_action_n.clone()

        # Track per-frame effective w (for selective CFG + decay)
        frames_since_inject = 999  # large = no recent injection

        for i in range(num_frames):
            ctx = seq[:, -self.block_size:]
            act_m_ctx = action_m_seq[:, -self.block_size:]
            act_n_ctx = action_n_seq[:, -self.block_size:]

            # Check if this is an injection frame
            is_inject = (npc_control is not None and i < len(npc_control)
                         and npc_control[i] > 0)
            if is_inject:
                frames_since_inject = 0
            else:
                frames_since_inject += 1

            # Compute effective w
            if frames_since_inject == 0:
                w_eff = W
            elif cfg_decay > 0:
                import math
                w_eff = W * math.exp(-frames_since_inject / cfg_decay)
                if w_eff < 0.05:
                    w_eff = 0.0
            else:
                w_eff = 0.0

            # Always run conditional forward (so action embeddings are visible)
            ctx_cond = self._concat_action_emb(ctx, act_m_ctx, act_n_ctx)
            root_cond, _, hidden_cond = self.root_gpt(ctx_cond)
            ctx_s2_cond = self._inject_root(ctx_cond, root_cond)
            body_cond, _, _ = self.body_gpt(ctx_s2_cond)

            if w_eff > 0:
                # CFG boost: also run unconditional, interpolate
                ctx_uncond = self._concat_zero_action(ctx)
                root_uncond, _, _ = self.root_gpt(ctx_uncond)
                ctx_s2_uncond = self._inject_root(ctx_uncond, root_uncond)
                body_uncond, _, _ = self.body_gpt(ctx_s2_uncond)

                root_g = (1 + w_eff) * root_cond[:, -1:] - w_eff * root_uncond[:, -1:]
                body_g = (1 + w_eff) * body_cond[:, -1:] - w_eff * body_uncond[:, -1:]
            else:
                # Conditional only (action embeddings active, no CFG)
                root_g = root_cond[:, -1:]
                body_g = body_cond[:, -1:]

            # Action prediction with echo gate
            m_logits = self.m_action_head(hidden_cond[:, -1:])
            n_logits = self.n_action_head(hidden_cond[:, -1:])
            if self.use_echo:
                m_logits, n_logits = self._apply_echo(
                    m_logits, n_logits, hidden_cond[:, -1:],
                    act_m_ctx[:, -1:], act_n_ctx[:, -1:])
            next_act_m = m_logits.argmax(dim=-1)

            npc_rot = gt_frames[:, i:i+1, npc_rot_start:npc_rot_end]
            next_frame = self._assemble_full(root_g, body_g, npc_rot)
            seq = torch.cat([seq, next_frame], dim=1)

            # NPC action: user control or autoregressive with echo
            if is_inject:
                next_act_n = torch.tensor([[npc_control[i]]],
                                          device=device, dtype=torch.long)
            else:
                next_act_n = n_logits.argmax(dim=-1)

            action_m_seq = torch.cat([action_m_seq, next_act_m], dim=1)
            action_n_seq = torch.cat([action_n_seq, next_act_n], dim=1)

        S = seed_action_m.shape[1]
        return (seq,
                action_m_seq[0, S:].cpu().numpy(),
                action_n_seq[0, S:].cpu().numpy())

    @torch.no_grad()
    def generate(self, seed, num_frames, clamp_std=None):
        """Free generation (no GT NPC rotation). For ablation only."""
        self.eval()
        seq = seed.clone()

        for _ in range(num_frames):
            ctx = seq[:, -self.block_size:]

            root_pred, _, _ = self.root_gpt(ctx)
            ctx_s2 = self._inject_root(ctx, root_pred)
            body_pred, _, _ = self.body_gpt(ctx_s2)

            root_next = root_pred[:, -1:, :]
            body_next = body_pred[:, -1:, :]
            npc_rot = ctx[:, -1:, 270:276]
            next_frame = self._assemble_full(root_next, body_next, npc_rot)

            if clamp_std is not None:
                next_frame = next_frame.clamp(-clamp_std, clamp_std)

            seq = torch.cat([seq, next_frame], dim=1)

        return seq

    @torch.no_grad()
    def generate_with_gt_rot(self, seed, gt_frames, rot_dim_start=264):
        """Generation with all GT rotations (Monster + NPC). For ablation."""
        self.eval()
        num_frames = gt_frames.shape[1]
        seq = seed.clone()

        for i in range(num_frames):
            ctx = seq[:, -self.block_size:]

            root_pred, _, _ = self.root_gpt(ctx)
            ctx_s2 = self._inject_root(ctx, root_pred)
            body_pred, _, _ = self.body_gpt(ctx_s2)

            root_next = root_pred[:, -1:, :]
            body_next = body_pred[:, -1:, :]
            npc_rot = gt_frames[:, i:i+1, 270:276]
            next_frame = self._assemble_full(root_next, body_next, npc_rot)

            next_frame[:, :, rot_dim_start:] = gt_frames[:, i:i+1, rot_dim_start:]

            seq = torch.cat([seq, next_frame], dim=1)

        return seq

    @staticmethod
    def _evict_kv_cache(kvs, max_len):
        """Evict oldest entries from KV caches to keep length <= max_len."""
        if kvs is None or kvs[0].size(3) <= max_len:
            return kvs
        return [kv[:, :, :, -max_len:, :] for kv in kvs]

    @torch.no_grad()
    def generate_streaming(self, seed, npc_rot_seq,
                           npc_rot_start=270, npc_rot_end=276):
        """Streaming generation with KV cache.

        Args:
            seed: (1, S, 276) seed sequence
            npc_rot_seq: (1, G, 6) per-frame NPC rot6d control signal
        Returns:
            (1, S + G, 276) full generated sequence
        """
        self.eval()
        num_frames = npc_rot_seq.shape[1]
        seq = seed.clone()

        # Prefill: process entire seed at once to populate KV caches
        root_pred, root_kvs, _ = self.root_gpt(seed)
        ctx_s2 = self._inject_root(seed, root_pred)
        body_pred, body_kvs, _ = self.body_gpt(ctx_s2)

        # Track current cache length for position encoding offset
        cache_len = seed.shape[1]

        for i in range(num_frames):
            root_next = root_pred[:, -1:, :]
            body_next = body_pred[:, -1:, :]

            npc_rot = npc_rot_seq[:, i:i+1, :]
            next_frame = self._assemble_full(root_next, body_next, npc_rot)
            seq = torch.cat([seq, next_frame], dim=1)

            if cache_len >= self.block_size:
                root_kvs = self._evict_kv_cache(root_kvs, self.block_size - 1)
                body_kvs = self._evict_kv_cache(body_kvs, self.block_size - 1)
                cache_len = self.block_size - 1

            root_pred, root_kvs, _ = self.root_gpt(next_frame, past_kvs=root_kvs)

            frame_s2 = self._inject_root(next_frame, root_pred)
            body_pred, body_kvs, _ = self.body_gpt(frame_s2, past_kvs=body_kvs)

            cache_len += 1

        return seq
