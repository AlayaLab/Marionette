import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, block_size=257, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % n_head == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        self.n_head = n_head

    def forward(self, x, past_kv=None):
        """
        Args:
            x: (B, T_new, C) input embeddings
            past_kv: optional (2, B, n_head, T_past, head_dim) cached keys/values
        Returns:
            y: (B, T_new, C) output
            new_kv: (2, B, n_head, T_past+T_new, head_dim) updated cache
        """
        B, T_new, C = x.size()
        head_dim = C // self.n_head

        k = self.key(x).view(B, T_new, self.n_head, head_dim).transpose(1, 2)
        q = self.query(x).view(B, T_new, self.n_head, head_dim).transpose(1, 2)
        v = self.value(x).view(B, T_new, self.n_head, head_dim).transpose(1, 2)

        if past_kv is not None:
            # Concatenate past keys/values with current
            k = torch.cat([past_kv[0], k], dim=2)  # (B, n_head, T_past+T_new, head_dim)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = torch.stack([k, v])  # (2, B, n_head, T_total, head_dim)

        T_total = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        # Causal mask: each query position can attend to keys at positions <= its own
        # Query positions correspond to the last T_new positions in the full sequence
        causal_mask = self.mask[:, :, T_total - T_new:T_total, :T_total]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        y = self.resid_drop(self.proj(y))
        return y, new_kv


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, block_size=257, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Encoding (RoPE).

    Applied per-head to q and k vectors in attention.
    Supports position offset for KV cache incremental decoding.
    """

    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        # Precompute cos/sin for max_seq_len positions
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
        cos_cache = freqs.cos()  # (max_seq_len, head_dim/2)
        sin_cache = freqs.sin()
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)

    def forward(self, x, offset=0):
        """Apply RoPE to x.

        Args:
            x: (B, n_head, T, head_dim)
            offset: position offset (for KV cache)
        Returns:
            rotated x with same shape
        """
        T = x.size(2)
        cos = self.cos_cache[offset:offset + T]  # (T, head_dim/2)
        sin = self.sin_cache[offset:offset + T]

        # Reshape for broadcasting: (1, 1, T, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split x into pairs and rotate
        x1 = x[..., 0::2]  # even dims
        x2 = x[..., 1::2]  # odd dims
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1).flatten(-2)
        return rotated


class CausalSelfAttentionRoPE(nn.Module):
    """CausalSelfAttention with RoPE instead of additive position encoding."""

    def __init__(self, embed_dim=512, block_size=257, n_head=8, drop_out_rate=0.1, rope_base=10000.0):
        super().__init__()
        assert embed_dim % n_head == 0
        head_dim = embed_dim // n_head
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        self.n_head = n_head
        self.rope = RotaryPositionEncoding(head_dim, max_seq_len=block_size * 4, base=rope_base)

    def forward(self, x, past_kv=None, pos_offset=0):
        """
        Args:
            x: (B, T_new, C) input embeddings (NO position encoding added)
            past_kv: optional (2, B, n_head, T_past, head_dim) cached keys/values
            pos_offset: absolute position of the first token in x
        Returns:
            y: (B, T_new, C) output
            new_kv: (2, B, n_head, T_past+T_new, head_dim) updated cache
        """
        B, T_new, C = x.size()
        head_dim = C // self.n_head

        k = self.key(x).view(B, T_new, self.n_head, head_dim).transpose(1, 2)
        q = self.query(x).view(B, T_new, self.n_head, head_dim).transpose(1, 2)
        v = self.value(x).view(B, T_new, self.n_head, head_dim).transpose(1, 2)

        # Apply RoPE to q and k with correct position offsets
        q = self.rope(q, offset=pos_offset)
        k = self.rope(k, offset=pos_offset)

        if past_kv is not None:
            # Past keys already have RoPE applied at their original positions
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = torch.stack([k, v])

        T_total = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        causal_mask = self.mask[:, :, T_total - T_new:T_total, :T_total]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        y = self.resid_drop(self.proj(y))
        return y, new_kv


class TransformerBlockRoPE(nn.Module):
    """TransformerBlock using RoPE attention."""

    def __init__(self, embed_dim=512, block_size=257, n_head=8, drop_out_rate=0.1, fc_rate=4, rope_base=10000.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttentionRoPE(embed_dim, block_size, n_head, drop_out_rate, rope_base=rope_base)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, past_kv=None, pos_offset=0):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, pos_offset=pos_offset)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, block_size, embed_dim):
        super().__init__()
        pe = torch.zeros(block_size, embed_dim)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MotionGPT(nn.Module):
    """Autoregressive GPT for motion token prediction (no text conditioning).

    Vocab: 0..num_vq-1 = codebook, num_vq = EOS, num_vq+1 = PAD
    Output logits: num_vq+1 classes (codes + EOS, PAD is ignored)
    """

    def __init__(self, num_vq=512, embed_dim=512, block_size=257,
                 num_layers=6, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.num_vq = num_vq
        self.block_size = block_size

        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)  # codes + EOS + PAD
        self.pos_enc = SinusoidalPositionEncoding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)  # codes + EOS

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx):
        """Forward pass. idx: (B, T) token indices. Returns logits (B, T, num_vq+1)."""
        b, t = idx.size()
        assert t <= self.block_size
        token_embeddings = self.tok_emb(idx)
        x = self.pos_enc(token_embeddings)
        x = self.drop(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def sample(self, start_tokens, max_len=256, temperature=1.0, top_k=None, categorical=True):
        """Autoregressive sampling.

        Args:
            start_tokens: (1, S) initial token sequence
            max_len: max generation length
            temperature: sampling temperature
            top_k: if set, only sample from top-k logits
            categorical: if True use categorical sampling, else greedy
        Returns:
            generated token sequence (1, L) excluding EOS
        """
        self.eval()
        xs = start_tokens.clone()
        for _ in range(max_len):
            if xs.size(1) >= self.block_size:
                break
            logits = self.forward(xs)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            if categorical:
                idx = Categorical(probs).sample().unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
            if idx.item() == self.num_vq:  # EOS
                break
            xs = torch.cat((xs, idx), dim=1)
        return xs
