"""
Autoresearch pretraining script — MLX port for Apple Silicon.
Single-file, targets M-series Macs with unified memory via Apple MLX.
Usage: uv run train.py
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
import resource
from dataclasses import dataclass, asdict

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model helpers
# ---------------------------------------------------------------------------

def rms_norm(x):
    """
    Functional RMSNorm — no learnable weight, mirrors F.rms_norm(x, (x.size(-1),)).
    Computes norm in float32 to avoid bfloat16 underflow, then casts back.
    """
    return x * mx.rsqrt(
        mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6
    ).astype(x.dtype)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """
    Rotary positional embeddings.
    x:   (B, T, H, D)
    cos: (1, T, 1, D//2)
    sin: (1, T, 1, D//2)
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def make_causal_window_mask(T, window_size, dtype=mx.bfloat16):
    """
    Additive attention mask: 0.0 for allowed positions, -inf for masked.

    Uses arange comparisons rather than triu/tril — guaranteed to exist in MLX
    and idiomatic for Metal-compiled kernels.

    window_size: (size, 0) tuple matching original format; size < 0 means full context.
    """
    win = window_size[0]
    i = mx.arange(T)[:, None]   # (T, 1) — query positions
    j = mx.arange(T)[None, :]   # (1, T) — key positions
    # Causal: future keys are forbidden
    mask = mx.where(j > i, float('-inf'), 0.0)
    # Sliding window: keys more than `win` steps back are forbidden
    if 0 < win < T:
        window_mask = mx.where(i - j > win, float('-inf'), 0.0)
        mask = mask + window_mask
    return mask.astype(dtype)


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim  = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0

        self.c_q    = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.c_k    = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v    = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.ve_gate_channels = 32
        if has_ve(layer_idx, config.n_layer):
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

    def __call__(self, x, ve, cos_sin, mask):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head,    self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix value embedding with input-dependent gate
        if ve is not None and self.ve_gate is not None:
            ve_r = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate[..., None] * ve_r

        # RoPE + QK-norm
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        # GQA: expand k/v to match n_head if using grouped-query attention
        if self.n_kv_head < self.n_head:
            r = self.n_head // self.n_kv_head
            # mx.repeat may not exist in all MLX versions — use concatenate as fallback
            k = mx.concatenate([k] * r, axis=2)
            v = mx.concatenate([v] * r, axis=2)

        # Scaled dot-product attention: (B, T, H, D) → (B, H, T, D) for matmul
        scale = self.head_dim ** -0.5
        q = q.transpose(0, 2, 1, 3)            # (B, H, T, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * scale   # (B, H, T, T)
        scores = scores + mask                            # additive causal+window mask

        # Upcast to float32 for numerically stable softmax, then cast back
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        y = scores @ v                                    # (B, H, T, D)

        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)   # (B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.relu(x) ** 2    # ReLU²
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def __call__(self, x, ve, cos_sin, mask):
        x = x + self.attn(rms_norm(x), ve, cos_sin, mask)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._window_sizes = self._compute_window_sizes(config)

        # Transformer body — plain list/dict; MLX auto-registers these
        self.wte         = nn.Embedding(config.vocab_size, config.n_embd)
        self.h           = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head     = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-layer scalars (trainable) — assigned as mx.array attributes
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas    = mx.full((config.n_layer,), 0.1)

        # Value embeddings (subset of layers, based on has_ve pattern)
        head_dim = config.n_embd // config.n_head
        kv_dim   = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }

        # Pre-computed RoPE buffers — _-prefixed → not registered as trainable params
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len * 10, head_dim)
        self._cos = cos   # (1, max_seq, 1, head_dim//2)
        self._sin = sin

        # Attention mask cache — keyed by (T, window) tuple
        self._mask_cache = {}

    def _get_mask(self, T, window_size):
        key = (T, window_size[0])
        if key not in self._mask_cache:
            self._mask_cache[key] = make_causal_window_mask(T, window_size)
        return self._mask_cache[key]

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = mx.arange(0, head_dim, 2).astype(mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t        = mx.arange(seq_len).astype(mx.float32)
        freqs    = mx.outer(t, inv_freq)       # (seq_len, head_dim//2)
        cos      = mx.cos(freqs).astype(mx.bfloat16)
        sin      = mx.sin(freqs).astype(mx.bfloat16)
        # Reshape for broadcasting: (1, T, 1, head_dim//2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern    = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_win   = config.sequence_len
        short_win  = long_win // 2
        char_map   = {"L": (long_win, 0), "S": (short_win, 0)}
        sizes = [char_map[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_win, 0)   # last layer always full context
        return sizes

    def init_weights(self):
        """Initialise all parameters — called once before training."""
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5

        # Embeddings
        self.wte.weight     = mx.random.normal((self.config.vocab_size, n_embd))
        self.lm_head.weight = mx.random.normal((self.config.vocab_size, n_embd)) * 0.001

        # Transformer blocks
        for block in self.h:
            block.attn.c_q.weight    = mx.random.uniform(low=-s, high=s, shape=block.attn.c_q.weight.shape)
            block.attn.c_k.weight    = mx.random.uniform(low=-s, high=s, shape=block.attn.c_k.weight.shape)
            block.attn.c_v.weight    = mx.random.uniform(low=-s, high=s, shape=block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros(block.attn.c_proj.weight.shape)
            block.mlp.c_fc.weight    = mx.random.uniform(low=-s, high=s, shape=block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight  = mx.zeros(block.mlp.c_proj.weight.shape)
            if block.attn.ve_gate is not None:
                # gate init: zeros → sigmoid(0)=0.5, scaled by 2 → 1.0 (neutral)
                block.attn.ve_gate.weight = mx.zeros(block.attn.ve_gate.weight.shape)

        # Per-layer scalars
        self.resid_lambdas = mx.ones((self.config.n_layer,))
        self.x0_lambdas    = mx.full((self.config.n_layer,), 0.1)

        # Value embedding weights
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(low=-s, high=s, shape=ve.weight.shape)

        # Cast embedding tables to bfloat16 (match original)
        self.wte.weight = self.wte.weight.astype(mx.bfloat16)
        for ve in self.value_embeds.values():
            ve.weight = ve.weight.astype(mx.bfloat16)

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward = 6× params)."""
        nparams = sum(p.size for _, p in tree_flatten(self.parameters()))
        # Exclude non-scaling params from FLOPs count
        excl = (self.wte.weight.size
                + sum(ve.weight.size for ve in self.value_embeds.values())
                + self.resid_lambdas.size + self.x0_lambdas.size)
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = sum(
            12 * h * q * (t if ws[0] < 0 else min(ws[0], t))
            for ws in self._window_sizes
        )
        return 6 * (nparams - excl) + attn_flops

    def num_scaling_params(self):
        wte             = self.wte.weight.size
        value_embeds    = sum(ve.weight.size for ve in self.value_embeds.values())
        lm_head         = self.lm_head.weight.size
        transformer_mat = sum(p.size for _, p in tree_flatten(
                              [b.parameters() for b in self.h]))
        scalars         = self.resid_lambdas.size + self.x0_lambdas.size
        total           = wte + value_embeds + lm_head + transformer_mat + scalars
        return dict(wte=wte, value_embeds=value_embeds, lm_head=lm_head,
                    transformer_matrices=transformer_mat, scalars=scalars, total=total)

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape
        cos_sin = (self._cos[:, :T], self._sin[:, :T])

        x = self.wte(idx)
        x = rms_norm(x)
        x0 = x
        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve   = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            mask = self._get_mask(T, self._window_sizes[i])
            x    = block(x, ve, cos_sin, mask)
        x = rms_norm(x)

        # Soft-cap logits to [-15, 15] — prevents large cross-entropy inputs
        logits = self.lm_head(x).astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)

        if targets is not None:
            # Cross-entropy loss; MLX's cross_entropy takes (logits, targets) with int targets
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            if reduction == 'mean':
                return mx.mean(loss)
            return loss   # 'none' → shape [B*T]

        return logits


# ---------------------------------------------------------------------------
# Optimizer: MuonAdamW
# ---------------------------------------------------------------------------

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461,  -22.48329292557795,   15.878769915207462),
    (4.042929935166739,  -2.808917465908714,    0.5000178451051316),
    (3.8916678022926607, -2.772484153217685,    0.5060648178503393),
    (3.285753657755655,  -2.3681294933425376,   0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081,   0.42323551169305323),
]


def _orthogonalize(G, ns_steps):
    """
    Approximate polar decomposition via Newton-Schulz iterations.
    Brings gradient matrix toward the orthogonal group.
    Operates in bfloat16 (matches original); caller handles float32 conversion.
    """
    X    = G.astype(mx.bfloat16)
    norm = mx.sqrt(mx.sum(X ** 2, axis=(-2, -1), keepdims=True))
    X    = X / (norm * 1.02 + 1e-6)
    for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
        if X.shape[-2] >= X.shape[-1]:    # tall matrix: X^T X is smaller
            A = X.swapaxes(-2, -1) @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
        else:                              # wide matrix: X X^T is smaller
            A = X @ X.swapaxes(-2, -1)
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


class MuonAdamW:
    """
    Hybrid optimizer: Muon for 2D weight matrices in transformer blocks,
    AdamW for embeddings, lm_head, and scalar parameters.

    Muon applies Nesterov momentum + polar decomposition (Polar Express) +
    NorMuon variance reduction + cautious weight decay.

    Unlike the original, matrices are processed one-by-one rather than
    stacked by shape — MLX's lazy evaluation naturally parallelises
    independent array ops on the Metal GPU.
    """

    def __init__(self, model,
                 unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                 scalar_lr=0.5, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        # Scale learning rates by 1/sqrt(model_dim / 768) — tuned at d=768
        d = model.config.n_embd
        lr_scale = (d / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({d}/768) = {lr_scale:.6f}")

        self._muon_group = dict(
            lr=matrix_lr, momentum=0.95, ns_steps=5,
            beta2=0.95, weight_decay=weight_decay,
            initial_lr=matrix_lr,
        )
        # Per-path AdamW configs; paths resolved at first update
        self._adamw_defaults = dict(
            lm_head      = dict(lr=unembedding_lr * lr_scale, betas=adam_betas,
                                eps=1e-10, weight_decay=0.0,
                                initial_lr=unembedding_lr * lr_scale),
            wte          = dict(lr=embedding_lr  * lr_scale, betas=adam_betas,
                                eps=1e-10, weight_decay=0.0,
                                initial_lr=embedding_lr  * lr_scale),
            value_embeds = dict(lr=embedding_lr  * lr_scale, betas=adam_betas,
                                eps=1e-10, weight_decay=0.0,
                                initial_lr=embedding_lr  * lr_scale),
            resid_lambdas= dict(lr=scalar_lr * 0.01, betas=adam_betas,
                                eps=1e-10, weight_decay=0.0,
                                initial_lr=scalar_lr * 0.01),
            x0_lambdas   = dict(lr=scalar_lr, betas=(0.96, 0.95),
                                eps=1e-10, weight_decay=0.0,
                                initial_lr=scalar_lr),
        )
        self.state = {}             # keyed by param path string
        self._muon_paths  = None    # populated on first update
        self._adamw_paths = None

    def _classify_params(self, flat_params):
        """
        Identify Muon vs AdamW parameters.
        Muon: 2D weight matrices inside transformer blocks, excluding
              embeddings and lm_head (which have sparse gradient structure).
        AdamW: everything else (embeddings, scalars, lm_head).
        """
        muon_paths  = set()
        adamw_paths = set()
        for path, param in flat_params.items():
            is_2d_matrix = (param.ndim == 2)
            is_embedding = (
                path.startswith("wte.")
                or "value_embeds" in path
                or path.startswith("lm_head.")
            )
            if is_2d_matrix and not is_embedding:
                muon_paths.add(path)
            else:
                adamw_paths.add(path)
        return muon_paths, adamw_paths

    def _adamw_group_for(self, path):
        """Return the AdamW hyperparameter dict for a given parameter path."""
        for key, cfg in self._adamw_defaults.items():
            if key in path:
                return cfg
        # Fallback: use x0_lambdas config (scalar-like lr)
        return self._adamw_defaults['x0_lambdas']

    def _adamw_step(self, path, param, grad):
        group = self._adamw_group_for(path)
        st    = self.state.setdefault(path, {})
        if 'step' not in st:
            st['step']       = 0
            st['exp_avg']    = mx.zeros_like(param)
            st['exp_avg_sq'] = mx.zeros_like(param)
        st['step'] += 1
        t        = st['step']
        b1, b2   = group['betas']
        lr, wd   = group['lr'], group['weight_decay']
        eps      = group['eps']

        param = param * (1.0 - lr * wd)
        st['exp_avg']    = (1 - b1) * grad      + b1 * st['exp_avg']
        st['exp_avg_sq'] = (1 - b2) * grad ** 2 + b2 * st['exp_avg_sq']
        bias1 = 1.0 - b1 ** t
        bias2 = 1.0 - b2 ** t
        denom = mx.sqrt(st['exp_avg_sq'] / bias2) + eps
        return param - (lr / bias1) * st['exp_avg'] / denom

    def _muon_step(self, path, param, grad):
        group = self._muon_group
        st    = self.state.setdefault(path, {})

        if 'momentum_buf' not in st:
            st['momentum_buf'] = mx.zeros_like(grad)
        if 'variance_buf' not in st:
            # Reduce along the longer dimension for NorMuon variance estimate
            red_dim = -1 if grad.shape[-2] >= grad.shape[-1] else -2
            vshape  = list(grad.shape)
            vshape[red_dim] = 1
            st['variance_buf'] = mx.zeros(vshape, dtype=mx.float32)
            st['red_dim']      = red_dim

        momentum = group['momentum']
        # Scale LR by sqrt(max(1, rows/cols)) — Muon convention for non-square matrices
        lr       = group['lr'] * max(1.0, grad.shape[-2] / grad.shape[-1]) ** 0.5
        wd       = group['weight_decay']
        beta2    = group['beta2']
        ns_steps = group['ns_steps']
        red_dim  = st['red_dim']

        # Step 1: Nesterov momentum
        st['momentum_buf'] = (1 - momentum) * st['momentum_buf'] + momentum * grad
        g = (1 - momentum) * grad + momentum * st['momentum_buf']

        # Step 2: Polar Express orthogonalization — approximate polar decomposition
        g = _orthogonalize(g, ns_steps)

        # Step 3: NorMuon variance reduction — adaptive per-row/col scaling
        g_f32       = g.astype(mx.float32)
        red_dim_sz  = grad.shape[red_dim]
        v_mean      = mx.mean(g_f32 ** 2, axis=red_dim, keepdims=True)
        v_norm      = mx.sqrt(mx.sum(v_mean) * red_dim_sz)

        st['variance_buf'] = (1 - beta2) * st['variance_buf'] + beta2 * v_mean
        step_size    = mx.rsqrt(mx.maximum(st['variance_buf'], 1e-10))
        scaled_sq    = (v_mean * red_dim_sz) * step_size.astype(mx.float32) ** 2
        v_norm_new   = mx.sqrt(mx.sum(scaled_sq))
        final_scale  = step_size * v_norm / mx.maximum(v_norm_new, 1e-10)
        g = (g_f32 * final_scale).astype(grad.dtype)

        # Step 4: Cautious weight decay + parameter update
        # Decay only where gradient and parameter have the same sign
        mask  = ((g * param) >= 0).astype(g.dtype)
        return param - lr * g - lr * wd * param * mask

    def update(self, model, grads):
        """Apply one optimizer step. Call mx.eval() after this."""
        flat_params = dict(tree_flatten(model.trainable_parameters()))
        flat_grads  = dict(tree_flatten(grads))

        # Classify on first call (shapes are stable after init)
        if self._muon_paths is None:
            self._muon_paths, self._adamw_paths = self._classify_params(flat_params)

        updated = {}
        for path, grad in flat_grads.items():
            if grad is None:
                continue
            param = flat_params[path]
            if path in self._muon_paths:
                updated[path] = self._muon_step(path, param, grad)
            else:
                updated[path] = self._adamw_step(path, param, grad)

        # Apply all updates to model parameters in bulk
        model.update(tree_unflatten(list(updated.items())))

    def set_lr_multiplier(self, lrm):
        """Scale all learning rates by lrm relative to their initial values."""
        self._muon_group['lr'] = self._muon_group['initial_lr'] * lrm
        for cfg in self._adamw_defaults.values():
            cfg['lr'] = cfg['initial_lr'] * lrm

    def set_muon_momentum(self, momentum):
        self._muon_group['momentum'] = momentum

    def set_muon_weight_decay(self, wd):
        self._muon_group['weight_decay'] = wd


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO  = 64        # model_dim = depth * ASPECT_RATIO
HEAD_DIM      = 128       # target head dimension for attention
WINDOW_PATTERN = "SSSL"   # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE  = 2**19  # ~524K tokens per optimizer step
EMBEDDING_LR      = 0.6   # learning rate for token embeddings (AdamW)
UNEMBEDDING_LR    = 0.004  # learning rate for lm_head (AdamW)
MATRIX_LR         = 0.04   # learning rate for weight matrices (Muon)
SCALAR_LR         = 0.5    # learning rate for per-layer scalars (AdamW)
WEIGHT_DECAY      = 0.2    # cautious weight decay for Muon
ADAM_BETAS        = (0.8, 0.95)
WARMUP_RATIO      = 0.0    # fraction of time budget for LR warmup
WARMDOWN_RATIO    = 0.5    # fraction of time budget for LR cooldown
FINAL_LR_FRAC     = 0.0    # final LR as fraction of initial

# Model size
DEPTH             = 8      # number of transformer layers
# Reduced from 128 (H100) — Apple Silicon has less compute throughput than H100
# despite having more unified memory. Increase if runs are stable.
DEVICE_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

# M-series theoretical bfloat16 peak throughput (approximate; varies by chip)
# Used for MFU reporting relative to Apple Silicon rather than H100.
# M4 Max ~= 38 TFLOPS bf16; adjust for your chip if desired.
APPLE_BF16_PEAK_FLOPS = 38.0e12

tokenizer  = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):
    base_dim  = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())  # materialise all weights before timing starts

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params          = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, (
    f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by "
    f"DEVICE_BATCH_SIZE × MAX_SEQ_LEN ({tokens_per_fwdbwd})"
)
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = MuonAdamW(
    model,
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    weight_decay=WEIGHT_DECAY,
)

train_loader       = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x_np, y_np, epoch = next(train_loader)   # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# LR / schedule helpers (unchanged from original)
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


# ---------------------------------------------------------------------------
# Autograd setup
# ---------------------------------------------------------------------------

def loss_fn(model, x, y):
    return model(x, y, reduction='mean')

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training    = time.time()
smooth_train_loss   = 0.0
total_training_time = 0.0
step                = 0

while True:
    # mx.eval() is the synchronization point — time after it completes
    # (mirrors torch.cuda.synchronize() placement in the original)
    t0 = time.time()

    # --- Gradient accumulation ---
    # MLX has no .backward() accumulation; we manually average gradients
    # across micro-steps by dividing each contribution by grad_accum_steps.
    accumulated_grads = None
    accumulated_loss  = mx.array(0.0)

    for _ in range(grad_accum_steps):
        x = mx.array(x_np)
        y = mx.array(y_np)
        loss, grads = loss_and_grad_fn(model, x, y)
        accumulated_loss = accumulated_loss + loss / grad_accum_steps
        flat_g = dict(tree_flatten(grads))
        if accumulated_grads is None:
            accumulated_grads = {k: v / grad_accum_steps for k, v in flat_g.items()}
        else:
            for k in accumulated_grads:
                accumulated_grads[k] = accumulated_grads[k] + flat_g[k] / grad_accum_steps
        # Prefetch next batch during accumulation
        x_np, y_np, epoch = next(train_loader)

    # --- Schedules ---
    progress         = min(total_training_time / TIME_BUDGET, 1.0)
    lrm              = get_lr_multiplier(progress)
    muon_momentum    = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    optimizer.set_lr_multiplier(lrm)
    optimizer.set_muon_momentum(muon_momentum)
    optimizer.set_muon_weight_decay(muon_weight_decay)

    # --- Optimizer step ---
    optimizer.update(model, tree_unflatten(list(accumulated_grads.items())))

    # --- Synchronise: force execution of the entire lazy computation graph ---
    # This is where Metal GPU work actually happens.
    mx.eval(model.parameters(), accumulated_loss)

    t1 = time.time()
    dt = t1 - t0

    train_loss_f = float(accumulated_loss)

    # Fast-fail: abort if loss is NaN or exploding
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    if step > 10:
        total_training_time += dt

    # --- Logging ---
    ema_beta           = 0.9
    smooth_train_loss  = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased           = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done           = 100 * progress
    tok_per_sec        = int(TOTAL_BATCH_SIZE / dt)
    mfu                = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / APPLE_BF16_PEAK_FLOPS
    remaining          = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased:.6f} | lrm: {lrm:.2f} | "
        f"dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    # GC management — Python GC can cause ~500ms stalls; freeze after first step
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Stop after time budget (skip warmup steps so compilation isn't included)
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()   # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# --- Final evaluation ---
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# --- Summary ---
t_end          = time.time()
startup_time   = t_start_training - t_start
steady_mfu     = (100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10)
                  / total_training_time / APPLE_BF16_PEAK_FLOPS
                  if total_training_time > 0 else 0)
# On macOS, ru_maxrss is in bytes (Linux: kilobytes)
peak_ram_mb    = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_ram_mb:      {peak_ram_mb:.1f}")
print(f"mfu_percent:      {steady_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
