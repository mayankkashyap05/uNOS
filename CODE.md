# 🗂️ Codebase

<!-- AUTO-GENERATED — do not edit by hand -->

| Field | Value |
| ----- | ----- |
| **Generated** | `2026-04-26 21:28:20` |
| **Source mode** | YAML config (`codebase.yaml`) |
| **Base directory** | `C:\Users\kashy\OneDrive\Documents\uNOS` |
| **Total files** | 6 |

## 📑 Table of Contents

1. [Project Structure](#-project-structure)
2. [File Contents](#-file-contents)
   - [configs/config.yaml](#configsconfigyaml)
   - [model/module.py](#modelmodulepy)
   - [model/nos.py](#modelnospy)
   - [model/__init__.py](#model--init--py)
   - [config_loader.py](#config-loaderpy)
   - [hpo_tuner.py](#hpo-tunerpy)

## 📁 Project Structure

```
.
├── configs
│   └── config.yaml
├── model
│   ├── module.py
│   ├── nos.py
│   └── __init__.py
├── config_loader.py
└── hpo_tuner.py
```

## 📄 File Contents

### `configs/config.yaml`

```yaml

data:
  data_path: "data/1h.csv"
  lookback_window: 512
  predict_window: 48
  max_context: 512
  clip: 5.0
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
  batch_size: 32
  log_interval: 50
  num_workers: 6
  seed: 42

  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001

  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1

  accumulation_steps: 1

  # Scheduler params (previously hardcoded)
  tokenizer_pct_start: 0.03
  tokenizer_div_factor: 10.0
  basemodel_pct_start: 0.03
  basemodel_div_factor: 10.0

  # Gradient clipping (previously hardcoded)
  tokenizer_max_grad_norm: 2.0
  basemodel_max_grad_norm: 3.0

  # Dropout overrides (optional — leave null to use pretrained values)
  ffn_dropout_p: null
  attn_dropout_p: null
  resid_dropout_p: null
  token_dropout_p: null

  # BSQ loss weight overrides (optional — leave null to use pretrained values)
  bsq_beta: null
  bsq_gamma0: null
  bsq_gamma: null
  bsq_zeta: null

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"
  exp_name: "test_1h"
  base_path: "finetuned"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "Nos_custom_finetune"
  description: "Custom finetune for 1h stock data"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false

device:
  use_cuda: true
  device_id: 0

# ── Hyperparameter Search Space ──────────────────────────────────
hpo:
  enabled: false                    # Set true to activate HPO
  n_trials: 30                      # Optuna trials
  direction: "minimize"             # minimize val_loss
  sampler: "tpe"                    # tpe | random | cmaes
  pruner: "median"                  # median | hyperband | none
  storage: null                     # optuna DB URI or null for in-memory
  study_name: "nos_finetune_hpo"

  # What to optimize
  optimize_tokenizer: true
  optimize_basemodel: true

  # Fast evaluation mode during HPO (reduced epochs)
  hpo_tokenizer_epochs: 5
  hpo_basemodel_epochs: 3

  search_space:
    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer_learning_rate:
      type: float
      low: 1.0e-5
      high: 5.0e-3
      log: true

    # ── Predictor ──────────────────────────────────────────────
    predictor_learning_rate:
      type: float
      low: 1.0e-7
      high: 1.0e-4
      log: true

    # ── Shared Optimizer ───────────────────────────────────────
    adam_weight_decay:
      type: float
      low: 0.001
      high: 0.3
      log: true

    adam_beta1:
      type: float
      low: 0.85
      high: 0.95
      log: false

    adam_beta2:
      type: float
      low: 0.90
      high: 0.999
      log: false

    # ── Batch & Accumulation ───────────────────────────────────
    batch_size:
      type: categorical
      choices: [16, 32, 64, 128]

    accumulation_steps:
      type: categorical
      choices: [1, 2, 4]

    # ── Scheduler ─────────────────────────────────────────────
    tokenizer_pct_start:
      type: float
      low: 0.01
      high: 0.15
      log: false

    basemodel_pct_start:
      type: float
      low: 0.01
      high: 0.15
      log: false

    tokenizer_div_factor:
      type: categorical
      choices: [5.0, 10.0, 25.0, 50.0]

    basemodel_div_factor:
      type: categorical
      choices: [5.0, 10.0, 25.0, 50.0]

    # ── Gradient Clipping ─────────────────────────────────────
    tokenizer_max_grad_norm:
      type: float
      low: 0.5
      high: 5.0
      log: false

    basemodel_max_grad_norm:
      type: float
      low: 0.5
      high: 5.0
      log: false

    # ── Dropout (finetuning regularization) ────────────────────
    ffn_dropout_p:
      type: float
      low: 0.0
      high: 0.4
      log: false

    attn_dropout_p:
      type: float
      low: 0.0
      high: 0.2
      log: false

    resid_dropout_p:
      type: float
      low: 0.0
      high: 0.3
      log: false

    token_dropout_p:
      type: float
      low: 0.0
      high: 0.2
      log: false

    # ── BSQ Loss Weights (careful tuning) ─────────────────────
    bsq_beta:
      type: float
      low: 0.01
      high: 0.2
      log: true

    bsq_gamma0:
      type: float
      low: 0.5
      high: 2.0
      log: false

    bsq_gamma:
      type: float
      low: 0.8
      high: 2.0
      log: false

    bsq_zeta:
      type: float
      low: 0.01
      high: 0.2
      log: true

    # ── Data Params ────────────────────────────────────────────
    clip:
      type: float
      low: 3.0
      high: 10.0
      log: false
```

---

### `model/module.py`

```python
import math

from einops import rearrange, reduce
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        zb = (zq + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)
        cnt = torch.scatter_reduce(torch.zeros(2 ** K, device=zq.device, dtype=zq.dtype),
                                   0,
                                   zi.flatten(),
                                   torch.ones_like(zi.flatten()).to(zq.dtype),
                                   'sum')
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-4):
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


class BinarySphericalQuantizer(nn.Module):
    def __init__(self, embed_dim, beta, gamma0, gamma, zeta,
                 input_format='bchw',
                 soft_entropy=True, group_size=9,
                 persample_entropy_compute='analytical',
                 cb_entropy_compute='group',
                 l2_norm=True,
                 inv_temperature=1):
        """
        Paper link: https://arxiv.org/pdf/2406.07548.pdf
        Here we use the official implementation of the BinarySphericalQuantizer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta  # loss weight for commit loss
        self.gamma0 = gamma0  # loss weight for entropy penalty
        self.gamma = gamma  # loss weight for entropy penalty
        self.zeta = zeta  # loss weight for entire entropy penalty
        self.input_format = input_format
        assert self.embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
        self.num_groups = self.embed_dim // group_size
        self.group_size = group_size
        assert persample_entropy_compute in ['group', 'analytical'], "persample_entropy_compute must be either 'group' or 'analytical'"
        assert cb_entropy_compute in ['group', 'nce'], "cb_entropy_compute must be either 'group' or 'nce'"
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature

        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer('group_basis', 2 ** torch.arange(group_size - 1, -1, -1))

        self.num_dimensions = 2 ** embed_dim
        self.bits_per_index = embed_dim

        # we only need to keep the codebook portion up to the group size
        # because we approximate the H loss with this subcode
        group_codes = torch.arange(2 ** self.group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer('group_codebook', group_codebook, persistent=False)

        self.soft_entropy = soft_entropy  # soft_entropy: Sec 3.2 of https://arxiv.org/pdf/1911.05894.pdf

    def quantize(self, z):
        assert z.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def forward(self, z):
        # if self.input_format == 'bchw':
        #     z = rearrange(z, 'b c h w -> b h w c')
        zq = self.quantize(z)

        indices = self.codes_to_indexes(zq.detach())
        group_indices = self.codes_to_group_indexes(zq.detach())
        if not self.training:
            used_codes = torch.unique(indices, return_counts=False)
        else:
            used_codes = None

        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.

        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb_by_sample = ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

        zq = zq * q_scale

        # commit loss
        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        # if self.input_format == 'bchw':
        #     zq = rearrange(zq, 'b h w c -> b c h w')

        return (
            zq,
            commit_loss + self.zeta * entropy_penalty / self.inv_temperature,
            {"H": cb_entropy, "used_codes": used_codes, "indices": indices, "group_indices": group_indices,
             "avg_prob": avg_prob}
        )

    def soft_entropy_loss(self, z):
        # if we divide the code in subgroups of size group_size, the codebook will be of size 2 ** group_size
        # the sub-code is the last group_size bits of the full code
        group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1)
        divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)

        # we calculate the distance between the divided_z and the codebook for each subgroup
        distance = - 2 * torch.einsum('... g c, d c ->... g d', divided_z, group_code_book)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)
        if self.persample_entropy_compute == 'analytical':
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, '... g d ->g d', 'mean')
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample):
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = - probs_per_dim * torch.log(probs_per_dim + 1e-8) - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
        persample_entropy = persample_entropy.sum(-1)
        return persample_entropy.mean()

    def codes_to_indexes(self, zhat):
        """Converts a `code` to an index in the codebook.
        Args:
            zhat: A tensor of shape (B, ..., C) containing the codes. must be in {-1, 1}
        """
        assert zhat.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}"
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat):
        """Converts a `code` to a list of indexes (in groups) in the codebook.
        Args:
            zhat: A tensor of shape (B, ..., C) containing the codes. must be in {-1, 1}
        """
        zhat_in_group = rearrange(zhat, 'b ... (g c) -> b ... g c', c=self.group_size)
        return ((zhat_in_group + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        """Inverse of `indexes_to_codes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(indices, self.basis), 2
        )
        return codes_non_centered * 2 - 1

    def group_indexes_to_codes(self, group_indices):
        """Inverse of `group_indexes_to_codes`."""
        group_indices = group_indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(group_indices, self.group_basis), 2
        )
        codes_non_centered = rearrange(codes_non_centered, 'b ... g c -> b ... (g c)')
        return codes_non_centered * 2 - 1

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def get_group_codebook_entry(self, group_indices):
        z_q = self.group_indexes_to_codes(group_indices)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        z_q = z_q * q_scale
        if self.input_format == 'bchw':
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q

    def get_codebook_entry(self, indices):
        z_q = self.indexes_to_codes(indices)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        z_q = z_q * q_scale
        if self.input_format == 'bchw':
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q


class BSQuantizer(nn.Module):

    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.codebook_dim = s1_bits + s2_bits
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size)

    def bits_to_indices(self, bits):
        bits = (bits >= 0).to(torch.long)
        indices = 2 ** torch.arange(
            0,
            bits.shape[-1],
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def forward(self, z, half=False):
        z = F.normalize(z, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(z)
        if half:
            q_pre = quantized[:, :, :self.s1_bits]
            q_post = quantized[:, :, self.s1_bits:]
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()

        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        cos, sin = self._update_cos_sin_cache(q, q.shape[-2])
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, training=True) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias

    if attn_mask is not None:
        attn_mask_bias = torch.zeros_like(attn_weight)
        if attn_mask.dtype == torch.bool:
            attn_mask_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_mask_bias += attn_mask
        attn_weight += attn_mask_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=training)
    return attn_weight @ value


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)  # [batch, n_heads, q_len, k_len]
        else:
            attn_mask = None

        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            is_causal=True,
            training=self.training
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_len, _ = query.shape
        _, seq_len, _ = key.shape

        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        is_causal_flag = self.training

        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            is_causal=is_causal_flag,
            training=self.training
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model=256):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits

        vocab_s1 = 2 ** s1_bits
        vocab_s2 = 2 ** s2_bits

        self.emb_s1 = nn.Embedding(vocab_s1, d_model)
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)
        self.d_model = d_model
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        nn.init.normal_(self.emb_s1.weight, mean=0, std=d_model ** -0.5)
        nn.init.normal_(self.emb_s2.weight, mean=0, std=d_model ** -0.5)

    def forward(self, token_ids):
        """Inputs:
        token_ids: [batch_size, seq_len] token ID
        Output: [batch_size, seq_len, d_model]
        """
        if isinstance(token_ids, tuple) or isinstance(token_ids, list):
            s1_ids, s2_ids = token_ids
        else:
            s1_ids, s2_ids = self.split_token(token_ids, self.s2_bits)
        s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
        s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
        return self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))


class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        """hidden_states: [batch, seq_len, d_model]
        sibling_embed: Embedding from another subtoken
        """
        attn_out = self.cross_attn(
            query=sibling_embed,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        return self.norm(hidden_states + attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x


class DualHead(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        super().__init__()
        self.vocab_s1 = 2 ** s1_bits
        self.vocab_s2 = 2 ** s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)

    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        if padding_mask is not None:
            valid_mask = (padding_mask == 0)
            s1_logits = s1_logits[valid_mask]
            s2_logits = s2_logits[valid_mask]
            s1_targets = s1_targets[valid_mask]
            s2_targets = s2_targets[valid_mask]
            ce_s1 = F.cross_entropy(s1_logits, s1_targets)
            ce_s2 = F.cross_entropy(s2_logits, s2_targets)
        else:
            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        ce_loss = (ce_s1 + ce_s2) / 2
        return ce_loss, ce_s1, ce_s2

    def forward(self, x):
        return self.proj_s1(x)

    def cond_forward(self, x2):
        return self.proj_s2(x2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, learn_pe):
        super(TemporalEmbedding, self).__init__()

        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if not learn_pe else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 0])
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 3])
        month_x = self.month_embed(x[:, :, 4])

        return hour_x + weekday_x + day_x + month_x + minute_x









```

---

### `model/nos.py`

```python
import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
import sys

from tqdm import trange

sys.path.append("../")
from model.module import *


class NosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    NosTokenizer module for tokenizing input data using a hybrid quantization approach.

    This tokenizer utilizes a combination of encoder and decoder Transformer blocks
    along with the Binary Spherical Quantization (BSQuantizer) to compress and decompress input data.

    Args:
           d_in (int): Input dimension.
           d_model (int): Model dimension.
           n_heads (int): Number of attention heads.
           ff_dim (int): Feed-forward dimension.
           n_enc_layers (int): Number of encoder layers.
           n_dec_layers (int): Number of decoder layers.
           ffn_dropout_p (float): Dropout probability for feed-forward networks.
           attn_dropout_p (float): Dropout probability for attention mechanisms.
           resid_dropout_p (float): Dropout probability for residual connections.
           s1_bits (int): Number of bits for the pre token in BSQuantizer.
           s2_bits (int): Number of bits for the post token in BSQuantizer.
           beta (float): Beta parameter for BSQuantizer.
           gamma0 (float): Gamma0 parameter for BSQuantizer.
           gamma (float): Gamma parameter for BSQuantizer.
           zeta (float): Zeta parameter for BSQuantizer.
           group_size (int): Group size parameter for BSQuantizer.

    """

    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers, ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):

        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits # Total dimension of the codebook after quantization
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder Transformer Blocks
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        # Decoder Transformer Blocks
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim) # Linear layer before quantization
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model) # Linear layer after quantization (pre part - s1 bits)
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model) # Linear layer after quantization (full codebook)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size) # BSQuantizer module

    def forward(self, x):
        """
        Forward pass of the NosTokenizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            tuple: A tuple containing:
                - tuple: (z_pre, z) - Reconstructed outputs from decoder with s1_bits and full codebook respectively,
                         both of shape (batch_size, seq_len, d_in).
                - torch.Tensor: bsq_loss - Loss from the BSQuantizer.
                - torch.Tensor: quantized - Quantized representation from BSQuantizer.
                - torch.Tensor: z_indices - Indices from the BSQuantizer.
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z) # (B, T, codebook)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits] # Extract the first part of quantized representation (s1_bits)
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        # Decoder layers (for pre part - s1 bits)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # Decoder layers (for full codebook)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        """
        Converts indices to bit representations and scales them.

        Args:
            x (torch.Tensor): Indices tensor.
            half (bool, optional): Whether to process only half of the codebook dimension. Defaults to False.

        Returns:
            torch.Tensor: Bit representation tensor.
        """
        if half:
            x1 = x[0] # Assuming x is a tuple of indices if half is True
            x2 = x[1]
            mask = 2 ** torch.arange(self.codebook_dim//2, device=x1.device, dtype=torch.long) # Create a mask for bit extraction
            x1 = (x1.unsqueeze(-1) & mask) != 0 # Extract bits for the first half
            x2 = (x2.unsqueeze(-1) & mask) != 0 # Extract bits for the second half
            x = torch.cat([x1, x2], dim=-1) # Concatenate the bit representations
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long) # Create a mask for bit extraction
            x = (x.unsqueeze(-1) & mask) != 0 # Extract bits

        x = x.float() * 2 - 1 # Convert boolean to bipolar (-1, 1)
        q_scale = 1. / (self.codebook_dim ** 0.5) # Scaling factor
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """
        Encodes the input data into quantized indices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).
            half (bool, optional): Whether to use half quantization in BSQuantizer. Defaults to False.

        Returns:
            torch.Tensor: Quantized indices from BSQuantizer.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z, half)
        return z_indices

    def decode(self, x, half=False):
        """
        Decodes quantized indices back to the input data space.

        Args:
            x (torch.Tensor): Quantized indices tensor.
            half (bool, optional): Whether the indices were generated with half quantization. Defaults to False.

        Returns:
            torch.Tensor: Reconstructed output tensor of shape (batch_size, seq_len, d_in).
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Nos(nn.Module, PyTorchModelHubMixin):
    """
    Nos Model.

    Args:
        s1_bits (int): Number of bits for pre tokens.
        s2_bits (int): Number of bits for post tokens.
        n_layers (int): Number of Transformer blocks.
        d_model (int): Dimension of the model's embeddings and hidden states.
        n_heads (int): Number of attention heads in the MultiheadAttention layers.
        ff_dim (int): Dimension of the feedforward network in the Transformer blocks.
        ffn_dropout_p (float): Dropout probability for the feedforward network.
        attn_dropout_p (float): Dropout probability for the attention layers.
        resid_dropout_p (float): Dropout probability for residual connections.
        token_dropout_p (float): Dropout probability for token embeddings.
        learn_te (bool): Whether to use learnable temporal embeddings.
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing for s1 decoding. Defaults to False.
            s1_targets (torch.Tensor, optional): Target s1 token IDs for teacher forcing. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - s2_logits: Logits for s2 token predictions, conditioned on s1. Shape: [batch_size, seq_len, s2_vocab_size]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask) # Dependency Aware Layer: Condition on s1 embeddings
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Decodes only the s1 tokens.

        This method performs a forward pass to predict only s1 tokens. It returns the s1 logits
        and the context representation from the Transformer, which can be used for subsequent s2 decoding.

        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - context: Context representation from the Transformer. Shape: [batch_size, seq_len, d_model]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """
        Decodes the s2 tokens, conditioned on the context and s1 tokens.

        This method decodes s2 tokens based on a pre-computed context representation (typically from `decode_s1`)
        and the s1 token IDs. It uses the dependency-aware layer and the conditional s2 head to predict s2 tokens.

        Args:
            context (torch.Tensor): Context representation from the transformer (output of decode_s1).
                                     Shape: [batch_size, seq_len, d_model]
            s1_ids (torch.torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            torch.Tensor: s2 logits. Shape: [batch_size, seq_len, s2_vocab_size]
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x


def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5, verbose=False):
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)

        def get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, pred_step):

            if current_seq_len <= max_context - pred_step:
                return torch.cat([x_stamp, y_stamp[:, :pred_step, :]], dim=1)
            else:
                start_idx = max_context - pred_step
                return torch.cat([x_stamp[:, -start_idx:, :], y_stamp[:, :pred_step, :]], dim=1)

        if verbose:
            ran = trange
        else:
            ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i

            if current_seq_len <= max_context:
                input_tokens = x_token
            else:
                input_tokens = [t[:, -max_context:].contiguous() for t in x_token]

            current_stamp = get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, i)

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            x_token[0] = torch.cat([x_token[0], sample_pre], dim=1)
            x_token[1] = torch.cat([x_token[1], sample_post], dim=1)

            torch.cuda.empty_cache()

        input_tokens = [t[:, -max_context:].contiguous() for t in x_token]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(batch_size, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        return preds


def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute
    time_df['hour'] = x_timestamp.dt.hour
    time_df['weekday'] = x_timestamp.dt.weekday
    time_df['day'] = x_timestamp.dt.day
    time_df['month'] = x_timestamp.dt.month
    return time_df


class NosPredictor:

    def __init__(self, model, tokenizer, device="cuda:0", max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        self.device = device

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):

        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len,
                                          self.clip, T, top_k, top_p, sample_count, verbose)
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0  # Fill missing volume with zeros
            df[self.amt_vol] = 0.0  # Fill missing amount with zeros
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp)
        return pred_df


    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        """
        Perform parallel (batch) prediction on multiple time series. All series must have the same historical length and prediction length (pred_len).

        Args:
            df_list (List[pd.DataFrame]): List of input DataFrames, each containing price columns and optional volume/amount columns.
            x_timestamp_list (List[pd.DatetimeIndex or Series]): List of timestamps corresponding to historical data, length should match the number of rows in each DataFrame.
            y_timestamp_list (List[pd.DatetimeIndex or Series]): List of future prediction timestamps, length should equal pred_len.
            pred_len (int): Number of prediction steps.
            T (float): Sampling temperature.
            top_k (int): Top-k filtering threshold.
            top_p (float): Top-p (nucleus sampling) threshold.
            sample_count (int): Number of parallel samples per series, automatically averaged internally.
            verbose (bool): Whether to display autoregressive progress.

        Returns:
            List[pd.DataFrame]: List of prediction results in the same order as input, each DataFrame contains
                                `open, high, low, close, volume, amount` columns, indexed by corresponding `y_timestamp`.
        """
        # Basic validation
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        # Require all series to have consistent historical and prediction lengths for batch processing
        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)           # (B, seq_len, feat)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32) # (B, seq_len, time_feat)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32) # (B, pred_len, time_feat)

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)
        # preds: (B, pred_len, feat)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(preds_i, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp_list[i])
            pred_dfs.append(pred_df)

        return pred_dfs


```

---

### `model/__init__.py`

```python
from .nos import NosTokenizer, Nos, NosPredictor

model_dict = {
    'nos_tokenizer': NosTokenizer,
    'nos': Nos,
    'nos_predictor': NosPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError



```

---

### `config_loader.py`

```python
import os
import yaml
from typing import Dict, Any


class ConfigLoader:

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        config = self._resolve_dynamic_paths(config)
        return config

    def _resolve_dynamic_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        exp_name = config.get('model_paths', {}).get('exp_name', '')
        if not exp_name:
            return config

        base_path = config.get('model_paths', {}).get('base_path', '')
        path_templates = {
            'base_save_path': f"{base_path}/{exp_name}",
            'finetuned_tokenizer': f"{base_path}/{exp_name}/tokenizer/best_model"
        }

        if 'model_paths' in config:
            for key, template in path_templates.items():
                if key in config['model_paths']:
                    current_value = config['model_paths'][key]
                    if current_value == "" or current_value is None:
                        config['model_paths'][key] = template
                    else:
                        if isinstance(current_value, str) and '{exp_name}' in current_value:
                            config['model_paths'][key] = current_value.format(exp_name=exp_name)

        return config

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})

    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})

    def get_model_paths(self) -> Dict[str, str]:
        return self.config.get('model_paths', {})

    def get_experiment_config(self) -> Dict[str, Any]:
        return self.config.get('experiment', {})

    def get_device_config(self) -> Dict[str, Any]:
        return self.config.get('device', {})

    def get_distributed_config(self) -> Dict[str, Any]:
        return self.config.get('distributed', {})

    def get_dynamic_tuning_config(self) -> Dict[str, Any]:
        return self.config.get('dynamic_tuning', {})

    def update_config(self, updates: Dict[str, Any]):
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        self.config = update_nested_dict(self.config, updates)

    def save_config(self, save_path: str = None):
        if save_path is None:
            save_path = self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False,
                      allow_unicode=True, indent=2)

    def print_config(self):
        print("=" * 50)
        print("Current configuration:")
        print("=" * 50)
        yaml.dump(self.config, default_flow_style=False,
                  allow_unicode=True, indent=2)
        print("=" * 50)


class CustomFinetuneConfig:

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        self.loader = ConfigLoader(config_path)
        self._load_all_configs()

    def _load_all_configs(self):
        # ── Data ──────────────────────────────────────────
        data_config = self.loader.get_data_config()
        self.data_path = data_config.get('data_path')
        self.lookback_window = data_config.get('lookback_window', 512)
        self.predict_window = data_config.get('predict_window', 48)
        self.max_context = data_config.get('max_context', 512)
        self.clip = data_config.get('clip', 5.0)
        self.train_ratio = data_config.get('train_ratio', 0.9)
        self.val_ratio = data_config.get('val_ratio', 0.1)
        self.test_ratio = data_config.get('test_ratio', 0.0)

        # ── Training ──────────────────────────────────────
        training_config = self.loader.get_training_config()

        self.tokenizer_epochs = training_config.get('tokenizer_epochs', 30)
        self.basemodel_epochs = training_config.get('basemodel_epochs', 30)
        if 'epochs' in training_config and 'tokenizer_epochs' not in training_config:
            self.tokenizer_epochs = training_config.get('epochs', 30)
        if 'epochs' in training_config and 'basemodel_epochs' not in training_config:
            self.basemodel_epochs = training_config.get('epochs', 30)

        self.batch_size = training_config.get('batch_size', 160)
        self.log_interval = training_config.get('log_interval', 50)
        self.num_workers = training_config.get('num_workers', 6)
        self.seed = training_config.get('seed', 100)

        # TIER 1 FIX 1: betas now loaded and passed to both optimizers
        self.adam_beta1 = training_config.get('adam_beta1', 0.9)
        self.adam_beta2 = training_config.get('adam_beta2', 0.95)
        self.adam_weight_decay = training_config.get('adam_weight_decay', 0.1)

        # TIER 2 FIX 7: Adam epsilon
        self.adam_eps = training_config.get('adam_eps', 1e-6)

        # Learning rates
        self.tokenizer_learning_rate = training_config.get('tokenizer_learning_rate', 2e-4)
        self.predictor_learning_rate = training_config.get('predictor_learning_rate', 4e-5)

        # TIER 1 FIX 2: Grad clipping from config
        self.tokenizer_grad_clip = training_config.get('tokenizer_grad_clip', 2.0)
        self.basemodel_grad_clip = training_config.get('basemodel_grad_clip', 3.0)
        self.grad_clip_norm_type = training_config.get('grad_clip_norm_type', 2.0)

        # TIER 1 FIX 3 & 12: Scheduler params from config
        self.scheduler_type = training_config.get('scheduler_type', 'cosine_warmup')
        self.scheduler_pct_start = training_config.get('scheduler_pct_start', 0.05)
        self.scheduler_div_factor = training_config.get('scheduler_div_factor', 25.0)
        self.scheduler_final_div_factor = training_config.get('scheduler_final_div_factor', 1000.0)

        # TIER 1 FIX 5: Loss weights from config
        self.tokenizer_recon_pre_weight = training_config.get('tokenizer_recon_pre_weight', 1.0)
        self.tokenizer_recon_all_weight = training_config.get('tokenizer_recon_all_weight', 1.0)
        self.tokenizer_bsq_weight = training_config.get('tokenizer_bsq_weight', 0.5)
        self.tokenizer_recon_weight = training_config.get('tokenizer_recon_weight', 0.5)
        self.basemodel_s1_weight = training_config.get('basemodel_s1_weight', 0.5)
        self.basemodel_s2_weight = training_config.get('basemodel_s2_weight', 0.5)

        # TIER 2 FIX 8: Label smoothing
        self.label_smoothing = training_config.get('label_smoothing', 0.0)

        # TIER 2 FIX 9: BSQ inverse temperature
        self.bsq_inv_temperature = training_config.get('bsq_inv_temperature', 1.0)

        # TIER 2 FIX 10: RoPE base
        self.rope_base = training_config.get('rope_base', 10000)

        self.accumulation_steps = training_config.get('accumulation_steps', 1)

        # ── Model paths ───────────────────────────────────
        model_paths = self.loader.get_model_paths()
        self.exp_name = model_paths.get('exp_name', 'default_experiment')
        self.pretrained_tokenizer_path = model_paths.get('pretrained_tokenizer')
        self.pretrained_predictor_path = model_paths.get('pretrained_predictor')
        self.base_save_path = model_paths.get('base_save_path') or ''
        self.tokenizer_save_name = model_paths.get('tokenizer_save_name', 'tokenizer')
        self.basemodel_save_name = model_paths.get('basemodel_save_name', 'basemodel')
        self.finetuned_tokenizer_path = model_paths.get('finetuned_tokenizer')

        # ── Experiment ────────────────────────────────────
        experiment_config = self.loader.get_experiment_config()
        self.experiment_name = experiment_config.get('name', 'Nos_custom_finetune')
        self.experiment_description = experiment_config.get('description', '')
        self.use_comet = experiment_config.get('use_comet', False)
        self.train_tokenizer = experiment_config.get('train_tokenizer', True)
        self.train_basemodel = experiment_config.get('train_basemodel', True)
        self.skip_existing = experiment_config.get('skip_existing', False)

        unified_pretrained = experiment_config.get('pre_trained', None)
        self.pre_trained_tokenizer = experiment_config.get(
            'pre_trained_tokenizer',
            unified_pretrained if unified_pretrained is not None else True)
        self.pre_trained_predictor = experiment_config.get(
            'pre_trained_predictor',
            unified_pretrained if unified_pretrained is not None else True)

        # ── Device ────────────────────────────────────────
        device_config = self.loader.get_device_config()
        self.use_cuda = device_config.get('use_cuda', True)
        self.device_id = device_config.get('device_id', 0)

        # ── Distributed ───────────────────────────────────
        distributed_config = self.loader.get_distributed_config()
        self.use_ddp = distributed_config.get('use_ddp', False)
        self.ddp_backend = distributed_config.get('backend', 'nccl')

        # ── Dynamic tuning ────────────────────────────────
        dt = self.loader.get_dynamic_tuning_config()
        self.dynamic_tuning_enabled = dt.get('enabled', False)
        self.dynamic_tuning_mode = dt.get('mode', 'moderate')

        self.tune_learning_rate = dt.get('tune_learning_rate', True)
        self.tune_grad_clip = dt.get('tune_grad_clip', True)
        self.tune_betas = dt.get('tune_betas', True)
        self.tune_loss_weights = dt.get('tune_loss_weights', True)
        self.tune_label_smoothing = dt.get('tune_label_smoothing', True)
        self.tune_bsq_temperature = dt.get('tune_bsq_temperature', True)
        self.tune_scheduler_params = dt.get('tune_scheduler_params', True)
        self.tune_weight_decay = dt.get('tune_weight_decay', True)
        self.tune_dropout = dt.get('tune_dropout', True)
        self.tune_adam_eps = dt.get('tune_adam_eps', True)

        # Bounds
        self.dt_lr_min = dt.get('lr_min', 1e-8)
        self.dt_lr_max = dt.get('lr_max', 1e-3)
        self.dt_beta1_min = dt.get('beta1_min', 0.85)
        self.dt_beta1_max = dt.get('beta1_max', 0.95)
        self.dt_beta2_min = dt.get('beta2_min', 0.90)
        self.dt_beta2_max = dt.get('beta2_max', 0.999)
        self.dt_grad_clip_min = dt.get('grad_clip_min', 0.5)
        self.dt_grad_clip_max = dt.get('grad_clip_max', 10.0)
        self.dt_loss_weight_min = dt.get('loss_weight_min', 0.1)
        self.dt_loss_weight_max = dt.get('loss_weight_max', 2.0)
        self.dt_label_smoothing_min = dt.get('label_smoothing_min', 0.0)
        self.dt_label_smoothing_max = dt.get('label_smoothing_max', 0.15)
        self.dt_bsq_inv_temp_min = dt.get('bsq_inv_temp_min', 0.5)
        self.dt_bsq_inv_temp_max = dt.get('bsq_inv_temp_max', 5.0)
        self.dt_weight_decay_min = dt.get('weight_decay_min', 0.0)
        self.dt_weight_decay_max = dt.get('weight_decay_max', 0.5)

        # Detection thresholds
        self.dt_plateau_window = dt.get('plateau_window_epochs', 4)
        self.dt_plateau_threshold = dt.get('plateau_threshold', 5e-4)
        self.dt_overfit_gap_ratio = dt.get('overfit_gap_ratio', 0.10)
        self.dt_severe_overfit_gap_ratio = dt.get('severe_overfit_gap_ratio', 0.30)
        self.dt_grad_explosion_threshold = dt.get('grad_explosion_threshold', 8.0)

        # Cooldowns
        self.dt_lr_cooldown = dt.get('lr_cooldown_epochs', 2)
        self.dt_beta_cooldown = dt.get('beta_cooldown_epochs', 4)
        self.dt_loss_weight_cooldown = dt.get('loss_weight_cooldown_epochs', 3)
        self.dt_label_smooth_cooldown = dt.get('label_smooth_cooldown_epochs', 4)
        self.dt_bsq_temp_cooldown = dt.get('bsq_temp_cooldown_epochs', 5)
        self.dt_grad_clip_cooldown = dt.get('grad_clip_cooldown_epochs', 1)
        self.dt_scheduler_swap_cooldown = dt.get('scheduler_swap_cooldown_epochs', 5)
        self.dt_tuning_warmup = dt.get('tuning_warmup_epochs', 2)

        self._compute_full_paths()

        # ── HPO ───────────────────────────────────────────
        hpo_config = self.loader.get('hpo', {})
        self.hpo_enabled = hpo_config.get('enabled', False)
        self.hpo_n_trials = hpo_config.get('n_trials', 10)
        self.hpo_direction = hpo_config.get('direction', 'minimize')
        self.hpo_sampler = hpo_config.get('sampler', 'tpe')
        self.hpo_pruner = hpo_config.get('pruner', 'median')
        self.hpo_storage = hpo_config.get('storage', None)
        self.hpo_study_name = hpo_config.get('study_name', 'nos_hpo')
        
        self.hpo_optimize_tokenizer = hpo_config.get('optimize_tokenizer', True)
        self.hpo_optimize_basemodel = hpo_config.get('optimize_basemodel', True)
        self.hpo_tokenizer_epochs = hpo_config.get('hpo_tokenizer_epochs', 5)
        self.hpo_basemodel_epochs = hpo_config.get('hpo_basemodel_epochs', 3)
        self.hpo_search_space = hpo_config.get('search_space', {})

    def _compute_full_paths(self):
        self.tokenizer_save_path = os.path.join(
            self.base_save_path, self.tokenizer_save_name)
        self.tokenizer_best_model_path = os.path.join(
            self.tokenizer_save_path, 'best_model')
        self.basemodel_save_path = os.path.join(
            self.base_save_path, self.basemodel_save_name)
        self.basemodel_best_model_path = os.path.join(
            self.basemodel_save_path, 'best_model')

    def clone_with_overrides(self, overrides: Dict[str, Any]):
        """Creates a new config instance with Optuna-suggested parameters."""
        import copy
        new_config = copy.copy(self)
        for k, v in overrides.items():
            setattr(new_config, k, v)
        return new_config    

    def get_tokenizer_config(self):
        return {
            'data_path': self.data_path,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.tokenizer_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'adam_eps': self.adam_eps,
            'accumulation_steps': self.accumulation_steps,
            'pretrained_model_path': self.pretrained_tokenizer_path,
            'save_path': self.tokenizer_save_path,
            'use_comet': self.use_comet,
            'tokenizer_grad_clip': self.tokenizer_grad_clip,
            'scheduler_type': self.scheduler_type,
            'scheduler_pct_start': self.scheduler_pct_start,
            'scheduler_div_factor': self.scheduler_div_factor,
            'scheduler_final_div_factor': self.scheduler_final_div_factor,
            'tokenizer_recon_pre_weight': self.tokenizer_recon_pre_weight,
            'tokenizer_recon_all_weight': self.tokenizer_recon_all_weight,
            'tokenizer_bsq_weight': self.tokenizer_bsq_weight,
            'tokenizer_recon_weight': self.tokenizer_recon_weight,
            'label_smoothing': self.label_smoothing,
            'bsq_inv_temperature': self.bsq_inv_temperature,
        }

    def get_basemodel_config(self):
        return {
            'data_path': self.data_path,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.basemodel_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'predictor_learning_rate': self.predictor_learning_rate,
            'tokenizer_learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'adam_eps': self.adam_eps,
            'pretrained_tokenizer_path': self.finetuned_tokenizer_path,
            'pretrained_predictor_path': self.pretrained_predictor_path,
            'save_path': self.basemodel_save_path,
            'use_comet': self.use_comet,
            'basemodel_grad_clip': self.basemodel_grad_clip,
            'scheduler_type': self.scheduler_type,
            'scheduler_pct_start': self.scheduler_pct_start,
            'scheduler_div_factor': self.scheduler_div_factor,
            'scheduler_final_div_factor': self.scheduler_final_div_factor,
            'basemodel_s1_weight': self.basemodel_s1_weight,
            'basemodel_s2_weight': self.basemodel_s2_weight,
            'label_smoothing': self.label_smoothing,
        }

    def print_config_summary(self):
        print("=" * 60)
        print("Nos finetuning configuration summary")
        print("=" * 60)
        print(f"Experiment name:          {self.exp_name}")
        print(f"Data path:                {self.data_path}")
        print(f"Lookback window:          {self.lookback_window}")
        print(f"Predict window:           {self.predict_window}")
        print(f"Tokenizer epochs:         {self.tokenizer_epochs}")
        print(f"Basemodel epochs:         {self.basemodel_epochs}")
        print(f"Batch size:               {self.batch_size}")
        print(f"Tokenizer LR:             {self.tokenizer_learning_rate}")
        print(f"Predictor LR:             {self.predictor_learning_rate}")
        print(f"Adam betas:               ({self.adam_beta1}, {self.adam_beta2})")
        print(f"Adam eps:                 {self.adam_eps}")
        print(f"Scheduler type:           {self.scheduler_type}")
        print(f"Scheduler pct_start:      {self.scheduler_pct_start}")
        print(f"Tokenizer grad clip:      {self.tokenizer_grad_clip}")
        print(f"Basemodel grad clip:      {self.basemodel_grad_clip}")
        print(f"Label smoothing:          {self.label_smoothing}")
        print(f"BSQ inv temperature:      {self.bsq_inv_temperature}")
        print(f"RoPE base:                {self.rope_base}")
        print(f"Dynamic tuning enabled:   {self.dynamic_tuning_enabled}")
        print(f"Dynamic tuning mode:      {self.dynamic_tuning_mode}")
        print(f"Base save path:           {self.base_save_path}")
        print("=" * 60)
```

---

### `hpo_tuner.py`

```python
# hpo_tuner.py
"""
Automatic Hyperparameter Tuning for Nos Model Finetuning.

Usage:
    python hpo_tuner.py --config configs/config_test_1h.yaml
    python hpo_tuner.py --config configs/config_test_1h.yaml --n-trials 50
    python hpo_tuner.py --config configs/config_test_1h.yaml --phase tokenizer
    python hpo_tuner.py --config configs/config_test_1h.yaml --phase basemodel
    python hpo_tuner.py --config configs/config_test_1h.yaml --apply-best
"""

import os
import sys
import json
import copy
import logging
import argparse
import time
import tempfile
import shutil
import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import numpy as np
import yaml

sys.path.append('../')

# ── Optional Optuna import ─────────────────────────────────────────────────
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: optuna not installed. Run: pip install optuna")

from config_loader import CustomFinetuneConfig
from model import Nos, NosTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Apply dropout overrides to a loaded model
# ══════════════════════════════════════════════════════════════════════════════

def apply_dropout_overrides(model: torch.nn.Module, config: CustomFinetuneConfig):
    """
    Walk through all submodules and override dropout values
    when config specifies non-None values.
    Works without changing architecture (weights untouched).
    """
    import torch.nn as nn

    ffn_dp = getattr(config, 'ffn_dropout_p', None)
    attn_dp = getattr(config, 'attn_dropout_p', None)
    resid_dp = getattr(config, 'resid_dropout_p', None)
    token_dp = getattr(config, 'token_dropout_p', None)

    for name, module in model.named_modules():
        # FeedForward ffn_dropout
        if ffn_dp is not None and hasattr(module, 'ffn_dropout'):
            module.ffn_dropout.p = ffn_dp

        # Residual dropout in attention
        if resid_dp is not None and hasattr(module, 'resid_dropout'):
            module.resid_dropout.p = resid_dp

        # Token dropout (only on Nos predictor)
        if token_dp is not None and hasattr(module, 'token_drop'):
            module.token_drop.p = token_dp

        # Attention dropout — stored as a float, not nn.Dropout
        if attn_dp is not None and hasattr(module, 'attn_dropout_p'):
            module.attn_dropout_p = attn_dp

    return model


def apply_bsq_overrides(tokenizer: NosTokenizer, config: CustomFinetuneConfig):
    """
    Override BSQ loss weights on a loaded tokenizer.
    Only modifies scalar attributes, not weight tensors.
    """
    bsq = tokenizer.tokenizer.bsq  # BinarySphericalQuantizer instance

    if getattr(config, 'bsq_beta', None) is not None:
        bsq.beta = config.bsq_beta
    if getattr(config, 'bsq_gamma0', None) is not None:
        bsq.gamma0 = config.bsq_gamma0
    if getattr(config, 'bsq_gamma', None) is not None:
        bsq.gamma = config.bsq_gamma
    if getattr(config, 'bsq_zeta', None) is not None:
        bsq.zeta = config.bsq_zeta

    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Search Space Builder
# ══════════════════════════════════════════════════════════════════════════════

class SearchSpaceBuilder:
    """
    Reads the `hpo.search_space` block from config and samples
    Optuna trial suggestions for each parameter.
    """

    PARAM_GROUPS = {
        'tokenizer': [
            'tokenizer_learning_rate',
            'tokenizer_pct_start',
            'tokenizer_div_factor',
            'tokenizer_max_grad_norm',
            'bsq_beta', 'bsq_gamma0', 'bsq_gamma', 'bsq_zeta',
        ],
        'basemodel': [
            'predictor_learning_rate',
            'basemodel_pct_start',
            'basemodel_div_factor',
            'basemodel_max_grad_norm',
            'ffn_dropout_p', 'attn_dropout_p',
            'resid_dropout_p', 'token_dropout_p',
        ],
        'shared': [
            'adam_weight_decay', 'adam_beta1', 'adam_beta2',
            'batch_size', 'accumulation_steps', 'clip',
        ],
    }

    def __init__(self, search_space: Dict[str, Any], phase: str = 'both'):
        self.search_space = search_space
        self.phase = phase  # 'tokenizer' | 'basemodel' | 'both'

    def _should_include(self, param_name: str) -> bool:
        if self.phase == 'both':
            return True
        shared = self.PARAM_GROUPS['shared']
        group = self.PARAM_GROUPS.get(self.phase, [])
        return param_name in group or param_name in shared

    def suggest(self, trial, param_name: str) -> Any:
        """Suggest a value for param_name using the trial object."""
        if param_name not in self.search_space:
            return None

        spec = self.search_space[param_name]
        ptype = spec['type']

        if ptype == 'float':
            return trial.suggest_float(
                param_name,
                spec['low'],
                spec['high'],
                log=spec.get('log', False)
            )
        elif ptype == 'int':
            return trial.suggest_int(
                param_name,
                spec['low'],
                spec['high'],
                log=spec.get('log', False)
            )
        elif ptype == 'categorical':
            return trial.suggest_categorical(param_name, spec['choices'])
        else:
            raise ValueError(f"Unknown search space type: {ptype}")

    def build_overrides(self, trial) -> Dict[str, Any]:
        """
        Returns a dict of {param_name: suggested_value} for all
        parameters in the search space relevant to this phase.
        """
        overrides = {}
        for param_name in self.search_space:
            if self._should_include(param_name):
                value = self.suggest(trial, param_name)
                if value is not None:
                    overrides[param_name] = value
        return overrides


# ══════════════════════════════════════════════════════════════════════════════
# Objective Functions
# ══════════════════════════════════════════════════════════════════════════════

class TokenizerObjective:
    """Optuna objective for tokenizer HPO."""

    def __init__(self, base_config: CustomFinetuneConfig,
                 search_space_builder: SearchSpaceBuilder,
                 device: torch.device,
                 trial_base_dir: str):
        self.base_config = base_config
        self.ssb = search_space_builder
        self.device = device
        self.trial_base_dir = trial_base_dir

    def __call__(self, trial) -> float:
        # ── 1. Sample hyperparameters ──────────────────────────────
        overrides = self.ssb.build_overrides(trial)

        # HPO uses reduced epochs for speed
        overrides['tokenizer_epochs'] = self.base_config.hpo_tokenizer_epochs
        overrides['log_interval'] = 999999  # suppress step-level logging

        # ── 2. Build trial config ──────────────────────────────────
        trial_config = self.base_config.clone_with_overrides(overrides)

        # Unique save dir per trial to avoid conflicts
        trial_dir = os.path.join(
            self.trial_base_dir, f"tokenizer_trial_{trial.number}"
        )
        trial_config.tokenizer_save_path = trial_dir
        os.makedirs(trial_dir, exist_ok=True)

        # ── 3. Load model (always from original pretrained) ────────
        try:
            if getattr(trial_config, 'pre_trained_tokenizer', True):
                tokenizer = NosTokenizer.from_pretrained(
                    self.base_config.pretrained_tokenizer_path
                )
            else:
                cfg_path = os.path.join(
                    self.base_config.pretrained_tokenizer_path, 'config.json'
                )
                with open(cfg_path, 'r') as f:
                    arch = json.load(f)
                tokenizer = NosTokenizer(**{k: arch[k] for k in [
                    'd_in', 'd_model', 'n_heads', 'ff_dim',
                    'n_enc_layers', 'n_dec_layers', 'ffn_dropout_p',
                    'attn_dropout_p', 'resid_dropout_p', 's1_bits',
                    's2_bits', 'beta', 'gamma0', 'gamma', 'zeta', 'group_size'
                ] if k in arch})

            # Apply BSQ and dropout overrides
            tokenizer = apply_bsq_overrides(tokenizer, trial_config)
            tokenizer = apply_dropout_overrides(tokenizer, trial_config)
            tokenizer = tokenizer.to(self.device)

        except Exception as e:
            print(f"Trial {trial.number}: Model loading failed: {e}")
            raise optuna.exceptions.TrialPruned()

        # ── 4. Run training ────────────────────────────────────────
        logger = _get_silent_logger(f"hpo_tok_trial_{trial.number}")

        try:
            from finetune_tokenizer import train_tokenizer
            val_loss = train_tokenizer(
                tokenizer, self.device, trial_config, trial_dir, logger
            )
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()
        finally:
            # ── Clean up trial artifacts to save disk ──────────────
            if os.path.exists(trial_dir):
                shutil.rmtree(trial_dir, ignore_errors=True)

        print(f"  Trial {trial.number}: val_loss={val_loss:.6f} | {overrides}")
        return val_loss


class BasemodelObjective:
    """Optuna objective for predictor/basemodel HPO."""

    def __init__(self, base_config: CustomFinetuneConfig,
                 search_space_builder: SearchSpaceBuilder,
                 device: torch.device,
                 trial_base_dir: str,
                 tokenizer_path: str):
        self.base_config = base_config
        self.ssb = search_space_builder
        self.device = device
        self.trial_base_dir = trial_base_dir
        self.tokenizer_path = tokenizer_path  # path to best finetuned tokenizer

    def __call__(self, trial) -> float:
        overrides = self.ssb.build_overrides(trial)
        overrides['basemodel_epochs'] = self.base_config.hpo_basemodel_epochs
        overrides['log_interval'] = 999999

        trial_config = self.base_config.clone_with_overrides(overrides)
        trial_config.finetuned_tokenizer_path = self.tokenizer_path

        trial_dir = os.path.join(
            self.trial_base_dir, f"basemodel_trial_{trial.number}"
        )
        trial_config.basemodel_save_path = trial_dir
        os.makedirs(trial_dir, exist_ok=True)

        try:
            # Load tokenizer (frozen during basemodel HPO)
            tokenizer = NosTokenizer.from_pretrained(self.tokenizer_path)
            tokenizer = tokenizer.to(self.device)
            tokenizer.eval()
            for p in tokenizer.parameters():
                p.requires_grad_(False)

            # Load predictor
            if getattr(trial_config, 'pre_trained_predictor', True):
                model = Nos.from_pretrained(
                    self.base_config.pretrained_predictor_path
                )
            else:
                cfg_path = os.path.join(
                    self.base_config.pretrained_predictor_path, 'config.json'
                )
                with open(cfg_path, 'r') as f:
                    arch = json.load(f)
                model = Nos(**{k: arch[k] for k in [
                    's1_bits', 's2_bits', 'n_layers', 'd_model', 'n_heads',
                    'ff_dim', 'ffn_dropout_p', 'attn_dropout_p',
                    'resid_dropout_p', 'token_dropout_p', 'learn_te'
                ] if k in arch})

            model = apply_dropout_overrides(model, trial_config)
            model = model.to(self.device)

        except Exception as e:
            print(f"Trial {trial.number}: Model loading failed: {e}")
            raise optuna.exceptions.TrialPruned()

        logger = _get_silent_logger(f"hpo_base_trial_{trial.number}")

        try:
            from finetune_base_model import train_model
            val_loss = train_model(
                model, tokenizer, self.device,
                trial_config, trial_dir, logger
            )
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()
        finally:
            if os.path.exists(trial_dir):
                shutil.rmtree(trial_dir, ignore_errors=True)

        print(f"  Trial {trial.number}: val_loss={val_loss:.6f} | {overrides}")
        return val_loss


# ══════════════════════════════════════════════════════════════════════════════
# HPO Runner
# ══════════════════════════════════════════════════════════════════════════════

class NosHPOTuner:
    """
    Main HPO orchestrator. Builds Optuna study, runs trials,
    reports best params, and optionally writes them back to config.
    """

    def __init__(self, config_path: str):
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Please install optuna: pip install optuna")

        self.config_path = config_path
        self.config = CustomFinetuneConfig(config_path)
        self.device = self._setup_device()

        self.trial_base_dir = os.path.join(
            self.config.base_save_path, "hpo_trials"
        )
        os.makedirs(self.trial_base_dir, exist_ok=True)

        # Results storage
        self.results: Dict[str, Any] = {}

    def _setup_device(self) -> torch.device:
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.config.device_id)
            return torch.device(f"cuda:{self.config.device_id}")
        return torch.device("cpu")

    def _build_sampler(self):
        name = self.config.hpo_sampler.lower()
        seed = self.config.seed
        if name == 'tpe':
            return TPESampler(seed=seed, multivariate=True)
        elif name == 'random':
            return RandomSampler(seed=seed)
        elif name == 'cmaes':
            return CmaEsSampler(seed=seed)
        else:
            return TPESampler(seed=seed)

    def _build_pruner(self):
        name = self.config.hpo_pruner.lower()
        if name == 'median':
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1,
                interval_steps=1
            )
        elif name == 'hyperband':
            return HyperbandPruner()
        else:
            return NopPruner()

    def _create_study(self, study_name_suffix: str = "") -> optuna.Study:
        study_name = f"{self.config.hpo_study_name}{study_name_suffix}"
        storage = self.config.hpo_storage

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=self.config.hpo_direction,
            sampler=self._build_sampler(),
            pruner=self._build_pruner(),
            load_if_exists=True
        )
        return study

    # ── Phase: Tokenizer ──────────────────────────────────────────────────

    def tune_tokenizer(self, n_trials: Optional[int] = None) -> Dict[str, Any]:
        n = n_trials or self.config.hpo_n_trials
        print(f"\n{'='*60}")
        print(f"HPO Phase 1: Tokenizer ({n} trials)")
        print(f"{'='*60}")

        ssb = SearchSpaceBuilder(
            self.config.hpo_search_space, phase='tokenizer'
        )
        objective = TokenizerObjective(
            self.config, ssb, self.device, self.trial_base_dir
        )

        study = self._create_study("_tokenizer")
        study.optimize(
            objective,
            n_trials=n,
            show_progress_bar=True,
            catch=(Exception,)
        )

        best = {
            'value': study.best_value,
            'params': study.best_params
        }
        self.results['tokenizer'] = best

        print(f"\n✅ Best tokenizer val_loss: {best['value']:.6f}")
        print(f"   Best params: {best['params']}")

        # Save study results
        self._save_study_results(study, "tokenizer")
        return best

    # ── Phase: Basemodel ──────────────────────────────────────────────────

    def tune_basemodel(self,
                       tokenizer_path: Optional[str] = None,
                       n_trials: Optional[int] = None) -> Dict[str, Any]:
        n = n_trials or self.config.hpo_n_trials

        # Determine tokenizer to use
        if tokenizer_path is None:
            tokenizer_path = self.config.finetuned_tokenizer_path
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at: {tokenizer_path}\n"
                f"Run tokenizer finetuning first."
            )

        print(f"\n{'='*60}")
        print(f"HPO Phase 2: Basemodel ({n} trials)")
        print(f"Using tokenizer: {tokenizer_path}")
        print(f"{'='*60}")

        ssb = SearchSpaceBuilder(
            self.config.hpo_search_space, phase='basemodel'
        )
        objective = BasemodelObjective(
            self.config, ssb, self.device,
            self.trial_base_dir, tokenizer_path
        )

        study = self._create_study("_basemodel")
        study.optimize(
            objective,
            n_trials=n,
            show_progress_bar=True,
            catch=(Exception,)
        )

        best = {
            'value': study.best_value,
            'params': study.best_params
        }
        self.results['basemodel'] = best

        print(f"\n✅ Best basemodel val_loss: {best['value']:.6f}")
        print(f"   Best params: {best['params']}")

        self._save_study_results(study, "basemodel")
        return best

    # ── Full Pipeline ─────────────────────────────────────────────────────

    def tune_full_pipeline(self) -> Dict[str, Any]:
        """
        Run HPO for tokenizer, then use best tokenizer params
        to run HPO for basemodel.
        """
        all_results = {}

        if self.config.hpo_optimize_tokenizer:
            tok_best = self.tune_tokenizer()
            all_results['tokenizer'] = tok_best

            # Train full tokenizer with best params before basemodel HPO
            print("\n📦 Training tokenizer with best params for basemodel HPO...")
            best_tok_config = self.config.clone_with_overrides(
                tok_best['params']
            )
            self._train_phase_with_config(best_tok_config, 'tokenizer')

        if self.config.hpo_optimize_basemodel:
            tok_path = self.config.tokenizer_best_model_path
            base_best = self.tune_basemodel(tokenizer_path=tok_path)
            all_results['basemodel'] = base_best

        self._print_final_report(all_results)
        return all_results

    def _train_phase_with_config(self, config: CustomFinetuneConfig,
                                  phase: str):
        """Run a full training phase (not HPO) with given config."""
        if phase == 'tokenizer':
            from finetune_tokenizer import train_tokenizer
            if getattr(config, 'pre_trained_tokenizer', True):
                tokenizer = NosTokenizer.from_pretrained(
                    config.pretrained_tokenizer_path
                )
            tokenizer = apply_bsq_overrides(tokenizer, config)
            tokenizer = apply_dropout_overrides(tokenizer, config)
            tokenizer = tokenizer.to(self.device)
            logger = _get_silent_logger("hpo_best_tokenizer")
            os.makedirs(config.tokenizer_save_path, exist_ok=True)
            train_tokenizer(
                tokenizer, self.device, config,
                config.tokenizer_save_path, logger
            )

    # ── Results & Reporting ───────────────────────────────────────────────

    def _save_study_results(self, study: optuna.Study, phase: str):
        results_dir = os.path.join(self.config.base_save_path, "hpo_results")
        os.makedirs(results_dir, exist_ok=True)

        # Save all trials as JSON
        trials_data = []
        for trial in study.trials:
            if trial.value is not None:
                trials_data.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': str(trial.state)
                })

        results_file = os.path.join(results_dir, f"{phase}_trials.json")
        with open(results_file, 'w') as f:
            json.dump({
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'trials': trials_data,
                'timestamp': datetime.datetime.now().isoformat()
            }, f, indent=2)

        print(f"📊 Results saved to: {results_file}")

        # Save importances plot if possible
        try:
            import optuna.visualization as vis
            fig = vis.plot_param_importances(study)
            fig_path = os.path.join(results_dir, f"{phase}_importances.html")
            fig.write_html(fig_path)

            fig2 = vis.plot_optimization_history(study)
            fig2_path = os.path.join(
                results_dir, f"{phase}_history.html"
            )
            fig2.write_html(fig2_path)
            print(f"📈 Plots saved to: {results_dir}")
        except Exception:
            pass

    def _print_final_report(self, results: Dict[str, Any]):
        print(f"\n{'='*60}")
        print("HPO FINAL REPORT")
        print(f"{'='*60}")

        for phase, result in results.items():
            print(f"\n{'─'*40}")
            print(f"Phase: {phase.upper()}")
            print(f"  Best val loss: {result['value']:.6f}")
            print(f"  Best hyperparameters:")
            for k, v in result['params'].items():
                print(f"    {k}: {v}")

        print(f"\n{'='*60}")

    def apply_best_to_config(self, output_config_path: Optional[str] = None):
        """
        Write best found hyperparameters back into the YAML config file.
        Creates a new config file by default.
        """
        if not self.results:
            print("No HPO results to apply. Run tune_* methods first.")
            return

        # Load raw YAML
        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Merge best params into training section
        training_updates = {}
        for phase_results in self.results.values():
            training_updates.update(phase_results.get('params', {}))

        for k, v in training_updates.items():
            raw_config.setdefault('training', {})[k] = v

        # Determine output path
        if output_config_path is None:
            base, ext = os.path.splitext(self.config_path)
            output_config_path = f"{base}_hpo_best{ext}"

        with open(output_config_path, 'w') as f:
            yaml.dump(raw_config, f, default_flow_style=False,
                      allow_unicode=True, indent=2)

        print(f"\n✅ Best config written to: {output_config_path}")
        return output_config_path

    def print_importance_report(self):
        """
        Print a ranked list of hyperparameter importances
        using Optuna's fANOVA estimator (if available).
        """
        if not OPTUNA_AVAILABLE:
            return

        print(f"\n{'='*60}")
        print("HYPERPARAMETER IMPORTANCE RANKING")
        print(f"{'='*60}")

        for phase in ['tokenizer', 'basemodel']:
            study_name = f"{self.config.hpo_study_name}_{phase}"
            try:
                storage = self.config.hpo_storage
                if storage:
                    study = optuna.load_study(
                        study_name=study_name, storage=storage
                    )
                    importances = optuna.importance.get_param_importances(study)
                    print(f"\n{phase.upper()} importances:")
                    for param, imp in sorted(
                        importances.items(),
                        key=lambda x: x[1], reverse=True
                    ):
                        bar = '█' * int(imp * 30)
                        print(f"  {param:<40} {imp:.3f} {bar}")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def _get_silent_logger(name: str) -> logging.Logger:
    """Logger that only writes to a temp file (no console spam during HPO)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        tmp_log = os.path.join(tempfile.gettempdir(), f"{name}.log")
        handler = logging.FileHandler(tmp_log, encoding='utf-8')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Automatic Hyperparameter Tuning for Nos Finetuning'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config_test_1h.yaml',
        help='Path to config YAML'
    )
    parser.add_argument(
        '--phase', type=str, default='both',
        choices=['tokenizer', 'basemodel', 'both'],
        help='Which phase to tune'
    )
    parser.add_argument(
        '--n-trials', type=int, default=None,
        help='Override number of trials from config'
    )
    parser.add_argument(
        '--apply-best', action='store_true',
        help='Write best params back to a new config file after tuning'
    )
    parser.add_argument(
        '--tokenizer-path', type=str, default=None,
        help='Path to finetuned tokenizer (for basemodel-only HPO)'
    )
    parser.add_argument(
        '--output-config', type=str, default=None,
        help='Output path for best config (default: original_name_hpo_best.yaml)'
    )
    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("ERROR: Install optuna first: pip install optuna plotly")
        sys.exit(1)

    tuner = NosHPOTuner(args.config)

    print(f"HPO Configuration:")
    print(f"  Phase: {args.phase}")
    print(f"  Trials: {args.n_trials or tuner.config.hpo_n_trials}")
    print(f"  Sampler: {tuner.config.hpo_sampler}")
    print(f"  Pruner: {tuner.config.hpo_pruner}")
    print(f"  Device: {tuner.device}")

    start = time.time()

    if args.phase == 'both':
        results = tuner.tune_full_pipeline()
    elif args.phase == 'tokenizer':
        results = tuner.tune_tokenizer(n_trials=args.n_trials)
    elif args.phase == 'basemodel':
        results = tuner.tune_basemodel(
            tokenizer_path=args.tokenizer_path,
            n_trials=args.n_trials
        )

    elapsed = time.time() - start
    print(f"\n⏱  Total HPO time: {elapsed/60:.1f} minutes")

    if args.apply_best:
        tuner.apply_best_to_config(args.output_config)

    tuner.print_importance_report()


if __name__ == "__main__":
    main()
```

---
