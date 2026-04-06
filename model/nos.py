import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
import sys
from tqdm import trange

sys.path.append("../")
from model.module import (
    BSQuantizer,
    RMSNorm,
    HierarchicalEmbedding,
    TemporalEmbedding,
    TransformerBlock,
    DependencyAwareLayer,
    DualHead,
    top_k_top_p_filtering,
    sample_from_logits,
)


class NosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    NosTokenizer: encodes time-series into discrete tokens
    via BSQ quantization, and decodes them back.
    """

    def __init__(self, d_in, d_model, n_heads, ff_dim,
                 n_enc_layers, n_dec_layers,
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p,
                 s1_bits, s2_bits,
                 beta, gamma0, gamma, zeta, group_size,
                 inv_temperature=1.0,   # TIER 2 FIX 9
                 rope_base=10000):      # TIER 2 FIX 10
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
        self.codebook_dim = s1_bits + s2_bits
        self.inv_temperature = inv_temperature
        self.rope_base = rope_base

        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        self.encoder = nn.ModuleList([
            TransformerBlock(
                self.d_model, self.n_heads, self.ff_dim,
                self.ffn_dropout_p, self.attn_dropout_p,
                self.resid_dropout_p,
                rope_base=rope_base)
            for _ in range(self.enc_layers - 1)
        ])

        self.decoder = nn.ModuleList([
            TransformerBlock(
                self.d_model, self.n_heads, self.ff_dim,
                self.ffn_dropout_p, self.attn_dropout_p,
                self.resid_dropout_p,
                rope_base=rope_base)
            for _ in range(self.dec_layers - 1)
        ])

        self.quant_embed = nn.Linear(
            self.d_model, self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(
            self.s1_bits, self.d_model)
        self.post_quant_embed = nn.Linear(
            self.codebook_dim, self.d_model)

        self.tokenizer = BSQuantizer(
            self.s1_bits, self.s2_bits,
            beta, gamma0, gamma, zeta, group_size,
            inv_temperature=inv_temperature)

    def forward(self, x):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        z_full = self.post_quant_embed(quantized)

        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        for layer in self.decoder:
            z_full = layer(z_full)
        z_full = self.head(z_full)

        return (z_pre, z_full), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        if half:
            x1, x2 = x[0], x[1]
            mask = 2 ** torch.arange(
                self.codebook_dim // 2,
                device=x1.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask) != 0
            x2 = (x2.unsqueeze(-1) & mask) != 0
            x_bits = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(
                self.codebook_dim,
                device=x.device, dtype=torch.long)
            x_bits = (x.unsqueeze(-1) & mask) != 0

        x_bits = x_bits.float() * 2 - 1
        q_scale = 1.0 / (self.codebook_dim ** 0.5)
        return x_bits * q_scale

    def encode(self, x, half=False):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        _, _, z_indices = self.tokenizer(z, half)
        return z_indices

    def decode(self, x, half=False):
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        return self.head(z)


class Nos(nn.Module, PyTorchModelHubMixin):
    """
    Nos: autoregressive predictor over BSQ token sequences.
    """

    def __init__(self, s1_bits, s2_bits, n_layers,
                 d_model, n_heads, ff_dim,
                 ffn_dropout_p, attn_dropout_p,
                 resid_dropout_p, token_dropout_p,
                 learn_te,
                 rope_base=10000,          # TIER 2 FIX 10
                 label_smoothing=0.0,      # TIER 2 FIX 8
                 dep_layer_n_heads=None):  # TIER 1 FIX 6
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
        self.rope_base = rope_base
        self.label_smoothing = label_smoothing

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(
            self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(
            self.d_model, self.learn_te)

        self.transformer = nn.ModuleList([
            TransformerBlock(
                self.d_model, self.n_heads, self.ff_dim,
                self.ffn_dropout_p, self.attn_dropout_p,
                self.resid_dropout_p,
                rope_base=rope_base)
            for _ in range(self.n_layers)
        ])

        self.norm = RMSNorm(self.d_model)

        # TIER 1 FIX 6: use model n_heads for dep_layer
        _dep_heads = (dep_layer_n_heads
                      if dep_layer_n_heads is not None
                      else n_heads)
        self.dep_layer = DependencyAwareLayer(
            self.d_model,
            n_heads=_dep_heads,
            rope_base=rope_base)

        # TIER 2 FIX 8: label_smoothing in DualHead
        self.head = DualHead(
            self.s1_bits, self.s2_bits, self.d_model,
            label_smoothing=label_smoothing)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0,
                            std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None,
                padding_mask=None,
                use_teacher_forcing=False,
                s1_targets=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)

        s1_logits = self.head(x)

        if use_teacher_forcing and s1_targets is not None:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1 = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1
            ).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1)

        x2 = self.dep_layer(
            x, sibling_embed,
            key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None,
                  padding_mask=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        return self.head(x), x

    def decode_s2(self, context, s1_ids,
                  padding_mask=None):
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(
            context, sibling_embed,
            key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


# ── Autoregressive inference ──────────────────────────────────

def auto_regressive_inference(tokenizer, model, x, x_stamp,
                               y_stamp, max_context, pred_len,
                               clip=5, T=1.0, top_k=0,
                               top_p=0.99, sample_count=5,
                               verbose=False):
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        x = torch.clip(x, -clip, clip)
        device = x.device

        x = (x.unsqueeze(1)
             .repeat(1, sample_count, 1, 1)
             .reshape(-1, x.size(1), x.size(2))
             .to(device))
        x_stamp = (x_stamp.unsqueeze(1)
                   .repeat(1, sample_count, 1, 1)
                   .reshape(-1, x_stamp.size(1), x_stamp.size(2))
                   .to(device))
        y_stamp = (y_stamp.unsqueeze(1)
                   .repeat(1, sample_count, 1, 1)
                   .reshape(-1, y_stamp.size(1), y_stamp.size(2))
                   .to(device))

        x_token = tokenizer.encode(x, half=True)

        def get_dynamic_stamp(xs, ys, cur_len, step):
            if cur_len <= max_context - step:
                return torch.cat(
                    [xs, ys[:, :step, :]], dim=1)
            start = max_context - step
            return torch.cat(
                [xs[:, -start:, :], ys[:, :step, :]], dim=1)

        ran = trange if verbose else range
        for i in ran(pred_len):
            cur_len = initial_seq_len + i
            if cur_len <= max_context:
                inp = x_token
            else:
                inp = [t[:, -max_context:].contiguous()
                       for t in x_token]

            stamp = get_dynamic_stamp(
                x_stamp, y_stamp, cur_len, i)

            s1_logits, ctx = model.decode_s1(
                inp[0], inp[1], stamp)
            s1_logits = s1_logits[:, -1, :]
            samp_pre = sample_from_logits(
                s1_logits, temperature=T,
                top_k=top_k, top_p=top_p,
                sample_logits=True)

            s2_logits = model.decode_s2(ctx, samp_pre)
            s2_logits = s2_logits[:, -1, :]
            samp_post = sample_from_logits(
                s2_logits, temperature=T,
                top_k=top_k, top_p=top_p,
                sample_logits=True)

            x_token[0] = torch.cat(
                [x_token[0], samp_pre], dim=1)
            x_token[1] = torch.cat(
                [x_token[1], samp_post], dim=1)
            torch.cuda.empty_cache()

        final_inp = [t[:, -max_context:].contiguous()
                     for t in x_token]
        z = tokenizer.decode(final_inp, half=True)
        z = z.reshape(
            batch_size, sample_count,
            z.size(1), z.size(2))
        preds = np.mean(z.cpu().numpy(), axis=1)
        return preds


def calc_time_stamps(x_timestamp):
    df = pd.DataFrame()
    df['minute'] = x_timestamp.dt.minute
    df['hour'] = x_timestamp.dt.hour
    df['weekday'] = x_timestamp.dt.weekday
    df['day'] = x_timestamp.dt.day
    df['month'] = x_timestamp.dt.month
    return df


class NosPredictor:
    def __init__(self, model, tokenizer,
                 device="cuda:0",
                 max_context=512, clip=5):
        self.tokenizer = tokenizer.to(device)
        self.model = model.to(device)
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'
        self.device = device

    def generate(self, x, x_stamp, y_stamp, pred_len,
                 T, top_k, top_p, sample_count, verbose):
        xt = torch.from_numpy(
            np.array(x).astype(np.float32)).to(self.device)
        xst = torch.from_numpy(
            np.array(x_stamp).astype(np.float32)).to(self.device)
        yst = torch.from_numpy(
            np.array(y_stamp).astype(np.float32)).to(self.device)
        preds = auto_regressive_inference(
            self.tokenizer, self.model,
            xt, xst, yst,
            self.max_context, pred_len,
            self.clip, T, top_k, top_p,
            sample_count, verbose)
        return preds[:, -pred_len:, :]

    def predict(self, df, x_timestamp, y_timestamp,
                pred_len, T=1.0, top_k=0, top_p=0.9,
                sample_count=1, verbose=True):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a DataFrame.")
        cols = self.price_cols + [self.vol_col, self.amt_vol]
        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns:
            df[self.amt_vol] = (
                df[self.vol_col]
                * df[self.price_cols].mean(axis=1))

        x_time = calc_time_stamps(x_timestamp)
        y_time = calc_time_stamps(y_timestamp)
        x = df[cols].values.astype(np.float32)
        x_stamp = x_time.values.astype(np.float32)
        y_stamp = y_time.values.astype(np.float32)

        xm = np.mean(x, axis=0)
        xs = np.std(x, axis=0)
        x = np.clip((x - xm) / (xs + 1e-5),
                    -self.clip, self.clip)

        preds = self.generate(
            x[np.newaxis], x_stamp[np.newaxis],
            y_stamp[np.newaxis], pred_len,
            T, top_k, top_p,
            sample_count, verbose).squeeze(0)
        preds = preds * (xs + 1e-5) + xm
        return pd.DataFrame(
            preds, columns=cols, index=y_timestamp)

    def predict_batch(self, df_list, x_ts_list, y_ts_list,
                      pred_len, T=1.0, top_k=0, top_p=0.9,
                      sample_count=1, verbose=True):
        if len(df_list) != len(x_ts_list) != len(y_ts_list):
            raise ValueError("Lists must have same length.")
        cols = self.price_cols + [self.vol_col, self.amt_vol]
        x_list, xs_list, ys_list = [], [], []
        means, stds = [], []

        for i, (df, xts, yts) in enumerate(
                zip(df_list, x_ts_list, y_ts_list)):
            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            x = df[cols].values.astype(np.float32)
            xm = np.mean(x, axis=0)
            xs = np.std(x, axis=0)
            x = np.clip((x - xm) / (xs + 1e-5),
                        -self.clip, self.clip)
            x_list.append(x)
            xs_list.append(
                calc_time_stamps(xts).values.astype(np.float32))
            ys_list.append(
                calc_time_stamps(yts).values.astype(np.float32))
            means.append(xm)
            stds.append(xs)

        x_batch = np.stack(x_list).astype(np.float32)
        xs_batch = np.stack(xs_list).astype(np.float32)
        ys_batch = np.stack(ys_list).astype(np.float32)

        preds = self.generate(
            x_batch, xs_batch, ys_batch, pred_len,
            T, top_k, top_p, sample_count, verbose)

        return [
            pd.DataFrame(
                preds[i] * (stds[i] + 1e-5) + means[i],
                columns=cols, index=y_ts_list[i])
            for i in range(len(df_list))
        ]