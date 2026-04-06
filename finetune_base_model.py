import os
import sys
import json
import math
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from logging.handlers import RotatingFileHandler
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from model import Nos, NosTokenizer, NosPredictor
from config_loader import CustomFinetuneConfig


class CustomKlineDataset(Dataset):
    def __init__(self, data_path, data_type='train',
                 lookback_window=90, predict_window=10,
                 clip=5.0, seed=100, train_ratio=0.7,
                 val_ratio=0.15, test_ratio=0.15):
        self.data_path = data_path
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.feature_list = [
            'open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = [
            'minute', 'hour', 'weekday', 'day', 'month']
        self.py_rng = random.Random(seed)
        self._load_and_preprocess_data()
        self._split_data_by_time()
        self.n_samples = len(self.data) - self.window + 1
        print(f"[{data_type.upper()}] len={len(self.data)}, "
              f"samples={self.n_samples}")

    def _load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)
        self.timestamps = df['timestamps'].copy()
        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month
        self.data = df[self.feature_list
                       + self.time_feature_list].copy()
        if self.data.isnull().any().any():
            self.data = self.data.fillna(method='ffill')

    def _split_data_by_time(self):
        n = len(self.data)
        te = int(n * self.train_ratio)
        ve = int(n * (self.train_ratio + self.val_ratio))
        if self.data_type == 'train':
            self.data = self.data.iloc[:te].copy()
            self.timestamps = self.timestamps.iloc[:te].copy()
        elif self.data_type == 'val':
            self.data = self.data.iloc[te:ve].copy()
            self.timestamps = self.timestamps.iloc[te:ve].copy()
        elif self.data_type == 'test':
            self.data = self.data.iloc[ve:].copy()
            self.timestamps = self.timestamps.iloc[ve:].copy()
        print(f"[{self.data_type.upper()}] "
              f"after split: {len(self.data)} records")

    def set_epoch_seed(self, epoch):
        self.py_rng.seed(self.seed + epoch)
        self.current_epoch = epoch

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        max_start = len(self.data) - self.window
        if max_start <= 0:
            raise ValueError("Data too short")
        if self.data_type == 'train':
            epoch = getattr(self, 'current_epoch', 0)
            start_idx = ((idx * 9973 + (epoch + 1) * 104729)
                         % (max_start + 1))
        else:
            start_idx = idx % (max_start + 1)
        end_idx = start_idx + self.window
        wd = self.data.iloc[start_idx:end_idx]
        x = wd[self.feature_list].values.astype(np.float32)
        xs = wd[self.time_feature_list].values.astype(np.float32)
        xm, xstd = np.mean(x, axis=0), np.std(x, axis=0)
        x = np.clip((x - xm) / (xstd + 1e-5),
                    -self.clip, self.clip)
        return torch.from_numpy(x), torch.from_numpy(xs)


def setup_logging(exp_name: str, log_dir: str,
                  rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(
        f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    log_file = os.path.join(
        log_dir, f"basemodel_training_rank_{rank}.log")
    fh = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024,
        backupCount=5, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = None
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    if ch:
        ch.setFormatter(fmt)
    logger.addHandler(fh)
    if ch:
        logger.addHandler(ch)
    logger.info("=== Basemodel Training Started ===")
    logger.info(f"Experiment: {exp_name} | Rank: {rank}")
    return logger


def create_dataloaders(config):
    rank0 = (not dist.is_available()
             or not dist.is_initialized()
             or dist.get_rank() == 0)
    if rank0:
        print("Creating basemodel dataloaders...")

    train_ds = CustomKlineDataset(
        data_path=config.data_path, data_type='train',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip, seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio)

    val_ds = CustomKlineDataset(
        data_path=config.data_path, data_type='val',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip, seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio)

    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = (
        DistributedSampler(train_ds,
                           num_replicas=dist.get_world_size(),
                           rank=dist.get_rank(), shuffle=True)
        if use_ddp else None)
    val_sampler = (
        DistributedSampler(val_ds,
                           num_replicas=dist.get_world_size(),
                           rank=dist.get_rank(), shuffle=False,
                           drop_last=False)
        if use_ddp else None)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True, drop_last=True,
        sampler=train_sampler)

    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True, drop_last=False,
        sampler=val_sampler)

    if rank0:
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    return (train_loader, val_loader, train_ds, val_ds,
            train_sampler, val_sampler)


def _build_scheduler(optimizer, config, total_steps: int,
                     lr: float = None):
    """Same factory as tokenizer, uses basemodel LR."""
    stype = getattr(config, 'scheduler_type', 'cosine_warmup')
    pct = getattr(config, 'scheduler_pct_start', 0.05)
    div = getattr(config, 'scheduler_div_factor', 25.0)
    final_div = getattr(
        config, 'scheduler_final_div_factor', 1000.0)
    lr = lr or config.predictor_learning_rate

    if stype == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            total_steps=total_steps,
            pct_start=pct,           # TIER 1 FIX 3
            div_factor=div,          # TIER 1 FIX 3
            final_div_factor=final_div)  # TIER 2 FIX 12

    elif stype == 'cosine_warmup':
        warmup = int(pct * total_steps)

        def lr_lambda(step):
            if step < warmup:
                return max(step / max(1, warmup), 1.0 / div)
            prog = ((step - warmup)
                    / max(1, total_steps - warmup))
            return max(1.0 / final_div,
                       0.5 * (1 + math.cos(math.pi * prog)))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda)

    elif stype == 'cosine_restart':
        T0 = max(total_steps // 300, 5)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T0, T_mult=1,
            eta_min=lr / final_div)

    elif stype == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=3, min_lr=1e-8, verbose=False)

    else:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0)


def train_model(model, tokenizer, device, config,
                save_dir, logger):
    logger.info("Starting basemodel training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0

    (train_loader, val_loader, train_ds, val_ds,
     train_sampler, val_sampler) = create_dataloaders(config)

    # TIER 1 FIX 1: eps added; betas were already correct here
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=getattr(config, 'adam_eps', 1e-6))   # FIX

    total_steps = (len(train_loader)
                   * config.basemodel_epochs)
    # TIER 1 FIX 3: from config
    scheduler = _build_scheduler(optimizer, config, total_steps)

    # ── Dynamic tuner ─────────────────────────────────────
    tuner = None
    if getattr(config, 'dynamic_tuning_enabled', False):
        try:
            from dynamic_tuner_v2 import DynamicTunerV2
            raw_model = model.module if use_ddp else model
            tuner = DynamicTunerV2(
                optimizer=optimizer,
                model=raw_model,
                config=config,
                logger=logger,
                save_dir=os.path.join(save_dir, 'tuner'),
                training_mode='basemodel',
                total_steps_per_epoch=len(train_loader),
                total_epochs=config.basemodel_epochs)
            tuner.set_scheduler(scheduler)
            logger.info("[DynamicTunerV2] Basemodel tuner active")
        except Exception as e:
            logger.warning(
                f"DynamicTunerV2 init failed: {e}")
            tuner = None

    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False)

    raw_model = model.module if use_ddp else model
    best_val_loss = float('inf')

    for epoch in range(config.basemodel_epochs):
        t0 = time.time()
        model.train()
        train_ds.set_epoch_seed(epoch * 10000)
        val_ds.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_train_loss = 0.0
        train_batches = 0

        for batch_idx, (batch_x, batch_x_stamp) in enumerate(
                train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(
                device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(
                    batch_x, half=True)

            token_in = [token_seq_0[:, :-1],
                        token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:],
                         token_seq_1[:, 1:]]

            logits = raw_model(
                token_in[0], token_in[1],
                batch_x_stamp[:, :-1, :])

            # ── TIER 1 FIX 5 + TIER 2 FIX 8: dynamic weights ─
            if tuner is not None:
                w = tuner.get_loss_weights()
                ls = tuner.get_label_smoothing()
            else:
                w = {
                    's1': config.basemodel_s1_weight,
                    's2': config.basemodel_s2_weight,
                }
                ls = getattr(config, 'label_smoothing', 0.0)

            loss, s1_loss, s2_loss = raw_model.head.compute_loss(
                logits[0], logits[1],
                token_out[0], token_out[1],
                s1_weight=w['s1'],       # TIER 1 FIX 5
                s2_weight=w['s2'],       # TIER 1 FIX 5
                label_smoothing=ls)      # TIER 2 FIX 8

            optimizer.zero_grad()
            loss.backward()

            # TIER 1 FIX 2: dynamic grad clip
            grad_clip_val = (
                tuner.get_grad_clip() if tuner is not None
                else config.basemodel_grad_clip)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                max_norm=grad_clip_val,
                norm_type=getattr(
                    config, 'grad_clip_norm_type', 2.0)
            ).item()

            optimizer.step()

            if tuner is None:
                if not isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()

            if tuner is not None:
                tuner.on_step_end(loss.item(), grad_norm)
                tuner.track_loss_components({
                    's1_loss': s1_loss.item(),
                    's2_loss': s2_loss.item(),
                })
                # Apply label smoothing live to head
                raw_model.head.label_smoothing = \
                    tuner.get_label_smoothing()

            epoch_train_loss += loss.item()
            train_batches += 1

            if (batch_idx + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                msg = (
                    f"[Epoch {epoch+1}/"
                    f"{config.basemodel_epochs}, "
                    f"Step {batch_idx+1}/"
                    f"{len(train_loader)}] "
                    f"LR={lr:.2e} "
                    f"Loss={loss.item():.4f} "
                    f"(s1={s1_loss.item():.4f},"
                    f"s2={s2_loss.item():.4f}) "
                    f"GN={grad_norm:.3f}")
                logger.info(msg)
                if rank == 0:
                    print(msg)

        # ── Validation ───────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(
                    device, non_blocking=True)
                t0_seq, t1_seq = tokenizer.encode(
                    batch_x, half=True)
                ti = [t0_seq[:, :-1], t1_seq[:, :-1]]
                to_ = [t0_seq[:, 1:], t1_seq[:, 1:]]
                vlogits = raw_model(ti[0], ti[1],
                                    batch_x_stamp[:, :-1, :])
                vl, _, _ = raw_model.head.compute_loss(
                    vlogits[0], vlogits[1],
                    to_[0], to_[1])
                val_loss_sum += vl.item()
                val_batches += 1

        if use_ddp:
            t = torch.tensor(
                [epoch_train_loss, train_batches,
                 val_loss_sum, val_batches],
                dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_train_loss = t[0].item()
            train_batches = int(t[1].item())
            val_loss_sum = t[2].item()
            val_batches = int(t[3].item())

        avg_train = (epoch_train_loss / train_batches
                     if train_batches > 0 else 0.0)
        avg_val = (val_loss_sum / val_batches
                   if val_batches > 0 else 0.0)

        # ── Tuner epoch hook ─────────────────────────────
        changes = {}
        if tuner is not None:
            try:
                changes = tuner.on_epoch_end(
                    epoch, avg_train, avg_val)
                tuner.scheduler_step(metric=avg_val)
            except Exception as e:
                logger.warning(f"Tuner epoch_end error: {e}")
        else:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(avg_val)

        ep_time = time.time() - t0
        summary = (
            f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} ---"
            f"\n  Train: {avg_train:.6f}"
            f"\n  Val:   {avg_val:.6f}"
            f"\n  Gap:   {avg_val-avg_train:.6f}"
            f"\n  Time:  {ep_time:.1f}s"
            f"\n  Tuner: {list(changes.keys()) or 'none'}")
        logger.info(summary)
        if rank == 0:
            print(summary)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            if rank == 0:
                mp = os.path.join(save_dir, "best_model")
                os.makedirs(mp, exist_ok=True)
                raw_model.save_pretrained(mp)
                msg = (f"Best model saved "
                       f"(val={best_val_loss:.6f})")
                logger.info(msg)
                print(msg)

    if tuner is not None and rank == 0:
        summary = tuner.get_full_summary()
        logger.info(
            f"\n[DynamicTunerV2 Final — Basemodel]\n"
            f"{json.dumps(summary, indent=2, default=str)}")

    return best_val_loss


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str,
        default='configs/config_test_1h.yaml')
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    config = CustomFinetuneConfig(args.config)
    os.makedirs(config.basemodel_save_path, exist_ok=True)
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if getattr(config, 'pre_trained_tokenizer', True):
        tokenizer = NosTokenizer.from_pretrained(
            config.finetuned_tokenizer_path)
    else:
        with open(os.path.join(
                config.pretrained_tokenizer_path,
                'config.json'), 'r') as f:
            arch = json.load(f)
        tokenizer = NosTokenizer(**{
            k: arch.get(k, v) for k, v in {
                'd_in': 6, 'd_model': 256, 'n_heads': 4,
                'ff_dim': 512, 'n_enc_layers': 4,
                'n_dec_layers': 4, 'ffn_dropout_p': 0.0,
                'attn_dropout_p': 0.0, 'resid_dropout_p': 0.0,
                's1_bits': 10, 's2_bits': 10, 'beta': 0.05,
                'gamma0': 1.0, 'gamma': 1.1, 'zeta': 0.05,
                'group_size': 4,
                'inv_temperature': getattr(
                    config, 'bsq_inv_temperature', 1.0),
                'rope_base': getattr(config, 'rope_base', 10000),
            }.items()})

    if getattr(config, 'pre_trained_predictor', True):
        model = Nos.from_pretrained(
            config.pretrained_predictor_path)
    else:
        with open(os.path.join(
                config.pretrained_predictor_path,
                'config.json'), 'r') as f:
            arch = json.load(f)
        model = Nos(
            s1_bits=arch.get('s1_bits', 10),
            s2_bits=arch.get('s2_bits', 10),
            n_layers=arch.get('n_layers', 12),
            d_model=arch.get('d_model', 832),
            n_heads=arch.get('n_heads', 16),
            ff_dim=arch.get('ff_dim', 2048),
            ffn_dropout_p=arch.get('ffn_dropout_p', 0.2),
            attn_dropout_p=arch.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch.get('resid_dropout_p', 0.2),
            token_dropout_p=arch.get('token_dropout_p', 0.0),
            learn_te=arch.get('learn_te', True),
            rope_base=getattr(config, 'rope_base', 10000),
            label_smoothing=getattr(
                config, 'label_smoothing', 0.0),
            dep_layer_n_heads=arch.get('n_heads', 16))

    tokenizer = tokenizer.to(device)
    model = model.to(device)

    logger.info(
        f"Model params: "
        f"{sum(p.numel() for p in model.parameters()):,}")

    best = train_model(
        model, tokenizer, device, config,
        config.basemodel_save_path, logger)
    print(f"Done. Best val loss: {best:.6f}")


if __name__ == "__main__":
    main()