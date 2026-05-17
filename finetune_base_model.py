import os
import sys
import json
import time
import pickle
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import logging
from logging.handlers import RotatingFileHandler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from model import Nos, NosTokenizer, NosPredictor
from config_loader import CustomFinetuneConfig


# ── Utilities ─────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string."""
    return str(datetime.timedelta(seconds=int(seconds)))


def get_model_size(model: torch.nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1_000_000_000:
        return f"{total_params / 1e9:.1f}B"
    elif total_params >= 1_000_000:
        return f"{total_params / 1e6:.1f}M"
    else:
        return f"{total_params / 1e3:.1f}K"


# ── Dataset ───────────────────────────────────────────────────────────────────

class CustomKlineDataset(Dataset):
    """
    Memory-mapped Kline dataset with Log Return transformation for price columns.
    """

    FEATURE_COLS      = ['open', 'high', 'low', 'close', 'volume', 'amount']
    TIME_FEATURE_COLS = ['minute', 'hour', 'weekday', 'day', 'month']

    def __init__(
        self,
        data_path: str,
        data_type: str = 'train',
        lookback_window: int = 90,
        predict_window: int = 10,
        clip: float = 5.0,
        seed: int = 100,
        train_ratio: float = 0.70,
        val_ratio: float  = 0.15,
        test_ratio: float = 0.15,
    ):
        self.data_path       = data_path
        self.data_type       = data_type
        self.lookback_window = lookback_window
        self.predict_window  = predict_window
        self.window          = lookback_window + predict_window + 1
        self.clip            = clip
        self.seed            = seed
        self.train_ratio     = train_ratio
        self.val_ratio       = val_ratio
        self.test_ratio      = test_ratio

        self.py_rng        = random.Random(seed)
        self.current_epoch = 0

        self._load_or_build_cache()

        self.n_samples = len(self.x_data) - self.window + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"[{data_type.upper()}] Not enough data rows ({len(self.x_data)}) "
                f"for window size {self.window}."
            )
        print(
            f"[{data_type.upper()}] Rows: {len(self.x_data):,}  "
            f"Samples: {self.n_samples:,}"
        )

    def _load_or_build_cache(self):
        cache_x     = f"{self.data_path}.{self.data_type}.x.npy"
        cache_stamp = f"{self.data_path}.{self.data_type}.stamp.npy"
        lock_path   = f"{self.data_path}.{self.data_type}.lock"

        # Fast path: lock-free check for 99% of workers
        if os.path.exists(cache_x) and os.path.exists(cache_stamp):
            self.x_data     = np.load(cache_x,     mmap_mode='r')
            self.stamp_data = np.load(cache_stamp, mmap_mode='r')
            return

        from filelock import FileLock
        with FileLock(lock_path, timeout=600):
            if os.path.exists(cache_x) and os.path.exists(cache_stamp):
                self.x_data     = np.load(cache_x,     mmap_mode='r')
                self.stamp_data = np.load(cache_stamp, mmap_mode='r')
                return

            print(f"[{self.data_type.upper()}] Building mmap cache …")
            df = pd.read_csv(self.data_path)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.sort_values('timestamps').reset_index(drop=True)

            # Temporal features
            df['minute']  = df['timestamps'].dt.minute
            df['hour']    = df['timestamps'].dt.hour
            df['weekday'] = df['timestamps'].dt.weekday
            df['day']     = df['timestamps'].dt.day
            df['month']   = df['timestamps'].dt.month

            if df.isnull().any().any():
                df = df.ffill()

            # ── PROFESSIONAL UPGRADE: Leak-Free Structural Returns ──
            price_cols = ['open', 'high', 'low', 'close']
            
            # Store absolute open to calculate intra-bar dynamics correctly
            abs_open = df['open'].copy()
            
            # 1. Open: Overnight return relative to previous Close
            df['open'] = np.log(df['open'] / df['close'].shift(1))
            
            # 2. High, Low, Close: Intra-bar returns relative to CURRENT Open
            df['high'] = np.log(df['high'] / abs_open)
            df['low'] = np.log(df['low'] / abs_open)
            df['close'] = np.log(df['close'] / abs_open)

            # Clean up the NaN created by shift on row 0, and any infinites
            df[price_cols] = df[price_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Chronological split
            n = len(df)
            train_end = int(n * self.train_ratio)
            val_end   = int(n * (self.train_ratio + self.val_ratio))

            slices = {'train': slice(None, train_end),
                      'val':   slice(train_end, val_end),
                      'test':  slice(val_end, None)}
            df = df.iloc[slices[self.data_type]]

            x_arr     = df[self.FEATURE_COLS].values.astype(np.float32)
            stamp_arr = df[self.TIME_FEATURE_COLS].values.astype(np.float32)

            del df

            # Atomic writes
            tmp_x     = f"{cache_x}.tmp.{os.getpid()}.npy"
            tmp_stamp = f"{cache_stamp}.tmp.{os.getpid()}.npy"

            np.save(tmp_x, x_arr)
            np.save(tmp_stamp, stamp_arr)

            os.replace(tmp_x, cache_x)
            os.replace(tmp_stamp, cache_stamp)

        self.x_data     = np.load(cache_x,     mmap_mode='r')
        self.stamp_data = np.load(cache_stamp, mmap_mode='r')

    def set_epoch_seed(self, epoch: int):
        self.py_rng.seed(self.seed + epoch)
        self.current_epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        max_start = len(self.x_data) - self.window
        start_idx = idx % (max_start + 1)
        end_idx = start_idx + self.window

        x       = self.x_data[start_idx:end_idx].copy()
        x_stamp = self.stamp_data[start_idx:end_idx].copy()

        # Calculate mean and std ONLY on the lookback window
        lookback_x = x[:self.lookback_window]
        x_mean = lookback_x.mean(axis=0)
        x_std  = lookback_x.std(axis=0)

        # Apply historical stats to the ENTIRE window
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)

    # Idempotent – do not attach duplicate handlers on re-import
    if logger.handlers:
        return logger

    log_file     = os.path.join(log_dir, f"basemodel_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler() if rank == 0 else None
    if console_handler:
        console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(formatter)
    if console_handler:
        console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    if console_handler:
        logger.addHandler(console_handler)

    logger.info("=== Basemodel Training Started ===")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Log dir:    {log_dir}")
    logger.info(f"Rank:       {rank}")
    logger.info(f"Timestamp:  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    return logger


# ── DataLoaders ───────────────────────────────────────────────────────────────

def create_dataloaders(config):
    is_main = (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    )
    if is_main:
        print("Creating data loaders …")

    shared_kw = dict(
        data_path       = config.data_path,
        lookback_window = config.lookback_window,
        predict_window  = config.predict_window,
        clip            = config.clip,
        train_ratio     = config.train_ratio,
        val_ratio       = config.val_ratio,
        test_ratio      = config.test_ratio,
    )

    train_dataset = CustomKlineDataset(data_type='train', seed=config.seed,     **shared_kw)
    val_dataset   = CustomKlineDataset(data_type='val',   seed=config.seed + 1, **shared_kw)

    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)  if use_ddp else None
    )
    # FIX: Use drop_last=False - validation loop correctly masks padding via valid_mask.
    # drop_last=True discards real data when batch sizes don't divide evenly, causing
    # inconsistent val_loss across different GPU configurations.
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False, drop_last=False) if use_ddp else None
    )

    loader_kw = dict(
        num_workers=config.num_workers, 
        pin_memory=True,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )

    train_loader = DataLoader(
        train_dataset,
        # FIX: Explicit cast from trial config to ensure HPO batch_size is respected
        batch_size = int(config.batch_size),
        shuffle    = (train_sampler is None),
        drop_last  = True,
        sampler    = train_sampler,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        # FIX: Explicit cast from trial config to ensure HPO batch_size is respected
        batch_size = int(config.batch_size),
        shuffle    = False,
        drop_last  = False,
        sampler    = val_sampler,
        **loader_kw,
    )

    if is_main:
        print(
            f"Train samples: {len(train_dataset):,}  "
            f"Val samples: {len(val_dataset):,}"
        )

    return (
        train_loader, val_loader,
        train_dataset, val_dataset,
        train_sampler, val_sampler,
    )


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_model(model, tokenizer, device, config, save_dir, logger, trial=None):
    """
    Fine-tunes *model* (NosPredictor / Nos) with the frozen *tokenizer*.

    True Gradient Accumulation
    ──────────────────────────
    Accumulates gradients across `accumulation_steps` DataLoader batches before
    performing an optimizer step. This achieves the effective batch size:
    effective_batch_size = batch_size * accumulation_steps

    The scheduler uses effective_steps_per_epoch to correctly count optimizer
    steps, not raw DataLoader batches.
    """
    logger.info("Starting base-model training …")

    use_ddp = dist.is_available() and dist.is_initialized()
    rank    = dist.get_rank() if use_ddp else 0

    (
        train_loader, val_loader,
        train_dataset, val_dataset,
        train_sampler, val_sampler,
    ) = create_dataloaders(config)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.predictor_learning_rate,
        betas        = (config.adam_beta1, config.adam_beta2),
        weight_decay = config.adam_weight_decay,
    )

    # ── Scheduler: use effective steps for true gradient accumulation ────────
    pct_start  = getattr(config, 'basemodel_pct_start',  0.03)
    div_factor = getattr(config, 'basemodel_div_factor', 10.0)
    accumulation_steps = getattr(config, 'accumulation_steps', 1)

    # Effective steps = ceil(total_batches / accumulation_steps)
    import math
    effective_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        # FIX: Guarantee max_lr matches the trial's exact optimizer LR
        max_lr          = [group['lr'] for group in optimizer.param_groups],
        steps_per_epoch = effective_steps_per_epoch,   # 1 step == accumulation_steps batches
        epochs          = config.basemodel_epochs,
        pct_start       = pct_start,
        div_factor      = div_factor,
    )

    # ── DDP wrapping (after scheduler construction) ────────────────────────────
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(
            model,
            device_ids          = [local_rank],
            output_device       = local_rank,
            find_unused_parameters = False,
        )

    # ── Hyper-parameters from config ──────────────────────────────────────────
    max_grad_norm     = getattr(config, 'basemodel_max_grad_norm', 3.0)

    # Convenience alias: unwrap DDP only once per call-site
    def raw_model():
        return model.module if use_ddp else model

    # ── Epoch loop ────────────────────────────────────────────────────────────
    best_val_loss   = float('inf')
    batch_idx_global = 0

    for epoch in range(config.basemodel_epochs):
        epoch_start = time.time()
        model.train()

        # ── NEW: Scheduled Sampling Probability ──
        # Linear decay from 100% to 0% over the first 75% of training.
        # The final 25% of epochs will be pure autoregressive training.
        tf_decay_epochs = max(1, int(config.basemodel_epochs * 0.75))
        tf_prob = max(0.0, 1.0 - (epoch / tf_decay_epochs))

        # Gumbel temperature decay: tau starts at 1.0, decays to 0.3 over training
        tau_decay_epochs = max(1, int(config.basemodel_epochs * 0.75))
        current_tau = max(0.3, 1.0 - (epoch / tau_decay_epochs) * 0.7)

        # Deterministic-but-varied shuffling per epoch
        train_dataset.set_epoch_seed(epoch * 10_000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_train_loss = 0.0
        train_batches    = 0
        current_accum_loss = 0.0

        # Zero gradients BEFORE the dataloader loop (true gradient accumulation)
        optimizer.zero_grad()

        # ── Dataloader loop (True Gradient Accumulation) ─────────────────────
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x       = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Tokenizer is frozen — skip its grad computation entirely
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in  = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:],  token_seq_1[:, 1:]]

            # ── Apply Scheduled Sampling (DDP Safe) ──
            if use_ddp:
                if rank == 0:
                    use_tf_tensor = torch.tensor([1.0 if random.random() < tf_prob else 0.0], device=device)
                else:
                    use_tf_tensor = torch.tensor([0.0], device=device)

                dist.broadcast(use_tf_tensor, src=0)
                use_tf = bool(use_tf_tensor.item())
            else:
                use_tf = random.random() < tf_prob

            logits = raw_model()(
                s1_ids=token_in[0],
                s2_ids=token_in[1],
                stamp=batch_x_stamp[:, :-1, :],
                use_teacher_forcing=use_tf,
                s1_targets=token_out[0] if use_tf else None,
                gumbel_tau=current_tau  # Injected dynamic temperature
            )

            loss, s1_loss, s2_loss = raw_model().head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            # ── TASK 1.2: Dynamic Micro-Batch & DDP Sync Storm Fix ──────────
            is_last_batch = (batch_idx + 1) == len(train_loader)
            is_step_time = (batch_idx + 1) % accumulation_steps == 0 or is_last_batch

            # Dynamic scalar to prevent inverse gradient accumulation on the tail batch
            current_micro_batches = (batch_idx % accumulation_steps) + 1 if is_last_batch else accumulation_steps
            loss_scaled = loss / current_micro_batches

            # DDP Context Manager to prevent all_reduce storms
            if use_ddp and not is_step_time:
                with model.no_sync():
                    loss_scaled.backward()
            else:
                loss_scaled.backward()

            current_accum_loss += loss.item()

            # ── Optimizer step (once per accumulation cycle) ────────────────
            if is_step_time:
                torch.nn.utils.clip_grad_norm_(
                    raw_model().parameters(),
                    max_norm=max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                avg_loss = current_accum_loss / current_micro_batches
                epoch_train_loss += avg_loss
                train_batches    += 1

                if (batch_idx_global + 1) % config.log_interval == 0:
                    lr      = optimizer.param_groups[0]['lr']
                    log_msg = (
                        f"[Epoch {epoch+1}/{config.basemodel_epochs}, "
                        f"Step {batch_idx_global+1}/{effective_steps_per_epoch}] "
                        f"LR: {lr:.6f}  Loss: {avg_loss:.4f}"
                    )
                    logger.info(log_msg)
                    if rank == 0:
                        print(log_msg)

                current_accum_loss = 0.0
                batch_idx_global += 1

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum   = 0.0
        val_sample_cnt = 0

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x       = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in  = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:],  token_seq_1[:, 1:]]

                logits = raw_model()(
                    s1_ids=token_in[0], 
                    s2_ids=token_in[1], 
                    stamp=batch_x_stamp[:, :-1, :],
                    use_teacher_forcing=False
                )
                loss, _, _ = raw_model().head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )

                # Detect valid samples (non-padding) to ignore DDP dummy padding
                valid_mask = (batch_x != 0).any(dim=-1).any(dim=-1)
                valid_count = valid_mask.sum().item()
                val_loss_sum   += loss.item() * valid_count
                val_sample_cnt += valid_count

        # ── DDP aggregation ───────────────────────────────────────────────────
        if use_ddp:
            agg = torch.tensor(
                [epoch_train_loss, float(train_batches),
                 val_loss_sum,     float(val_sample_cnt)],
                dtype=torch.float64, device=device,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            avg_train_loss = agg[0].item() / agg[1].item() if agg[1].item() > 0 else 0.0
            avg_val_loss   = agg[2].item() / agg[3].item() if agg[3].item() > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0.0
            avg_val_loss   = val_loss_sum / val_sample_cnt    if val_sample_cnt > 0 else 0.0

        epoch_time    = time.time() - epoch_start
        epoch_summary = (
            f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
            f"  Train Loss : {avg_train_loss:.6f}\n"
            f"  Val   Loss : {avg_val_loss:.6f}\n"
            f"  Epoch Time : {format_time(epoch_time)}\n"
        )
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)

        # ── TASK 1.1: Optuna Pruning Hook (Traceback OOM Fix) ─────────────────
        if trial is not None:
            import optuna
            trial.report(avg_val_loss, epoch)

            if trial.should_prune():
                prune_msg = f"Trial {trial.number} pruned at epoch {epoch} (val_loss: {avg_val_loss:.6f})"
                logger.info(prune_msg)
                if rank == 0:
                    print(prune_msg)

                # CRITICAL: Sever local variable references to prevent Traceback OOM leak
                locals().clear()
                raise optuna.exceptions.TrialPruned()    
        

        # ── Checkpoint (rank-0 only) ───────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                ckpt_dir = os.path.join(save_dir, "best_model")
                os.makedirs(ckpt_dir, exist_ok=True)
                raw_model().save_pretrained(ckpt_dir)
                save_msg = (
                    f"✓ Best model saved → {ckpt_dir}  "
                    f"(val loss: {best_val_loss:.6f})"
                )
                logger.info(save_msg)
                print(save_msg)

    return best_val_loss


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Nos Base-Model Fine-tuning')
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to YAML config file (default: config.yaml)',
    )
    args   = parser.parse_args()
    config = CustomFinetuneConfig(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(config.basemodel_save_path, exist_ok=True)

    log_dir = os.path.join(config.base_save_path, "logs")
    logger  = setup_logging(config.exp_name, log_dir, rank=0)

    # Reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer …")
    if getattr(config, 'pre_trained_tokenizer', True):
        tokenizer = NosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
    else:
        logger.info("pre_trained_tokenizer=False — random init")
        cfg_path = os.path.join(config.pretrained_tokenizer_path, 'config.json')
        with open(cfg_path) as fh:
            arch = json.load(fh)
        tokenizer = NosTokenizer(
            d_in            = arch.get('d_in',             6),
            d_model         = arch.get('d_model',         256),
            n_heads         = arch.get('n_heads',           4),
            ff_dim          = arch.get('ff_dim',          512),
            n_enc_layers    = arch.get('n_enc_layers',      4),
            n_dec_layers    = arch.get('n_dec_layers',      4),
            ffn_dropout_p   = arch.get('ffn_dropout_p',  0.0),
            attn_dropout_p  = arch.get('attn_dropout_p', 0.0),
            resid_dropout_p = arch.get('resid_dropout_p',0.0),
            s1_bits         = arch.get('s1_bits',          10),
            s2_bits         = arch.get('s2_bits',          10),
            beta            = arch.get('beta',           0.05),
            gamma0          = arch.get('gamma0',          1.0),
            gamma           = arch.get('gamma',           1.1),
            zeta            = arch.get('zeta',           0.05),
            group_size      = arch.get('group_size',        4),
        )

    # ── Predictor ─────────────────────────────────────────────────────────────
    logger.info("Loading predictor …")
    if getattr(config, 'pre_trained_predictor', True):
        model = Nos.from_pretrained(config.pretrained_predictor_path)
    else:
        logger.info("pre_trained_predictor=False — random init")
        cfg_path = os.path.join(config.pretrained_predictor_path, 'config.json')
        with open(cfg_path) as fh:
            arch = json.load(fh)
        model = Nos(
            s1_bits         = arch.get('s1_bits',          10),
            s2_bits         = arch.get('s2_bits',          10),
            n_layers        = arch.get('n_layers',          12),
            d_model         = arch.get('d_model',          832),
            n_heads         = arch.get('n_heads',           16),
            ff_dim          = arch.get('ff_dim',          2048),
            ffn_dropout_p   = arch.get('ffn_dropout_p',   0.2),
            attn_dropout_p  = arch.get('attn_dropout_p',  0.0),
            resid_dropout_p = arch.get('resid_dropout_p', 0.2),
            token_dropout_p = arch.get('token_dropout_p', 0.0),
            learn_te        = arch.get('learn_te',        True),
        )

    tokenizer = tokenizer.to(device)
    model     = model.to(device)

    logger.info(f"Tokenizer size : {get_model_size(tokenizer)}")
    logger.info(f"Model size     : {get_model_size(model)}")
    print(f"Tokenizer: {get_model_size(tokenizer)}  |  Model: {get_model_size(model)}")

    # ── Config summary ────────────────────────────────────────────────────────
    logger.info("=== Training Configuration ===")
    for key, val in [
        ("data_path",           config.data_path),
        ("lookback_window",     config.lookback_window),
        ("predict_window",      config.predict_window),
        ("batch_size",          config.batch_size),
        ("accumulation_steps",  getattr(config, 'accumulation_steps', 1)),
        ("learning_rate",       config.predictor_learning_rate),
        ("epochs",              config.basemodel_epochs),
        ("device",              device),
        ("tokenizer_path",      config.finetuned_tokenizer_path),
        ("predictor_path",      config.pretrained_predictor_path),
    ]:
        logger.info(f"  {key:<22}: {val}")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_val_loss = train_model(
        model, tokenizer, device, config,
        config.basemodel_save_path, logger,
    )

    final_msg = (
        f"Training complete.  Best val loss: {best_val_loss:.6f}\n"
        f"Checkpoint: {config.basemodel_save_path}"
    )
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()