import os
import sys
import json
import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import datetime
import logging
from logging.handlers import RotatingFileHandler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("../")
from model import NosTokenizer
from finetune_base_model import CustomKlineDataset
from config_loader import CustomFinetuneConfig


def set_seed(seed: int, rank: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters()
                if p.requires_grad)
    if total >= 1e9:
        return f"{total/1e9:.1f}B"
    elif total >= 1e6:
        return f"{total/1e6:.1f}M"
    return f"{total/1e3:.1f}K"


def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def setup_logging(exp_name: str, log_dir: str,
                  rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(
        f"tokenizer_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    log_file = os.path.join(
        log_dir, f"tokenizer_training_rank_{rank}.log")
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

    logger.info("=== Tokenizer Training Started ===")
    logger.info(f"Experiment: {exp_name} | Rank: {rank}")
    return logger


def create_dataloaders(config):
    rank0 = (not dist.is_available()
             or not dist.is_initialized()
             or dist.get_rank() == 0)
    if rank0:
        print("Creating tokenizer dataloaders...")

    train_ds = CustomKlineDataset(
        data_path=config.data_path,
        data_type="train",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip, seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio)

    val_ds = CustomKlineDataset(
        data_path=config.data_path,
        data_type="val",
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
                     current_lr: float = None):
    """Build scheduler from config. Supports all types."""
    stype = getattr(config, 'scheduler_type', 'cosine_warmup')
    pct = getattr(config, 'scheduler_pct_start', 0.05)
    div = getattr(config, 'scheduler_div_factor', 25.0)
    final_div = getattr(
        config, 'scheduler_final_div_factor', 1000.0)
    lr = current_lr or config.tokenizer_learning_rate

    if stype == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            total_steps=total_steps,
            pct_start=pct,            # TIER 1 FIX 3
            div_factor=div,           # TIER 1 FIX 3
            final_div_factor=final_div)  # TIER 2 FIX 12

    elif stype == 'cosine_warmup':
        warmup_steps = int(pct * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return max(step / max(1, warmup_steps),
                           1.0 / div)
            prog = ((step - warmup_steps)
                    / max(1, total_steps - warmup_steps))
            return max(1.0 / final_div,
                       0.5 * (1 + math.cos(math.pi * prog)))

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda)

    elif stype == 'cosine_restart':
        T0 = max(total_steps // (3 * max(1, 100)), 5)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T0, T_mult=1,
            eta_min=lr / final_div)

    elif stype == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=3, min_lr=1e-8, verbose=False)

    else:  # constant
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0)


def train_tokenizer(model, device, config, save_dir, logger):
    logger.info("Starting tokenizer training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0

    (train_loader, val_loader, train_ds, val_ds,
     train_sampler, val_sampler) = create_dataloaders(config)

    # ── TIER 1 FIX 1: betas now passed (was missing!) ────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),  # FIX
        weight_decay=config.adam_weight_decay,
        eps=getattr(config, 'adam_eps', 1e-6))         # FIX

    total_steps = (len(train_loader)
                   * config.tokenizer_epochs)
    # TIER 1 FIX 3: scheduler from config, not hardcoded
    scheduler = _build_scheduler(
        optimizer, config, total_steps)

    # ── Dynamic tuner setup ──────────────────────────────
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
                training_mode='tokenizer',
                total_steps_per_epoch=len(train_loader),
                total_epochs=config.tokenizer_epochs)
            tuner.set_scheduler(scheduler)
            logger.info("[DynamicTunerV2] Tokenizer tuner active")
        except Exception as e:
            logger.warning(
                f"DynamicTunerV2 init failed: {e}. "
                f"Continuing without tuning.")
            tuner = None

    # ── DDP wrap ─────────────────────────────────────────
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False)

    best_val_loss = float("inf")
    accumulation_steps = getattr(config, 'accumulation_steps', 1)
    raw_model = model.module if use_ddp else model

    for epoch in range(config.tokenizer_epochs):
        t0 = time.time()
        model.train()
        train_ds.set_epoch_seed(epoch * 10000)
        val_ds.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_batches = 0

        for batch_idx, (ori_batch_x, _) in enumerate(
                train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(
                device, non_blocking=True)

            optimizer.zero_grad()

            # Current loss weights from tuner (or config defaults)
            if tuner is not None:
                w = tuner.get_loss_weights()
            else:
                w = {
                    'recon_pre':
                        config.tokenizer_recon_pre_weight,
                    'recon_all':
                        config.tokenizer_recon_all_weight,
                    'bsq': config.tokenizer_bsq_weight,
                    'recon': config.tokenizer_recon_weight,
                }

            # Current grad clip from tuner or config
            grad_clip_val = (
                tuner.get_grad_clip() if tuner is not None
                else config.tokenizer_grad_clip)  # TIER 1 FIX 2

            # Accumulation
            cur_accum = accumulation_steps
            chunk = max(1, ori_batch_x.shape[0] // cur_accum)
            total_batch_loss = 0.0
            bsq_loss_item = recon_pre_item = recon_all_item = 0.0

            for j in range(cur_accum):
                start = j * chunk
                end = min((j + 1) * chunk,
                          ori_batch_x.shape[0])
                if start >= end:
                    break
                batch_x = ori_batch_x[start:end]

                zs, bsq_loss, _, _ = raw_model(batch_x)
                z_pre, z = zs

                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)

                # TIER 1 FIX 5: dynamic loss weights
                recon_loss = (w['recon_pre'] * recon_loss_pre
                              + w['recon_all'] * recon_loss_all)
                total_w = w['recon'] + w['bsq']
                loss = ((w['recon'] * recon_loss
                         + w['bsq'] * bsq_loss)
                        / max(total_w, 1e-8))

                (loss / cur_accum).backward()
                total_batch_loss += loss.item()
                bsq_loss_item += bsq_loss.item()
                recon_pre_item += recon_loss_pre.item()
                recon_all_item += recon_loss_all.item()

            # TIER 1 FIX 2: dynamic grad clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                max_norm=grad_clip_val,
                norm_type=getattr(
                    config, 'grad_clip_norm_type', 2.0)
            ).item()

            optimizer.step()

            # Scheduler step (only when tuner is not managing it)
            if tuner is None:
                if not isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()

            avg_batch_loss = total_batch_loss / cur_accum

            # Track individual components for tuner
            if tuner is not None:
                tuner.on_step_end(avg_batch_loss, grad_norm)
                tuner.track_loss_components({
                    'bsq_loss': bsq_loss_item / cur_accum,
                    'recon_pre': recon_pre_item / cur_accum,
                    'recon_all': recon_all_item / cur_accum,
                })
                # Apply live BSQ temperature
                try:
                    raw_model.tokenizer.set_inv_temperature(
                        tuner.get_bsq_temperature())
                except AttributeError:
                    pass

            epoch_loss_sum += avg_batch_loss
            epoch_batches += 1

            if (epoch_batches % config.log_interval) == 0:
                lr = optimizer.param_groups[0]['lr']
                msg = (
                    f"[Epoch {epoch+1}/{config.tokenizer_epochs}, "
                    f"Step {batch_idx+1}/{len(train_loader)}] "
                    f"LR={lr:.2e} Loss={avg_batch_loss:.4f} "
                    f"GradNorm={grad_norm:.3f}")
                logger.info(msg)
                if rank == 0:
                    print(msg)

        # ── Validation ───────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.squeeze(0).to(
                    device, non_blocking=True)
                zs, _, _, _ = raw_model(ori_batch_x)
                _, z = zs
                v_loss = F.mse_loss(z, ori_batch_x)
                val_loss_sum += v_loss.item() * ori_batch_x.size(0)
                val_count += ori_batch_x.size(0)

        if use_ddp:
            t = torch.tensor(
                [epoch_loss_sum, epoch_batches,
                 val_loss_sum, val_count],
                dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_loss_sum = t[0].item()
            epoch_batches = int(t[1].item())
            val_loss_sum = t[2].item()
            val_count = int(t[3].item())

        avg_train = (epoch_loss_sum / epoch_batches
                     if epoch_batches > 0 else 0.0)
        avg_val = (val_loss_sum / val_count
                   if val_count > 0 else 0.0)

        # ── Tuner epoch hook ─────────────────────────────
        changes = {}
        if tuner is not None:
            try:
                changes = tuner.on_epoch_end(
                    epoch, avg_train, avg_val)
                tuner.scheduler_step(metric=avg_val)
            except Exception as e:
                logger.warning(
                    f"Tuner epoch_end error: {e}")
        else:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(avg_val)

        ep_time = time.time() - t0
        summary = (
            f"\n--- Epoch {epoch+1}/{config.tokenizer_epochs} ---"
            f"\n  Train: {avg_train:.6f}"
            f"\n  Val:   {avg_val:.6f}"
            f"\n  Gap:   {avg_val-avg_train:.6f}"
            f"\n  Time:  {format_time(ep_time)}"
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
                msg = (f"Best tokenizer saved "
                       f"(val={best_val_loss:.6f})")
                logger.info(msg)
                print(msg)

    if tuner is not None and rank == 0:
        summary = tuner.get_full_summary()
        logger.info(
            f"\n[DynamicTunerV2 Final — Tokenizer]\n"
            f"{json.dumps(summary, indent=2, default=str)}")

    return best_val_loss


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/config_test_1h.yaml')
    args = parser.parse_args()

    config = CustomFinetuneConfig(args.config)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(config.tokenizer_save_path, exist_ok=True)
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)
    set_seed(config.seed)

    if getattr(config, 'pre_trained_tokenizer', True):
        tokenizer = NosTokenizer.from_pretrained(
            config.pretrained_tokenizer_path)
    else:
        cfg_path = os.path.join(
            config.pretrained_tokenizer_path, 'config.json')
        with open(cfg_path, 'r') as f:
            arch = json.load(f)
        tokenizer = NosTokenizer(
            d_in=arch.get('d_in', 6),
            d_model=arch.get('d_model', 256),
            n_heads=arch.get('n_heads', 4),
            ff_dim=arch.get('ff_dim', 512),
            n_enc_layers=arch.get('n_enc_layers', 4),
            n_dec_layers=arch.get('n_dec_layers', 4),
            ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
            attn_dropout_p=arch.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch.get('resid_dropout_p', 0.0),
            s1_bits=arch.get('s1_bits', 10),
            s2_bits=arch.get('s2_bits', 10),
            beta=arch.get('beta', 0.05),
            gamma0=arch.get('gamma0', 1.0),
            gamma=arch.get('gamma', 1.1),
            zeta=arch.get('zeta', 0.05),
            group_size=arch.get('group_size', 4),
            inv_temperature=getattr(config, 'bsq_inv_temperature', 1.0),
            rope_base=getattr(config, 'rope_base', 10000))

    tokenizer = tokenizer.to(device)
    logger.info(f"Tokenizer size: {get_model_size(tokenizer)}")

    best = train_tokenizer(
        tokenizer, device, config,
        config.tokenizer_save_path, logger)
    print(f"Done. Best val loss: {best:.6f}")


if __name__ == "__main__":
    main()