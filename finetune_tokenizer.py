import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
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
    actual_seed = seed
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    else:
        return f"{total_params / 1e3:.1f}K"


def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"tokenizer_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    log_file = os.path.join(log_dir, f"tokenizer_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)
    
    logger.info(f"=== Tokenizer Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def create_dataloaders(config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating tokenizer training data loaders...")
    
    train_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type="train",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    val_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type="val",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    # FIX: Use drop_last=False - validation loop correctly masks padding via valid_mask.
    # drop_last=True discards real data when batch sizes don't divide evenly, causing
    # inconsistent val_loss across different GPU configurations.
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        # FIX: Explicit cast from trial config to ensure HPO batch_size is respected
        batch_size=int(config.batch_size),
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=True,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        # FIX: Explicit cast from trial config to ensure HPO batch_size is respected
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=False,
        sampler=val_sampler
    )
    
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler



def train_tokenizer(model, device, config, save_dir, logger, trial=None):
    """
    All previously hardcoded values (pct_start, div_factor, max_norm)
    are now read from config with safe fallbacks.

    True Gradient Accumulation
    ──────────────────────────
    Accumulates gradients across `accumulation_steps` DataLoader batches before
    performing an optimizer step. This achieves the effective batch size:
    effective_batch_size = batch_size * accumulation_steps
    """
    logger.info("Starting tokenizer training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0

    train_loader, val_loader, train_dataset, val_dataset, \
        train_sampler, val_sampler = create_dataloaders(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )

    # ── Scheduler: use effective steps for true gradient accumulation ────────
    pct_start = getattr(config, 'tokenizer_pct_start', 0.03)
    div_factor = getattr(config, 'tokenizer_div_factor', 10.0)
    accumulation_steps = getattr(config, 'accumulation_steps', 1)

    import math
    effective_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        # FIX: Guarantee max_lr matches the trial's exact optimizer LR
        max_lr=[group['lr'] for group in optimizer.param_groups],
        steps_per_epoch=effective_steps_per_epoch,
        epochs=config.tokenizer_epochs,
        pct_start=pct_start,
        div_factor=div_factor
    )

    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=False)

    # ── Grad clip: read from config ───────────────────────────────
    max_grad_norm = getattr(config, 'tokenizer_max_grad_norm', 2.0)

    best_val_loss = float("inf")
    batch_idx_global = 0

    for epoch in range(config.tokenizer_epochs):
        epoch_start_time = time.time()
        model.train()

        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        current_accum_loss = 0.0

        # Zero gradients BEFORE the dataloader loop (true gradient accumulation)
        optimizer.zero_grad()

        # ── Dataloader loop (True Gradient Accumulation) ─────────────────────
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)

            zs, bsq_loss, _, _ = (model.module if use_ddp else model)(batch_x)
            z_pre, z = zs

            recon_loss_pre = F.mse_loss(z_pre, batch_x)
            recon_loss_all = F.mse_loss(z, batch_x)

            weighted_recon = (
                (config.tokenizer_recon_pre_weight * recon_loss_pre) +
                (config.tokenizer_recon_all_weight * recon_loss_all)
            )

            loss = (config.tokenizer_recon_weight * weighted_recon) + (config.tokenizer_bsq_weight * bsq_loss)

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
                # Grad clip from config
                torch.nn.utils.clip_grad_norm_(
                    (model.module if use_ddp else model).parameters(),
                    max_norm=max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                avg_loss = current_accum_loss / current_micro_batches

                if (batch_idx_global + 1) % config.log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    log_msg = (
                        f"[Epoch {epoch+1}/{config.tokenizer_epochs}, "
                        f"Step {batch_idx_global+1}/{effective_steps_per_epoch}] "
                        f"LR: {lr:.6f}, Loss: {avg_loss:.4f}"
                    )
                    logger.info(log_msg)
                    if rank == 0:
                        print(log_msg)

                current_accum_loss = 0.0
                batch_idx_global += 1

        # ── Validation ────────────────────────────────────────────
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0

        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                zs, _, _, _ = (model.module if use_ddp else model)(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)
                # Detect valid samples (non-padding) to ignore DDP dummy padding
                valid_mask = (ori_batch_x != 0).any(dim=-1).any(dim=-1)
                valid_count = valid_mask.sum().item()
                tot_val_loss_sum_rank += val_loss_item.item() * valid_count
                val_sample_count_rank += valid_count

        if use_ddp:
            tensor_sum = torch.tensor(
                [tot_val_loss_sum_rank, val_sample_count_rank],
                dtype=torch.float64, device=device
            )
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            avg_val_loss = (tensor_sum[0].item() / tensor_sum[1].item()
                            if tensor_sum[1].item() > 0 else 0.0)
        else:
            avg_val_loss = (tot_val_loss_sum_rank / val_sample_count_rank
                            if val_sample_count_rank > 0 else 0)

        epoch_time = time.time() - epoch_start_time
        epoch_summary = (
            f"\n--- Epoch {epoch+1}/{config.tokenizer_epochs} Summary ---\n"
            f"Validation Loss: {avg_val_loss:.6f}\n"
            f"Epoch Time: {format_time(epoch_time)}\n"
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = (f"Best model saved: {model_save_path} "
                            f"(val loss: {best_val_loss:.6f})")
                logger.info(save_msg)
                print(save_msg)

    return best_val_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Nos Tokenizer Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    config = CustomFinetuneConfig(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = CustomFinetuneConfig(args.config)
    
    os.makedirs(config.tokenizer_save_path, exist_ok=True)
    
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)
    
    set_seed(config.seed)
    
    # 加载预训练tokenizer
    if getattr(config, 'pre_trained_tokenizer', True):
        logger.info("Loading pretrained tokenizer...")
        print("Loading pretrained tokenizer...")
        tokenizer = NosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    else:
        print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture")
        import json, os
        cfg_path = os.path.join(config.pretrained_tokenizer_path, 'config.json')
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
            group_size=arch.get('group_size', 4)
        )
    tokenizer = tokenizer.to(device)
    
    model_size = get_model_size(tokenizer)
    logger.info(f"Tokenizer parameters: {model_size}")
    print(f"Tokenizer parameters: {model_size}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.tokenizer_learning_rate}")
    logger.info(f"Training epochs: {config.tokenizer_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Distributed training: False")
    
    logger.info("Starting tokenizer fine-tuning training...")
    print("Starting tokenizer fine-tuning training...")
    best_val_loss = train_tokenizer(tokenizer, device, config, config.tokenizer_save_path, logger)
    
    final_msg = f"Tokenizer training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.tokenizer_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
    
