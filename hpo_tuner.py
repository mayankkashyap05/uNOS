"""
Automatic Hyperparameter Tuning for Nos Model Finetuning.

Architecture: Asynchronous multi-process HPO where each GPU process runs
independently, sharing only an Optuna SQLite/PostgreSQL study database.
No DDP is used — process isolation is enforced at the launcher level.

Launch (single GPU, development):
    python hpo_tuner.py --config configs/config.yaml --phase both

Launch (8-GPU cluster, production):
    ./launch_hpo.sh --config configs/config.yaml --phase both --n-trials 30

Launch (specific phase):
    python hpo_tuner.py --config configs/config.yaml --phase tokenizer
    python hpo_tuner.py --config configs/config.yaml --phase basemodel --tokenizer-path path/to/tokenizer
    python hpo_tuner.py --config configs/config.yaml --apply-best
"""

from __future__ import annotations

import copy
import datetime
import gc
import json
import logging
import argparse
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import warnings  # <-- Correct standard library import
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.pool import NullPool

import torch
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Optional Optuna import ─────────────────────────────────────────────────
try:
    import optuna
    from optuna.exceptions import ExperimentalWarning  # <-- Moved inside the safe block
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
    from optuna.storages import RDBStorage
    
    # Suppress Optuna's experimental feature warnings safely
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from config_loader import CustomFinetuneConfig
from model import Nos, NosTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Module-level logger (console output for the orchestrator process)
# ══════════════════════════════════════════════════════════════════════════════

def _configure_root_logger(log_dir: str, worker_tag: str) -> logging.Logger:
    """
    Configures the root logger for this HPO worker process.

    Writes INFO+ to both console (WARNING+ only to reduce noise) and a
    dedicated per-worker log file with full DEBUG output.

    Args:
        log_dir:    Directory where the worker log file will be written.
        worker_tag: Unique identifier string (e.g., "pid12345_gpu0").

    Returns:
        Configured root logger for this process.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"worker_{worker_tag}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to prevent duplication on re-import
    root_logger.handlers.clear()

    # ── File handler: full DEBUG output per worker ─────────────────────────
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # ── Console handler: WARNING+ only to keep stdout clean ───────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def _get_trial_logger(name: str, trial_dir: str) -> logging.Logger:
    """
    Creates an isolated, non-propagating file logger for a single trial.

    Using a unique logger name (including PID) prevents Python's logging
    singleton from returning a cached logger from a previous trial, which
    would cause handler accumulation (each call adding another FileHandler).

    Args:
        name:      Base name for the logger (e.g., "hpo_tok_trial_7").
        trial_dir: Path to the trial's isolated working directory.

    Returns:
        Configured logger that writes only to trial_dir/trial.log.
    """
    unique_name = f"{name}_pid{os.getpid()}"
    logger = logging.getLogger(unique_name)

    # Always clear handlers — prevents accumulation if function is called
    # multiple times with the same effective logger name (e.g., trial restarts)
    logger.handlers.clear()
    logger.propagate = False  # Do not bubble up to root logger
    logger.setLevel(logging.INFO)

    log_path = os.path.join(trial_dir, "trial.log")
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Model Override Helpers
# ══════════════════════════════════════════════════════════════════════════════

def apply_dropout_overrides(
    model: torch.nn.Module,
    config: CustomFinetuneConfig,
) -> torch.nn.Module:
    """
    Walks all named submodules and overrides dropout probabilities in-place
    when the corresponding config attribute is set to a non-None value.

    This modifies only scalar dropout probability attributes and the `p`
    attribute of nn.Dropout instances. Model weights are never touched.

    Args:
        model:  Any PyTorch module (NosTokenizer or Nos predictor).
        config: Trial config carrying dropout override values.

    Returns:
        The same model instance with dropout values updated.
    """
    ffn_dp    = getattr(config, "ffn_dropout_p",   None)
    attn_dp   = getattr(config, "attn_dropout_p",  None)
    resid_dp  = getattr(config, "resid_dropout_p", None)
    token_dp  = getattr(config, "token_dropout_p", None)

    for _name, module in model.named_modules():
        # FeedForward ffn_dropout (nn.Dropout instance on FeedForward blocks)
        if ffn_dp is not None and hasattr(module, "ffn_dropout"):
            if isinstance(module.ffn_dropout, torch.nn.Dropout):
                module.ffn_dropout.p = float(ffn_dp)

        # Residual dropout in attention blocks (nn.Dropout instance)
        if resid_dp is not None and hasattr(module, "resid_dropout"):
            if isinstance(module.resid_dropout, torch.nn.Dropout):
                module.resid_dropout.p = float(resid_dp)

        # Token dropout on Nos predictor (nn.Dropout instance)
        if token_dp is not None and hasattr(module, "token_drop"):
            if isinstance(module.token_drop, torch.nn.Dropout):
                module.token_drop.p = float(token_dp)

        # Attention dropout stored as a plain float scalar (not nn.Dropout)
        if attn_dp is not None and hasattr(module, "attn_dropout_p"):
            if isinstance(module.attn_dropout_p, float):
                module.attn_dropout_p = float(attn_dp)

    return model


def apply_bsq_overrides(
    tokenizer: NosTokenizer,
    config: CustomFinetuneConfig,
) -> NosTokenizer:
    """
    Overrides BSQ (Binary Spherical Quantizer) loss weight scalars on a
    loaded tokenizer. Only modifies scalar hyperparameter attributes;
    quantizer weight tensors are never modified.

    Args:
        tokenizer: Loaded NosTokenizer instance.
        config:    Trial config carrying BSQ override values.

    Returns:
        The same tokenizer instance with BSQ weights updated.

    Raises:
        AttributeError: If tokenizer does not have the expected BSQ structure.
    """
    try:
        bsq = tokenizer.tokenizer.bsq  # BinarySphericalQuantizer instance
    except AttributeError as exc:
        raise AttributeError(
            "Could not access tokenizer.tokenizer.bsq. "
            "Verify NosTokenizer architecture has not changed."
        ) from exc

    override_map = {
        "bsq_beta":   "beta",
        "bsq_gamma0": "gamma0",
        "bsq_gamma":  "gamma",
        "bsq_zeta":   "zeta",
    }
    for config_attr, bsq_attr in override_map.items():
        value = getattr(config, config_attr, None)
        if value is not None:
            setattr(bsq, bsq_attr, float(value))

    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# GPU Memory Management
# ══════════════════════════════════════════════════════════════════════════════

def release_gpu_memory(*model_refs) -> None:
    """
    Performs deterministic GPU memory cleanup after a trial completes.

    Executes the full cleanup chain required to return memory to the CUDA
    driver (not just to PyTorch's internal allocator cache):
      1. Delete all passed model references from Python's reference count.
      2. Force Python's cyclic garbage collector to collect any cycles.
      3. Release PyTorch's allocator cache back to the CUDA driver.
      4. Synchronize the CUDA stream to ensure all ops have completed.

    Without step 3 (`empty_cache`), PyTorch holds memory in its internal
    pool. Across many HPO trials this accumulates until the next trial's
    allocation fails with OutOfMemoryError even though logically the memory
    was "freed" after the previous trial.

    Args:
        *model_refs: Any number of PyTorch module references to delete.
    """
    for ref in model_refs:
        if ref is not None:
            del ref

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ══════════════════════════════════════════════════════════════════════════════
# Search Space Builder
# ══════════════════════════════════════════════════════════════════════════════

class SearchSpaceBuilder:
    """
    Translates the `hpo.search_space` YAML block into Optuna trial suggestions.

    The search space config block uses a typed specification format:
        param_name:
            type: float | int | categorical
            low:  <number>          # float and int only
            high: <number>          # float and int only
            log:  true | false      # float and int only (log-scale sampling)
            choices: [a, b, c]      # categorical only

    Parameters are grouped by training phase so that tokenizer-only or
    basemodel-only HPO runs sample only the relevant hyperparameters.
    """

    # Maps each parameter to its owning training phase.
    # Parameters listed under 'shared' are always included regardless of phase.
    PARAM_GROUPS: Dict[str, List[str]] = {
        "tokenizer": [
            "tokenizer_learning_rate",
            "tokenizer_pct_start",
            "tokenizer_div_factor",
            "tokenizer_max_grad_norm",
            "bsq_beta",
            "bsq_gamma0",
            "bsq_gamma",
            "bsq_zeta",
        ],
        "basemodel": [
            "predictor_learning_rate",
            "basemodel_pct_start",
            "basemodel_div_factor",
            "basemodel_max_grad_norm",
            "ffn_dropout_p",
            "attn_dropout_p",
            "resid_dropout_p",
            "token_dropout_p",
        ],
        "shared": [
            "adam_weight_decay",
            "adam_beta1",
            "adam_beta2",
            "batch_size",
            "accumulation_steps",
            "clip",
        ],
    }

    def __init__(self, search_space: Dict[str, Any], phase: str = "both") -> None:
        """
        Args:
            search_space: The parsed `hpo.search_space` dict from config YAML.
            phase:        Which phase to sample for: 'tokenizer', 'basemodel', or 'both'.
        """
        if phase not in ("tokenizer", "basemodel", "both"):
            raise ValueError(
                f"phase must be 'tokenizer', 'basemodel', or 'both'. Got: '{phase}'"
            )
        self.search_space = search_space
        self.phase = phase

    def _should_include(self, param_name: str) -> bool:
        """Returns True if param_name is relevant to the current HPO phase."""
        if self.phase == "both":
            return True
        phase_params = self.PARAM_GROUPS.get(self.phase, [])
        shared_params = self.PARAM_GROUPS["shared"]
        return param_name in phase_params or param_name in shared_params

    def suggest(self, trial: "optuna.Trial", param_name: str) -> Optional[Any]:
        """
        Suggests a single hyperparameter value using the Optuna trial API.

        Args:
            trial:      The current Optuna trial object.
            param_name: Name of the parameter to suggest.

        Returns:
            Suggested value, or None if param_name is not in the search space.

        Raises:
            ValueError: If the spec type is unrecognised.
        """
        if param_name not in self.search_space:
            return None

        spec = self.search_space[param_name]
        ptype = spec.get("type", "").lower()

        if ptype == "float":
            return trial.suggest_float(
                param_name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "int":
            return trial.suggest_int(
                param_name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "categorical":
            return trial.suggest_categorical(param_name, spec["choices"])
        else:
            raise ValueError(
                f"Unknown search space type '{ptype}' for parameter '{param_name}'. "
                f"Valid types: float, int, categorical."
            )

    def build_overrides(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """
        Samples all relevant hyperparameters for this phase and returns
        them as a flat dict suitable for `config.clone_with_overrides()`.

        Args:
            trial: The current Optuna trial object.

        Returns:
            Dict mapping parameter names to their suggested values.
        """
        overrides: Dict[str, Any] = {}
        for param_name in self.search_space:
            if self._should_include(param_name):
                value = self.suggest(trial, param_name)
                if value is not None:
                    overrides[param_name] = value
        return overrides


# ══════════════════════════════════════════════════════════════════════════════
# Optuna Trial Failure Callback
# ══════════════════════════════════════════════════════════════════════════════

def _build_failure_rate_callback(
    max_failure_rate: float = 0.50,
    min_trials_before_check: int = 4,
) -> "Callable":
    """
    Builds an Optuna callback that aborts the study if the trial failure
    rate exceeds a threshold. This prevents silent budget exhaustion when
    a systematic bug causes every trial to crash.

    Without this guard, `catch=(RuntimeError, ...)` in study.optimize will
    silently mark all trials as FAIL and consume the full n_trials budget
    before the user notices nothing is working.

    Args:
        max_failure_rate:          Fraction of failed trials that triggers abort [0.0, 1.0].
        min_trials_before_check:   Minimum number of trials before rate is evaluated.

    Returns:
        Callable matching Optuna's callback signature: f(study, trial) -> None.
    """
    logger = logging.getLogger(__name__)

    def callback(study: "optuna.Study", trial: "optuna.Trial") -> None:
        if trial.state != optuna.trial.TrialState.FAIL:
            return

        completed_and_failed = [
            t for t in study.trials
            if t.state in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
            )
        ]
        total = len(completed_and_failed)
        if total < min_trials_before_check:
            return

        failed_count = sum(
            1 for t in completed_and_failed
            if t.state == optuna.trial.TrialState.FAIL
        )
        fail_rate = failed_count / total

        logger.warning(
            f"Trial {trial.number} FAILED. "
            f"Cumulative failure rate: {failed_count}/{total} ({fail_rate:.0%})"
        )

        if fail_rate > max_failure_rate:
            logger.error(
                f"Failure rate {fail_rate:.0%} exceeds {max_failure_rate:.0%} "
                f"threshold after {total} trials. Aborting study to prevent "
                f"wasting the remaining trial budget. "
                f"Inspect per-trial logs for root cause."
            )
            study.stop()

    return callback


# ══════════════════════════════════════════════════════════════════════════════
# Objective Functions
# ══════════════════════════════════════════════════════════════════════════════

class TokenizerObjective:
    """
    Optuna objective function for tokenizer HPO.

    Each call to __call__ corresponds to one Optuna trial. The lifecycle is:
      1. Sample hyperparameters from the search space.
      2. Build a fully isolated trial config (unique save directory, PID-stamped).
      3. Load the pretrained tokenizer and apply config overrides.
      4. Run abbreviated training via the external train_tokenizer function.
      5. Return the validation loss as the optimization target.
      6. Deterministically release all GPU memory regardless of outcome.

    The trial directory is always deleted in the finally block. Trial artifacts
    are never shared between workers because the directory name includes both
    the trial number (assigned by Optuna DB) and the worker PID.
    """

    def __init__(
        self,
        base_config: CustomFinetuneConfig,
        search_space_builder: SearchSpaceBuilder,
        device: torch.device,
        trial_base_dir: str,
        safe_num_workers: int,
    ) -> None:
        """
        Args:
            base_config:           Original config (never mutated).
            search_space_builder:  Configured SearchSpaceBuilder instance.
            device:                CUDA device for this worker process.
            trial_base_dir:        Root directory under which per-trial dirs are created.
            safe_num_workers:      Pre-computed safe DataLoader worker count.
        """
        self.base_config = base_config
        self.ssb = search_space_builder
        self.device = device
        self.trial_base_dir = trial_base_dir
        self.safe_num_workers = safe_num_workers
        self._logger = logging.getLogger(
            f"{__name__}.TokenizerObjective.pid{os.getpid()}"
        )

    def __call__(self, trial: "optuna.Trial") -> float:
        """
        Executes one HPO trial for the tokenizer.

        Returns:
            Validation loss (float) to be minimized by Optuna.

        Raises:
            optuna.exceptions.TrialPruned: On model load failure or training crash.
        """
        tokenizer: Optional[NosTokenizer] = None
        trial_dir: Optional[str] = None

        try:
            # ── Step 1: Sample hyperparameters ────────────────────────────
            overrides = self.ssb.build_overrides(trial)

            # Use reduced epochs for HPO speed; suppress per-step log spam
            overrides["tokenizer_epochs"] = self.base_config.hpo_tokenizer_epochs
            overrides["log_interval"]     = 999_999
            overrides["num_workers"]      = self.safe_num_workers

            self._logger.info(
                f"Trial {trial.number} | Sampled overrides: {overrides}"
            )

            # ── Step 2: Build fully isolated trial config ──────────────────
            # PID inclusion guarantees uniqueness even if the Optuna DB
            # assigns the same trial number to two workers simultaneously
            # (which can happen when a worker restarts mid-study).
            worker_tag = f"pid{os.getpid()}_trial{trial.number}"
            trial_dir  = os.path.join(
                self.trial_base_dir, f"tokenizer_{worker_tag}"
            )
            os.makedirs(trial_dir, exist_ok=True)

            trial_config = self.base_config.clone_with_overrides(overrides)

            # Force ALL save paths into the isolated trial directory.
            # This prevents any overlap with the canonical save paths
            # that other workers or the main training pipeline might use.
            trial_config.tokenizer_save_path       = trial_dir
            trial_config.tokenizer_best_model_path = os.path.join(trial_dir, "best_model")
            trial_config.base_save_path            = trial_dir

            # ── Step 3: Load model ────────────────────────────────────────
            try:
                if getattr(trial_config, "pre_trained_tokenizer", True):
                    tokenizer = NosTokenizer.from_pretrained(
                        self.base_config.pretrained_tokenizer_path
                    )
                else:
                    arch_path = os.path.join(
                        self.base_config.pretrained_tokenizer_path, "config.json"
                    )
                    with open(arch_path, "r", encoding="utf-8") as f:
                        arch = json.load(f)
                    valid_keys = [
                        "d_in", "d_model", "n_heads", "ff_dim",
                        "n_enc_layers", "n_dec_layers", "ffn_dropout_p",
                        "attn_dropout_p", "resid_dropout_p", "s1_bits",
                        "s2_bits", "beta", "gamma0", "gamma", "zeta", "group_size",
                    ]
                    tokenizer = NosTokenizer(
                        **{k: arch[k] for k in valid_keys if k in arch}
                    )

                tokenizer = apply_bsq_overrides(tokenizer, trial_config)
                tokenizer = apply_dropout_overrides(tokenizer, trial_config)
                tokenizer = tokenizer.to(self.device)

            except FileNotFoundError as exc:
                self._logger.error(
                    f"Trial {trial.number}: Pretrained tokenizer not found "
                    f"at '{self.base_config.pretrained_tokenizer_path}': {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Model loading failed: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            # ── Step 4: Run training ──────────────────────────────────────
            trial_logger = _get_trial_logger(
                f"hpo_tok_trial_{trial.number}", trial_dir
            )

            try:
                from finetune_tokenizer import train_tokenizer  # type: ignore
                val_loss: float = train_tokenizer(
                    tokenizer, self.device, trial_config, trial_dir, trial_logger, trial=trial
                )
            except optuna.exceptions.TrialPruned:
                raise  # Pruning signals must propagate — never catch them
            except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
                self._logger.warning(
                    f"Trial {trial.number}: Training failed with expected "
                    f"transient error ({type(exc).__name__}): {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Training failed with unexpected "
                    f"error: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            self._logger.info(
                f"Trial {trial.number} COMPLETE | "
                f"val_loss={val_loss:.6f} | overrides={overrides}"
            )
            print(
                f"  [GPU {self.device}] Trial {trial.number}: "
                f"val_loss={val_loss:.6f}"
            )
            return val_loss

        finally:
            # ── Step 5: Deterministic GPU memory release ───────────────────
            # This executes regardless of success, pruning, or exception.
            # Skipping this causes PyTorch allocator cache accumulation,
            # which leads to OOM on trial N+K even though trial N "freed" its memory.
            # Explicitly delete local references so GC can collect them.
            if 'tokenizer' in locals():
                del tokenizer

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # ── Step 6: Delete trial artifacts ────────────────────────────
            if trial_dir and os.path.exists(trial_dir):
                shutil.rmtree(trial_dir, ignore_errors=True)


class BasemodelObjective:
    """
    Optuna objective function for predictor/basemodel HPO.

    Assumes a finetuned (or pretrained) tokenizer is available at a fixed
    path. The tokenizer is frozen (eval mode, no gradients) during basemodel
    HPO to isolate the predictor hyperparameter search from tokenizer variance.

    The lifecycle mirrors TokenizerObjective with the addition of loading
    and freezing the tokenizer before loading the predictor.
    """

    def __init__(
        self,
        base_config: CustomFinetuneConfig,
        search_space_builder: SearchSpaceBuilder,
        device: torch.device,
        trial_base_dir: str,
        tokenizer_path: str,
        safe_num_workers: int,
    ) -> None:
        """
        Args:
            base_config:           Original config (never mutated).
            search_space_builder:  Configured SearchSpaceBuilder instance.
            device:                CUDA device for this worker process.
            trial_base_dir:        Root directory for per-trial isolation.
            tokenizer_path:        Path to the finetuned tokenizer checkpoint.
            safe_num_workers:      Pre-computed safe DataLoader worker count.
        """
        self.base_config = base_config
        self.ssb = search_space_builder
        self.device = device
        self.trial_base_dir = trial_base_dir
        self.tokenizer_path = tokenizer_path
        self.safe_num_workers = safe_num_workers
        self._logger = logging.getLogger(
            f"{__name__}.BasemodelObjective.pid{os.getpid()}"
        )

    def __call__(self, trial: "optuna.Trial") -> float:
        """
        Executes one HPO trial for the basemodel predictor.

        Returns:
            Validation loss (float) to be minimized by Optuna.

        Raises:
            optuna.exceptions.TrialPruned: On model load failure or training crash.
        """
        tokenizer: Optional[NosTokenizer] = None
        model: Optional[Nos] = None
        trial_dir: Optional[str] = None

        try:
            # ── Step 1: Sample hyperparameters ────────────────────────────
            overrides = self.ssb.build_overrides(trial)
            overrides["basemodel_epochs"] = self.base_config.hpo_basemodel_epochs
            overrides["log_interval"]     = 999_999
            overrides["num_workers"]      = self.safe_num_workers

            self._logger.info(
                f"Trial {trial.number} | Sampled overrides: {overrides}"
            )

            # ── Step 2: Build isolated trial config ────────────────────────
            worker_tag = f"pid{os.getpid()}_trial{trial.number}"
            trial_dir  = os.path.join(
                self.trial_base_dir, f"basemodel_{worker_tag}"
            )
            os.makedirs(trial_dir, exist_ok=True)

            trial_config = self.base_config.clone_with_overrides(overrides)
            trial_config.finetuned_tokenizer_path  = self.tokenizer_path
            trial_config.basemodel_save_path       = trial_dir
            trial_config.basemodel_best_model_path = os.path.join(trial_dir, "best_model")
            trial_config.base_save_path            = trial_dir

            # ── Step 3: Load and freeze tokenizer ─────────────────────────
            try:
                tokenizer = NosTokenizer.from_pretrained(self.tokenizer_path)
                tokenizer = tokenizer.to(self.device)
                tokenizer.eval()
                for param in tokenizer.parameters():
                    param.requires_grad_(False)
            except FileNotFoundError as exc:
                self._logger.error(
                    f"Trial {trial.number}: Tokenizer not found "
                    f"at '{self.tokenizer_path}': {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Tokenizer loading failed: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            # ── Step 4: Load predictor model ──────────────────────────────
            try:
                if getattr(trial_config, "pre_trained_predictor", True):
                    model = Nos.from_pretrained(
                        self.base_config.pretrained_predictor_path
                    )
                else:
                    arch_path = os.path.join(
                        self.base_config.pretrained_predictor_path, "config.json"
                    )
                    with open(arch_path, "r", encoding="utf-8") as f:
                        arch = json.load(f)
                    valid_keys = [
                        "s1_bits", "s2_bits", "n_layers", "d_model",
                        "n_heads", "ff_dim", "ffn_dropout_p", "attn_dropout_p",
                        "resid_dropout_p", "token_dropout_p", "learn_te",
                    ]
                    model = Nos(**{k: arch[k] for k in valid_keys if k in arch})

                model = apply_dropout_overrides(model, trial_config)
                model = model.to(self.device)

            except FileNotFoundError as exc:
                self._logger.error(
                    f"Trial {trial.number}: Predictor model not found "
                    f"at '{self.base_config.pretrained_predictor_path}': {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Predictor loading failed: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            # ── Step 5: Run training ──────────────────────────────────────
            trial_logger = _get_trial_logger(
                f"hpo_base_trial_{trial.number}", trial_dir
            )

            try:
                from finetune_base_model import train_model  # type: ignore
                val_loss: float = train_model(
                    model, tokenizer, self.device,
                    trial_config, trial_dir, trial_logger, trial=trial
                )
            except optuna.exceptions.TrialPruned:
                raise
            except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
                self._logger.warning(
                    f"Trial {trial.number}: Training failed with expected "
                    f"transient error ({type(exc).__name__}): {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Training failed with unexpected "
                    f"error: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            self._logger.info(
                f"Trial {trial.number} COMPLETE | "
                f"val_loss={val_loss:.6f} | overrides={overrides}"
            )
            print(
                f"  [GPU {self.device}] Trial {trial.number}: "
                f"val_loss={val_loss:.6f}"
            )
            return val_loss

        finally:
            # Explicitly delete local references so GC can collect them.
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if trial_dir and os.path.exists(trial_dir):
                shutil.rmtree(trial_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# HPO Runner
# ══════════════════════════════════════════════════════════════════════════════

class NosHPOTuner:
    """
    Main HPO orchestrator for the Nos two-phase training pipeline.

    Responsibilities:
    - Device setup with CUDA_VISIBLE_DEVICES awareness for multi-GPU isolation.
    - Optuna study creation with SQLite WAL mode and concurrency hardening.
    - Tokenizer and basemodel HPO phases with correct sampler configuration.
    - Atomic result persistence (race-condition-safe JSON writes).
    - Best parameter application back to YAML config.

    Multi-GPU Usage:
        Do NOT use torchrun. Instead, launch independent processes:

            CUDA_VISIBLE_DEVICES=0 python hpo_tuner.py --config cfg.yaml &
            CUDA_VISIBLE_DEVICES=1 python hpo_tuner.py --config cfg.yaml &
            ...
            # Or use launch_hpo.sh which automates this pattern.

        All workers share the same Optuna storage (SQLite or PostgreSQL) and
        independently pick up and execute trials. No synchronisation barriers.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialises the HPO tuner, sets up device, logging, and directories.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            RuntimeError: If Optuna is not installed.
            FileNotFoundError: If config_path does not exist.
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError(
                "Optuna is not installed. "
                "Run: pip install optuna plotly"
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        self.config_path = config_path
        self.config = CustomFinetuneConfig(config_path)

        # ── Device setup (must precede logging setup for log annotations) ─
        self.device = self._setup_device()
        self._worker_tag = f"pid{os.getpid()}_{str(self.device).replace(':', '')}"

        # ── Directory layout ───────────────────────────────────────────────
        self.trial_base_dir = os.path.join(
            self.config.base_save_path, "hpo_trials"
        )
        self.results_dir = os.path.join(
            self.config.base_save_path, "hpo_results"
        )
        self.log_dir = os.path.join(
            self.config.base_save_path, "hpo_logs"
        )
        for directory in (self.trial_base_dir, self.results_dir, self.log_dir):
            os.makedirs(directory, exist_ok=True)

        # ── Logging ───────────────────────────────────────────────────────
        _configure_root_logger(self.log_dir, self._worker_tag)
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f"NosHPOTuner initialised | config={config_path} | "
            f"device={self.device} | worker_tag={self._worker_tag}"
        )

        # ── Pre-compute safe DataLoader concurrency ────────────────────────
        self._safe_num_workers = _compute_safe_num_workers(
            n_gpu_workers=self._detect_n_concurrent_workers()
        )

        # Results accumulator for this worker's session
        self.results: Dict[str, Any] = {}

    # ── Device Setup ──────────────────────────────────────────────────────────

    def _setup_device(self) -> torch.device:
        """
        Selects the correct GPU device for this worker process.

        Priority resolution order:
          1. CUDA_VISIBLE_DEVICES environment variable (set by launch_hpo.sh).
             When set to a single GPU ID (e.g., "3"), CUDA remaps it to
             logical index 0 within this process. We use cuda:0 in this case.
          2. device_id from YAML config (for single-machine manual override).
          3. CPU fallback if CUDA is unavailable.

        Returns:
            Resolved torch.device for this worker.
        """
        if not (self.config.use_cuda and torch.cuda.is_available()):
            logging.getLogger(__name__).info(
                "CUDA not available or disabled in config. Using CPU."
            )
            return torch.device("cpu")

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        if cuda_visible is not None:
            # Process-isolation mode: the bash launcher set this env var.
            # CUDA remaps the assigned physical GPU to logical index 0.
            visible_list = [
                v.strip() for v in cuda_visible.split(",")
                if v.strip().lstrip("-").isdigit()
            ]

            if len(visible_list) == 0:
                # Malformed or empty CUDA_VISIBLE_DEVICES — fall back to config
                logical_id  = int(self.config.device_id)
                physical_id = logical_id
                logging.getLogger(__name__).warning(
                    f"CUDA_VISIBLE_DEVICES='{cuda_visible}' is not a valid "
                    f"GPU index list. Falling back to config device_id={logical_id}."
                )
            elif len(visible_list) == 1:
                # Exactly one GPU visible — correct process isolation
                logical_id  = 0
                physical_id = int(visible_list[0])
            else:
                # Multiple GPUs visible — warn and use first
                logical_id  = 0
                physical_id = int(visible_list[0])
                logging.getLogger(__name__).warning(
                    f"CUDA_VISIBLE_DEVICES='{cuda_visible}' exposes multiple GPUs. "
                    f"For true process isolation, assign one GPU per worker. "
                    f"Using logical cuda:0 (physical GPU {physical_id})."
                )
        else:
            # Single-machine mode — use config device_id directly
            logical_id  = int(self.config.device_id)
            physical_id = logical_id

        torch.cuda.set_device(logical_id)
        device = torch.device(f"cuda:{logical_id}")

        gpu_props = torch.cuda.get_device_properties(device)
        logging.getLogger(__name__).info(
            f"[PID {os.getpid()}] Using cuda:{logical_id} "
            f"(physical GPU {physical_id}: {gpu_props.name}, "
            f"{gpu_props.total_memory / 1e9:.1f} GB VRAM)"
        )
        return device

    # ── Worker Concurrency Detection ──────────────────────────────────────────

    def _detect_n_concurrent_workers(self) -> int:
        """
        Estimates the number of concurrent GPU workers in this HPO run.

        Used to compute a safe per-process DataLoader num_workers count.
        Reads CUDA_VISIBLE_DEVICES if set by the launcher; otherwise assumes
        single-worker mode.

        Returns:
            Estimated number of concurrent GPU workers (>=1).
        """
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and cuda_visible not in ("NoDevFiles", "-1"):
            n = len([v for v in cuda_visible.split(",") if v.strip().isdigit()])
            if n > 0:
                return n
        # If CUDA_VISIBLE_DEVICES is not set, count all available GPUs
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
        return 1

    # ── Optuna Internals ──────────────────────────────────────────────────────

    def _build_sampler(self) -> "optuna.samplers.BaseSampler":
        """
        Constructs the Optuna sampler.

        For async multi-worker HPO, TPESampler with constant_liar=True is
        strongly recommended. Without constant_liar, all workers sample
        simultaneously before any trial completes, so the TPE model has
        zero completed trials to learn from and every worker independently
        suggests the same near-identical parameters (correlated exploration).

        constant_liar=True tells TPE to treat in-progress trials as if they
        returned the current worst observed value, encouraging each worker
        to explore a different region of the search space.

        Returns:
            Configured sampler instance.
        """
        name = self.config.hpo_sampler.lower()
        seed = getattr(self.config, "seed", 42)
        # Need enough startup trials before TPE's probabilistic model is reliable.
        # Require at least 2 full rounds of workers to complete before TPE kicks in.
        n_workers  = self._detect_n_concurrent_workers()
        n_startup  = max(10, 2 * n_workers)

        if name == "tpe":
            return TPESampler(
                seed=seed,
                multivariate=True,      # Model parameter correlations
                constant_liar=True,     # Critical for async multi-worker HPO
                n_startup_trials=n_startup,
            )
        elif name == "random":
            return RandomSampler(seed=seed)
        elif name == "cmaes":
            self._logger.warning(
                "CMA-ES sampler does not support constant_liar. "
                "Workers will suggest correlated parameters in async mode. "
                "TPE is recommended for distributed HPO."
            )
            return CmaEsSampler(seed=seed)
        else:
            self._logger.warning(
                f"Unknown sampler '{name}'. Defaulting to TPE with constant_liar."
            )
            return TPESampler(
                seed=seed,
                constant_liar=True,
                n_startup_trials=n_startup,
            )

    def _build_pruner(self) -> "optuna.pruners.BasePruner":
        """
        Constructs the Optuna pruner.

        MedianPruner is conservative and works well for training workloads
        where intermediate values are reported via trial.report().
        HyperbandPruner is more aggressive and better for very large search spaces.

        Returns:
            Configured pruner instance.
        """
        name = self.config.hpo_pruner.lower()
        if name == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1,
                interval_steps=1,
            )
        elif name == "hyperband":
            return HyperbandPruner()
        else:
            return NopPruner()

    def _create_study(self, study_name_suffix: str = "") -> "optuna.Study":
        """
        Creates or loads an Optuna study from the configured storage backend.

        For SQLite storage, applies WAL (Write-Ahead Logging) mode via a
        connection event hook. WAL allows concurrent readers while one
        writer is active, dramatically reducing lock contention when 8
        workers simultaneously commit trial results.

        For non-SQLite storage (PostgreSQL, MySQL), the URI is passed
        directly without modification.

        load_if_exists=True is mandatory for multi-worker mode: all workers
        must join the same existing study rather than each creating a new one.

        Args:
            study_name_suffix: Appended to base study name (e.g., "_tokenizer").

        Returns:
            Configured optuna.Study instance.
        """
        study_name = f"{self.config.hpo_study_name}{study_name_suffix}"
        storage_uri = self.config.hpo_storage

        self._logger.info(
            f"Creating/loading study '{study_name}' | storage={storage_uri}"
        )

        if storage_uri is not None and storage_uri.startswith("sqlite"):
            storage = self._build_sqlite_storage(storage_uri)
        else:
            # None → in-memory (single process only)
            # postgresql:// or mysql:// → passed through directly
            storage = storage_uri

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=self.config.hpo_direction,
            sampler=self._build_sampler(),
            pruner=self._build_pruner(),
            load_if_exists=True,  # Mandatory for multi-worker mode
        )
        self._logger.info(
            f"Study ready: {len(study.trials)} existing trials loaded."
        )
        return study

    def _build_sqlite_storage(self, uri: str) -> "RDBStorage":
        db_path = uri.split("?")[0].replace("sqlite:///", "")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        storage = RDBStorage(
            url=uri,
            engine_kwargs={
                "connect_args": {
                     "timeout": 60,
                     "check_same_thread": False,
                },
                "poolclass": NullPool,  # CRITICAL FIX: Disable pooling to prevent lock storms
           },
           skip_compatibility_check=False,
       )

        try:
             from sqlalchemy import event as sa_event

             @sa_event.listens_for(storage.engine, "connect")
             def _apply_wal_pragmas(dbapi_connection, connection_record):
                 cursor = dbapi_connection.cursor()
                 cursor.execute("PRAGMA journal_mode=WAL")
                 cursor.execute("PRAGMA synchronous=NORMAL")
                 cursor.execute("PRAGMA busy_timeout=300000")
                 cursor.close()

             self._logger.info(
                 "SQLite WAL mode, NullPool, and busy_timeout=300s applied via connection hook."
             )
        except Exception as exc:
             self._logger.warning(
                 f"Could not apply WAL mode pragmas: {exc}. "
                 f"Falling back to URI timeout parameter only."
            )

        return storage

    # ── Phase: Tokenizer ──────────────────────────────────────────────────────

    def tune_tokenizer(
        self, n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs Optuna HPO for the tokenizer training phase.

        Args:
            n_trials: Number of trials to run. Overrides config if provided.

        Returns:
            Dict with keys 'value' (best val_loss) and 'params' (best hyperparams).
        """
        n = n_trials or self.config.hpo_n_trials
        self._logger.info(
            f"Starting tokenizer HPO: {n} trials | device={self.device}"
        )
        print(f"\n{'='*60}")
        print(f"HPO Phase 1: Tokenizer | {n} trials | {self.device}")
        print(f"{'='*60}")

        ssb = SearchSpaceBuilder(
            self.config.hpo_search_space, phase="tokenizer"
        )
        objective = TokenizerObjective(
            base_config=self.config,
            search_space_builder=ssb,
            device=self.device,
            trial_base_dir=self.trial_base_dir,
            safe_num_workers=self._safe_num_workers,
        )
        study = self._create_study("_tokenizer")
        failure_callback = _build_failure_rate_callback(
            max_failure_rate=0.5,
            min_trials_before_check=4,
        )

        study.optimize(
            objective,
            n_trials=n,
            show_progress_bar=True,
            # Only catch expected transient failures caused by extreme
            # hyperparameter values (OOM from huge batch, NaN from bad LR).
            # Systematic bugs (wrong data path, import error) will NOT be
            # caught here — they will raise and crash the worker intentionally.
            catch=(
                RuntimeError,
                ValueError,
                torch.cuda.OutOfMemoryError,
            ),
            callbacks=[failure_callback],
        )

        best = self._extract_best(study)
        self.results["tokenizer"] = best

        self._logger.info(
            f"Tokenizer HPO complete | best_val_loss={best['value']:.6f} | "
            f"best_params={best['params']}"
        )
        print(f"\n✅ Best tokenizer val_loss: {best['value']:.6f}")
        print(f"   Best params: {best['params']}")

        self._save_study_results(study, "tokenizer")
        return best

    # ── Phase: Basemodel ──────────────────────────────────────────────────────

    def tune_basemodel(
        self,
        tokenizer_path: Optional[str] = None,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs Optuna HPO for the basemodel predictor training phase.

        Requires a finetuned (or original pretrained) tokenizer checkpoint.
        The tokenizer is frozen throughout all basemodel trials.

        Args:
            tokenizer_path: Path to tokenizer checkpoint. Defaults to
                            config.finetuned_tokenizer_path.
            n_trials:       Number of trials. Overrides config if provided.

        Returns:
            Dict with keys 'value' (best val_loss) and 'params' (best hyperparams).

        Raises:
            FileNotFoundError: If tokenizer_path does not exist.
        """
        n = n_trials or self.config.hpo_n_trials

        if tokenizer_path is None:
            tokenizer_path = self.config.finetuned_tokenizer_path
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at: '{tokenizer_path}'\n"
                f"Run tokenizer finetuning or tokenizer HPO first, or "
                f"provide --tokenizer-path explicitly."
            )

        self._logger.info(
            f"Starting basemodel HPO: {n} trials | "
            f"device={self.device} | tokenizer={tokenizer_path}"
        )
        print(f"\n{'='*60}")
        print(f"HPO Phase 2: Basemodel | {n} trials | {self.device}")
        print(f"Tokenizer: {tokenizer_path}")
        print(f"{'='*60}")

        ssb = SearchSpaceBuilder(
            self.config.hpo_search_space, phase="basemodel"
        )
        objective = BasemodelObjective(
            base_config=self.config,
            search_space_builder=ssb,
            device=self.device,
            trial_base_dir=self.trial_base_dir,
            tokenizer_path=tokenizer_path,
            safe_num_workers=self._safe_num_workers,
        )
        study = self._create_study("_basemodel")
        failure_callback = _build_failure_rate_callback(
            max_failure_rate=0.5,
            min_trials_before_check=4,
        )

        study.optimize(
            objective,
            n_trials=n,
            show_progress_bar=True,
            catch=(
                RuntimeError,
                ValueError,
                torch.cuda.OutOfMemoryError,
            ),
            callbacks=[failure_callback],
        )

        best = self._extract_best(study)
        self.results["basemodel"] = best

        self._logger.info(
            f"Basemodel HPO complete | best_val_loss={best['value']:.6f} | "
            f"best_params={best['params']}"
        )
        print(f"\n✅ Best basemodel val_loss: {best['value']:.6f}")
        print(f"   Best params: {best['params']}")

        self._save_study_results(study, "basemodel")
        return best

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    def tune_full_pipeline(self) -> Dict[str, Any]:
        """
        Runs the complete two-phase HPO pipeline:
          1. Tokenizer HPO → find best tokenizer hyperparameters.
          2. Full tokenizer training with best params (to produce a stable checkpoint).
          3. Basemodel HPO using the stable tokenizer checkpoint.

        Returns:
            Dict containing results from both phases.
        """
        all_results: Dict[str, Any] = {}

        if self.config.hpo_optimize_tokenizer:
            tok_best = self.tune_tokenizer()
            all_results["tokenizer"] = tok_best

            # Train the tokenizer with best params at full epochs so the
            # basemodel HPO uses a well-converged tokenizer checkpoint.
            print("\n📦 Training tokenizer with best params for basemodel HPO...")
            self._logger.info(
                "Training full tokenizer with best HPO params before basemodel phase."
            )
            best_tok_config = self.config.clone_with_overrides(tok_best["params"])
            self._train_best_tokenizer(best_tok_config)

        if self.config.hpo_optimize_basemodel:
            tok_path = self.config.tokenizer_best_model_path
            base_best = self.tune_basemodel(tokenizer_path=tok_path)
            all_results["basemodel"] = base_best

        self._print_final_report(all_results)
        return all_results

    def _train_best_tokenizer(self, config: CustomFinetuneConfig) -> None:
        """
        Runs one full (non-HPO) tokenizer training pass with the given config.
        Used to produce a stable tokenizer checkpoint before basemodel HPO.

        Args:
            config: Config instance with best HPO hyperparameters applied.
        """
        tokenizer: Optional[NosTokenizer] = None
        try:
            if getattr(config, "pre_trained_tokenizer", True):
                tokenizer = NosTokenizer.from_pretrained(
                    config.pretrained_tokenizer_path
                )
            else:
                raise ValueError(
                    "pre_trained_tokenizer=False requires manual model "
                    "instantiation before calling _train_best_tokenizer."
                )
            tokenizer = apply_bsq_overrides(tokenizer, config)
            tokenizer = apply_dropout_overrides(tokenizer, config)
            tokenizer = tokenizer.to(self.device)

            os.makedirs(config.tokenizer_save_path, exist_ok=True)
            logger = _get_trial_logger("hpo_best_tokenizer", config.tokenizer_save_path)

            from finetune_tokenizer import train_tokenizer  # type: ignore
            train_tokenizer(
                tokenizer, self.device, config,
                config.tokenizer_save_path, logger,
            )
            self._logger.info(
                f"Best tokenizer training complete. "
                f"Saved to: {config.tokenizer_save_path}"
            )
        finally:
            release_gpu_memory(tokenizer)

    # ── Results Persistence ───────────────────────────────────────────────────

    @staticmethod
    def _extract_best(study: "optuna.Study") -> Dict[str, Any]:
        """
        Safely extracts best trial information from a study.

        Handles the edge case where no trials completed successfully
        (all pruned or failed), which would cause study.best_value to raise.

        Args:
            study: Completed Optuna study.

        Returns:
            Dict with 'value' and 'params' keys.
        """
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            logging.getLogger(__name__).error(
                "No trials completed successfully. "
                "All trials were pruned or failed. "
                "Inspect per-trial logs for root cause."
            )
            return {"value": float("inf"), "params": {}}

        return {
            "value": study.best_value,
            "params": study.best_params,
        }

    def _save_study_results(
        self, study: "optuna.Study", phase: str
    ) -> None:
        """
        Persists all trial results for a study phase to a JSON file.

        Uses an atomic write pattern (write to .tmp → os.replace to final path)
        to guarantee the output file is never in a partially-written state,
        even if 8 workers call this simultaneously. os.replace() is atomic
        on POSIX and near-atomic on Windows (the last writer wins, but the
        file is never corrupt).

        Individual worker results are tagged with the worker PID so they
        can be distinguished if needed. The final merged view comes from
        reading the Optuna study, which contains all workers' trials.

        Args:
            study: The completed Optuna study.
            phase: Phase label ('tokenizer' or 'basemodel').
        """
        trials_data = []
        for t in study.trials:
            if t.value is not None:
                trials_data.append({
                    "number":            t.number,
                    "value":             t.value,
                    "params":            t.params,
                    "state":             str(t.state),
                    "datetime_start":    (
                        t.datetime_start.isoformat()
                        if t.datetime_start else None
                    ),
                    "datetime_complete": (
                        t.datetime_complete.isoformat()
                        if t.datetime_complete else None
                    ),
                })

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        payload = {
            "best_value":   study.best_value  if completed else None,
            "best_params":  study.best_params if completed else {},
            "n_trials":     len(study.trials),
            "n_completed":  len(completed),
            "n_pruned":     sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ),
            "n_failed":     sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.FAIL
            ),
            "trials":       trials_data,
            "timestamp":    datetime.datetime.now().isoformat(),
            "worker_tag":   self._worker_tag,
        }

        # Atomic write: temp file → rename
        final_path = os.path.join(self.results_dir, f"{phase}_trials.json")
        tmp_path   = f"{final_path}.{self._worker_tag}.tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)

            # ── TASK 2.1: Windows race-condition backoff for atomic writes ──
            # OneDrive or other indexing services often hold transient read-locks
            # on newly created files. This ensures os.replace doesn't crash the worker.
            for attempt in range(5):
                try:
                    os.replace(tmp_path, final_path)  # Atomic on POSIX, near-atomic on Windows
                    self._logger.info(f"Results atomically written to: {final_path}")
                    print(f"📊 Results saved to: {final_path}")
                    break
                except PermissionError:
                    if attempt == 4:
                        raise  # Re-raise after exhausting retries
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s

        except Exception as exc:
            self._logger.error(
                f"Failed to save results to {final_path}: {exc}", exc_info=True
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # ── Optional: Save visualisation HTML plots ────────────────────────
        self._save_visualisations(study, phase)

    def _save_visualisations(
        self, study: "optuna.Study", phase: str
    ) -> None:
        """
        Attempts to save Optuna HTML visualisation plots.

        Silently skips if plotly is not installed or no completed trials exist.

        Args:
            study: Completed Optuna study.
            phase: Phase label for file naming.
        """
        try:
            import optuna.visualization as vis
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed) < 2:
                return  # Not enough trials for meaningful plots

            plots = {
                f"{phase}_param_importances.html": vis.plot_param_importances,
                f"{phase}_optimization_history.html": vis.plot_optimization_history,
                f"{phase}_parallel_coordinate.html": vis.plot_parallel_coordinate,
            }
            for filename, plot_fn in plots.items():
                try:
                    fig = plot_fn(study)
                    fig.write_html(os.path.join(self.results_dir, filename))
                except Exception:
                    pass  # Individual plot failure should not abort saving

            self._logger.info(
                f"Visualisation plots saved to: {self.results_dir}"
            )
            print(f"📈 Plots saved to: {self.results_dir}")
        except ImportError:
            pass  # plotly not installed — skip silently

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _print_final_report(self, results: Dict[str, Any]) -> None:
        """Prints a formatted summary of all HPO results to stdout."""
        print(f"\n{'='*60}")
        print("HPO FINAL REPORT")
        print(f"{'='*60}")
        for phase, result in results.items():
            print(f"\n{'─'*40}")
            print(f"Phase: {phase.upper()}")
            if result["value"] == float("inf"):
                print("  ⚠️  No trials completed successfully.")
            else:
                print(f"  Best val loss: {result['value']:.6f}")
                print(f"  Best hyperparameters:")
                for k, v in sorted(result["params"].items()):
                    print(f"    {k:<40}: {v}")
        print(f"\n{'='*60}")

    def print_importance_report(self) -> None:
        """
        Prints a ranked hyperparameter importance table using Optuna's
        fANOVA estimator. Requires at least 4 completed trials and a
        persistent storage backend (not in-memory).
        """
        if not OPTUNA_AVAILABLE:
            return

        storage_uri = self.config.hpo_storage
        if storage_uri is None:
            print(
                "\n⚠️  Parameter importance report requires persistent storage "
                "(set hpo.storage in config). Skipping."
            )
            return

        print(f"\n{'='*60}")
        print("HYPERPARAMETER IMPORTANCE RANKING")
        print(f"{'='*60}")

        for phase in ("tokenizer", "basemodel"):
            study_name = f"{self.config.hpo_study_name}_{phase}"
            try:
                study = optuna.load_study(
                    study_name=study_name, storage=storage_uri
                )
                completed = [
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if len(completed) < 4:
                    print(
                        f"\n{phase.upper()}: Not enough completed trials "
                        f"({len(completed)}/4 minimum) for importance analysis."
                    )
                    continue

                importances = optuna.importance.get_param_importances(study)
                print(f"\n{phase.upper()} importances:")
                for param, imp in sorted(
                    importances.items(), key=lambda x: x[1], reverse=True
                ):
                    bar = "█" * int(imp * 40)
                    print(f"  {param:<42} {imp:.4f}  {bar}")
            except Exception as exc:
                self._logger.debug(
                    f"Could not compute importances for {phase}: {exc}"
                )

    # ── Config Application ────────────────────────────────────────────────────

    def apply_best_to_config(
        self, output_config_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Writes the best found hyperparameters from all completed phases back
        into a new YAML config file. The original config is never modified.

        The `clip` parameter is a data-preprocessing scalar that belongs in
        the `data:` YAML block (where CustomFinetuneConfig reads it from).
        All other optimised parameters are written to the `training:` block.

        Args:
            output_config_path: Output path. Defaults to
                                <original_name>_hpo_best.yaml.

        Returns:
            Path to the written config file, or None if no results exist.
        """
        if not self.results:
            print(
                "No HPO results to apply. "
                "Run tune_tokenizer(), tune_basemodel(), or tune_full_pipeline() first."
            )
            return None

        # Load raw YAML to preserve comments and formatting as much as possible
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # ── Merge all phase best params into a single flat dict ────────────
        all_best_params: Dict[str, Any] = {}
        for phase_results in self.results.values():
            all_best_params.update(phase_results.get("params", {}))

        # ── FIX: Route `clip` to the `data` block, not `training` ─────────
        # `clip` is a data-preprocessing scalar read by CustomFinetuneConfig
        # exclusively from config["data"]["clip"]. Writing it into the
        # `training:` block causes it to be silently ignored at load time,
        # leaving the old (un-optimised) clip value in effect.
        #
        # .pop() removes `clip` from all_best_params so the subsequent
        # .update() call does NOT write a duplicate (and unreachable) `clip`
        # key into the `training:` block.
        if "clip" in all_best_params:
            optimized_clip = all_best_params.pop("clip")
            raw_config.setdefault("data", {})["clip"] = optimized_clip
            self._logger.info(
                f"Routed optimised clip={optimized_clip} → data.clip "
                f"(removed from training block to prevent shadowing)."
            )

        # ── Inject remaining parameters into the training block ────────────
        raw_config.setdefault("training", {}).update(all_best_params)

        # ── Stamp experiment metadata ──────────────────────────────────────
        raw_config.setdefault("experiment", {})["hpo_applied"] = True
        raw_config["experiment"]["hpo_applied_timestamp"] = (
            datetime.datetime.now().isoformat()
        )

        if output_config_path is None:
            base, ext = os.path.splitext(self.config_path)
            output_config_path = f"{base}_hpo_best{ext}"

        with open(output_config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                raw_config, f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
                sort_keys=False,
            )

        self._logger.info(f"Best config written to: {output_config_path}")
        print(f"\n✅ Best config written to: {output_config_path}")
        return output_config_path


# ══════════════════════════════════════════════════════════════════════════════
# Utility: Safe DataLoader Worker Count
# ══════════════════════════════════════════════════════════════════════════════

def _compute_safe_num_workers(n_gpu_workers: int = 1) -> int:
    """
    Computes a per-process DataLoader num_workers count that avoids
    CPU and disk I/O saturation when N GPU processes run simultaneously.

    With N=8 GPUs and num_workers=6 (from config), the system would spawn
    8×6=48 DataLoader processes competing for CPU cores and disk bandwidth,
    degrading all workers below single-process performance.

    Formula:
        available_cores = total_cpu_cores × 0.8  (20% reserved for OS/Optuna)
        per_process     = floor(available_cores / n_gpu_workers)
        safe_count      = clamp(per_process, 1, 4)

    The upper clamp of 4 reflects that DataLoader parallelism has strongly
    diminishing returns beyond 4 workers for most I/O workloads, particularly
    when reading from a shared NFS or NVMe that is already multi-reader bound.

    Args:
        n_gpu_workers: Number of concurrent GPU worker processes.

    Returns:
        Safe num_workers value for DataLoader in each worker process.
    """
    total_cores     = multiprocessing.cpu_count()
    available_cores = max(1, int(total_cores * 0.8))
    per_process     = max(1, available_cores // max(1, n_gpu_workers))
    safe_count      = min(per_process, 4)

    logging.getLogger(__name__).info(
        f"DataLoader num_workers: {safe_count} "
        f"(CPU cores={total_cores}, GPU workers={n_gpu_workers})"
    )
    return safe_count


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpo_tuner.py",
        description=(
            "Asynchronous Hyperparameter Tuning for Nos Model Finetuning.\n\n"
            "Single GPU:\n"
            "  python hpo_tuner.py --config configs/config.yaml\n\n"
            "Multi-GPU (8x A40 cluster):\n"
            "  ./launch_hpo.sh --config configs/config.yaml --n-trials 30"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="both",
        choices=["tokenizer", "basemodel", "both"],
        help="Which training phase to tune. Default: both.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override n_trials from config.",
    )
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write best params back to a new config file after tuning.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to finetuned tokenizer (required for --phase basemodel).",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="Output path for best config YAML (default: <config>_hpo_best.yaml).",
    )
    parser.add_argument(
        "--show-importance",
        action="store_true",
        help="Print hyperparameter importance ranking after tuning.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args   = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print(
            "ERROR: Optuna is not installed.\n"
            "Install it with: pip install optuna plotly"
        )
        sys.exit(1)

    # ── Validate phase/tokenizer-path combination ──────────────────────────
    if args.phase == "basemodel" and args.tokenizer_path is None:
        # Will fall back to config.finetuned_tokenizer_path in tune_basemodel;
        # log a warning but don't abort here since config may have it set.
        print(
            "⚠️  --phase basemodel without --tokenizer-path. "
            "Will use finetuned_tokenizer path from config."
        )

    # ── Initialise tuner ──────────────────────────────────────────────────
    try:
        tuner = NosHPOTuner(args.config)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("\nHPO Worker Configuration:")
    print(f"  Config:      {args.config}")
    print(f"  Phase:       {args.phase}")
    print(f"  Trials:      {args.n_trials or tuner.config.hpo_n_trials}")
    print(f"  Sampler:     {tuner.config.hpo_sampler}")
    print(f"  Pruner:      {tuner.config.hpo_pruner}")
    print(f"  Storage:     {tuner.config.hpo_storage or 'in-memory (single worker only)'}")
    print(f"  Device:      {tuner.device}")
    print(f"  DL Workers:  {tuner._safe_num_workers} per process")
    print(f"  Worker tag:  {tuner._worker_tag}")

    wall_start = time.perf_counter()

    # ── Run HPO ───────────────────────────────────────────────────────────
    try:
        if args.phase == "both":
            results = tuner.tune_full_pipeline()
        elif args.phase == "tokenizer":
            results = tuner.tune_tokenizer(n_trials=args.n_trials)
        elif args.phase == "basemodel":
            results = tuner.tune_basemodel(
                tokenizer_path=args.tokenizer_path,
                n_trials=args.n_trials,
            )
    except KeyboardInterrupt:
        print(
            "\n⚠️  HPO interrupted by user. "
            "Completed trials have been saved to the study database."
        )
        sys.exit(0)

    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n⏱  Total HPO wall time: {wall_elapsed / 60:.1f} minutes")

    # ── Post-run actions ──────────────────────────────────────────────────
    if args.apply_best:
        tuner.apply_best_to_config(args.output_config)

    if args.show_importance:
        tuner.print_importance_report()


if __name__ == "__main__":
    main()