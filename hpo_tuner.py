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