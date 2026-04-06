"""
Dynamic Hyperparameter Tuning Engine v2
Tunes Tier 1 & Tier 2 hyperparameters live during training.
"""

import os
import json
import math
import time
import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.optim as optim


# ─────────────────────────────────────────────────────────────
# Loss Signal Analyzer
# ─────────────────────────────────────────────────────────────

class LossSignalAnalyzer:
    """Maintains rolling history and computes diagnostic signals."""

    def __init__(self, max_step_history: int = 1000,
                 max_epoch_history: int = 200):
        self.train_losses: deque = deque(maxlen=max_epoch_history)
        self.val_losses: deque = deque(maxlen=max_epoch_history)
        self.grad_norms: deque = deque(maxlen=max_step_history)
        self.step_losses: deque = deque(maxlen=max_step_history)

    def add_step(self, train_loss: float, grad_norm: float = None):
        if math.isfinite(train_loss):
            self.step_losses.append(train_loss)
        if grad_norm is not None and math.isfinite(grad_norm):
            self.grad_norms.append(grad_norm)

    def add_epoch(self, train_loss: float, val_loss: float):
        if math.isfinite(train_loss):
            self.train_losses.append(train_loss)
        if math.isfinite(val_loss):
            self.val_losses.append(val_loss)

    def get_gap_ratio(self) -> float:
        """(val - train) / |train|"""
        if len(self.train_losses) == 0 or len(self.val_losses) == 0:
            return 0.0
        t = self.train_losses[-1]
        v = self.val_losses[-1]
        return (v - t) / (abs(t) + 1e-10)

    def is_plateau(self, window: int = 4,
                   threshold: float = 5e-4) -> bool:
        if len(self.val_losses) < window:
            return False
        recent = list(self.val_losses)[-window:]
        best = min(recent)
        worst = max(recent)
        if best < 1e-10:
            return False
        severity = (worst - best) / best
        return severity < threshold

    def is_overfitting(self, threshold: float = 0.10) -> bool:
        return self.get_gap_ratio() > threshold

    def is_severely_overfitting(self,
                                threshold: float = 0.30) -> bool:
        return self.get_gap_ratio() > threshold

    def is_grad_exploding(self,
                          threshold: float = 8.0) -> bool:
        if len(self.grad_norms) < 5:
            return False
        recent = list(self.grad_norms)[-5:]
        return np.mean(recent) > threshold

    def is_loss_spiking(self) -> bool:
        if len(self.step_losses) < 20:
            return False
        recent = list(self.step_losses)[-20:]
        mean = np.mean(recent)
        std = np.std(recent)
        return (mean > 1e-8) and (std / mean > 0.4)

    def get_trend(self, n_epochs: int = 5) -> str:
        if len(self.val_losses) < 2:
            return "unknown"
        recent = list(self.val_losses)[-n_epochs:]
        if len(recent) < 2:
            return "unknown"
        diffs = [recent[i+1] - recent[i]
                 for i in range(len(recent) - 1)]
        avg = np.mean(diffs)
        if avg < -1e-4:
            return "improving"
        elif avg > 1e-4:
            return "worsening"
        return "flat"

    def get_improvement_rate(self) -> float:
        if len(self.val_losses) < 2:
            return 0.0
        t = list(self.val_losses)
        return (t[-2] - t[-1]) / (abs(t[-2]) + 1e-10)

    def get_grad_norm_percentile(self,
                                  pct: float = 95,
                                  n: int = 50) -> float:
        if len(self.grad_norms) < 5:
            return 0.0
        recent = list(self.grad_norms)[-n:]
        return float(np.percentile(recent, pct))

    def steps_since_val_improvement(self) -> int:
        best = float('inf')
        counter = 0
        for v in self.val_losses:
            if v < best:
                best = v
                counter = 0
            else:
                counter += 1
        return counter


# ─────────────────────────────────────────────────────────────
# Dynamic Tuner V2
# ─────────────────────────────────────────────────────────────

class DynamicTunerV2:
    """
    Live hyperparameter tuning engine.
    Covers all Tier 1 and Tier 2 hyperparameters.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        config,                        # CustomFinetuneConfig
        logger: logging.Logger,
        save_dir: str,
        training_mode: str = 'basemodel',   # 'tokenizer' or 'basemodel'
        total_steps_per_epoch: int = 100,
        total_epochs: int = 20,
    ):
        self.optimizer = optimizer
        self.model = model
        self.cfg = config
        self.logger = logger
        self.save_dir = save_dir
        self.training_mode = training_mode
        self.total_steps_per_epoch = total_steps_per_epoch
        self.total_epochs = total_epochs

        self.analyzer = LossSignalAnalyzer()
        self.scheduler: Optional[Any] = None
        self.scheduler_type: str = getattr(
            config, 'scheduler_type', 'cosine_warmup')

        # ── Component loss history ────────────────────────
        self.component_history: Dict[str, deque] = {}

        # ── Live state ───────────────────────────────────
        self.state: Dict[str, Any] = {
            # Optimizer params
            'current_lr': self._get_lr(),
            'current_beta1': config.adam_beta1,
            'current_beta2': config.adam_beta2,
            'current_eps': config.adam_eps,
            'current_weight_decay': config.adam_weight_decay,

            # Grad clip
            'current_grad_clip': (
                config.tokenizer_grad_clip
                if training_mode == 'tokenizer'
                else config.basemodel_grad_clip),

            # Loss weights
            'current_tokenizer_recon_pre_w':
                config.tokenizer_recon_pre_weight,
            'current_tokenizer_recon_all_w':
                config.tokenizer_recon_all_weight,
            'current_tokenizer_bsq_w':
                config.tokenizer_bsq_weight,
            'current_tokenizer_recon_w':
                config.tokenizer_recon_weight,
            'current_s1_weight': config.basemodel_s1_weight,
            'current_s2_weight': config.basemodel_s2_weight,

            # Regularization
            'current_label_smoothing': config.label_smoothing,
            'current_bsq_inv_temperature':
                config.bsq_inv_temperature,

            # Cooldown tracking: epoch of last change
            'last_intervention_epoch': {
                'lr': -999, 'betas': -999, 'eps': -999,
                'grad_clip': -999, 'weight_decay': -999,
                'loss_weights': -999, 'label_smoothing': -999,
                'bsq_temp': -999, 'scheduler': -999,
                'dropout': -999,
            },

            # Counters
            'plateau_streak': 0,
            'overfit_streak': 0,
            'current_epoch': -1,
            'current_step': 0,
            'eps_corrected_once': False,

            # History
            'intervention_history': [],
        }

        # ── One-time eps fix for transformers ────────────
        # Apply immediately if eps is still at PyTorch default 1e-8
        if abs(self._get_eps() - 1e-8) < 1e-15:
            self._set_eps(config.adam_eps)
            self.state['current_eps'] = config.adam_eps

        self._log(
            f"[DynamicTunerV2] Init | mode={training_mode} | "
            f"LR={self.state['current_lr']:.2e} | "
            f"clip={self.state['current_grad_clip']:.2f} | "
            f"beta2={self.state['current_beta2']:.4f} | "
            f"eps={self.state['current_eps']:.2e}")

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def on_step_end(self, train_loss: float,
                    grad_norm: float = None):
        self.state['current_step'] += 1
        self.analyzer.add_step(train_loss, grad_norm)
        if grad_norm is not None:
            self._emergency_grad_clip_check(grad_norm)

    def on_epoch_end(self, epoch: int, avg_train_loss: float,
                     avg_val_loss: float) -> Dict[str, Any]:
        self.state['current_epoch'] = epoch
        self.analyzer.add_epoch(avg_train_loss, avg_val_loss)
        changes: Dict[str, Any] = {}

        if epoch < self.cfg.dt_tuning_warmup:
            self._log(
                f"[DynamicTunerV2] Epoch {epoch+1}: "
                f"warmup phase — monitoring only")
            return changes

        try:
            changes = self._run_all_interventions(
                epoch, avg_train_loss, avg_val_loss)
        except Exception as e:
            self._log(
                f"[DynamicTunerV2] WARNING: intervention failed: {e}")
            import traceback
            traceback.print_exc()

        self._log_epoch_summary(
            epoch, avg_train_loss, avg_val_loss, changes)
        self._save_state(epoch)
        return changes

    def track_loss_components(self,
                               components: Dict[str, float]):
        for key, val in components.items():
            if not math.isfinite(val):
                continue
            if key not in self.component_history:
                self.component_history[key] = deque(maxlen=500)
            self.component_history[key].append(val)

    def get_component_ratio(self, key_a: str, key_b: str,
                             n: int = 50) -> float:
        if (key_a not in self.component_history
                or key_b not in self.component_history):
            return 1.0
        mean_a = np.mean(list(self.component_history[key_a])[-n:])
        mean_b = np.mean(list(self.component_history[key_b])[-n:])
        return mean_a / (mean_b + 1e-10)

    # ── Getters for training loop ─────────────────────────

    def get_loss_weights(self) -> Dict[str, float]:
        return {
            'recon_pre':
                self.state['current_tokenizer_recon_pre_w'],
            'recon_all':
                self.state['current_tokenizer_recon_all_w'],
            'bsq':
                self.state['current_tokenizer_bsq_w'],
            'recon':
                self.state['current_tokenizer_recon_w'],
            's1': self.state['current_s1_weight'],
            's2': self.state['current_s2_weight'],
        }

    def get_grad_clip(self) -> float:
        return self.state['current_grad_clip']

    def get_label_smoothing(self) -> float:
        return self.state['current_label_smoothing']

    def get_bsq_temperature(self) -> float:
        return self.state['current_bsq_inv_temperature']

    def scheduler_step(self, metric: float = None):
        if self.scheduler is None:
            return
        try:
            if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
            # Sync lr state after scheduler step
            self.state['current_lr'] = self._get_lr()
        except Exception as e:
            self._log(f"[DynamicTunerV2] scheduler_step error: {e}")

    def get_full_summary(self) -> Dict:
        return {
            'training_mode': self.training_mode,
            'total_interventions':
                len(self.state['intervention_history']),
            'final_lr': self.state['current_lr'],
            'final_beta1': self.state['current_beta1'],
            'final_beta2': self.state['current_beta2'],
            'final_eps': self.state['current_eps'],
            'final_weight_decay': self.state['current_weight_decay'],
            'final_grad_clip': self.state['current_grad_clip'],
            'final_label_smoothing':
                self.state['current_label_smoothing'],
            'final_bsq_temperature':
                self.state['current_bsq_inv_temperature'],
            'final_loss_weights': self.get_loss_weights(),
            'plateau_streak': self.state['plateau_streak'],
            'overfit_streak': self.state['overfit_streak'],
        }

    # ─────────────────────────────────────────────────────
    # Intervention Orchestrator
    # ─────────────────────────────────────────────────────

    def _run_all_interventions(
        self, epoch: int,
        train_loss: float, val_loss: float
    ) -> Dict[str, Any]:

        changes: Dict[str, Any] = {}

        def _merge(d: Optional[Dict]):
            if d:
                changes.update(d)

        # Order matters — most critical first
        _merge(self._check_and_fix_grad_clip(epoch))
        _merge(self._check_and_fix_betas(
            epoch, train_loss, val_loss))
        _merge(self._check_and_fix_adam_eps(epoch, train_loss))
        _merge(self._check_and_fix_lr(
            epoch, train_loss, val_loss))
        _merge(self._check_and_fix_loss_weights(
            epoch, train_loss, val_loss))
        _merge(self._check_and_fix_label_smoothing(
            epoch, train_loss, val_loss))
        _merge(self._check_and_fix_bsq_temperature(
            epoch, train_loss, val_loss))
        _merge(self._check_and_fix_scheduler(
            epoch, train_loss, val_loss))
        _merge(self._check_and_fix_weight_decay(
            epoch, train_loss, val_loss))

        if changes:
            record = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'gap_ratio': self.analyzer.get_gap_ratio(),
                'changes': {
                    k: str(v) for k, v in changes.items()
                },
            }
            self.state['intervention_history'].append(record)

        return changes

    # ─────────────────────────────────────────────────────
    # Individual Intervention Methods
    # ─────────────────────────────────────────────────────

    def _check_and_fix_grad_clip(
            self, epoch: int) -> Optional[Dict]:
        if not self.cfg.tune_grad_clip:
            return None

        changes = {}
        is_exploding = self.analyzer.is_grad_exploding(
            self.cfg.dt_grad_explosion_threshold)

        if is_exploding and self._cooldown_ok('grad_clip', epoch):
            old = self.state['current_grad_clip']
            new = self._clamp(
                old * 0.6,
                self.cfg.dt_grad_clip_min,
                self.cfg.dt_grad_clip_max)
            if abs(new - old) > 0.01:
                self.state['current_grad_clip'] = new
                self._mark_cooldown('grad_clip', epoch)
                self._log_intervention(
                    epoch, 'grad_clip', old, new,
                    'gradient explosion detected')
                changes['grad_clip'] = (old, new)
            return changes

        # Smart clip: target 95th percentile + headroom
        if (self._cooldown_ok('grad_clip', epoch)
                and len(self.analyzer.grad_norms) >= 20):
            p95 = self.analyzer.get_grad_norm_percentile(95, 50)
            target = p95 * 1.5  # 50% headroom above p95
            current = self.state['current_grad_clip']

            if target < current * 0.8 and target > self.cfg.dt_grad_clip_min:
                # Can tighten
                new = self._clamp(
                    target, self.cfg.dt_grad_clip_min,
                    self.cfg.dt_grad_clip_max)
                if abs(new - current) > 0.05:
                    self.state['current_grad_clip'] = new
                    self._mark_cooldown('grad_clip', epoch)
                    self._log_intervention(
                        epoch, 'grad_clip', current, new,
                        f'smart clip: p95={p95:.2f}')
                    changes['grad_clip'] = (current, new)
            elif target > current * 1.2:
                # Need to relax
                new = self._clamp(
                    min(target, current * self.cfg.dt_grad_clip_max),
                    self.cfg.dt_grad_clip_min,
                    self.cfg.dt_grad_clip_max)
                if abs(new - current) > 0.05:
                    self.state['current_grad_clip'] = new
                    self._mark_cooldown('grad_clip', epoch)
                    self._log_intervention(
                        epoch, 'grad_clip', current, new,
                        f'smart relax: p95={p95:.2f}')
                    changes['grad_clip_relaxed'] = (current, new)

        return changes if changes else None

    def _check_and_fix_betas(
        self, epoch: int, train_loss: float, val_loss: float
    ) -> Optional[Dict]:
        if not self.cfg.tune_betas:
            return None
        if not self._cooldown_ok('betas', epoch):
            return None

        b1 = self.state['current_beta1']
        b2 = self.state['current_beta2']
        new_b1, new_b2 = b1, b2
        reason_parts = []

        is_spiking = self.analyzer.is_loss_spiking()
        is_overfit = self.analyzer.is_overfitting(
            self.cfg.dt_overfit_gap_ratio)
        is_plateau = self.analyzer.is_plateau(
            self.cfg.dt_plateau_window,
            self.cfg.dt_plateau_threshold)
        trend = self.analyzer.get_trend()

        # TIER 1 FIX 1: beta2 dynamics
        if is_spiking:
            new_b2 = self._clamp(
                b2 * 1.02,
                self.cfg.dt_beta2_min,
                self.cfg.dt_beta2_max)
            reason_parts.append('loss spiking→↑beta2')
        elif trend == 'improving' and not is_overfit:
            # Faster adaptation: slightly lower beta2
            new_b2 = self._clamp(
                b2 * 0.998,
                self.cfg.dt_beta2_min,
                self.cfg.dt_beta2_max)
            reason_parts.append('stable improve→↓beta2')

        # beta1 dynamics
        if is_overfit:
            new_b1 = self._clamp(
                b1 - 0.01,
                self.cfg.dt_beta1_min,
                self.cfg.dt_beta1_max)
            reason_parts.append('overfit→↓beta1')
        elif is_plateau:
            new_b1 = self._clamp(
                b1 + 0.005,
                self.cfg.dt_beta1_min,
                self.cfg.dt_beta1_max)
            reason_parts.append('plateau→↑beta1 momentum')

        changed = (abs(new_b1 - b1) > 1e-5
                   or abs(new_b2 - b2) > 1e-5)
        if not changed:
            return None

        self._set_betas(new_b1, new_b2)
        self.state['current_beta1'] = new_b1
        self.state['current_beta2'] = new_b2
        self._mark_cooldown('betas', epoch)
        reason = '; '.join(reason_parts)
        self._log_intervention(
            epoch, 'betas',
            f'({b1:.4f},{b2:.4f})',
            f'({new_b1:.4f},{new_b2:.4f})',
            reason)
        return {'betas': ((b1, b2), (new_b1, new_b2))}

    def _check_and_fix_adam_eps(
        self, epoch: int, train_loss: float
    ) -> Optional[Dict]:
        if not self.cfg.tune_adam_eps:
            return None

        # One-time correction at first tuning epoch
        if not self.state['eps_corrected_once']:
            current_eps = self._get_eps()
            target_eps = self.cfg.adam_eps
            if abs(current_eps - target_eps) > 1e-12:
                self._set_eps(target_eps)
                self.state['current_eps'] = target_eps
                self.state['eps_corrected_once'] = True
                self._log_intervention(
                    epoch, 'adam_eps', current_eps,
                    target_eps, 'one-time transformer correction')
                return {'adam_eps': (current_eps, target_eps)}
            self.state['eps_corrected_once'] = True

        # If loss very small and might have numerical issues
        if (train_loss < 0.001
                and self._cooldown_ok('eps', epoch)):
            current_eps = self.state['current_eps']
            new_eps = min(current_eps * 10, 1e-5)
            if new_eps != current_eps:
                self._set_eps(new_eps)
                self.state['current_eps'] = new_eps
                self._mark_cooldown('eps', epoch)
                self._log_intervention(
                    epoch, 'adam_eps', current_eps, new_eps,
                    'very small loss — numerical safety')
                return {'adam_eps': (current_eps, new_eps)}

        return None

    def _check_and_fix_lr(
        self, epoch: int, train_loss: float, val_loss: float
    ) -> Optional[Dict]:
        if not self.cfg.tune_learning_rate:
            return None
        if not self._cooldown_ok('lr', epoch):
            return None

        current_lr = self._get_lr()
        new_lr = current_lr
        reason = ''

        is_severe_of = self.analyzer.is_severely_overfitting(
            self.cfg.dt_severe_overfit_gap_ratio)
        is_of = self.analyzer.is_overfitting(
            self.cfg.dt_overfit_gap_ratio)
        is_plateau = self.analyzer.is_plateau(
            self.cfg.dt_plateau_window,
            self.cfg.dt_plateau_threshold)
        trend = self.analyzer.get_trend()
        gap_ratio = self.analyzer.get_gap_ratio()

        # Update streak counters
        if is_plateau:
            self.state['plateau_streak'] += 1
        else:
            self.state['plateau_streak'] = max(
                0, self.state['plateau_streak'] - 1)

        if is_severe_of:
            new_lr = current_lr * 0.4
            reason = f'SEVERE overfit (gap={gap_ratio:.3f})'
        elif is_plateau:
            streak = self.state['plateau_streak']
            if streak >= 3:
                # Warm restart: spike then let scheduler decay
                new_lr = min(current_lr * 2.0,
                             self.cfg.dt_lr_max)
                self.state['plateau_streak'] = 0
                reason = f'plateau streak {streak} → warm restart'
            else:
                new_lr = current_lr * 0.55
                reason = f'plateau (streak={streak})'
        elif is_of:
            new_lr = current_lr * 0.72
            reason = f'moderate overfit (gap={gap_ratio:.3f})'
        elif (not is_of and train_loss > 2.0
              and trend != 'worsening'):
            new_lr = min(current_lr * 1.2, self.cfg.dt_lr_max)
            reason = 'underfitting → increase LR'
        elif trend == 'improving' and not is_of:
            # Healthy: gentle boost
            new_lr = min(current_lr * 1.03, self.cfg.dt_lr_max)
            reason = 'healthy improving → gentle LR boost'

        new_lr = self._clamp(
            new_lr, self.cfg.dt_lr_min, self.cfg.dt_lr_max)

        if abs(new_lr - current_lr) / (current_lr + 1e-15) < 0.01:
            return None  # < 1% change — not worth it

        self._set_lr(new_lr)
        self.state['current_lr'] = new_lr
        self._mark_cooldown('lr', epoch)
        self._log_intervention(epoch, 'learning_rate',
                               current_lr, new_lr, reason)
        return {'lr': (current_lr, new_lr, reason)}

    def _check_and_fix_loss_weights(
        self, epoch: int, train_loss: float, val_loss: float
    ) -> Optional[Dict]:
        if not self.cfg.tune_loss_weights:
            return None
        if not self._cooldown_ok('loss_weights', epoch):
            return None

        changes = {}

        if self.training_mode == 'tokenizer':
            changes.update(
                self._tune_tokenizer_loss_weights(epoch))
        else:
            changes.update(
                self._tune_basemodel_loss_weights(epoch))

        if changes:
            self._mark_cooldown('loss_weights', epoch)
        return changes if changes else None

    def _tune_tokenizer_loss_weights(
            self, epoch: int) -> Dict:
        changes = {}
        wmin = self.cfg.dt_loss_weight_min
        wmax = self.cfg.dt_loss_weight_max

        has_bsq = 'bsq_loss' in self.component_history
        has_pre = 'recon_pre' in self.component_history
        has_all = 'recon_all' in self.component_history

        if has_bsq and (has_pre or has_all):
            bsq_ratio = self.get_component_ratio(
                'bsq_loss', 'recon_all'
                if has_all else 'recon_pre', 100)

            # BSQ loss dominates → rebalance
            if bsq_ratio > 2.0:
                old = self.state['current_tokenizer_bsq_w']
                new = self._clamp(old * 0.85, wmin, wmax)
                if abs(new - old) > 0.01:
                    self.state['current_tokenizer_bsq_w'] = new
                    self._log_intervention(
                        epoch, 'tokenizer_bsq_w', old, new,
                        f'bsq_ratio={bsq_ratio:.2f} too high')
                    changes['tokenizer_bsq_weight'] = (old, new)
            elif bsq_ratio < 0.3:
                # Recon dominates → boost bsq weight
                old = self.state['current_tokenizer_bsq_w']
                new = self._clamp(old * 1.15, wmin, wmax)
                if abs(new - old) > 0.01:
                    self.state['current_tokenizer_bsq_w'] = new
                    self._log_intervention(
                        epoch, 'tokenizer_bsq_w', old, new,
                        f'bsq_ratio={bsq_ratio:.2f} too low')
                    changes['tokenizer_bsq_weight'] = (old, new)

        # pre vs all recon balance
        if has_pre and has_all:
            pre_all_ratio = self.get_component_ratio(
                'recon_pre', 'recon_all', 100)
            if pre_all_ratio > 1.5:
                old = self.state['current_tokenizer_recon_pre_w']
                new = self._clamp(old * 1.1, wmin, wmax)
                if abs(new - old) > 0.01:
                    self.state[
                        'current_tokenizer_recon_pre_w'] = new
                    self._log_intervention(
                        epoch, 'recon_pre_w', old, new,
                        f'recon_pre >> recon_all '
                        f'({pre_all_ratio:.2f})')
                    changes['recon_pre_weight'] = (old, new)

        return changes

    def _tune_basemodel_loss_weights(
            self, epoch: int) -> Dict:
        changes = {}
        wmin = self.cfg.dt_loss_weight_min
        wmax = self.cfg.dt_loss_weight_max

        has_s1 = 's1_loss' in self.component_history
        has_s2 = 's2_loss' in self.component_history

        if not (has_s1 and has_s2):
            return changes

        s1_mean = np.mean(
            list(self.component_history['s1_loss'])[-100:])
        s2_mean = np.mean(
            list(self.component_history['s2_loss'])[-100:])
        ratio = s1_mean / (s2_mean + 1e-10)

        old_s1 = self.state['current_s1_weight']
        old_s2 = self.state['current_s2_weight']

        if ratio > 1.3:
            # s1 higher loss → increase s1 weight
            new_s1 = self._clamp(old_s1 * 1.1, wmin, wmax)
            # Renormalize
            total = new_s1 + old_s2
            new_s1 = new_s1 / total
            new_s2 = old_s2 / total
            if abs(new_s1 - old_s1) > 0.01:
                self.state['current_s1_weight'] = new_s1
                self.state['current_s2_weight'] = new_s2
                self._log_intervention(
                    epoch, 's1/s2_weights',
                    f'({old_s1:.3f},{old_s2:.3f})',
                    f'({new_s1:.3f},{new_s2:.3f})',
                    f's1_loss({s1_mean:.4f}) > s2_loss({s2_mean:.4f})')
                changes['s1_s2_weights'] = (
                    (old_s1, old_s2), (new_s1, new_s2))

        elif ratio < 0.77:
            # s2 higher loss → increase s2 weight
            new_s2 = self._clamp(old_s2 * 1.1, wmin, wmax)
            total = old_s1 + new_s2
            new_s1 = old_s1 / total
            new_s2 = new_s2 / total
            if abs(new_s2 - old_s2) > 0.01:
                self.state['current_s1_weight'] = new_s1
                self.state['current_s2_weight'] = new_s2
                self._log_intervention(
                    epoch, 's1/s2_weights',
                    f'({old_s1:.3f},{old_s2:.3f})',
                    f'({new_s1:.3f},{new_s2:.3f})',
                    f's2_loss({s2_mean:.4f}) > s1_loss({s1_mean:.4f})')
                changes['s1_s2_weights'] = (
                    (old_s1, old_s2), (new_s1, new_s2))

        return changes

    def _check_and_fix_label_smoothing(
        self, epoch: int, train_loss: float, val_loss: float
    ) -> Optional[Dict]:
        if not self.cfg.tune_label_smoothing:
            return None
        if not self._cooldown_ok('label_smoothing', epoch):
            return None

        gap = self.analyzer.get_gap_ratio()
        current = self.state['current_label_smoothing']
        ls_min = self.cfg.dt_label_smoothing_min
        ls_max = self.cfg.dt_label_smoothing_max

        is_of = self.analyzer.is_overfitting(
            self.cfg.dt_overfit_gap_ratio)

        if is_of and current < ls_max:
            new = self._clamp(current + 0.01, ls_min, ls_max)
        elif not is_of and gap < 0.02 and current > 0.0:
            new = self._clamp(current - 0.005, ls_min, ls_max)
        else:
            return None

        if abs(new - current) < 1e-5:
            return None

        # Apply to model head directly
        try:
            self.model.head.label_smoothing = new
        except AttributeError:
            pass

        self.state['current_label_smoothing'] = new
        self._mark_cooldown('label_smoothing', epoch)
        direction = '↑' if new > current else '↓'
        self._log_intervention(
            epoch, 'label_smoothing', current, new,
            f'{direction} (gap={gap:.3f}, overfit={is_of})')
        return {'label_smoothing': (current, new)}

    def _check_and_fix_bsq_temperature(
        self, epoch: int, train_loss: float, val_loss: float
    ) -> Optional[Dict]:
        if self.training_mode != 'tokenizer':
            return None
        if not self.cfg.tune_bsq_temperature:
            return None
        if not self._cooldown_ok('bsq_temp', epoch):
            return None

        current = self.state['current_bsq_inv_temperature']
        t_min = self.cfg.dt_bsq_inv_temp_min
        t_max = self.cfg.dt_bsq_inv_temp_max

        has_bsq = 'bsq_loss' in self.component_history
        has_recon = 'recon_all' in self.component_history

        if not has_bsq:
            return None

        new = current
        reason = ''

        bsq_vals = list(self.component_history['bsq_loss'])[-50:]
        bsq_mean = np.mean(bsq_vals) if bsq_vals else 1.0

        # Codebook collapse detection: bsq near 0 but recon still high
        if has_recon:
            recon_vals = list(
                self.component_history['recon_all'])[-50:]
            recon_mean = np.mean(recon_vals) if recon_vals else 1.0
            if bsq_mean < 0.001 and recon_mean > 0.05:
                # Collapse: soften temperature
                new = self._clamp(current * 0.8, t_min, t_max)
                reason = (f'codebook collapse: '
                          f'bsq={bsq_mean:.4f}, '
                          f'recon={recon_mean:.4f}')
        elif bsq_mean > 1.0 and current < t_max * 0.8:
            # High bsq loss → sharpen quantization
            new = self._clamp(current * 1.2, t_min, t_max)
            reason = f'high bsq loss ({bsq_mean:.4f}) → sharpen'

        if abs(new - current) < 0.01:
            return None

        # Apply to model
        try:
            self.model.tokenizer.set_inv_temperature(new)
        except AttributeError:
            try:
                self.model.tokenizer.bsq.inv_temperature = new
            except AttributeError:
                pass

        self.state['current_bsq_inv_temperature'] = new
        self._mark_cooldown('bsq_temp', epoch)
        self._log_intervention(
            epoch, 'bsq_inv_temperature', current, new, reason)
        return {'bsq_inv_temperature': (current, new)}

    def _check_and_fix_scheduler(
        self, epoch: int, train_loss: float, val_loss: float
    ) -> Optional[Dict]:
        if not self.cfg.tune_scheduler_params:
            return None
        if not self._cooldown_ok('scheduler', epoch):
            return None
        if self.scheduler is None:
            return None

        is_plateau = self.analyzer.is_plateau(
            self.cfg.dt_plateau_window,
            self.cfg.dt_plateau_threshold)
        is_spiking = self.analyzer.is_loss_spiking()
        is_severe_of = self.analyzer.is_severely_overfitting(
            self.cfg.dt_severe_overfit_gap_ratio)

        current_type = self.scheduler_type
        new_type = current_type
        reason = ''

        if (is_plateau
                and current_type == 'onecycle'):
            new_type = 'cosine_restart'
            reason = 'plateau + OneCycle → CosineRestart'
        elif (is_spiking
              and current_type == 'cosine_warmup'):
            new_type = 'reduce_on_plateau'
            reason = 'spiking → ReduceOnPlateau for stability'
        elif (is_severe_of
              and current_type not in ('reduce_on_plateau',
                                       'constant')):
            new_type = 'reduce_on_plateau'
            reason = 'severe overfit → ReduceOnPlateau'

        if new_type == current_type:
            return None

        # Build and swap scheduler
        try:
            new_scheduler = self._build_scheduler(
                new_type,
                self._get_lr(),
                self.total_epochs - epoch,
                self.total_steps_per_epoch)
            self.scheduler = new_scheduler
            self.scheduler_type = new_type
            self._mark_cooldown('scheduler', epoch)
            self._log_intervention(
                epoch, 'scheduler',
                current_type, new_type, reason)
            return {'scheduler_type': (current_type, new_type)}
        except Exception as e:
            self._log(
                f"[DynamicTunerV2] scheduler swap failed: {e}")
            return None

    def _check_and_fix_weight_decay(
        self, epoch: int, train_loss: float, val_loss: float
    ) -> Optional[Dict]:
        if not self.cfg.tune_weight_decay:
            return None
        if not self._cooldown_ok('weight_decay', epoch):
            return None

        current = self.state['current_weight_decay']
        wd_min = self.cfg.dt_weight_decay_min
        wd_max = self.cfg.dt_weight_decay_max

        is_severe_of = self.analyzer.is_severely_overfitting(
            self.cfg.dt_severe_overfit_gap_ratio)
        is_of = self.analyzer.is_overfitting(
            self.cfg.dt_overfit_gap_ratio)
        is_plateau = self.analyzer.is_plateau(
            self.cfg.dt_plateau_window,
            self.cfg.dt_plateau_threshold)
        trend = self.analyzer.get_trend()

        new = current
        reason = ''

        if is_severe_of:
            new = self._clamp(current * 1.5, wd_min, wd_max)
            reason = 'severe overfit → ↑WD'
        elif is_of:
            new = self._clamp(current * 1.2, wd_min, wd_max)
            reason = 'moderate overfit → ↑WD'
        elif is_plateau and not is_of:
            new = self._clamp(current * 0.8, wd_min, wd_max)
            reason = 'plateau + not-overfit → ↓WD'
        elif (train_loss > 2.0 and not is_of
              and trend != 'worsening'):
            new = self._clamp(current * 0.75, wd_min, wd_max)
            reason = 'underfitting → ↓WD'

        if abs(new - current) / (current + 1e-10) < 0.02:
            return None  # < 2% change

        self._set_wd(new)
        self.state['current_weight_decay'] = new
        self._mark_cooldown('weight_decay', epoch)
        self._log_intervention(
            epoch, 'weight_decay', current, new, reason)
        return {'weight_decay': (current, new)}

    def _emergency_grad_clip_check(self, grad_norm: float):
        """Called every step. No cooldown — immediate response."""
        threshold = self.cfg.dt_grad_explosion_threshold
        if grad_norm > threshold * 3:
            epoch = self.state['current_epoch']
            # Emergency LR cut
            new_lr = max(
                self._get_lr() * 0.1, self.cfg.dt_lr_min)
            self._set_lr(new_lr)
            self.state['current_lr'] = new_lr
            # Emergency clip tighten
            new_clip = max(
                self.state['current_grad_clip'] * 0.3,
                self.cfg.dt_grad_clip_min)
            self.state['current_grad_clip'] = new_clip
            self._log(
                f"[EMERGENCY][Epoch {epoch+1}] "
                f"Catastrophic explosion! norm={grad_norm:.2f} "
                f"→ LR={new_lr:.2e}, clip={new_clip:.2f}")

    # ─────────────────────────────────────────────────────
    # Scheduler Factory
    # ─────────────────────────────────────────────────────

    def _build_scheduler(self, stype: str, current_lr: float,
                          remaining_epochs: int,
                          steps_per_epoch: int):
        pct = getattr(self.cfg, 'scheduler_pct_start', 0.05)
        div = getattr(self.cfg, 'scheduler_div_factor', 25.0)
        final_div = getattr(
            self.cfg, 'scheduler_final_div_factor', 1000.0)
        total_steps = steps_per_epoch * remaining_epochs

        if stype == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=current_lr * div,
                total_steps=total_steps,
                pct_start=pct,
                div_factor=div,
                final_div_factor=final_div)

        elif stype == 'cosine_warmup':
            warmup_steps = int(pct * total_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    return max(step / max(1, warmup_steps),
                               1.0 / div)
                progress = ((step - warmup_steps)
                            / max(1, total_steps - warmup_steps))
                return max(
                    1.0 / final_div,
                    0.5 * (1 + math.cos(math.pi * progress)))

            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda)

        elif stype == 'cosine_restart':
            T0 = max(remaining_epochs // 3, 3)
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T0, T_mult=1,
                eta_min=current_lr / final_div)

        elif stype == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min', factor=0.5, patience=3,
                min_lr=self.cfg.dt_lr_min, verbose=False)

        else:  # constant
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: 1.0)

    # ─────────────────────────────────────────────────────
    # Optimizer State Accessors
    # ─────────────────────────────────────────────────────

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = float(lr)

    def _get_betas(self) -> Tuple[float, float]:
        b = self.optimizer.param_groups[0].get('betas', (0.9, 0.999))
        return float(b[0]), float(b[1])

    def _set_betas(self, b1: float, b2: float):
        for pg in self.optimizer.param_groups:
            if 'betas' in pg:
                pg['betas'] = (float(b1), float(b2))

    def _get_wd(self) -> float:
        return float(
            self.optimizer.param_groups[0].get('weight_decay', 0.0))

    def _set_wd(self, wd: float):
        for pg in self.optimizer.param_groups:
            pg['weight_decay'] = float(wd)

    def _get_eps(self) -> float:
        return float(
            self.optimizer.param_groups[0].get('eps', 1e-8))

    def _set_eps(self, eps: float):
        for pg in self.optimizer.param_groups:
            if 'eps' in pg:
                pg['eps'] = float(eps)

    # ─────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────

    def _clamp(self, value: float, lo: float,
               hi: float) -> float:
        return float(max(lo, min(hi, value)))

    def _cooldown_ok(self, key: str, epoch: int) -> bool:
        cooldown_map = {
            'lr': self.cfg.dt_lr_cooldown,
            'betas': self.cfg.dt_beta_cooldown,
            'eps': 99999,          # one-time only
            'grad_clip': self.cfg.dt_grad_clip_cooldown,
            'weight_decay': self.cfg.dt_lr_cooldown,
            'loss_weights': self.cfg.dt_loss_weight_cooldown,
            'label_smoothing': self.cfg.dt_label_smooth_cooldown,
            'bsq_temp': self.cfg.dt_bsq_temp_cooldown,
            'scheduler': self.cfg.dt_scheduler_swap_cooldown,
            'dropout': 4,
        }
        last = self.state['last_intervention_epoch'].get(key, -999)
        cooldown = cooldown_map.get(key, 2)
        return (epoch - last) >= cooldown

    def _mark_cooldown(self, key: str, epoch: int):
        self.state['last_intervention_epoch'][key] = epoch

    def _log(self, msg: str):
        try:
            self.logger.info(msg)
        except Exception:
            print(msg)

    def _log_intervention(self, epoch: int, param: str,
                          old_val, new_val, reason: str):
        if isinstance(old_val, float):
            old_str = f'{old_val:.6g}'
            new_str = f'{new_val:.6g}'
        else:
            old_str = str(old_val)
            new_str = str(new_val)
        msg = (f"[TUNE][Epoch {epoch+1}][{self.training_mode.upper()}] "
               f"{param}: {old_str} → {new_str} | {reason}")
        self._log(msg)

    def _log_epoch_summary(self, epoch: int, train_loss: float,
                            val_loss: float, changes: Dict):
        gap = val_loss - train_loss
        gap_r = self.analyzer.get_gap_ratio()
        cd_remaining = {
            k: max(0, self.cfg.dt_lr_cooldown
                   - (epoch - v))
            for k, v in
            self.state['last_intervention_epoch'].items()
        }

        lines = [
            f"",
            f"[DynamicTunerV2] Epoch {epoch+1} "
            f"[{self.training_mode.upper()}]"
            f" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  train={train_loss:.6f} "
            f"val={val_loss:.6f} "
            f"gap={gap:.6f} "
            f"gap_ratio={gap_r:.4f}",
            f"  LR={self.state['current_lr']:.3e} "
            f"WD={self.state['current_weight_decay']:.4f} "
            f"clip={self.state['current_grad_clip']:.3f}",
            f"  beta=({self.state['current_beta1']:.4f},"
            f"{self.state['current_beta2']:.4f}) "
            f"eps={self.state['current_eps']:.2e}",
            f"  smooth={self.state['current_label_smoothing']:.4f} "
            f"bsq_temp={self.state['current_bsq_inv_temperature']:.3f}",
        ]

        if self.training_mode == 'tokenizer':
            w = self.get_loss_weights()
            lines.append(
                f"  loss_w: recon_pre={w['recon_pre']:.3f} "
                f"recon_all={w['recon_all']:.3f} "
                f"bsq={w['bsq']:.3f} "
                f"recon={w['recon']:.3f}")
        else:
            w = self.get_loss_weights()
            lines.append(
                f"  loss_w: s1={w['s1']:.4f} s2={w['s2']:.4f}")

        if changes:
            lines.append(f"  Changes: {list(changes.keys())}")
        else:
            lines.append(f"  Changes: none")

        # Show cooldown status for key params
        no_cd = [
            k for k in ['lr', 'betas', 'grad_clip',
                         'loss_weights', 'label_smoothing']
            if not self._cooldown_ok(k, epoch)]
        if no_cd:
            lines.append(f"  On cooldown: {no_cd}")

        lines.append(
            f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self._log('\n'.join(lines))

    def _save_state(self, epoch: int):
        if not self.save_dir:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, 'tuner_state_v2.json')
        out = {
            'epoch': epoch,
            'training_mode': self.training_mode,
            'current_state': {
                k: (v if not isinstance(v, dict) else str(v))
                for k, v in self.state.items()
                if k != 'intervention_history'
            },
            'intervention_history': (
                self.state['intervention_history'][-30:]),
            'loss_history': {
                'train': list(self.analyzer.train_losses)[-50:],
                'val': list(self.analyzer.val_losses)[-50:],
            },
            'grad_norm_history': list(
                self.analyzer.grad_norms)[-100:],
        }
        try:
            with open(path, 'w') as f:
                json.dump(out, f, indent=2, default=str)
        except Exception as e:
            self._log(f"[DynamicTunerV2] save_state error: {e}")