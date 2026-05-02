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
        """
        Retrieves a (possibly nested) config value using dot-notation keys.

        Examples:
            loader.get('training.batch_size')
            loader.get('hpo.search_space.tokenizer_learning_rate.low')

        Args:
            key:     Dot-separated key path into the config dict.
            default: Value to return if any key in the path is missing.

        Returns:
            Config value at the given path, or default if not found.
        """
        keys  = key.split('.')
        value = self.config
        try:
            for k in keys:
                if not isinstance(value, dict):
                    return default
                value = value[k]
            # Treat explicit YAML null (None) as absent — return default
            return value if value is not None else default
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
    
    def get_hpo_config(self) -> Dict[str, Any]:
        """
        Returns the full HPO configuration block.

        Returns an empty dict (not None) if the block is absent, so callers
        can safely use .get() without None-checks.
        """
        return self.config.get('hpo') or {}

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

        # ── Training ──────────────────────────────────────────────
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

        self.adam_beta1 = training_config.get('adam_beta1', 0.9)
        self.adam_beta2 = training_config.get('adam_beta2', 0.95)
        self.adam_weight_decay = training_config.get('adam_weight_decay', 0.1)
        self.adam_eps = training_config.get('adam_eps', 1e-6)

        self.tokenizer_learning_rate = training_config.get('tokenizer_learning_rate', 2e-4)
        self.predictor_learning_rate = training_config.get('predictor_learning_rate', 4e-5)

        self.tokenizer_grad_clip = float(
            training_config.get('tokenizer_grad_clip')
            or training_config.get('tokenizer_max_grad_norm')
            or 2.0
        )
        self.basemodel_grad_clip = float(
            training_config.get('basemodel_grad_clip')
            or training_config.get('basemodel_max_grad_norm')
            or 3.0
        )
        self.grad_clip_norm_type = float(
            training_config.get('grad_clip_norm_type', 2.0)
        )

        self.scheduler_type = training_config.get('scheduler_type', 'cosine_warmup')
        self.scheduler_pct_start = training_config.get('scheduler_pct_start', 0.05)
        self.scheduler_div_factor = training_config.get('scheduler_div_factor', 25.0)
        self.scheduler_final_div_factor = training_config.get('scheduler_final_div_factor', 1000.0)

        self.tokenizer_recon_pre_weight = training_config.get('tokenizer_recon_pre_weight', 1.0)
        self.tokenizer_recon_all_weight = training_config.get('tokenizer_recon_all_weight', 1.0)
        self.tokenizer_bsq_weight = training_config.get('tokenizer_bsq_weight', 0.5)
        self.tokenizer_recon_weight = training_config.get('tokenizer_recon_weight', 0.5)
        self.basemodel_s1_weight = training_config.get('basemodel_s1_weight', 0.5)
        self.basemodel_s2_weight = training_config.get('basemodel_s2_weight', 0.5)

        self.label_smoothing = training_config.get('label_smoothing', 0.0)
        self.bsq_inv_temperature = training_config.get('bsq_inv_temperature', 1.0)
        self.rope_base = training_config.get('rope_base', 10000)
        self.accumulation_steps = training_config.get('accumulation_steps', 1)

        # ── NEW: HPO search space params that must exist as real attributes ──
        # Without these, getattr() fallbacks in training code ignore YAML values
        # whenever HPO doesn't sample a given parameter in a trial.
        self.tokenizer_pct_start = training_config.get(
            'tokenizer_pct_start', self.scheduler_pct_start
        )
        self.tokenizer_div_factor = training_config.get(
            'tokenizer_div_factor', self.scheduler_div_factor
        )
        self.tokenizer_max_grad_norm = training_config.get(
            'tokenizer_max_grad_norm', self.tokenizer_grad_clip
        )
        self.basemodel_pct_start = training_config.get(
            'basemodel_pct_start', self.scheduler_pct_start
        )
        self.basemodel_div_factor = training_config.get(
            'basemodel_div_factor', self.scheduler_div_factor
        )
        self.basemodel_max_grad_norm = training_config.get(
            'basemodel_max_grad_norm', self.basemodel_grad_clip
        )
        self.bsq_beta   = training_config.get('bsq_beta',   None)
        self.bsq_gamma0 = training_config.get('bsq_gamma0', None)
        self.bsq_gamma  = training_config.get('bsq_gamma',  None)
        self.bsq_zeta   = training_config.get('bsq_zeta',   None)
        self.ffn_dropout_p   = training_config.get('ffn_dropout_p',   None)
        self.attn_dropout_p  = training_config.get('attn_dropout_p',  None)
        self.resid_dropout_p = training_config.get('resid_dropout_p', None)
        self.token_dropout_p = training_config.get('token_dropout_p', None)
        # ── END NEW ──────────────────────────────────────────────────────────

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
        hpo_config = self.loader.get('hpo') or {}

        self.hpo_enabled            = bool(hpo_config.get('enabled', False))
        self.hpo_n_trials           = int(hpo_config.get('n_trials', 10))
        self.hpo_direction          = str(hpo_config.get('direction', 'minimize'))
        self.hpo_sampler            = str(hpo_config.get('sampler', 'tpe'))
        self.hpo_pruner             = str(hpo_config.get('pruner', 'median'))
        self.hpo_storage            = hpo_config.get('storage', None)
        self.hpo_study_name         = str(hpo_config.get('study_name', 'nos_hpo'))

        self.hpo_optimize_tokenizer = bool(hpo_config.get('optimize_tokenizer', True))
        self.hpo_optimize_basemodel = bool(hpo_config.get('optimize_basemodel', True))
        self.hpo_tokenizer_epochs   = int(hpo_config.get('hpo_tokenizer_epochs', 5))
        self.hpo_basemodel_epochs   = int(hpo_config.get('hpo_basemodel_epochs', 3))

        # Deep-copy the search space so that trial clones can never mutate
        # the base config's search space dict via shared reference.
        import copy as _copy
        raw_search_space = hpo_config.get('search_space', {}) or {}
        self.hpo_search_space: Dict[str, Any] = _copy.deepcopy(raw_search_space)

        # ── Search space key aliasing ─────────────────────────────────────────
        # The YAML uses tokenizer_max_grad_norm / basemodel_max_grad_norm but
        # the training code reads tokenizer_grad_clip / basemodel_grad_clip.
        # Normalise here so both YAML spellings work transparently.
        _alias_map = {
            'tokenizer_max_grad_norm': 'tokenizer_grad_clip',
            'basemodel_max_grad_norm': 'basemodel_grad_clip',
        }
        for yaml_key, code_key in _alias_map.items():
            if yaml_key in self.hpo_search_space and code_key not in self.hpo_search_space:
                self.hpo_search_space[code_key] = _copy.deepcopy(
                    self.hpo_search_space[yaml_key]
                )

        # ── Validate storage URI ──────────────────────────────────────────────
        # Warn loudly if SQLite is configured without a timeout parameter.
        # The timeout prevents "database is locked" errors under 8 concurrent
        # workers. See configs/config.yaml for the recommended URI format.
        if (
            self.hpo_storage is not None
            and self.hpo_storage.startswith('sqlite')
            and 'timeout=' not in self.hpo_storage
        ):
            import logging
            logging.getLogger(__name__).warning(
                f"HPO storage URI '{self.hpo_storage}' is SQLite but does not "
                f"include a timeout parameter. Under concurrent multi-GPU HPO "
                f"this will cause 'database is locked' errors. "
                f"Recommended: append '?timeout=60' to the URI."
            )

        # ── Validate direction ────────────────────────────────────────────────
        if self.hpo_direction not in ('minimize', 'maximize'):
            raise ValueError(
                f"hpo.direction must be 'minimize' or 'maximize'. "
                f"Got: '{self.hpo_direction}'"
            )

    def _compute_full_paths(self):
        self.tokenizer_save_path = os.path.join(
            self.base_save_path, self.tokenizer_save_name)
        self.tokenizer_best_model_path = os.path.join(
            self.tokenizer_save_path, 'best_model')
        self.basemodel_save_path = os.path.join(
            self.base_save_path, self.basemodel_save_name)
        self.basemodel_best_model_path = os.path.join(
            self.basemodel_save_path, 'best_model')

    def clone_with_overrides(self, overrides: Dict[str, Any]) -> 'CustomFinetuneConfig':
        
        import copy
        import logging

        new_config = copy.deepcopy(self)

        for k, v in overrides.items():
            if not hasattr(new_config, k):
                logging.getLogger(__name__).warning(
                    f"clone_with_overrides: attribute '{k}' does not exist "
                    f"on CustomFinetuneConfig. Adding dynamically. "
                    f"Verify this parameter name matches the training code."
                )
            setattr(new_config, k, v)

        new_config._compute_full_paths()

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