# 🗂️ Codebase

<!-- AUTO-GENERATED — do not edit by hand -->

| Field | Value |
| ----- | ----- |
| **Generated** | `2026-05-15 22:24:34` |
| **Source mode** | YAML config (`codebase.yaml`) |
| **Base directory** | `C:\Users\kashy\OneDrive\Documents\uNOS` |
| **Total files** | 43 |

## 📑 Table of Contents

1. [Project Structure](#-project-structure)
2. [File Contents](#-file-contents)
   - [configs/config.yaml](#configsconfigyaml)
   - [configs/config_mx110_hpo_fulltest.yaml](#configsconfig-mx110-hpo-fulltestyaml)
   - [configs/config_mx110_hpo_winner.yaml](#configsconfig-mx110-hpo-winneryaml)
   - [configs/config_t40_hpo_fulltest.yaml](#configsconfig-t40-hpo-fulltestyaml)
   - [data/1d.csv](#data1dcsv)
   - [finetuned/model1.zip](#finetunedmodel1zip)
   - [finetuned/smoke_test_full_hpo/basemodel/best_model/config.json](#finetunedsmoke-test-full-hpobasemodelbest-modelconfigjson)
   - [finetuned/smoke_test_full_hpo/basemodel/best_model/model.safetensors](#finetunedsmoke-test-full-hpobasemodelbest-modelmodelsafetensors)
   - [finetuned/smoke_test_full_hpo/basemodel/best_model/README.md](#finetunedsmoke-test-full-hpobasemodelbest-modelreadmemd)
   - [finetuned/smoke_test_full_hpo/hpo_logs/worker_pid20088_cuda0.log](#finetunedsmoke-test-full-hpohpo-logsworker-pid20088-cuda0log)
   - [finetuned/smoke_test_full_hpo/hpo_results/basemodel_trials.json](#finetunedsmoke-test-full-hpohpo-resultsbasemodel-trialsjson)
   - [finetuned/smoke_test_full_hpo/hpo_results/tokenizer_trials.json](#finetunedsmoke-test-full-hpohpo-resultstokenizer-trialsjson)
   - [finetuned/smoke_test_full_hpo/hpo_trials/basemodel_pid20088_trial0/trial.log](#finetunedsmoke-test-full-hpohpo-trialsbasemodel-pid20088-trial0triallog)
   - [finetuned/smoke_test_full_hpo/hpo_trials/basemodel_pid20088_trial1/trial.log](#finetunedsmoke-test-full-hpohpo-trialsbasemodel-pid20088-trial1triallog)
   - [finetuned/smoke_test_full_hpo/hpo_trials/tokenizer_pid20088_trial0/trial.log](#finetunedsmoke-test-full-hpohpo-trialstokenizer-pid20088-trial0triallog)
   - [finetuned/smoke_test_full_hpo/hpo_trials/tokenizer_pid20088_trial1/trial.log](#finetunedsmoke-test-full-hpohpo-trialstokenizer-pid20088-trial1triallog)
   - [finetuned/smoke_test_full_hpo/logs/basemodel_training_rank_0.log](#finetunedsmoke-test-full-hpologsbasemodel-training-rank-0log)
   - [finetuned/smoke_test_full_hpo/logs/tokenizer_training_rank_0.log](#finetunedsmoke-test-full-hpologstokenizer-training-rank-0log)
   - [finetuned/smoke_test_full_hpo/tokenizer/trial.log](#finetunedsmoke-test-full-hpotokenizertriallog)
   - [finetuned/smoke_test_full_hpo/tokenizer/best_model/config.json](#finetunedsmoke-test-full-hpotokenizerbest-modelconfigjson)
   - [finetuned/smoke_test_full_hpo/tokenizer/best_model/model.safetensors](#finetunedsmoke-test-full-hpotokenizerbest-modelmodelsafetensors)
   - [finetuned/smoke_test_full_hpo/tokenizer/best_model/README.md](#finetunedsmoke-test-full-hpotokenizerbest-modelreadmemd)
   - [model/__init__.py](#model--init--py)
   - [model/module.py](#modelmodulepy)
   - [model/nos.py](#modelnospy)
   - [models/nos_base/config.json](#modelsnos-baseconfigjson)
   - [models/nos_base/model.safetensors](#modelsnos-basemodelsafetensors)
   - [models/nos_mini/config.json](#modelsnos-miniconfigjson)
   - [models/nos_mini/model.safetensors](#modelsnos-minimodelsafetensors)
   - [models/nos_small/config.json](#modelsnos-smallconfigjson)
   - [models/nos_small/model.safetensors](#modelsnos-smallmodelsafetensors)
   - [models/nos_tokenizer_2k/config.json](#modelsnos-tokenizer-2kconfigjson)
   - [models/nos_tokenizer_2k/model.safetensors](#modelsnos-tokenizer-2kmodelsafetensors)
   - [models/nos_tokenizer_base/config.json](#modelsnos-tokenizer-baseconfigjson)
   - [models/nos_tokenizer_base/model.safetensors](#modelsnos-tokenizer-basemodelsafetensors)
   - [config_loader.py](#config-loaderpy)
   - [hpo_tuner.py](#hpo-tunerpy)
   - [launch_hpo.sh](#launch-hposh)
   - [finetune_base_model.py](#finetune-base-modelpy)
   - [finetune_tokenizer.py](#finetune-tokenizerpy)
   - [train_sequential.py](#train-sequentialpy)
   - [requirements.txt](#requirementstxt)
   - [CLAUDE.md](#claudemd)

## 📁 Project Structure

```
.
├── configs
│   ├── config.yaml
│   ├── config_mx110_hpo_fulltest.yaml
│   ├── config_mx110_hpo_winner.yaml
│   └── config_t40_hpo_fulltest.yaml
├── data
│   └── 1d.csv
├── finetuned
│   ├── model1.zip
│   └── smoke_test_full_hpo
│       ├── basemodel
│       │   └── best_model
│       │       ├── config.json
│       │       ├── model.safetensors
│       │       └── README.md
│       ├── hpo_logs
│       │   └── worker_pid20088_cuda0.log
│       ├── hpo_results
│       │   ├── basemodel_trials.json
│       │   └── tokenizer_trials.json
│       ├── hpo_trials
│       │   ├── basemodel_pid20088_trial0
│       │   │   └── trial.log
│       │   ├── basemodel_pid20088_trial1
│       │   │   └── trial.log
│       │   ├── tokenizer_pid20088_trial0
│       │   │   └── trial.log
│       │   └── tokenizer_pid20088_trial1
│       │       └── trial.log
│       ├── logs
│       │   ├── basemodel_training_rank_0.log
│       │   └── tokenizer_training_rank_0.log
│       └── tokenizer
│           ├── trial.log
│           └── best_model
│               ├── config.json
│               ├── model.safetensors
│               └── README.md
├── model
│   ├── __init__.py
│   ├── module.py
│   └── nos.py
├── models
│   ├── nos_base
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── nos_mini
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── nos_small
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── nos_tokenizer_2k
│   │   ├── config.json
│   │   └── model.safetensors
│   └── nos_tokenizer_base
│       ├── config.json
│       └── model.safetensors
├── config_loader.py
├── hpo_tuner.py
├── launch_hpo.sh
├── finetune_base_model.py
├── finetune_tokenizer.py
├── train_sequential.py
├── requirements.txt
└── CLAUDE.md
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
    # num_workers: DataLoader subprocess count per GPU process.
  # hpo_tuner.py overrides this dynamically based on CPU count and number
  # of concurrent GPU workers. The value here is used for standalone
  # (non-HPO) training runs only.
  # Formula used by HPO: floor(cpu_count * 0.8 / n_gpu_workers), capped at 4.
  num_workers: 4

  # pin_memory: Pins DataLoader output tensors to page-locked (pinned) CPU
  # memory, enabling faster async CPU→GPU transfers via DMA. Always true
  # when training on GPU.
  pin_memory: true

  # persistent_workers: Keeps DataLoader worker processes alive between
  # epochs instead of restarting them. Eliminates the worker process
  # spawn overhead at the start of each epoch (~0.5-2s per epoch on large
  # datasets). Requires num_workers > 0.
  persistent_workers: true
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

  # Gradient clipping.
  # Both spellings are supported in config and HPO search space.
  # The training code reads tokenizer_grad_clip / basemodel_grad_clip.
  # config_loader.py aliases tokenizer_max_grad_norm → tokenizer_grad_clip
  # automatically, so either key works in YAML and in HPO search_space.
  tokenizer_max_grad_norm: 2.0   # alias → tokenizer_grad_clip
  basemodel_max_grad_norm: 3.0   # alias → basemodel_grad_clip
  tokenizer_grad_clip: 2.0
  basemodel_grad_clip: 3.0

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
  # SQLite URI with 60-second busy timeout.
  # The ?timeout=60 parameter prevents "database is locked" errors when
  # 8 GPU workers finish trials simultaneously. WAL journal mode is applied
  # programmatically by hpo_tuner.py via a SQLAlchemy connection hook,
  # which allows concurrent reads while one worker writes.
  #
  # For production clusters with >8 GPUs, switch to PostgreSQL:
  #   storage: "postgresql://user:password@localhost:5432/nos_hpo"
  #
  # For single-GPU development (in-memory, no persistence):
  #   storage: null
  storage: null
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

### `configs/config_mx110_hpo_fulltest.yaml`

```yaml
data:
  data_path: "data/1d.csv"
  lookback_window: 3
  predict_window: 2
  max_context: 5
  clip: 5.0
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

training:
  # Base epochs kept at 1 for the final "apply best" retraining phase
  tokenizer_epochs: 2
  basemodel_epochs: 2
  
  # CRITICAL for 2GB VRAM: keep base batch_size tiny
  batch_size: 512
  log_interval: 10

  num_workers: 2

  pin_memory: true
  persistent_workers: true
  seed: 42

  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001

  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1

  accumulation_steps: 1

  tokenizer_pct_start: 0.03
  tokenizer_div_factor: 10.0
  basemodel_pct_start: 0.03
  basemodel_div_factor: 10.0

  tokenizer_max_grad_norm: 2.0   
  basemodel_max_grad_norm: 3.0   
  tokenizer_grad_clip: 2.0
  basemodel_grad_clip: 3.0

  ffn_dropout_p: null
  attn_dropout_p: null
  resid_dropout_p: null
  token_dropout_p: null

  bsq_beta: null
  bsq_gamma0: null
  bsq_gamma: null
  bsq_zeta: null

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"
  exp_name: "smoke_test_full_hpo"
  base_path: "finetuned"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "Nos_smoke_test_full"
  description: "2GB VRAM Full HPO Structure Test"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false

device:
  use_cuda: true
  device_id: 0

# ── Hyperparameter Search Space ──────────────────────────────────
hpo:
  enabled: true                    
  # 5 trials is enough for TPE to sample a wide variety of combinations
  # without taking all day on a local GPU.
  n_trials: 2                      
  direction: "minimize"            
  sampler: "tpe"                   
  pruner: "median"                 
  
  # Actively testing the SQLite WAL & NullPool lock fix locally
  storage: "sqlite:///nos_smoke_test.db?timeout=60"
  study_name: "nos_finetune_hpo_full"

  optimize_tokenizer: true
  optimize_basemodel: true

  # 1 epoch per trial to burn through the 5 trials rapidly
  hpo_tokenizer_epochs: 1
  hpo_basemodel_epochs: 1

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
    # CRITICAL VRAM GUARD: Limits batch to what a 2GB card can survive.
    batch_size:
      type: categorical
      choices: [1, 2, 4]

    # Accumulation doesn't cost extra VRAM, so we can test larger sizes.
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

### `configs/config_mx110_hpo_winner.yaml`

```yaml
data:
  data_path: data/1d.csv
  lookback_window: 3
  predict_window: 2
  max_context: 5
  clip: 7.282970263056656
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0
training:
  tokenizer_epochs: 2
  basemodel_epochs: 2
  batch_size: 1
  log_interval: 10
  num_workers: 2
  pin_memory: true
  persistent_workers: true
  seed: 42
  tokenizer_learning_rate: 0.0001025350969016849
  predictor_learning_rate: 1.3292918943162153e-06
  adam_beta1: 0.9231993941811405
  adam_beta2: 0.9592671899355066
  adam_weight_decay: 0.22648248189516848
  accumulation_steps: 1
  tokenizer_pct_start: 0.012881829201412343
  tokenizer_div_factor: 5.0
  basemodel_pct_start: 0.012881829201412343
  basemodel_div_factor: 5.0
  tokenizer_max_grad_norm: 1.325320294340452
  basemodel_max_grad_norm: 1.325320294340452
  tokenizer_grad_clip: 2.0
  basemodel_grad_clip: 3.0
  ffn_dropout_p: 0.1216968971838151
  attn_dropout_p: 0.10495128632644757
  resid_dropout_p: 0.12958350559263473
  token_dropout_p: 0.058245828039608386
  bsq_beta: 0.024878734419814436
  bsq_gamma0: 1.2871346474483567
  bsq_gamma: 1.3183340223705389
  bsq_zeta: 0.023927528765580644
model_paths:
  pretrained_tokenizer: models/nos_tokenizer_2k
  pretrained_predictor: models/nos_mini
  exp_name: smoke_test_full_hpo
  base_path: finetuned
  base_save_path: ''
  finetuned_tokenizer: ''
  tokenizer_save_name: tokenizer
  basemodel_save_name: basemodel
experiment:
  name: Nos_smoke_test_full
  description: 2GB VRAM Full HPO Structure Test
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false
  hpo_applied: true
  hpo_applied_timestamp: '2026-05-15T21:48:23.384812'
device:
  use_cuda: true
  device_id: 0
hpo:
  enabled: true
  n_trials: 2
  direction: minimize
  sampler: tpe
  pruner: median
  storage: sqlite:///nos_smoke_test.db?timeout=60
  study_name: nos_finetune_hpo_full
  optimize_tokenizer: true
  optimize_basemodel: true
  hpo_tokenizer_epochs: 1
  hpo_basemodel_epochs: 1
  search_space:
    tokenizer_learning_rate:
      type: float
      low: 1.0e-05
      high: 0.005
      log: true
    predictor_learning_rate:
      type: float
      low: 1.0e-07
      high: 0.0001
      log: true
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
      low: 0.9
      high: 0.999
      log: false
    batch_size:
      type: categorical
      choices:
      - 1
      - 2
      - 4
    accumulation_steps:
      type: categorical
      choices:
      - 1
      - 2
      - 4
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
      choices:
      - 5.0
      - 10.0
      - 25.0
      - 50.0
    basemodel_div_factor:
      type: categorical
      choices:
      - 5.0
      - 10.0
      - 25.0
      - 50.0
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
    clip:
      type: float
      low: 3.0
      high: 10.0
      log: false

```

---

### `configs/config_t40_hpo_fulltest.yaml`

```yaml
data:
  data_path: "data/1d.csv"
  # 5 years of daily data is roughly 1250-1825 rows.
  # 90 days lookback, 14 days predict is optimal for capturing macro trends.
  lookback_window: 90
  predict_window: 14
  max_context: 90
  clip: 5.0
  train_ratio: 0.8
  val_ratio: 0.2
  test_ratio: 0.0

training:
  # Base epochs used for the final "apply best" retraining phase
  tokenizer_epochs: 30
  basemodel_epochs: 20
  
  # Scaled up for T4 16GB VRAM
  batch_size: 64
  log_interval: 10

  num_workers: 4
  pin_memory: true
  persistent_workers: true
  seed: 42

  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.00004

  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1

  accumulation_steps: 1

  tokenizer_pct_start: 0.03
  tokenizer_div_factor: 10.0
  basemodel_pct_start: 0.03
  basemodel_div_factor: 10.0

  tokenizer_max_grad_norm: 2.0   
  basemodel_max_grad_norm: 3.0   

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"
  exp_name: "1d_t4_hpo"
  base_path: "finetuned"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "Nos_1d_T4_HPO"
  description: "T4 GPU Full HPO for 5-Year Daily Data"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false

device:
  use_cuda: true
  device_id: 0

# ── Hyperparameter Search Space ──────────────────────────────────
hpo:
  enabled: true                    
  n_trials: 40                      
  direction: "minimize"            
  sampler: "tpe"                   
  pruner: "hyperband"              
  
  storage: "sqlite:///nos_1d_t4_hpo.db?timeout=60"
  study_name: "nos_finetune_1d_hpo"

  optimize_tokenizer: true
  optimize_basemodel: true

  # 1D datasets pass quickly; 5 epochs gives Hyperband enough data to prune accurately
  hpo_tokenizer_epochs: 5
  hpo_basemodel_epochs: 5

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
      low: 1.0e-6
      high: 1.0e-4
      log: true

    # ── Shared Optimizer ───────────────────────────────────────
    adam_weight_decay:
      type: float
      low: 0.001
      high: 0.3
      log: true

    # ── Batch & Accumulation ───────────────────────────────────
    batch_size:
      type: categorical
      choices: [32, 64, 128]

    accumulation_steps:
      type: categorical
      choices: [1, 2]

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
      choices: [10.0, 25.0, 50.0]

    basemodel_div_factor:
      type: categorical
      choices: [10.0, 25.0, 50.0]

    # ── Dropout (Crucial for smaller 5-yr datasets) ────────────
    ffn_dropout_p:
      type: float
      low: 0.0
      high: 0.4
      log: false

    resid_dropout_p:
      type: float
      low: 0.0
      high: 0.3
      log: false

    # ── BSQ Loss Weights ──────────────────────────────────────
    bsq_beta:
      type: float
      low: 0.05
      high: 0.2
      log: true

    bsq_zeta:
      type: float
      low: 0.05
      high: 0.2
      log: true
```

---

### `data/1d.csv`

```text
timestamps,open,high,low,close,volume,amount
2020-08-11,2.85,3.5208,2.8433,3.2985,1552384.78,4941872.660548333
2020-08-12,3.2985,3.9289,3.08,3.7558,1737042.95,6176664.409993667
2020-08-13,3.75,4.1387,3.5003,3.73,1685759.24,6446323.368316334
2020-08-14,3.7207,3.7676,3.321,3.4099,1474161.79,5210030.644981333
2020-08-15,3.4181,3.74,3.15,3.173,1070233.2,3656268.640716667

# ⚠️  Preview — showing 5 of 1999 data rows (1994 rows hidden).
```

---

### `finetuned/model1.zip`

```text
[Binary file — content not displayed]
```

---

### `finetuned/smoke_test_full_hpo/basemodel/best_model/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.2,
  "learn_te": true,
  "n_heads": 4,
  "n_layers": 4,
  "resid_dropout_p": 0.2,
  "s1_bits": 10,
  "s2_bits": 10,
  "token_dropout_p": 0.0
}
```

---

### `finetuned/smoke_test_full_hpo/basemodel/best_model/model.safetensors`

```text
[File too large to display: 15.7 MB]
```

---

### `finetuned/smoke_test_full_hpo/basemodel/best_model/README.md`

```markdown
---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: [More Information Needed]
- Paper: [More Information Needed]
- Docs: [More Information Needed]
```

---

### `finetuned/smoke_test_full_hpo/hpo_logs/worker_pid20088_cuda0.log`

```text
2026-05-15 21:36:27 [INFO    ] [__main__] NosHPOTuner initialised | config=configs/config_mx110_hpo_fulltest.yaml | device=cuda:0 | worker_tag=pid20088_cuda0
2026-05-15 21:36:27 [INFO    ] [__main__] DataLoader num_workers: 4 (CPU cores=8, GPU workers=1)
2026-05-15 21:36:27 [INFO    ] [__main__] Starting tokenizer HPO: 2 trials | device=cuda:0
2026-05-15 21:36:27 [INFO    ] [__main__] Creating/loading study 'nos_finetune_hpo_full_tokenizer' | storage=sqlite:///nos_smoke_test.db?timeout=60
2026-05-15 21:36:27 [INFO    ] [__main__] SQLite WAL mode, NullPool, and busy_timeout=60s applied via connection hook.
2026-05-15 21:36:27 [INFO    ] [__main__] Study ready: 0 existing trials loaded.
2026-05-15 21:36:27 [INFO    ] [__main__.TokenizerObjective.pid20088] Trial 0 | Sampled overrides: {'tokenizer_learning_rate': 0.0001025350969016849, 'adam_weight_decay': 0.22648248189516848, 'adam_beta1': 0.9231993941811405, 'adam_beta2': 0.9592671899355066, 'batch_size': 1, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.012881829201412343, 'tokenizer_div_factor': 5.0, 'tokenizer_max_grad_norm': 1.325320294340452, 'bsq_beta': 0.024878734419814436, 'bsq_gamma0': 1.2871346474483567, 'bsq_gamma': 1.3183340223705389, 'bsq_zeta': 0.023927528765580644, 'clip': 7.282970263056656, 'tokenizer_epochs': 1, 'log_interval': 999999, 'num_workers': 4}
2026-05-15 21:36:28 [DEBUG   ] [filelock] Attempting to acquire lock 2509863254352 on data/1d.csv.train.lock
2026-05-15 21:36:28 [DEBUG   ] [filelock] Lock 2509863254352 acquired on data/1d.csv.train.lock
2026-05-15 21:36:28 [DEBUG   ] [filelock] Attempting to release lock 2509863254352 on data/1d.csv.train.lock
2026-05-15 21:36:28 [DEBUG   ] [filelock] Lock 2509863254352 released on data/1d.csv.train.lock
2026-05-15 21:36:28 [DEBUG   ] [filelock] Attempting to acquire lock 2509864379280 on data/1d.csv.val.lock
2026-05-15 21:36:28 [DEBUG   ] [filelock] Lock 2509864379280 acquired on data/1d.csv.val.lock
2026-05-15 21:36:28 [DEBUG   ] [filelock] Attempting to release lock 2509864379280 on data/1d.csv.val.lock
2026-05-15 21:36:28 [DEBUG   ] [filelock] Lock 2509864379280 released on data/1d.csv.val.lock
2026-05-15 21:38:40 [INFO    ] [__main__.TokenizerObjective.pid20088] Trial 0 COMPLETE | val_loss=0.457058 | overrides={'tokenizer_learning_rate': 0.0001025350969016849, 'adam_weight_decay': 0.22648248189516848, 'adam_beta1': 0.9231993941811405, 'adam_beta2': 0.9592671899355066, 'batch_size': 1, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.012881829201412343, 'tokenizer_div_factor': 5.0, 'tokenizer_max_grad_norm': 1.325320294340452, 'bsq_beta': 0.024878734419814436, 'bsq_gamma0': 1.2871346474483567, 'bsq_gamma': 1.3183340223705389, 'bsq_zeta': 0.023927528765580644, 'clip': 7.282970263056656, 'tokenizer_epochs': 1, 'log_interval': 999999, 'num_workers': 4}
2026-05-15 21:38:41 [INFO    ] [__main__.TokenizerObjective.pid20088] Trial 1 | Sampled overrides: {'tokenizer_learning_rate': 2.3795221163877236e-05, 'adam_weight_decay': 0.005292705365436975, 'adam_beta1': 0.8866361843293691, 'adam_beta2': 0.9451509284374866, 'batch_size': 1, 'accumulation_steps': 4, 'tokenizer_pct_start': 0.03387337731622081, 'tokenizer_div_factor': 25.0, 'tokenizer_max_grad_norm': 1.870761961280168, 'bsq_beta': 0.013399060561509796, 'bsq_gamma0': 1.5263495397682354, 'bsq_gamma': 1.3281829924875215, 'bsq_zeta': 0.014413697528610409, 'clip': 6.466238370778891, 'tokenizer_epochs': 1, 'log_interval': 999999, 'num_workers': 4}
2026-05-15 21:38:41 [DEBUG   ] [filelock] Attempting to acquire lock 2510045362896 on data/1d.csv.train.lock
2026-05-15 21:38:41 [DEBUG   ] [filelock] Lock 2510045362896 acquired on data/1d.csv.train.lock
2026-05-15 21:38:41 [DEBUG   ] [filelock] Attempting to release lock 2510045362896 on data/1d.csv.train.lock
2026-05-15 21:38:41 [DEBUG   ] [filelock] Lock 2510045362896 released on data/1d.csv.train.lock
2026-05-15 21:38:41 [DEBUG   ] [filelock] Attempting to acquire lock 2508479293904 on data/1d.csv.val.lock
2026-05-15 21:38:41 [DEBUG   ] [filelock] Lock 2508479293904 acquired on data/1d.csv.val.lock
2026-05-15 21:38:41 [DEBUG   ] [filelock] Attempting to release lock 2508479293904 on data/1d.csv.val.lock
2026-05-15 21:38:41 [DEBUG   ] [filelock] Lock 2508479293904 released on data/1d.csv.val.lock
2026-05-15 21:40:41 [INFO    ] [__main__.TokenizerObjective.pid20088] Trial 1 COMPLETE | val_loss=0.589649 | overrides={'tokenizer_learning_rate': 2.3795221163877236e-05, 'adam_weight_decay': 0.005292705365436975, 'adam_beta1': 0.8866361843293691, 'adam_beta2': 0.9451509284374866, 'batch_size': 1, 'accumulation_steps': 4, 'tokenizer_pct_start': 0.03387337731622081, 'tokenizer_div_factor': 25.0, 'tokenizer_max_grad_norm': 1.870761961280168, 'bsq_beta': 0.013399060561509796, 'bsq_gamma0': 1.5263495397682354, 'bsq_gamma': 1.3281829924875215, 'bsq_zeta': 0.014413697528610409, 'clip': 6.466238370778891, 'tokenizer_epochs': 1, 'log_interval': 999999, 'num_workers': 4}
2026-05-15 21:40:41 [INFO    ] [__main__] Tokenizer HPO complete | best_val_loss=0.457058 | best_params={'tokenizer_learning_rate': 0.0001025350969016849, 'adam_weight_decay': 0.22648248189516848, 'adam_beta1': 0.9231993941811405, 'adam_beta2': 0.9592671899355066, 'batch_size': 1, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.012881829201412343, 'tokenizer_div_factor': 5.0, 'tokenizer_max_grad_norm': 1.325320294340452, 'bsq_beta': 0.024878734419814436, 'bsq_gamma0': 1.2871346474483567, 'bsq_gamma': 1.3183340223705389, 'bsq_zeta': 0.023927528765580644, 'clip': 7.282970263056656}
2026-05-15 21:40:41 [INFO    ] [__main__] Results atomically written to: finetuned/smoke_test_full_hpo\hpo_results\tokenizer_trials.json
2026-05-15 21:40:41 [DEBUG   ] [matplotlib] matplotlib data path: C:\Users\kashy\.conda\envs\.conda\Lib\site-packages\matplotlib\mpl-data
2026-05-15 21:40:41 [DEBUG   ] [matplotlib] CONFIGDIR=C:\Users\kashy\.matplotlib
2026-05-15 21:40:41 [DEBUG   ] [matplotlib] interactive is False
2026-05-15 21:40:41 [DEBUG   ] [matplotlib] platform is win32
2026-05-15 21:40:41 [DEBUG   ] [matplotlib] CACHEDIR=C:\Users\kashy\.matplotlib
2026-05-15 21:40:41 [DEBUG   ] [matplotlib.font_manager] Using fontManager instance from C:\Users\kashy\.matplotlib\fontlist-v390.json
2026-05-15 21:40:44 [INFO    ] [__main__] Visualisation plots saved to: finetuned/smoke_test_full_hpo\hpo_results
2026-05-15 21:40:44 [INFO    ] [__main__] Training full tokenizer with best HPO params before basemodel phase.
2026-05-15 21:40:45 [DEBUG   ] [filelock] Attempting to acquire lock 2508796623312 on data/1d.csv.train.lock
2026-05-15 21:40:45 [DEBUG   ] [filelock] Lock 2508796623312 acquired on data/1d.csv.train.lock
2026-05-15 21:40:45 [DEBUG   ] [filelock] Attempting to release lock 2508796623312 on data/1d.csv.train.lock
2026-05-15 21:40:45 [DEBUG   ] [filelock] Lock 2508796623312 released on data/1d.csv.train.lock
2026-05-15 21:40:45 [DEBUG   ] [filelock] Attempting to acquire lock 2508792556048 on data/1d.csv.val.lock
2026-05-15 21:40:45 [DEBUG   ] [filelock] Lock 2508792556048 acquired on data/1d.csv.val.lock
2026-05-15 21:40:45 [DEBUG   ] [filelock] Attempting to release lock 2508792556048 on data/1d.csv.val.lock
2026-05-15 21:40:45 [DEBUG   ] [filelock] Lock 2508792556048 released on data/1d.csv.val.lock
2026-05-15 21:44:18 [INFO    ] [__main__] Best tokenizer training complete. Saved to: finetuned/smoke_test_full_hpo\tokenizer
2026-05-15 21:44:19 [INFO    ] [__main__] Starting basemodel HPO: 2 trials | device=cuda:0 | tokenizer=finetuned/smoke_test_full_hpo\tokenizer\best_model
2026-05-15 21:44:19 [INFO    ] [__main__] Creating/loading study 'nos_finetune_hpo_full_basemodel' | storage=sqlite:///nos_smoke_test.db?timeout=60
2026-05-15 21:44:19 [INFO    ] [__main__] SQLite WAL mode, NullPool, and busy_timeout=60s applied via connection hook.
2026-05-15 21:44:19 [INFO    ] [__main__] Study ready: 0 existing trials loaded.
2026-05-15 21:44:19 [INFO    ] [__main__.BasemodelObjective.pid20088] Trial 0 | Sampled overrides: {'predictor_learning_rate': 1.3292918943162153e-06, 'adam_weight_decay': 0.22648248189516848, 'adam_beta1': 0.9231993941811405, 'adam_beta2': 0.9592671899355066, 'batch_size': 1, 'accumulation_steps': 1, 'basemodel_pct_start': 0.012881829201412343, 'basemodel_div_factor': 5.0, 'basemodel_max_grad_norm': 1.325320294340452, 'ffn_dropout_p': 0.1216968971838151, 'attn_dropout_p': 0.10495128632644757, 'resid_dropout_p': 0.12958350559263473, 'token_dropout_p': 0.058245828039608386, 'clip': 7.282970263056656, 'basemodel_epochs': 1, 'log_interval': 999999, 'num_workers': 4}
2026-05-15 21:44:19 [DEBUG   ] [filelock] Attempting to acquire lock 2508803439888 on data/1d.csv.train.lock
2026-05-15 21:44:19 [DEBUG   ] [filelock] Lock 2508803439888 acquired on data/1d.csv.train.lock
2026-05-15 21:44:19 [DEBUG   ] [filelock] Attempting to release lock 2508803439888 on data/1d.csv.train.lock
2026-05-15 21:44:19 [DEBUG   ] [filelock] Lock 2508803439888 released on data/1d.csv.train.lock
2026-05-15 21:44:19 [DEBUG   ] [filelock] Attempting to acquire lock 2509864538448 on data/1d.csv.val.lock
2026-05-15 21:44:19 [DEBUG   ] [filelock] Lock 2509864538448 acquired on data/1d.csv.val.lock
2026-05-15 21:44:19 [DEBUG   ] [filelock] Attempting to release lock 2509864538448 on data/1d.csv.val.lock
2026-05-15 21:44:19 [DEBUG   ] [filelock] Lock 2509864538448 released on data/1d.csv.val.lock
2026-05-15 21:46:31 [INFO    ] [__main__.BasemodelObjective.pid20088] Trial 0 COMPLETE | val_loss=6.592907 | overrides={'predictor_learning_rate': 1.3292918943162153e-06, 'adam_weight_decay': 0.22648248189516848, 'adam_beta1': 0.9231993941811405, 'adam_beta2': 0.9592671899355066, 'batch_size': 1, 'accumulation_steps': 1, 'basemodel_pct_start': 0.012881829201412343, 'basemodel_div_factor': 5.0, 'basemodel_max_grad_norm': 1.325320294340452, 'ffn_dropout_p': 0.1216968971838151, 'attn_dropout_p': 0.10495128632644757, 'resid_dropout_p': 0.12958350559263473, 'token_dropout_p': 0.058245828039608386, 'clip': 7.282970263056656, 'basemodel_epochs': 1, 'log_interval': 999999, 'num_workers': 4}
2026-05-15 21:46:32 [INFO    ] [__main__.BasemodelObjective.pid20088] Trial 1 | Sampled overrides: {'predictor_learning_rate': 2.621087878265437e-07, 'adam_weight_decay': 0.005292705365436975, 'adam_beta1': 0.8866361843293691, 'adam_beta2': 0.9451509284374866, 'batch_size': 1, 'accumulation_steps': 4, 'basemodel_pct_start': 0.03387337731622081, 'basemodel_div_factor': 25.0, 'basemodel_max_grad_norm': 1.870761961280168, 'ffn_dropout_p': 0.039068845602553554, 'attn_dropout_p': 0.1368466053024314, 'resid_dropout_p': 0.13204574812188039, 'token_dropout_p': 0.024407646968955768, 'clip': 6.466238370778891, 'basemodel_epochs': 1, 'log_interval': 999999, 'num_workers': 4}
2026-05-15 21:46:32 [DEBUG   ] [filelock] Attempting to acquire lock 2508478293520 on data/1d.csv.train.lock
2026-05-15 21:46:32 [DEBUG   ] [filelock] Lock 2508478293520 acquired on data/1d.csv.train.lock
2026-05-15 21:46:32 [DEBUG   ] [filelock] Attempting to release lock 2508478293520 on data/1d.csv.train.lock
2026-05-15 21:46:32 [DEBUG   ] [filelock] Lock 2508478293520 released on data/1d.csv.train.lock
2026-05-15 21:46:32 [DEBUG   ] [filelock] Attempting to acquire lock 2508382663120 on data/1d.csv.val.lock
2026-05-15 21:46:32 [DEBUG   ] [filelock] Lock 2508382663120 acquired on data/1d.csv.val.lock
2026-05-15 21:46:32 [DEBUG   ] [filelock] Attempting to release lock 2508382663120 on data/1d.csv.val.lock
2026-05-15 21:46:32 [DEBUG   ] [filelock] Lock 2508382663120 released on data/1d.csv.val.lock
2026-05-15 21:48:21 [INFO    ] [__main__.BasemodelObjective.pid20088] Trial 1 COMPLETE | val_loss=6.711541 | overrides={'predictor_learning_rate': 2.621087878265437e-07, 'adam_weight_decay': 0.005292705365436975, 'adam_beta1': 0.8866361843293691, 'adam_beta2': 0.9451509284374866, 'batch_size': 1, 'accumulation_steps': 4, 'basemodel_pct_start': 0.03387337731622081, 'basemodel_div_factor': 25.0, 'basemodel_max_grad_norm': 1.870761961280168, 'ffn_dropout_p': 0.039068845602553554, 'attn_dropout_p': 0.1368466053024314, 'resid_dropout_p': 0.13204574812188039, 'token_dropout_p': 0.024407646968955768, 'clip': 6.466238370778891, 'basemodel_epochs': 1, 'log_interval': 999999, 'num_workers': 4}
2026-05-15 21:48:22 [INFO    ] [__main__] Basemodel HPO complete | best_val_loss=6.592907 | best_params={'predictor_learning_rate': 1.3292918943162153e-06, 'adam_weight_decay': 0.22648248189516848, 'adam_beta1': 0.9231993941811405, 'adam_beta2': 0.9592671899355066, 'batch_size': 1, 'accumulation_steps': 1, 'basemodel_pct_start': 0.012881829201412343, 'basemodel_div_factor': 5.0, 'basemodel_max_grad_norm': 1.325320294340452, 'ffn_dropout_p': 0.1216968971838151, 'attn_dropout_p': 0.10495128632644757, 'resid_dropout_p': 0.12958350559263473, 'token_dropout_p': 0.058245828039608386, 'clip': 7.282970263056656}
2026-05-15 21:48:22 [INFO    ] [__main__] Results atomically written to: finetuned/smoke_test_full_hpo\hpo_results\basemodel_trials.json
2026-05-15 21:48:23 [INFO    ] [__main__] Visualisation plots saved to: finetuned/smoke_test_full_hpo\hpo_results
2026-05-15 21:48:23 [INFO    ] [__main__] Routed optimised clip=7.282970263056656 → data.clip (removed from training block to prevent shadowing).
2026-05-15 21:48:23 [INFO    ] [__main__] Best config written to: configs/config_mx110_hpo_winner.yaml

```

---

### `finetuned/smoke_test_full_hpo/hpo_results/basemodel_trials.json`

```json
{
  "best_value": 6.592906690255189,
  "best_params": {
    "predictor_learning_rate": 1.3292918943162153e-06,
    "adam_weight_decay": 0.22648248189516848,
    "adam_beta1": 0.9231993941811405,
    "adam_beta2": 0.9592671899355066,
    "batch_size": 1,
    "accumulation_steps": 1,
    "basemodel_pct_start": 0.012881829201412343,
    "basemodel_div_factor": 5.0,
    "basemodel_max_grad_norm": 1.325320294340452,
    "ffn_dropout_p": 0.1216968971838151,
    "attn_dropout_p": 0.10495128632644757,
    "resid_dropout_p": 0.12958350559263473,
    "token_dropout_p": 0.058245828039608386,
    "clip": 7.282970263056656
  },
  "n_trials": 2,
  "n_completed": 2,
  "n_pruned": 0,
  "n_failed": 0,
  "trials": [
    {
      "number": 0,
      "value": 6.592906690255189,
      "params": {
        "predictor_learning_rate": 1.3292918943162153e-06,
        "adam_weight_decay": 0.22648248189516848,
        "adam_beta1": 0.9231993941811405,
        "adam_beta2": 0.9592671899355066,
        "batch_size": 1,
        "accumulation_steps": 1,
        "basemodel_pct_start": 0.012881829201412343,
        "basemodel_div_factor": 5.0,
        "basemodel_max_grad_norm": 1.325320294340452,
        "ffn_dropout_p": 0.1216968971838151,
        "attn_dropout_p": 0.10495128632644757,
        "resid_dropout_p": 0.12958350559263473,
        "token_dropout_p": 0.058245828039608386,
        "clip": 7.282970263056656
      },
      "state": "TrialState.COMPLETE",
      "datetime_start": "2026-05-15T21:44:19.180226",
      "datetime_complete": "2026-05-15T21:46:32.171509"
    },
    {
      "number": 1,
      "value": 6.711540647653433,
      "params": {
        "predictor_learning_rate": 2.621087878265437e-07,
        "adam_weight_decay": 0.005292705365436975,
        "adam_beta1": 0.8866361843293691,
        "adam_beta2": 0.9451509284374866,
        "batch_size": 1,
        "accumulation_steps": 4,
        "basemodel_pct_start": 0.03387337731622081,
        "basemodel_div_factor": 25.0,
        "basemodel_max_grad_norm": 1.870761961280168,
        "ffn_dropout_p": 0.039068845602553554,
        "attn_dropout_p": 0.1368466053024314,
        "resid_dropout_p": 0.13204574812188039,
        "token_dropout_p": 0.024407646968955768,
        "clip": 6.466238370778891
      },
      "state": "TrialState.COMPLETE",
      "datetime_start": "2026-05-15T21:46:32.243462",
      "datetime_complete": "2026-05-15T21:48:22.116180"
    }
  ],
  "timestamp": "2026-05-15T21:48:22.216430",
  "worker_tag": "pid20088_cuda0"
}
```

---

### `finetuned/smoke_test_full_hpo/hpo_results/tokenizer_trials.json`

```json
{
  "best_value": 0.4570583648311022,
  "best_params": {
    "tokenizer_learning_rate": 0.0001025350969016849,
    "adam_weight_decay": 0.22648248189516848,
    "adam_beta1": 0.9231993941811405,
    "adam_beta2": 0.9592671899355066,
    "batch_size": 1,
    "accumulation_steps": 1,
    "tokenizer_pct_start": 0.012881829201412343,
    "tokenizer_div_factor": 5.0,
    "tokenizer_max_grad_norm": 1.325320294340452,
    "bsq_beta": 0.024878734419814436,
    "bsq_gamma0": 1.2871346474483567,
    "bsq_gamma": 1.3183340223705389,
    "bsq_zeta": 0.023927528765580644,
    "clip": 7.282970263056656
  },
  "n_trials": 2,
  "n_completed": 2,
  "n_pruned": 0,
  "n_failed": 0,
  "trials": [
    {
      "number": 0,
      "value": 0.4570583648311022,
      "params": {
        "tokenizer_learning_rate": 0.0001025350969016849,
        "adam_weight_decay": 0.22648248189516848,
        "adam_beta1": 0.9231993941811405,
        "adam_beta2": 0.9592671899355066,
        "batch_size": 1,
        "accumulation_steps": 1,
        "tokenizer_pct_start": 0.012881829201412343,
        "tokenizer_div_factor": 5.0,
        "tokenizer_max_grad_norm": 1.325320294340452,
        "bsq_beta": 0.024878734419814436,
        "bsq_gamma0": 1.2871346474483567,
        "bsq_gamma": 1.3183340223705389,
        "bsq_zeta": 0.023927528765580644,
        "clip": 7.282970263056656
      },
      "state": "TrialState.COMPLETE",
      "datetime_start": "2026-05-15T21:36:27.620410",
      "datetime_complete": "2026-05-15T21:38:40.823190"
    },
    {
      "number": 1,
      "value": 0.5896485339563626,
      "params": {
        "tokenizer_learning_rate": 2.3795221163877236e-05,
        "adam_weight_decay": 0.005292705365436975,
        "adam_beta1": 0.8866361843293691,
        "adam_beta2": 0.9451509284374866,
        "batch_size": 1,
        "accumulation_steps": 4,
        "tokenizer_pct_start": 0.03387337731622081,
        "tokenizer_div_factor": 25.0,
        "tokenizer_max_grad_norm": 1.870761961280168,
        "bsq_beta": 0.013399060561509796,
        "bsq_gamma0": 1.5263495397682354,
        "bsq_gamma": 1.3281829924875215,
        "bsq_zeta": 0.014413697528610409,
        "clip": 6.466238370778891
      },
      "state": "TrialState.COMPLETE",
      "datetime_start": "2026-05-15T21:38:40.924198",
      "datetime_complete": "2026-05-15T21:40:41.307166"
    }
  ],
  "timestamp": "2026-05-15T21:40:41.430166",
  "worker_tag": "pid20088_cuda0"
}
```

---

### `finetuned/smoke_test_full_hpo/hpo_trials/basemodel_pid20088_trial0/trial.log`

```text
2026-05-15 21:44:19 [INFO] Starting base-model training …
2026-05-15 21:46:29 [INFO] 
--- Epoch 1/1 Summary ---
  Train Loss : 6.765360
  Val   Loss : 6.592907
  Epoch Time : 0:02:10

2026-05-15 21:46:29 [INFO] ✓ Best model saved → finetuned/smoke_test_full_hpo\hpo_trials\basemodel_pid20088_trial0\best_model  (val loss: 6.592907)

```

---

### `finetuned/smoke_test_full_hpo/hpo_trials/basemodel_pid20088_trial1/trial.log`

```text
2026-05-15 21:46:32 [INFO] Starting base-model training …
2026-05-15 21:48:19 [INFO] 
--- Epoch 1/1 Summary ---
  Train Loss : 6.812303
  Val   Loss : 6.711541
  Epoch Time : 0:01:47

2026-05-15 21:48:19 [INFO] ✓ Best model saved → finetuned/smoke_test_full_hpo\hpo_trials\basemodel_pid20088_trial1\best_model  (val loss: 6.711541)

```

---

### `finetuned/smoke_test_full_hpo/hpo_trials/tokenizer_pid20088_trial0/trial.log`

```text
2026-05-15 21:36:28 [INFO] Starting tokenizer training...
2026-05-15 21:38:38 [INFO] 
--- Epoch 1/1 Summary ---
Validation Loss: 0.457058
Epoch Time: 0:02:07

2026-05-15 21:38:38 [INFO] Best model saved: finetuned/smoke_test_full_hpo\hpo_trials\tokenizer_pid20088_trial0\best_model (val loss: 0.457058)

```

---

### `finetuned/smoke_test_full_hpo/hpo_trials/tokenizer_pid20088_trial1/trial.log`

```text
2026-05-15 21:38:41 [INFO] Starting tokenizer training...
2026-05-15 21:40:39 [INFO] 
--- Epoch 1/1 Summary ---
Validation Loss: 0.589649
Epoch Time: 0:01:57

2026-05-15 21:40:39 [INFO] Best model saved: finetuned/smoke_test_full_hpo\hpo_trials\tokenizer_pid20088_trial1\best_model (val loss: 0.589649)

```

---

### `finetuned/smoke_test_full_hpo/logs/basemodel_training_rank_0.log`

```text
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - === Basemodel Training Started ===
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Experiment: smoke_test_full_hpo
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Log dir:    finetuned/smoke_test_full_hpo\logs
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Rank:       0
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Timestamp:  2026-05-15 21:58:00
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Loading fine-tuned tokenizer...
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Loading pretrained predictor...
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Model parameters: 4,108,032
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - === Training Configuration ===
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Data path: data/1d.csv
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Lookback window: 3
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Predict window: 2
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Batch size: 1
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Learning rate: 1.3292918943162153e-06
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Training epochs: 2
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Device: cuda:0
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Tokenizer path: finetuned/smoke_test_full_hpo/tokenizer/best_model
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Pretrained model path: models/nos_mini
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Starting fine-tuning training...
2026-05-15 21:58:00 - basemodel_training_rank_0 - INFO - Starting base-model training …
2026-05-15 21:58:11 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 10/1794] LR: 0.000000  Loss: 6.0878
2026-05-15 21:58:11 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 20/1794] LR: 0.000001  Loss: 7.8186
2026-05-15 21:58:12 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 30/1794] LR: 0.000001  Loss: 9.0534
2026-05-15 21:58:12 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 40/1794] LR: 0.000001  Loss: 7.5791
2026-05-15 21:58:12 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 50/1794] LR: 0.000001  Loss: 7.7302
2026-05-15 21:58:13 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 60/1794] LR: 0.000001  Loss: 8.3658
2026-05-15 21:58:13 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 70/1794] LR: 0.000001  Loss: 8.3715
2026-05-15 21:58:14 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 80/1794] LR: 0.000001  Loss: 7.7935
2026-05-15 21:58:14 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 90/1794] LR: 0.000001  Loss: 6.7388
2026-05-15 21:58:15 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 100/1794] LR: 0.000001  Loss: 6.6046
2026-05-15 21:58:15 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 110/1794] LR: 0.000001  Loss: 5.7785
2026-05-15 21:58:15 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 120/1794] LR: 0.000001  Loss: 7.4133
2026-05-15 21:58:16 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 130/1794] LR: 0.000001  Loss: 4.3124
2026-05-15 21:58:16 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 140/1794] LR: 0.000001  Loss: 6.2734
2026-05-15 21:58:17 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 150/1794] LR: 0.000001  Loss: 8.1511
2026-05-15 21:58:17 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 160/1794] LR: 0.000001  Loss: 6.5824
2026-05-15 21:58:18 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 170/1794] LR: 0.000001  Loss: 7.9326
2026-05-15 21:58:18 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 180/1794] LR: 0.000001  Loss: 6.9654
2026-05-15 21:58:18 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 190/1794] LR: 0.000001  Loss: 6.3368
2026-05-15 21:58:19 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 200/1794] LR: 0.000001  Loss: 8.1660
2026-05-15 21:58:19 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 210/1794] LR: 0.000001  Loss: 7.3577
2026-05-15 21:58:20 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 220/1794] LR: 0.000001  Loss: 7.1350
2026-05-15 21:58:20 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 230/1794] LR: 0.000001  Loss: 5.3583
2026-05-15 21:58:20 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 240/1794] LR: 0.000001  Loss: 5.9936
2026-05-15 21:58:21 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 250/1794] LR: 0.000001  Loss: 9.5113
2026-05-15 21:58:21 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 260/1794] LR: 0.000001  Loss: 7.4828
2026-05-15 21:58:22 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 270/1794] LR: 0.000001  Loss: 4.2376
2026-05-15 21:58:22 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 280/1794] LR: 0.000001  Loss: 7.0154
2026-05-15 21:58:23 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 290/1794] LR: 0.000001  Loss: 6.2785
2026-05-15 21:58:23 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 300/1794] LR: 0.000001  Loss: 8.1898
2026-05-15 21:58:24 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 310/1794] LR: 0.000001  Loss: 6.7221
2026-05-15 21:58:24 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 320/1794] LR: 0.000001  Loss: 8.4432
2026-05-15 21:58:24 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 330/1794] LR: 0.000001  Loss: 5.6606
2026-05-15 21:58:25 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 340/1794] LR: 0.000001  Loss: 6.9166
2026-05-15 21:58:25 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 350/1794] LR: 0.000001  Loss: 7.9993
2026-05-15 21:58:26 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 360/1794] LR: 0.000001  Loss: 7.2830
2026-05-15 21:58:26 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 370/1794] LR: 0.000001  Loss: 7.3009
2026-05-15 21:58:26 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 380/1794] LR: 0.000001  Loss: 6.0146
2026-05-15 21:58:27 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 390/1794] LR: 0.000001  Loss: 6.2142
2026-05-15 21:58:27 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 400/1794] LR: 0.000001  Loss: 6.8091
2026-05-15 21:58:28 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 410/1794] LR: 0.000001  Loss: 8.3810
2026-05-15 21:58:28 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 420/1794] LR: 0.000001  Loss: 7.8329
2026-05-15 21:58:28 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 430/1794] LR: 0.000001  Loss: 8.2790
2026-05-15 21:58:29 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 440/1794] LR: 0.000001  Loss: 6.8179
2026-05-15 21:58:29 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 450/1794] LR: 0.000001  Loss: 6.6183
2026-05-15 21:58:30 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 460/1794] LR: 0.000001  Loss: 8.2882
2026-05-15 21:58:30 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 470/1794] LR: 0.000001  Loss: 6.6144
2026-05-15 21:58:30 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 480/1794] LR: 0.000001  Loss: 8.5469
2026-05-15 21:58:31 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 490/1794] LR: 0.000001  Loss: 5.7255
2026-05-15 21:58:31 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 500/1794] LR: 0.000001  Loss: 7.0486
2026-05-15 21:58:32 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 510/1794] LR: 0.000001  Loss: 6.7122
2026-05-15 21:58:32 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 520/1794] LR: 0.000001  Loss: 8.6196
2026-05-15 21:58:32 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 530/1794] LR: 0.000001  Loss: 5.7079
2026-05-15 21:58:33 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 540/1794] LR: 0.000001  Loss: 6.3289
2026-05-15 21:58:33 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 550/1794] LR: 0.000001  Loss: 6.7723
2026-05-15 21:58:34 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 560/1794] LR: 0.000001  Loss: 5.2375
2026-05-15 21:58:34 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 570/1794] LR: 0.000001  Loss: 5.5144
2026-05-15 21:58:35 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 580/1794] LR: 0.000001  Loss: 8.9952
2026-05-15 21:58:35 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 590/1794] LR: 0.000001  Loss: 7.0892
2026-05-15 21:58:35 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 600/1794] LR: 0.000001  Loss: 7.0084
2026-05-15 21:58:36 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 610/1794] LR: 0.000001  Loss: 6.8636
2026-05-15 21:58:36 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 620/1794] LR: 0.000001  Loss: 5.1036
2026-05-15 21:58:37 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 630/1794] LR: 0.000001  Loss: 7.8978
2026-05-15 21:58:37 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 640/1794] LR: 0.000001  Loss: 5.4780
2026-05-15 21:58:37 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 650/1794] LR: 0.000001  Loss: 7.8894
2026-05-15 21:58:38 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 660/1794] LR: 0.000001  Loss: 6.7433
2026-05-15 21:58:38 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 670/1794] LR: 0.000001  Loss: 7.9692
2026-05-15 21:58:39 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 680/1794] LR: 0.000001  Loss: 7.5640
2026-05-15 21:58:39 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 690/1794] LR: 0.000001  Loss: 8.2115
2026-05-15 21:58:39 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 700/1794] LR: 0.000001  Loss: 6.7912
2026-05-15 21:58:40 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 710/1794] LR: 0.000001  Loss: 5.7785
2026-05-15 21:58:40 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 720/1794] LR: 0.000001  Loss: 7.0543
2026-05-15 21:58:41 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 730/1794] LR: 0.000001  Loss: 4.2969
2026-05-15 21:58:41 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 740/1794] LR: 0.000001  Loss: 6.4855
2026-05-15 21:58:41 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 750/1794] LR: 0.000001  Loss: 6.5109
2026-05-15 21:58:42 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 760/1794] LR: 0.000001  Loss: 5.7851
2026-05-15 21:58:42 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 770/1794] LR: 0.000001  Loss: 5.0529
2026-05-15 21:58:43 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 780/1794] LR: 0.000001  Loss: 6.4990
2026-05-15 21:58:43 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 790/1794] LR: 0.000001  Loss: 7.5906
2026-05-15 21:58:43 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 800/1794] LR: 0.000001  Loss: 5.7993
2026-05-15 21:58:44 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 810/1794] LR: 0.000001  Loss: 7.5439
2026-05-15 21:58:44 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 820/1794] LR: 0.000001  Loss: 6.0796
2026-05-15 21:58:45 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 830/1794] LR: 0.000001  Loss: 8.9155
2026-05-15 21:58:45 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 840/1794] LR: 0.000001  Loss: 7.9628
2026-05-15 21:58:46 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 850/1794] LR: 0.000001  Loss: 7.2383
2026-05-15 21:58:46 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 860/1794] LR: 0.000001  Loss: 5.8134
2026-05-15 21:58:46 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 870/1794] LR: 0.000001  Loss: 6.0238
2026-05-15 21:58:47 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 880/1794] LR: 0.000001  Loss: 6.5158
2026-05-15 21:58:47 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 890/1794] LR: 0.000001  Loss: 5.8257
2026-05-15 21:58:48 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 900/1794] LR: 0.000001  Loss: 7.2646
2026-05-15 21:58:48 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 910/1794] LR: 0.000001  Loss: 7.1047
2026-05-15 21:58:48 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 920/1794] LR: 0.000001  Loss: 7.1466
2026-05-15 21:58:49 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 930/1794] LR: 0.000001  Loss: 7.7298
2026-05-15 21:58:49 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 940/1794] LR: 0.000001  Loss: 7.5728
2026-05-15 21:58:50 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 950/1794] LR: 0.000001  Loss: 7.4751
2026-05-15 21:58:50 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 960/1794] LR: 0.000001  Loss: 4.7443
2026-05-15 21:58:50 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 970/1794] LR: 0.000001  Loss: 7.1207
2026-05-15 21:58:51 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 980/1794] LR: 0.000001  Loss: 8.0237
2026-05-15 21:58:51 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 990/1794] LR: 0.000001  Loss: 7.4085
2026-05-15 21:58:52 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1000/1794] LR: 0.000001  Loss: 6.0280
2026-05-15 21:58:52 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1010/1794] LR: 0.000001  Loss: 7.1263
2026-05-15 21:58:52 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1020/1794] LR: 0.000001  Loss: 7.5011
2026-05-15 21:58:53 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1030/1794] LR: 0.000001  Loss: 6.8924
2026-05-15 21:58:53 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1040/1794] LR: 0.000001  Loss: 5.0690
2026-05-15 21:58:54 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1050/1794] LR: 0.000001  Loss: 5.7510
2026-05-15 21:58:54 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1060/1794] LR: 0.000001  Loss: 8.0840
2026-05-15 21:58:54 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1070/1794] LR: 0.000001  Loss: 6.1376
2026-05-15 21:58:55 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1080/1794] LR: 0.000001  Loss: 8.0532
2026-05-15 21:58:55 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1090/1794] LR: 0.000001  Loss: 6.7824
2026-05-15 21:58:56 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1100/1794] LR: 0.000001  Loss: 8.1990
2026-05-15 21:58:56 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1110/1794] LR: 0.000001  Loss: 4.2648
2026-05-15 21:58:57 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1120/1794] LR: 0.000001  Loss: 3.5304
2026-05-15 21:58:57 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1130/1794] LR: 0.000001  Loss: 5.6564
2026-05-15 21:58:57 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1140/1794] LR: 0.000001  Loss: 7.1262
2026-05-15 21:58:58 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1150/1794] LR: 0.000001  Loss: 6.4940
2026-05-15 21:58:58 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1160/1794] LR: 0.000001  Loss: 7.7888
2026-05-15 21:58:59 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1170/1794] LR: 0.000001  Loss: 6.8624
2026-05-15 21:58:59 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1180/1794] LR: 0.000001  Loss: 6.1779
2026-05-15 21:59:00 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1190/1794] LR: 0.000001  Loss: 6.9867
2026-05-15 21:59:00 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1200/1794] LR: 0.000001  Loss: 5.5341
2026-05-15 21:59:00 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1210/1794] LR: 0.000001  Loss: 7.0560
2026-05-15 21:59:01 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1220/1794] LR: 0.000001  Loss: 5.1570
2026-05-15 21:59:01 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1230/1794] LR: 0.000001  Loss: 6.7527
2026-05-15 21:59:02 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1240/1794] LR: 0.000001  Loss: 7.9329
2026-05-15 21:59:02 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1250/1794] LR: 0.000001  Loss: 6.4122
2026-05-15 21:59:02 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1260/1794] LR: 0.000001  Loss: 7.1196
2026-05-15 21:59:03 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1270/1794] LR: 0.000001  Loss: 6.2920
2026-05-15 21:59:03 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1280/1794] LR: 0.000001  Loss: 5.4692
2026-05-15 21:59:04 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1290/1794] LR: 0.000001  Loss: 4.6464
2026-05-15 21:59:04 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1300/1794] LR: 0.000001  Loss: 7.1967
2026-05-15 21:59:04 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1310/1794] LR: 0.000001  Loss: 6.8948
2026-05-15 21:59:05 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1320/1794] LR: 0.000001  Loss: 6.4065
2026-05-15 21:59:05 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1330/1794] LR: 0.000001  Loss: 7.9638
2026-05-15 21:59:06 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1340/1794] LR: 0.000001  Loss: 6.3921
2026-05-15 21:59:06 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1350/1794] LR: 0.000001  Loss: 6.7522
2026-05-15 21:59:07 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1360/1794] LR: 0.000001  Loss: 5.9398
2026-05-15 21:59:07 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1370/1794] LR: 0.000001  Loss: 7.5382
2026-05-15 21:59:07 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1380/1794] LR: 0.000001  Loss: 7.4681
2026-05-15 21:59:08 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1390/1794] LR: 0.000001  Loss: 6.0841
2026-05-15 21:59:08 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1400/1794] LR: 0.000001  Loss: 8.0195
2026-05-15 21:59:09 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1410/1794] LR: 0.000001  Loss: 7.5665
2026-05-15 21:59:09 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1420/1794] LR: 0.000001  Loss: 8.0281
2026-05-15 21:59:10 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1430/1794] LR: 0.000001  Loss: 7.6966
2026-05-15 21:59:10 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1440/1794] LR: 0.000001  Loss: 6.8888
2026-05-15 21:59:11 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1450/1794] LR: 0.000001  Loss: 5.7377
2026-05-15 21:59:11 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1460/1794] LR: 0.000001  Loss: 4.4185
2026-05-15 21:59:12 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1470/1794] LR: 0.000001  Loss: 5.8494
2026-05-15 21:59:12 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1480/1794] LR: 0.000001  Loss: 7.8785
2026-05-15 21:59:13 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1490/1794] LR: 0.000001  Loss: 6.9940
2026-05-15 21:59:13 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1500/1794] LR: 0.000001  Loss: 7.9382
2026-05-15 21:59:13 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1510/1794] LR: 0.000001  Loss: 7.5699
2026-05-15 21:59:14 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1520/1794] LR: 0.000001  Loss: 7.5138
2026-05-15 21:59:14 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1530/1794] LR: 0.000001  Loss: 6.5208
2026-05-15 21:59:15 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1540/1794] LR: 0.000001  Loss: 6.2944
2026-05-15 21:59:15 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1550/1794] LR: 0.000001  Loss: 6.3023
2026-05-15 21:59:15 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1560/1794] LR: 0.000001  Loss: 4.0611
2026-05-15 21:59:16 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1570/1794] LR: 0.000001  Loss: 6.7449
2026-05-15 21:59:17 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1580/1794] LR: 0.000001  Loss: 5.4866
2026-05-15 21:59:17 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1590/1794] LR: 0.000001  Loss: 7.5962
2026-05-15 21:59:18 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1600/1794] LR: 0.000001  Loss: 7.8451
2026-05-15 21:59:18 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1610/1794] LR: 0.000001  Loss: 6.0204
2026-05-15 21:59:19 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1620/1794] LR: 0.000001  Loss: 5.5739
2026-05-15 21:59:19 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1630/1794] LR: 0.000001  Loss: 5.7568
2026-05-15 21:59:20 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1640/1794] LR: 0.000001  Loss: 7.0386
2026-05-15 21:59:21 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1650/1794] LR: 0.000001  Loss: 5.6961
2026-05-15 21:59:21 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1660/1794] LR: 0.000001  Loss: 7.4736
2026-05-15 21:59:21 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1670/1794] LR: 0.000001  Loss: 6.0850
2026-05-15 21:59:22 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1680/1794] LR: 0.000001  Loss: 6.2303
2026-05-15 21:59:22 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1690/1794] LR: 0.000001  Loss: 7.3153
2026-05-15 21:59:23 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1700/1794] LR: 0.000001  Loss: 8.0360
2026-05-15 21:59:23 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1710/1794] LR: 0.000001  Loss: 6.4830
2026-05-15 21:59:23 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1720/1794] LR: 0.000001  Loss: 7.4346
2026-05-15 21:59:24 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1730/1794] LR: 0.000001  Loss: 6.9003
2026-05-15 21:59:24 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1740/1794] LR: 0.000001  Loss: 6.6870
2026-05-15 21:59:25 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1750/1794] LR: 0.000001  Loss: 5.9802
2026-05-15 21:59:25 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1760/1794] LR: 0.000001  Loss: 6.5568
2026-05-15 21:59:26 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1770/1794] LR: 0.000001  Loss: 5.9597
2026-05-15 21:59:26 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1780/1794] LR: 0.000001  Loss: 7.3455
2026-05-15 21:59:26 - basemodel_training_rank_0 - INFO - [Epoch 1/2, Step 1790/1794] LR: 0.000001  Loss: 5.6959
2026-05-15 21:59:40 - basemodel_training_rank_0 - INFO - 
--- Epoch 1/2 Summary ---
  Train Loss : 6.938168
  Val   Loss : 6.556541
  Epoch Time : 0:01:40

2026-05-15 21:59:41 - basemodel_training_rank_0 - INFO - ✓ Best model saved → finetuned/smoke_test_full_hpo\basemodel\best_model  (val loss: 6.556541)
2026-05-15 21:59:41 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1800/1794] LR: 0.000001  Loss: 6.8747
2026-05-15 21:59:41 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1810/1794] LR: 0.000001  Loss: 7.4042
2026-05-15 21:59:42 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1820/1794] LR: 0.000001  Loss: 6.1652
2026-05-15 21:59:42 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1830/1794] LR: 0.000001  Loss: 7.4556
2026-05-15 21:59:43 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1840/1794] LR: 0.000001  Loss: 7.0307
2026-05-15 21:59:43 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1850/1794] LR: 0.000001  Loss: 7.1201
2026-05-15 21:59:44 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1860/1794] LR: 0.000001  Loss: 7.9635
2026-05-15 21:59:44 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1870/1794] LR: 0.000001  Loss: 8.3787
2026-05-15 21:59:44 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1880/1794] LR: 0.000001  Loss: 8.6691
2026-05-15 21:59:45 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1890/1794] LR: 0.000001  Loss: 6.0956
2026-05-15 21:59:45 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1900/1794] LR: 0.000001  Loss: 7.4894
2026-05-15 21:59:46 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1910/1794] LR: 0.000001  Loss: 6.3062
2026-05-15 21:59:46 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1920/1794] LR: 0.000001  Loss: 6.7310
2026-05-15 21:59:47 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1930/1794] LR: 0.000001  Loss: 4.4663
2026-05-15 21:59:47 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1940/1794] LR: 0.000001  Loss: 6.6806
2026-05-15 21:59:48 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1950/1794] LR: 0.000001  Loss: 7.6941
2026-05-15 21:59:48 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1960/1794] LR: 0.000001  Loss: 5.9783
2026-05-15 21:59:48 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1970/1794] LR: 0.000001  Loss: 6.6295
2026-05-15 21:59:49 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1980/1794] LR: 0.000001  Loss: 6.5137
2026-05-15 21:59:49 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 1990/1794] LR: 0.000001  Loss: 7.1209
2026-05-15 21:59:50 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2000/1794] LR: 0.000001  Loss: 7.9844
2026-05-15 21:59:50 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2010/1794] LR: 0.000001  Loss: 7.3541
2026-05-15 21:59:50 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2020/1794] LR: 0.000001  Loss: 7.0681
2026-05-15 21:59:51 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2030/1794] LR: 0.000001  Loss: 5.2895
2026-05-15 21:59:51 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2040/1794] LR: 0.000001  Loss: 6.1039
2026-05-15 21:59:52 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2050/1794] LR: 0.000001  Loss: 5.7074
2026-05-15 21:59:52 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2060/1794] LR: 0.000001  Loss: 6.7788
2026-05-15 21:59:52 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2070/1794] LR: 0.000001  Loss: 7.2636
2026-05-15 21:59:53 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2080/1794] LR: 0.000001  Loss: 6.1499
2026-05-15 21:59:53 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2090/1794] LR: 0.000001  Loss: 5.5407
2026-05-15 21:59:54 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2100/1794] LR: 0.000000  Loss: 6.1092
2026-05-15 21:59:54 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2110/1794] LR: 0.000000  Loss: 6.4248
2026-05-15 21:59:55 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2120/1794] LR: 0.000000  Loss: 5.2587
2026-05-15 21:59:55 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2130/1794] LR: 0.000000  Loss: 7.0053
2026-05-15 21:59:55 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2140/1794] LR: 0.000000  Loss: 7.8562
2026-05-15 21:59:56 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2150/1794] LR: 0.000000  Loss: 7.8888
2026-05-15 21:59:56 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2160/1794] LR: 0.000000  Loss: 7.5895
2026-05-15 21:59:57 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2170/1794] LR: 0.000000  Loss: 6.2289
2026-05-15 21:59:57 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2180/1794] LR: 0.000000  Loss: 7.4013
2026-05-15 21:59:58 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2190/1794] LR: 0.000000  Loss: 7.2676
2026-05-15 21:59:59 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2200/1794] LR: 0.000000  Loss: 6.3565
2026-05-15 21:59:59 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2210/1794] LR: 0.000000  Loss: 6.5330
2026-05-15 22:00:00 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2220/1794] LR: 0.000000  Loss: 6.3500
2026-05-15 22:00:00 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2230/1794] LR: 0.000000  Loss: 5.8350
2026-05-15 22:00:01 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2240/1794] LR: 0.000000  Loss: 6.1169
2026-05-15 22:00:01 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2250/1794] LR: 0.000000  Loss: 6.7367
2026-05-15 22:00:02 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2260/1794] LR: 0.000000  Loss: 5.7155
2026-05-15 22:00:02 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2270/1794] LR: 0.000000  Loss: 6.7260
2026-05-15 22:00:03 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2280/1794] LR: 0.000000  Loss: 7.2162
2026-05-15 22:00:03 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2290/1794] LR: 0.000000  Loss: 6.4210
2026-05-15 22:00:04 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2300/1794] LR: 0.000000  Loss: 6.5946
2026-05-15 22:00:04 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2310/1794] LR: 0.000000  Loss: 6.3798
2026-05-15 22:00:05 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2320/1794] LR: 0.000000  Loss: 7.8730
2026-05-15 22:00:06 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2330/1794] LR: 0.000000  Loss: 5.6484
2026-05-15 22:00:06 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2340/1794] LR: 0.000000  Loss: 5.1258
2026-05-15 22:00:07 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2350/1794] LR: 0.000000  Loss: 7.3026
2026-05-15 22:00:07 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2360/1794] LR: 0.000000  Loss: 6.6431
2026-05-15 22:00:08 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2370/1794] LR: 0.000000  Loss: 8.2434
2026-05-15 22:00:08 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2380/1794] LR: 0.000000  Loss: 5.6608
2026-05-15 22:00:09 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2390/1794] LR: 0.000000  Loss: 6.4952
2026-05-15 22:00:09 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2400/1794] LR: 0.000000  Loss: 6.1842
2026-05-15 22:00:09 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2410/1794] LR: 0.000000  Loss: 6.5156
2026-05-15 22:00:10 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2420/1794] LR: 0.000000  Loss: 5.9064
2026-05-15 22:00:10 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2430/1794] LR: 0.000000  Loss: 6.8604
2026-05-15 22:00:11 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2440/1794] LR: 0.000000  Loss: 6.2352
2026-05-15 22:00:11 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2450/1794] LR: 0.000000  Loss: 8.7647
2026-05-15 22:00:12 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2460/1794] LR: 0.000000  Loss: 6.7053
2026-05-15 22:00:12 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2470/1794] LR: 0.000000  Loss: 5.2143
2026-05-15 22:00:12 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2480/1794] LR: 0.000000  Loss: 8.9139
2026-05-15 22:00:13 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2490/1794] LR: 0.000000  Loss: 6.2923
2026-05-15 22:00:13 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2500/1794] LR: 0.000000  Loss: 7.3140
2026-05-15 22:00:14 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2510/1794] LR: 0.000000  Loss: 6.2111
2026-05-15 22:00:14 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2520/1794] LR: 0.000000  Loss: 4.6267
2026-05-15 22:00:14 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2530/1794] LR: 0.000000  Loss: 6.4367
2026-05-15 22:00:15 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2540/1794] LR: 0.000000  Loss: 6.5726
2026-05-15 22:00:15 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2550/1794] LR: 0.000000  Loss: 5.0899
2026-05-15 22:00:16 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2560/1794] LR: 0.000000  Loss: 5.3149
2026-05-15 22:00:16 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2570/1794] LR: 0.000000  Loss: 7.3770
2026-05-15 22:00:17 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2580/1794] LR: 0.000000  Loss: 8.4491
2026-05-15 22:00:17 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2590/1794] LR: 0.000000  Loss: 6.0128
2026-05-15 22:00:17 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2600/1794] LR: 0.000000  Loss: 7.9546
2026-05-15 22:00:18 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2610/1794] LR: 0.000000  Loss: 7.0301
2026-05-15 22:00:18 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2620/1794] LR: 0.000000  Loss: 6.2088
2026-05-15 22:00:19 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2630/1794] LR: 0.000000  Loss: 7.6196
2026-05-15 22:00:19 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2640/1794] LR: 0.000000  Loss: 6.2946
2026-05-15 22:00:20 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2650/1794] LR: 0.000000  Loss: 6.6772
2026-05-15 22:00:20 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2660/1794] LR: 0.000000  Loss: 5.3274
2026-05-15 22:00:20 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2670/1794] LR: 0.000000  Loss: 6.7526
2026-05-15 22:00:21 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2680/1794] LR: 0.000000  Loss: 6.1773
2026-05-15 22:00:21 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2690/1794] LR: 0.000000  Loss: 6.9623
2026-05-15 22:00:22 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2700/1794] LR: 0.000000  Loss: 6.7525
2026-05-15 22:00:22 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2710/1794] LR: 0.000000  Loss: 5.8814
2026-05-15 22:00:22 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2720/1794] LR: 0.000000  Loss: 5.7925
2026-05-15 22:00:23 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2730/1794] LR: 0.000000  Loss: 6.4242
2026-05-15 22:00:23 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2740/1794] LR: 0.000000  Loss: 6.5647
2026-05-15 22:00:24 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2750/1794] LR: 0.000000  Loss: 6.5257
2026-05-15 22:00:24 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2760/1794] LR: 0.000000  Loss: 6.1524
2026-05-15 22:00:25 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2770/1794] LR: 0.000000  Loss: 4.7907
2026-05-15 22:00:25 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2780/1794] LR: 0.000000  Loss: 6.4771
2026-05-15 22:00:26 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2790/1794] LR: 0.000000  Loss: 6.3547
2026-05-15 22:00:26 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2800/1794] LR: 0.000000  Loss: 8.4004
2026-05-15 22:00:27 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2810/1794] LR: 0.000000  Loss: 8.4364
2026-05-15 22:00:28 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2820/1794] LR: 0.000000  Loss: 7.4172
2026-05-15 22:00:28 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2830/1794] LR: 0.000000  Loss: 7.7326
2026-05-15 22:00:29 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2840/1794] LR: 0.000000  Loss: 5.7115
2026-05-15 22:00:29 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2850/1794] LR: 0.000000  Loss: 7.6926
2026-05-15 22:00:30 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2860/1794] LR: 0.000000  Loss: 7.5321
2026-05-15 22:00:31 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2870/1794] LR: 0.000000  Loss: 7.2155
2026-05-15 22:00:31 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2880/1794] LR: 0.000000  Loss: 5.9637
2026-05-15 22:00:32 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2890/1794] LR: 0.000000  Loss: 5.3592
2026-05-15 22:00:32 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2900/1794] LR: 0.000000  Loss: 6.9166
2026-05-15 22:00:33 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2910/1794] LR: 0.000000  Loss: 5.5388
2026-05-15 22:00:33 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2920/1794] LR: 0.000000  Loss: 7.4923
2026-05-15 22:00:34 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2930/1794] LR: 0.000000  Loss: 7.1705
2026-05-15 22:00:35 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2940/1794] LR: 0.000000  Loss: 7.5704
2026-05-15 22:00:35 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2950/1794] LR: 0.000000  Loss: 6.5248
2026-05-15 22:00:36 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2960/1794] LR: 0.000000  Loss: 7.3530
2026-05-15 22:00:36 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2970/1794] LR: 0.000000  Loss: 6.9696
2026-05-15 22:00:37 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2980/1794] LR: 0.000000  Loss: 7.1953
2026-05-15 22:00:38 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 2990/1794] LR: 0.000000  Loss: 4.4221
2026-05-15 22:00:39 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3000/1794] LR: 0.000000  Loss: 5.4496
2026-05-15 22:00:40 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3010/1794] LR: 0.000000  Loss: 7.2656
2026-05-15 22:00:41 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3020/1794] LR: 0.000000  Loss: 6.9236
2026-05-15 22:00:41 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3030/1794] LR: 0.000000  Loss: 7.2571
2026-05-15 22:00:42 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3040/1794] LR: 0.000000  Loss: 8.8563
2026-05-15 22:00:43 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3050/1794] LR: 0.000000  Loss: 5.5690
2026-05-15 22:00:43 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3060/1794] LR: 0.000000  Loss: 7.4525
2026-05-15 22:00:44 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3070/1794] LR: 0.000000  Loss: 5.6516
2026-05-15 22:00:44 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3080/1794] LR: 0.000000  Loss: 6.1751
2026-05-15 22:00:45 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3090/1794] LR: 0.000000  Loss: 6.1530
2026-05-15 22:00:46 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3100/1794] LR: 0.000000  Loss: 4.6695
2026-05-15 22:00:46 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3110/1794] LR: 0.000000  Loss: 7.4454
2026-05-15 22:00:47 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3120/1794] LR: 0.000000  Loss: 5.8209
2026-05-15 22:00:47 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3130/1794] LR: 0.000000  Loss: 5.8700
2026-05-15 22:00:48 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3140/1794] LR: 0.000000  Loss: 7.3106
2026-05-15 22:00:49 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3150/1794] LR: 0.000000  Loss: 7.6604
2026-05-15 22:00:50 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3160/1794] LR: 0.000000  Loss: 6.3596
2026-05-15 22:00:50 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3170/1794] LR: 0.000000  Loss: 6.3808
2026-05-15 22:00:51 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3180/1794] LR: 0.000000  Loss: 5.0820
2026-05-15 22:00:52 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3190/1794] LR: 0.000000  Loss: 8.6529
2026-05-15 22:00:52 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3200/1794] LR: 0.000000  Loss: 8.4370
2026-05-15 22:00:53 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3210/1794] LR: 0.000000  Loss: 7.5671
2026-05-15 22:00:54 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3220/1794] LR: 0.000000  Loss: 8.2981
2026-05-15 22:00:55 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3230/1794] LR: 0.000000  Loss: 6.4124
2026-05-15 22:00:56 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3240/1794] LR: 0.000000  Loss: 6.9811
2026-05-15 22:00:56 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3250/1794] LR: 0.000000  Loss: 7.4438
2026-05-15 22:00:57 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3260/1794] LR: 0.000000  Loss: 6.1068
2026-05-15 22:00:58 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3270/1794] LR: 0.000000  Loss: 8.1449
2026-05-15 22:00:58 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3280/1794] LR: 0.000000  Loss: 6.0273
2026-05-15 22:00:59 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3290/1794] LR: 0.000000  Loss: 5.0708
2026-05-15 22:01:00 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3300/1794] LR: 0.000000  Loss: 5.7382
2026-05-15 22:01:00 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3310/1794] LR: 0.000000  Loss: 6.7520
2026-05-15 22:01:01 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3320/1794] LR: 0.000000  Loss: 7.2662
2026-05-15 22:01:01 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3330/1794] LR: 0.000000  Loss: 6.8461
2026-05-15 22:01:02 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3340/1794] LR: 0.000000  Loss: 5.8791
2026-05-15 22:01:02 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3350/1794] LR: 0.000000  Loss: 6.8828
2026-05-15 22:01:03 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3360/1794] LR: 0.000000  Loss: 9.3677
2026-05-15 22:01:03 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3370/1794] LR: 0.000000  Loss: 6.3464
2026-05-15 22:01:04 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3380/1794] LR: 0.000000  Loss: 5.5902
2026-05-15 22:01:04 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3390/1794] LR: 0.000000  Loss: 6.3147
2026-05-15 22:01:05 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3400/1794] LR: 0.000000  Loss: 7.5460
2026-05-15 22:01:06 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3410/1794] LR: 0.000000  Loss: 6.9013
2026-05-15 22:01:06 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3420/1794] LR: 0.000000  Loss: 6.5338
2026-05-15 22:01:07 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3430/1794] LR: 0.000000  Loss: 7.3296
2026-05-15 22:01:07 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3440/1794] LR: 0.000000  Loss: 7.8815
2026-05-15 22:01:08 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3450/1794] LR: 0.000000  Loss: 7.1921
2026-05-15 22:01:08 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3460/1794] LR: 0.000000  Loss: 6.8186
2026-05-15 22:01:09 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3470/1794] LR: 0.000000  Loss: 6.2331
2026-05-15 22:01:09 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3480/1794] LR: 0.000000  Loss: 7.2265
2026-05-15 22:01:10 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3490/1794] LR: 0.000000  Loss: 7.1151
2026-05-15 22:01:11 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3500/1794] LR: 0.000000  Loss: 7.9452
2026-05-15 22:01:11 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3510/1794] LR: 0.000000  Loss: 8.0360
2026-05-15 22:01:12 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3520/1794] LR: 0.000000  Loss: 8.0401
2026-05-15 22:01:13 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3530/1794] LR: 0.000000  Loss: 6.2872
2026-05-15 22:01:13 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3540/1794] LR: 0.000000  Loss: 5.7967
2026-05-15 22:01:14 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3550/1794] LR: 0.000000  Loss: 7.0809
2026-05-15 22:01:14 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3560/1794] LR: 0.000000  Loss: 5.5050
2026-05-15 22:01:15 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3570/1794] LR: 0.000000  Loss: 4.8762
2026-05-15 22:01:15 - basemodel_training_rank_0 - INFO - [Epoch 2/2, Step 3580/1794] LR: 0.000000  Loss: 7.5143
2026-05-15 22:01:21 - basemodel_training_rank_0 - INFO - 
--- Epoch 2/2 Summary ---
  Train Loss : 6.717719
  Val   Loss : 6.567409
  Epoch Time : 0:01:40

2026-05-15 22:01:23 - basemodel_training_rank_0 - INFO - Basemodel training completed! Best validation loss: 6.5565
Training time: 3.38 minutes
Model saved to: finetuned/smoke_test_full_hpo\basemodel

```

---

### `finetuned/smoke_test_full_hpo/logs/tokenizer_training_rank_0.log`

```text
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - === Tokenizer Training Started ===
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Experiment Name: smoke_test_full_hpo
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Log Directory: finetuned/smoke_test_full_hpo\logs
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Rank: 0
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Timestamp: 2026-05-15 21:54:15
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Loading pretrained tokenizer...
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Tokenizer parameters: 3,958,042
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - === Training Configuration ===
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Data path: data/1d.csv
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Lookback window: 3
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Predict window: 2
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Batch size: 1
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Learning rate: 0.0001025350969016849
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Training epochs: 2
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Device: cuda:0
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Distributed training: False
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Starting tokenizer fine-tuning training...
2026-05-15 21:54:15 - tokenizer_training_rank_0 - INFO - Starting tokenizer training...
2026-05-15 21:54:26 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 10/1794] LR: 0.000030, Loss: 2.3433
2026-05-15 21:54:27 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 20/1794] LR: 0.000054, Loss: 0.7943
2026-05-15 21:54:27 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 30/1794] LR: 0.000082, Loss: 1.2861
2026-05-15 21:54:28 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 40/1794] LR: 0.000100, Loss: 1.3876
2026-05-15 21:54:28 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 50/1794] LR: 0.000103, Loss: 7.0425
2026-05-15 21:54:29 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 60/1794] LR: 0.000103, Loss: 6.1461
2026-05-15 21:54:29 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 70/1794] LR: 0.000103, Loss: 2.7023
2026-05-15 21:54:30 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 80/1794] LR: 0.000103, Loss: 0.5079
2026-05-15 21:54:31 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 90/1794] LR: 0.000102, Loss: 2.2205
2026-05-15 21:54:31 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 100/1794] LR: 0.000102, Loss: 6.5140
2026-05-15 21:54:32 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 110/1794] LR: 0.000102, Loss: 0.8430
2026-05-15 21:54:32 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 120/1794] LR: 0.000102, Loss: 1.5388
2026-05-15 21:54:33 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 130/1794] LR: 0.000102, Loss: 5.1875
2026-05-15 21:54:33 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 140/1794] LR: 0.000102, Loss: 1.0642
2026-05-15 21:54:34 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 150/1794] LR: 0.000102, Loss: 1.2118
2026-05-15 21:54:35 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 160/1794] LR: 0.000102, Loss: 0.2296
2026-05-15 21:54:35 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 170/1794] LR: 0.000102, Loss: 1.7278
2026-05-15 21:54:36 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 180/1794] LR: 0.000102, Loss: 3.2777
2026-05-15 21:54:36 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 190/1794] LR: 0.000102, Loss: 4.1288
2026-05-15 21:54:37 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 200/1794] LR: 0.000102, Loss: 5.2337
2026-05-15 21:54:37 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 210/1794] LR: 0.000102, Loss: 0.7865
2026-05-15 21:54:38 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 220/1794] LR: 0.000102, Loss: 5.0216
2026-05-15 21:54:39 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 230/1794] LR: 0.000102, Loss: 1.0496
2026-05-15 21:54:39 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 240/1794] LR: 0.000102, Loss: 1.5882
2026-05-15 21:54:40 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 250/1794] LR: 0.000102, Loss: 0.6286
2026-05-15 21:54:40 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 260/1794] LR: 0.000102, Loss: 0.5014
2026-05-15 21:54:41 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 270/1794] LR: 0.000102, Loss: 0.8294
2026-05-15 21:54:41 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 280/1794] LR: 0.000101, Loss: 2.3095
2026-05-15 21:54:42 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 290/1794] LR: 0.000101, Loss: 3.0071
2026-05-15 21:54:42 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 300/1794] LR: 0.000101, Loss: 0.3673
2026-05-15 21:54:43 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 310/1794] LR: 0.000101, Loss: 1.9188
2026-05-15 21:54:43 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 320/1794] LR: 0.000101, Loss: 0.8183
2026-05-15 21:54:44 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 330/1794] LR: 0.000101, Loss: 6.9320
2026-05-15 21:54:44 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 340/1794] LR: 0.000101, Loss: 0.7634
2026-05-15 21:54:45 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 350/1794] LR: 0.000101, Loss: 1.3306
2026-05-15 21:54:45 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 360/1794] LR: 0.000101, Loss: 1.4901
2026-05-15 21:54:46 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 370/1794] LR: 0.000100, Loss: 3.3172
2026-05-15 21:54:46 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 380/1794] LR: 0.000100, Loss: 0.6478
2026-05-15 21:54:47 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 390/1794] LR: 0.000100, Loss: 0.5142
2026-05-15 21:54:47 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 400/1794] LR: 0.000100, Loss: 1.0890
2026-05-15 21:54:48 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 410/1794] LR: 0.000100, Loss: 1.0484
2026-05-15 21:54:48 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 420/1794] LR: 0.000100, Loss: 0.1895
2026-05-15 21:54:49 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 430/1794] LR: 0.000100, Loss: 2.0020
2026-05-15 21:54:49 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 440/1794] LR: 0.000099, Loss: 2.3173
2026-05-15 21:54:50 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 450/1794] LR: 0.000099, Loss: 3.4154
2026-05-15 21:54:50 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 460/1794] LR: 0.000099, Loss: 0.2529
2026-05-15 21:54:51 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 470/1794] LR: 0.000099, Loss: 1.2558
2026-05-15 21:54:51 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 480/1794] LR: 0.000099, Loss: 1.4201
2026-05-15 21:54:52 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 490/1794] LR: 0.000099, Loss: 0.5216
2026-05-15 21:54:52 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 500/1794] LR: 0.000098, Loss: 2.5613
2026-05-15 21:54:53 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 510/1794] LR: 0.000098, Loss: 2.0690
2026-05-15 21:54:53 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 520/1794] LR: 0.000098, Loss: 2.5403
2026-05-15 21:54:54 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 530/1794] LR: 0.000098, Loss: 0.5441
2026-05-15 21:54:55 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 540/1794] LR: 0.000098, Loss: 2.3873
2026-05-15 21:54:55 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 550/1794] LR: 0.000097, Loss: 2.6983
2026-05-15 21:54:56 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 560/1794] LR: 0.000097, Loss: 1.6089
2026-05-15 21:54:56 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 570/1794] LR: 0.000097, Loss: 1.4164
2026-05-15 21:54:57 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 580/1794] LR: 0.000097, Loss: 3.6365
2026-05-15 21:54:57 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 590/1794] LR: 0.000097, Loss: 0.5469
2026-05-15 21:54:58 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 600/1794] LR: 0.000096, Loss: 0.9913
2026-05-15 21:54:58 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 610/1794] LR: 0.000096, Loss: 3.1757
2026-05-15 21:54:59 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 620/1794] LR: 0.000096, Loss: 2.4659
2026-05-15 21:54:59 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 630/1794] LR: 0.000096, Loss: 0.4025
2026-05-15 21:55:00 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 640/1794] LR: 0.000096, Loss: 2.3968
2026-05-15 21:55:00 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 650/1794] LR: 0.000095, Loss: 1.4598
2026-05-15 21:55:01 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 660/1794] LR: 0.000095, Loss: 2.1596
2026-05-15 21:55:01 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 670/1794] LR: 0.000095, Loss: 0.9511
2026-05-15 21:55:02 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 680/1794] LR: 0.000095, Loss: 1.8131
2026-05-15 21:55:02 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 690/1794] LR: 0.000094, Loss: 2.6519
2026-05-15 21:55:03 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 700/1794] LR: 0.000094, Loss: 0.3701
2026-05-15 21:55:03 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 710/1794] LR: 0.000094, Loss: 0.8293
2026-05-15 21:55:04 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 720/1794] LR: 0.000094, Loss: 2.3030
2026-05-15 21:55:04 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 730/1794] LR: 0.000093, Loss: 3.3153
2026-05-15 21:55:05 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 740/1794] LR: 0.000093, Loss: 0.5490
2026-05-15 21:55:05 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 750/1794] LR: 0.000093, Loss: 0.8746
2026-05-15 21:55:06 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 760/1794] LR: 0.000093, Loss: 0.4113
2026-05-15 21:55:06 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 770/1794] LR: 0.000092, Loss: 1.8034
2026-05-15 21:55:07 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 780/1794] LR: 0.000092, Loss: 0.8812
2026-05-15 21:55:08 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 790/1794] LR: 0.000092, Loss: 3.9816
2026-05-15 21:55:08 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 800/1794] LR: 0.000091, Loss: 2.1863
2026-05-15 21:55:09 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 810/1794] LR: 0.000091, Loss: 2.1920
2026-05-15 21:55:09 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 820/1794] LR: 0.000091, Loss: 0.7929
2026-05-15 21:55:10 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 830/1794] LR: 0.000091, Loss: 1.5705
2026-05-15 21:55:10 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 840/1794] LR: 0.000090, Loss: 1.4789
2026-05-15 21:55:11 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 850/1794] LR: 0.000090, Loss: 4.2083
2026-05-15 21:55:11 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 860/1794] LR: 0.000090, Loss: 2.0064
2026-05-15 21:55:12 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 870/1794] LR: 0.000089, Loss: 4.8620
2026-05-15 21:55:12 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 880/1794] LR: 0.000089, Loss: 0.9438
2026-05-15 21:55:13 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 890/1794] LR: 0.000089, Loss: 2.5550
2026-05-15 21:55:14 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 900/1794] LR: 0.000088, Loss: 2.9100
2026-05-15 21:55:14 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 910/1794] LR: 0.000088, Loss: 0.7832
2026-05-15 21:55:15 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 920/1794] LR: 0.000088, Loss: 0.4368
2026-05-15 21:55:15 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 930/1794] LR: 0.000088, Loss: 1.3638
2026-05-15 21:55:16 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 940/1794] LR: 0.000087, Loss: 3.1887
2026-05-15 21:55:16 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 950/1794] LR: 0.000087, Loss: 1.3974
2026-05-15 21:55:17 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 960/1794] LR: 0.000087, Loss: 0.8057
2026-05-15 21:55:17 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 970/1794] LR: 0.000086, Loss: 0.4860
2026-05-15 21:55:18 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 980/1794] LR: 0.000086, Loss: 2.3625
2026-05-15 21:55:19 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 990/1794] LR: 0.000086, Loss: 0.1833
2026-05-15 21:55:19 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1000/1794] LR: 0.000085, Loss: 3.9907
2026-05-15 21:55:20 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1010/1794] LR: 0.000085, Loss: 0.3734
2026-05-15 21:55:20 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1020/1794] LR: 0.000085, Loss: 1.1820
2026-05-15 21:55:21 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1030/1794] LR: 0.000084, Loss: 0.1762
2026-05-15 21:55:21 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1040/1794] LR: 0.000084, Loss: 1.5707
2026-05-15 21:55:22 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1050/1794] LR: 0.000083, Loss: 0.8985
2026-05-15 21:55:22 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1060/1794] LR: 0.000083, Loss: 0.1198
2026-05-15 21:55:23 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1070/1794] LR: 0.000083, Loss: 1.1233
2026-05-15 21:55:24 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1080/1794] LR: 0.000082, Loss: 0.1700
2026-05-15 21:55:24 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1090/1794] LR: 0.000082, Loss: 1.7746
2026-05-15 21:55:25 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1100/1794] LR: 0.000082, Loss: 1.7770
2026-05-15 21:55:25 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1110/1794] LR: 0.000081, Loss: 2.5000
2026-05-15 21:55:26 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1120/1794] LR: 0.000081, Loss: 0.7703
2026-05-15 21:55:26 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1130/1794] LR: 0.000081, Loss: 2.4563
2026-05-15 21:55:27 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1140/1794] LR: 0.000080, Loss: 1.0107
2026-05-15 21:55:27 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1150/1794] LR: 0.000080, Loss: 2.7023
2026-05-15 21:55:28 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1160/1794] LR: 0.000079, Loss: 3.8378
2026-05-15 21:55:28 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1170/1794] LR: 0.000079, Loss: 2.0122
2026-05-15 21:55:29 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1180/1794] LR: 0.000079, Loss: 1.3875
2026-05-15 21:55:29 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1190/1794] LR: 0.000078, Loss: 1.0579
2026-05-15 21:55:30 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1200/1794] LR: 0.000078, Loss: 0.9012
2026-05-15 21:55:30 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1210/1794] LR: 0.000078, Loss: 2.3192
2026-05-15 21:55:31 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1220/1794] LR: 0.000077, Loss: 3.8064
2026-05-15 21:55:31 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1230/1794] LR: 0.000077, Loss: 0.7891
2026-05-15 21:55:32 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1240/1794] LR: 0.000076, Loss: 1.9115
2026-05-15 21:55:33 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1250/1794] LR: 0.000076, Loss: 2.7201
2026-05-15 21:55:33 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1260/1794] LR: 0.000076, Loss: 0.1943
2026-05-15 21:55:34 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1270/1794] LR: 0.000075, Loss: 1.4699
2026-05-15 21:55:34 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1280/1794] LR: 0.000075, Loss: 1.1149
2026-05-15 21:55:35 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1290/1794] LR: 0.000074, Loss: 0.6376
2026-05-15 21:55:35 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1300/1794] LR: 0.000074, Loss: 1.2081
2026-05-15 21:55:36 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1310/1794] LR: 0.000074, Loss: 3.4864
2026-05-15 21:55:36 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1320/1794] LR: 0.000073, Loss: 0.2395
2026-05-15 21:55:37 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1330/1794] LR: 0.000073, Loss: 2.0649
2026-05-15 21:55:37 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1340/1794] LR: 0.000072, Loss: 3.4442
2026-05-15 21:55:38 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1350/1794] LR: 0.000072, Loss: 0.3660
2026-05-15 21:55:38 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1360/1794] LR: 0.000071, Loss: 0.4066
2026-05-15 21:55:39 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1370/1794] LR: 0.000071, Loss: 3.6787
2026-05-15 21:55:39 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1380/1794] LR: 0.000071, Loss: 2.8982
2026-05-15 21:55:40 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1390/1794] LR: 0.000070, Loss: 1.5900
2026-05-15 21:55:40 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1400/1794] LR: 0.000070, Loss: 2.1567
2026-05-15 21:55:41 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1410/1794] LR: 0.000069, Loss: 2.2690
2026-05-15 21:55:42 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1420/1794] LR: 0.000069, Loss: 0.8841
2026-05-15 21:55:42 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1430/1794] LR: 0.000068, Loss: 1.6766
2026-05-15 21:55:43 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1440/1794] LR: 0.000068, Loss: 0.2725
2026-05-15 21:55:43 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1450/1794] LR: 0.000068, Loss: 2.8324
2026-05-15 21:55:44 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1460/1794] LR: 0.000067, Loss: 1.5426
2026-05-15 21:55:44 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1470/1794] LR: 0.000067, Loss: 1.1839
2026-05-15 21:55:45 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1480/1794] LR: 0.000066, Loss: 0.2866
2026-05-15 21:55:45 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1490/1794] LR: 0.000066, Loss: 0.6184
2026-05-15 21:55:46 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1500/1794] LR: 0.000065, Loss: 0.4147
2026-05-15 21:55:47 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1510/1794] LR: 0.000065, Loss: 1.5428
2026-05-15 21:55:47 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1520/1794] LR: 0.000065, Loss: 4.2842
2026-05-15 21:55:48 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1530/1794] LR: 0.000064, Loss: 0.2330
2026-05-15 21:55:48 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1540/1794] LR: 0.000064, Loss: 1.6186
2026-05-15 21:55:49 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1550/1794] LR: 0.000063, Loss: 3.4042
2026-05-15 21:55:49 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1560/1794] LR: 0.000063, Loss: 1.2969
2026-05-15 21:55:50 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1570/1794] LR: 0.000062, Loss: 1.7230
2026-05-15 21:55:50 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1580/1794] LR: 0.000062, Loss: 0.6542
2026-05-15 21:55:51 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1590/1794] LR: 0.000061, Loss: 0.8374
2026-05-15 21:55:51 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1600/1794] LR: 0.000061, Loss: 1.0742
2026-05-15 21:55:52 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1610/1794] LR: 0.000061, Loss: 0.6512
2026-05-15 21:55:52 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1620/1794] LR: 0.000060, Loss: 1.0854
2026-05-15 21:55:53 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1630/1794] LR: 0.000060, Loss: 1.0582
2026-05-15 21:55:53 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1640/1794] LR: 0.000059, Loss: 0.4105
2026-05-15 21:55:54 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1650/1794] LR: 0.000059, Loss: 2.7758
2026-05-15 21:55:55 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1660/1794] LR: 0.000058, Loss: 2.1717
2026-05-15 21:55:55 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1670/1794] LR: 0.000058, Loss: 2.6133
2026-05-15 21:55:56 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1680/1794] LR: 0.000057, Loss: 0.6220
2026-05-15 21:55:56 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1690/1794] LR: 0.000057, Loss: 1.8014
2026-05-15 21:55:57 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1700/1794] LR: 0.000057, Loss: 0.2278
2026-05-15 21:55:57 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1710/1794] LR: 0.000056, Loss: 0.7974
2026-05-15 21:55:58 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1720/1794] LR: 0.000056, Loss: 2.5021
2026-05-15 21:55:58 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1730/1794] LR: 0.000055, Loss: 0.1977
2026-05-15 21:55:59 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1740/1794] LR: 0.000055, Loss: 0.2237
2026-05-15 21:55:59 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1750/1794] LR: 0.000054, Loss: 0.5298
2026-05-15 21:56:00 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1760/1794] LR: 0.000054, Loss: 4.9111
2026-05-15 21:56:00 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1770/1794] LR: 0.000053, Loss: 0.2248
2026-05-15 21:56:01 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1780/1794] LR: 0.000053, Loss: 0.9455
2026-05-15 21:56:02 - tokenizer_training_rank_0 - INFO - [Epoch 1/2, Step 1790/1794] LR: 0.000052, Loss: 1.2231
2026-05-15 21:56:14 - tokenizer_training_rank_0 - INFO - 
--- Epoch 1/2 Summary ---
Validation Loss: 0.615959
Epoch Time: 0:01:57

2026-05-15 21:56:14 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/smoke_test_full_hpo\tokenizer\best_model (val loss: 0.615959)
2026-05-15 21:56:14 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1800/1794] LR: 0.000052, Loss: 0.2977
2026-05-15 21:56:15 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1810/1794] LR: 0.000052, Loss: 0.1778
2026-05-15 21:56:16 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1820/1794] LR: 0.000051, Loss: 3.9521
2026-05-15 21:56:16 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1830/1794] LR: 0.000051, Loss: 1.2857
2026-05-15 21:56:17 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1840/1794] LR: 0.000050, Loss: 1.1588
2026-05-15 21:56:17 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1850/1794] LR: 0.000050, Loss: 0.1830
2026-05-15 21:56:18 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1860/1794] LR: 0.000049, Loss: 0.3476
2026-05-15 21:56:18 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1870/1794] LR: 0.000049, Loss: 0.6507
2026-05-15 21:56:19 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1880/1794] LR: 0.000048, Loss: 1.8438
2026-05-15 21:56:19 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1890/1794] LR: 0.000048, Loss: 1.3943
2026-05-15 21:56:20 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1900/1794] LR: 0.000047, Loss: 1.0754
2026-05-15 21:56:20 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1910/1794] LR: 0.000047, Loss: 1.9065
2026-05-15 21:56:21 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1920/1794] LR: 0.000047, Loss: 0.2941
2026-05-15 21:56:21 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1930/1794] LR: 0.000046, Loss: 3.6046
2026-05-15 21:56:22 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1940/1794] LR: 0.000046, Loss: 0.2206
2026-05-15 21:56:23 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1950/1794] LR: 0.000045, Loss: 1.9597
2026-05-15 21:56:23 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1960/1794] LR: 0.000045, Loss: 1.4106
2026-05-15 21:56:24 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1970/1794] LR: 0.000044, Loss: 1.3728
2026-05-15 21:56:24 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1980/1794] LR: 0.000044, Loss: 2.4945
2026-05-15 21:56:25 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 1990/1794] LR: 0.000043, Loss: 0.2393
2026-05-15 21:56:25 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2000/1794] LR: 0.000043, Loss: 0.3872
2026-05-15 21:56:26 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2010/1794] LR: 0.000042, Loss: 1.0203
2026-05-15 21:56:26 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2020/1794] LR: 0.000042, Loss: 2.0481
2026-05-15 21:56:27 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2030/1794] LR: 0.000042, Loss: 0.4110
2026-05-15 21:56:27 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2040/1794] LR: 0.000041, Loss: 3.1853
2026-05-15 21:56:28 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2050/1794] LR: 0.000041, Loss: 1.4692
2026-05-15 21:56:29 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2060/1794] LR: 0.000040, Loss: 1.2761
2026-05-15 21:56:29 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2070/1794] LR: 0.000040, Loss: 0.5946
2026-05-15 21:56:30 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2080/1794] LR: 0.000039, Loss: 4.6789
2026-05-15 21:56:30 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2090/1794] LR: 0.000039, Loss: 2.7726
2026-05-15 21:56:31 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2100/1794] LR: 0.000038, Loss: 2.0857
2026-05-15 21:56:31 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2110/1794] LR: 0.000038, Loss: 1.8000
2026-05-15 21:56:32 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2120/1794] LR: 0.000038, Loss: 0.1474
2026-05-15 21:56:32 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2130/1794] LR: 0.000037, Loss: 1.3305
2026-05-15 21:56:33 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2140/1794] LR: 0.000037, Loss: 0.3396
2026-05-15 21:56:33 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2150/1794] LR: 0.000036, Loss: 3.1193
2026-05-15 21:56:34 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2160/1794] LR: 0.000036, Loss: 0.7854
2026-05-15 21:56:34 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2170/1794] LR: 0.000035, Loss: 0.5174
2026-05-15 21:56:35 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2180/1794] LR: 0.000035, Loss: 2.4320
2026-05-15 21:56:36 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2190/1794] LR: 0.000035, Loss: 2.0438
2026-05-15 21:56:36 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2200/1794] LR: 0.000034, Loss: 0.4376
2026-05-15 21:56:37 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2210/1794] LR: 0.000034, Loss: 0.3862
2026-05-15 21:56:37 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2220/1794] LR: 0.000033, Loss: 0.7790
2026-05-15 21:56:38 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2230/1794] LR: 0.000033, Loss: 0.3568
2026-05-15 21:56:38 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2240/1794] LR: 0.000032, Loss: 1.7608
2026-05-15 21:56:39 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2250/1794] LR: 0.000032, Loss: 1.9485
2026-05-15 21:56:40 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2260/1794] LR: 0.000032, Loss: 0.9995
2026-05-15 21:56:40 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2270/1794] LR: 0.000031, Loss: 0.3120
2026-05-15 21:56:41 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2280/1794] LR: 0.000031, Loss: 0.7003
2026-05-15 21:56:41 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2290/1794] LR: 0.000030, Loss: 1.5609
2026-05-15 21:56:42 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2300/1794] LR: 0.000030, Loss: 1.4056
2026-05-15 21:56:42 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2310/1794] LR: 0.000030, Loss: 1.3968
2026-05-15 21:56:43 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2320/1794] LR: 0.000029, Loss: 1.3855
2026-05-15 21:56:43 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2330/1794] LR: 0.000029, Loss: 1.1748
2026-05-15 21:56:44 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2340/1794] LR: 0.000028, Loss: 0.2092
2026-05-15 21:56:44 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2350/1794] LR: 0.000028, Loss: 0.5808
2026-05-15 21:56:45 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2360/1794] LR: 0.000027, Loss: 0.2573
2026-05-15 21:56:46 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2370/1794] LR: 0.000027, Loss: 3.5308
2026-05-15 21:56:46 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2380/1794] LR: 0.000027, Loss: 1.7458
2026-05-15 21:56:47 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2390/1794] LR: 0.000026, Loss: 0.8730
2026-05-15 21:56:47 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2400/1794] LR: 0.000026, Loss: 1.9835
2026-05-15 21:56:48 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2410/1794] LR: 0.000025, Loss: 0.6289
2026-05-15 21:56:48 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2420/1794] LR: 0.000025, Loss: 1.4762
2026-05-15 21:56:49 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2430/1794] LR: 0.000025, Loss: 1.7674
2026-05-15 21:56:49 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2440/1794] LR: 0.000024, Loss: 1.1428
2026-05-15 21:56:50 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2450/1794] LR: 0.000024, Loss: 0.9088
2026-05-15 21:56:50 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2460/1794] LR: 0.000024, Loss: 3.2989
2026-05-15 21:56:51 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2470/1794] LR: 0.000023, Loss: 0.3570
2026-05-15 21:56:52 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2480/1794] LR: 0.000023, Loss: 0.8197
2026-05-15 21:56:52 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2490/1794] LR: 0.000022, Loss: 2.3705
2026-05-15 21:56:53 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2500/1794] LR: 0.000022, Loss: 2.0822
2026-05-15 21:56:53 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2510/1794] LR: 0.000022, Loss: 3.6792
2026-05-15 21:56:54 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2520/1794] LR: 0.000021, Loss: 0.3575
2026-05-15 21:56:54 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2530/1794] LR: 0.000021, Loss: 3.0172
2026-05-15 21:56:55 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2540/1794] LR: 0.000021, Loss: 1.1172
2026-05-15 21:56:55 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2550/1794] LR: 0.000020, Loss: 1.6670
2026-05-15 21:56:56 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2560/1794] LR: 0.000020, Loss: 2.5001
2026-05-15 21:56:56 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2570/1794] LR: 0.000019, Loss: 1.4365
2026-05-15 21:56:57 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2580/1794] LR: 0.000019, Loss: 0.4521
2026-05-15 21:56:58 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2590/1794] LR: 0.000019, Loss: 0.7723
2026-05-15 21:56:58 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2600/1794] LR: 0.000018, Loss: 4.0757
2026-05-15 21:56:59 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2610/1794] LR: 0.000018, Loss: 0.4973
2026-05-15 21:56:59 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2620/1794] LR: 0.000018, Loss: 0.8383
2026-05-15 21:57:00 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2630/1794] LR: 0.000017, Loss: 1.2434
2026-05-15 21:57:00 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2640/1794] LR: 0.000017, Loss: 0.1972
2026-05-15 21:57:01 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2650/1794] LR: 0.000017, Loss: 0.4647
2026-05-15 21:57:01 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2660/1794] LR: 0.000016, Loss: 2.1016
2026-05-15 21:57:02 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2670/1794] LR: 0.000016, Loss: 1.7797
2026-05-15 21:57:02 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2680/1794] LR: 0.000016, Loss: 1.2606
2026-05-15 21:57:03 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2690/1794] LR: 0.000015, Loss: 1.5311
2026-05-15 21:57:04 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2700/1794] LR: 0.000015, Loss: 1.0364
2026-05-15 21:57:04 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2710/1794] LR: 0.000015, Loss: 0.2310
2026-05-15 21:57:05 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2720/1794] LR: 0.000014, Loss: 1.2254
2026-05-15 21:57:05 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2730/1794] LR: 0.000014, Loss: 0.7537
2026-05-15 21:57:06 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2740/1794] LR: 0.000014, Loss: 0.3321
2026-05-15 21:57:07 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2750/1794] LR: 0.000013, Loss: 1.2203
2026-05-15 21:57:07 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2760/1794] LR: 0.000013, Loss: 1.7602
2026-05-15 21:57:08 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2770/1794] LR: 0.000013, Loss: 1.8017
2026-05-15 21:57:08 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2780/1794] LR: 0.000013, Loss: 2.2819
2026-05-15 21:57:09 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2790/1794] LR: 0.000012, Loss: 0.8650
2026-05-15 21:57:09 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2800/1794] LR: 0.000012, Loss: 1.5406
2026-05-15 21:57:10 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2810/1794] LR: 0.000012, Loss: 1.3338
2026-05-15 21:57:11 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2820/1794] LR: 0.000011, Loss: 0.6097
2026-05-15 21:57:11 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2830/1794] LR: 0.000011, Loss: 0.6356
2026-05-15 21:57:12 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2840/1794] LR: 0.000011, Loss: 1.0146
2026-05-15 21:57:12 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2850/1794] LR: 0.000011, Loss: 2.0886
2026-05-15 21:57:13 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2860/1794] LR: 0.000010, Loss: 0.1597
2026-05-15 21:57:13 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2870/1794] LR: 0.000010, Loss: 0.3865
2026-05-15 21:57:14 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2880/1794] LR: 0.000010, Loss: 0.2864
2026-05-15 21:57:14 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2890/1794] LR: 0.000009, Loss: 1.4818
2026-05-15 21:57:15 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2900/1794] LR: 0.000009, Loss: 0.3737
2026-05-15 21:57:16 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2910/1794] LR: 0.000009, Loss: 0.3744
2026-05-15 21:57:16 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2920/1794] LR: 0.000009, Loss: 0.3389
2026-05-15 21:57:17 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2930/1794] LR: 0.000008, Loss: 0.7624
2026-05-15 21:57:17 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2940/1794] LR: 0.000008, Loss: 1.1356
2026-05-15 21:57:18 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2950/1794] LR: 0.000008, Loss: 3.0660
2026-05-15 21:57:19 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2960/1794] LR: 0.000008, Loss: 2.3993
2026-05-15 21:57:19 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2970/1794] LR: 0.000007, Loss: 4.4355
2026-05-15 21:57:20 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2980/1794] LR: 0.000007, Loss: 0.7552
2026-05-15 21:57:20 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 2990/1794] LR: 0.000007, Loss: 4.4602
2026-05-15 21:57:21 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3000/1794] LR: 0.000007, Loss: 0.3369
2026-05-15 21:57:22 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3010/1794] LR: 0.000007, Loss: 0.1387
2026-05-15 21:57:22 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3020/1794] LR: 0.000006, Loss: 0.6729
2026-05-15 21:57:23 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3030/1794] LR: 0.000006, Loss: 3.1809
2026-05-15 21:57:23 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3040/1794] LR: 0.000006, Loss: 0.3831
2026-05-15 21:57:24 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3050/1794] LR: 0.000006, Loss: 0.2780
2026-05-15 21:57:24 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3060/1794] LR: 0.000006, Loss: 0.2683
2026-05-15 21:57:25 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3070/1794] LR: 0.000005, Loss: 0.3906
2026-05-15 21:57:26 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3080/1794] LR: 0.000005, Loss: 0.8128
2026-05-15 21:57:26 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3090/1794] LR: 0.000005, Loss: 1.8650
2026-05-15 21:57:27 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3100/1794] LR: 0.000005, Loss: 0.2816
2026-05-15 21:57:27 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3110/1794] LR: 0.000005, Loss: 0.2632
2026-05-15 21:57:28 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3120/1794] LR: 0.000004, Loss: 1.3896
2026-05-15 21:57:29 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3130/1794] LR: 0.000004, Loss: 2.1945
2026-05-15 21:57:29 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3140/1794] LR: 0.000004, Loss: 1.0838
2026-05-15 21:57:30 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3150/1794] LR: 0.000004, Loss: 0.4745
2026-05-15 21:57:30 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3160/1794] LR: 0.000004, Loss: 0.2575
2026-05-15 21:57:31 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3170/1794] LR: 0.000003, Loss: 0.4368
2026-05-15 21:57:31 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3180/1794] LR: 0.000003, Loss: 0.2905
2026-05-15 21:57:32 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3190/1794] LR: 0.000003, Loss: 1.3888
2026-05-15 21:57:33 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3200/1794] LR: 0.000003, Loss: 0.3012
2026-05-15 21:57:33 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3210/1794] LR: 0.000003, Loss: 0.9543
2026-05-15 21:57:34 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3220/1794] LR: 0.000003, Loss: 0.4134
2026-05-15 21:57:34 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3230/1794] LR: 0.000003, Loss: 1.3708
2026-05-15 21:57:35 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3240/1794] LR: 0.000002, Loss: 0.6229
2026-05-15 21:57:35 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3250/1794] LR: 0.000002, Loss: 1.4378
2026-05-15 21:57:36 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3260/1794] LR: 0.000002, Loss: 3.1213
2026-05-15 21:57:36 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3270/1794] LR: 0.000002, Loss: 0.6572
2026-05-15 21:57:37 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3280/1794] LR: 0.000002, Loss: 1.2480
2026-05-15 21:57:38 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3290/1794] LR: 0.000002, Loss: 1.0981
2026-05-15 21:57:38 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3300/1794] LR: 0.000002, Loss: 0.4321
2026-05-15 21:57:39 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3310/1794] LR: 0.000002, Loss: 2.3303
2026-05-15 21:57:39 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3320/1794] LR: 0.000001, Loss: 0.9430
2026-05-15 21:57:40 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3330/1794] LR: 0.000001, Loss: 5.1340
2026-05-15 21:57:40 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3340/1794] LR: 0.000001, Loss: 2.7393
2026-05-15 21:57:41 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3350/1794] LR: 0.000001, Loss: 0.8171
2026-05-15 21:57:42 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3360/1794] LR: 0.000001, Loss: 0.8582
2026-05-15 21:57:42 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3370/1794] LR: 0.000001, Loss: 1.5488
2026-05-15 21:57:43 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3380/1794] LR: 0.000001, Loss: 1.7059
2026-05-15 21:57:43 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3390/1794] LR: 0.000001, Loss: 0.3557
2026-05-15 21:57:44 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3400/1794] LR: 0.000001, Loss: 1.3660
2026-05-15 21:57:44 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3410/1794] LR: 0.000001, Loss: 0.2612
2026-05-15 21:57:45 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3420/1794] LR: 0.000001, Loss: 0.2913
2026-05-15 21:57:46 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3430/1794] LR: 0.000000, Loss: 0.4107
2026-05-15 21:57:46 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3440/1794] LR: 0.000000, Loss: 0.2296
2026-05-15 21:57:47 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3450/1794] LR: 0.000000, Loss: 3.2555
2026-05-15 21:57:47 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3460/1794] LR: 0.000000, Loss: 2.1433
2026-05-15 21:57:48 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3470/1794] LR: 0.000000, Loss: 0.9287
2026-05-15 21:57:48 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3480/1794] LR: 0.000000, Loss: 0.1606
2026-05-15 21:57:49 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3490/1794] LR: 0.000000, Loss: 1.1455
2026-05-15 21:57:49 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3500/1794] LR: 0.000000, Loss: 1.2582
2026-05-15 21:57:50 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3510/1794] LR: 0.000000, Loss: 1.5929
2026-05-15 21:57:51 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3520/1794] LR: 0.000000, Loss: 0.1948
2026-05-15 21:57:51 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3530/1794] LR: 0.000000, Loss: 0.5979
2026-05-15 21:57:52 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3540/1794] LR: 0.000000, Loss: 0.1246
2026-05-15 21:57:52 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3550/1794] LR: 0.000000, Loss: 0.4311
2026-05-15 21:57:53 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3560/1794] LR: 0.000000, Loss: 0.4204
2026-05-15 21:57:54 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3570/1794] LR: 0.000000, Loss: 0.6093
2026-05-15 21:57:54 - tokenizer_training_rank_0 - INFO - [Epoch 2/2, Step 3580/1794] LR: 0.000000, Loss: 1.4297
2026-05-15 21:57:58 - tokenizer_training_rank_0 - INFO - 
--- Epoch 2/2 Summary ---
Validation Loss: 0.581158
Epoch Time: 0:01:43

2026-05-15 21:57:58 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/smoke_test_full_hpo\tokenizer\best_model (val loss: 0.581158)
2026-05-15 21:58:00 - tokenizer_training_rank_0 - INFO - Tokenizer training completed! Best validation loss: 0.5812
Training time: 3.75 minutes
Model saved to: finetuned/smoke_test_full_hpo\tokenizer

```

---

### `finetuned/smoke_test_full_hpo/tokenizer/trial.log`

```text
2026-05-15 21:40:45 [INFO] Starting tokenizer training...
2026-05-15 21:40:54 [INFO] [Epoch 1/2, Step 10/1794] LR: 0.000030, Loss: 0.0934
2026-05-15 21:40:55 [INFO] [Epoch 1/2, Step 20/1794] LR: 0.000054, Loss: 0.1672
2026-05-15 21:40:55 [INFO] [Epoch 1/2, Step 30/1794] LR: 0.000082, Loss: 0.9532
2026-05-15 21:40:56 [INFO] [Epoch 1/2, Step 40/1794] LR: 0.000100, Loss: 0.1144
2026-05-15 21:40:56 [INFO] [Epoch 1/2, Step 50/1794] LR: 0.000103, Loss: 2.5285
2026-05-15 21:40:57 [INFO] [Epoch 1/2, Step 60/1794] LR: 0.000103, Loss: 1.0392
2026-05-15 21:40:57 [INFO] [Epoch 1/2, Step 70/1794] LR: 0.000103, Loss: 2.6632
2026-05-15 21:40:58 [INFO] [Epoch 1/2, Step 80/1794] LR: 0.000103, Loss: 0.9657
2026-05-15 21:40:58 [INFO] [Epoch 1/2, Step 90/1794] LR: 0.000102, Loss: 1.2757
2026-05-15 21:40:59 [INFO] [Epoch 1/2, Step 100/1794] LR: 0.000102, Loss: 1.2360
2026-05-15 21:41:00 [INFO] [Epoch 1/2, Step 110/1794] LR: 0.000102, Loss: 1.2800
2026-05-15 21:41:00 [INFO] [Epoch 1/2, Step 120/1794] LR: 0.000102, Loss: 2.1414
2026-05-15 21:41:01 [INFO] [Epoch 1/2, Step 130/1794] LR: 0.000102, Loss: 1.6991
2026-05-15 21:41:01 [INFO] [Epoch 1/2, Step 140/1794] LR: 0.000102, Loss: 2.6306
2026-05-15 21:41:02 [INFO] [Epoch 1/2, Step 150/1794] LR: 0.000102, Loss: 0.2886
2026-05-15 21:41:02 [INFO] [Epoch 1/2, Step 160/1794] LR: 0.000102, Loss: 0.7892
2026-05-15 21:41:03 [INFO] [Epoch 1/2, Step 170/1794] LR: 0.000102, Loss: 0.1407
2026-05-15 21:41:03 [INFO] [Epoch 1/2, Step 180/1794] LR: 0.000102, Loss: 1.1482
2026-05-15 21:41:04 [INFO] [Epoch 1/2, Step 190/1794] LR: 0.000102, Loss: 0.1159
2026-05-15 21:41:04 [INFO] [Epoch 1/2, Step 200/1794] LR: 0.000102, Loss: 0.2784
2026-05-15 21:41:05 [INFO] [Epoch 1/2, Step 210/1794] LR: 0.000102, Loss: 0.9711
2026-05-15 21:41:05 [INFO] [Epoch 1/2, Step 220/1794] LR: 0.000102, Loss: 1.8608
2026-05-15 21:41:06 [INFO] [Epoch 1/2, Step 230/1794] LR: 0.000102, Loss: 1.2936
2026-05-15 21:41:06 [INFO] [Epoch 1/2, Step 240/1794] LR: 0.000102, Loss: 0.8538
2026-05-15 21:41:07 [INFO] [Epoch 1/2, Step 250/1794] LR: 0.000102, Loss: 0.1301
2026-05-15 21:41:07 [INFO] [Epoch 1/2, Step 260/1794] LR: 0.000102, Loss: 0.1195
2026-05-15 21:41:08 [INFO] [Epoch 1/2, Step 270/1794] LR: 0.000102, Loss: 0.1528
2026-05-15 21:41:08 [INFO] [Epoch 1/2, Step 280/1794] LR: 0.000101, Loss: 2.0660
2026-05-15 21:41:09 [INFO] [Epoch 1/2, Step 290/1794] LR: 0.000101, Loss: 0.2262
2026-05-15 21:41:09 [INFO] [Epoch 1/2, Step 300/1794] LR: 0.000101, Loss: 0.7963
2026-05-15 21:41:10 [INFO] [Epoch 1/2, Step 310/1794] LR: 0.000101, Loss: 0.4644
2026-05-15 21:41:10 [INFO] [Epoch 1/2, Step 320/1794] LR: 0.000101, Loss: 0.1159
2026-05-15 21:41:11 [INFO] [Epoch 1/2, Step 330/1794] LR: 0.000101, Loss: 0.8148
2026-05-15 21:41:11 [INFO] [Epoch 1/2, Step 340/1794] LR: 0.000101, Loss: 0.1579
2026-05-15 21:41:12 [INFO] [Epoch 1/2, Step 350/1794] LR: 0.000101, Loss: 0.0235
2026-05-15 21:41:12 [INFO] [Epoch 1/2, Step 360/1794] LR: 0.000101, Loss: 0.1147
2026-05-15 21:41:13 [INFO] [Epoch 1/2, Step 370/1794] LR: 0.000100, Loss: 2.1354
2026-05-15 21:41:13 [INFO] [Epoch 1/2, Step 380/1794] LR: 0.000100, Loss: 0.0492
2026-05-15 21:41:14 [INFO] [Epoch 1/2, Step 390/1794] LR: 0.000100, Loss: 0.6292
2026-05-15 21:41:15 [INFO] [Epoch 1/2, Step 400/1794] LR: 0.000100, Loss: 0.9925
2026-05-15 21:41:15 [INFO] [Epoch 1/2, Step 410/1794] LR: 0.000100, Loss: 0.0856
2026-05-15 21:41:15 [INFO] [Epoch 1/2, Step 420/1794] LR: 0.000100, Loss: 0.3889
2026-05-15 21:41:16 [INFO] [Epoch 1/2, Step 430/1794] LR: 0.000100, Loss: 0.2932
2026-05-15 21:41:17 [INFO] [Epoch 1/2, Step 440/1794] LR: 0.000099, Loss: 0.2335
2026-05-15 21:41:17 [INFO] [Epoch 1/2, Step 450/1794] LR: 0.000099, Loss: 1.4390
2026-05-15 21:41:18 [INFO] [Epoch 1/2, Step 460/1794] LR: 0.000099, Loss: 0.8128
2026-05-15 21:41:18 [INFO] [Epoch 1/2, Step 470/1794] LR: 0.000099, Loss: 0.3874
2026-05-15 21:41:19 [INFO] [Epoch 1/2, Step 480/1794] LR: 0.000099, Loss: 0.5893
2026-05-15 21:41:19 [INFO] [Epoch 1/2, Step 490/1794] LR: 0.000099, Loss: 0.1835
2026-05-15 21:41:20 [INFO] [Epoch 1/2, Step 500/1794] LR: 0.000098, Loss: 1.0969
2026-05-15 21:41:20 [INFO] [Epoch 1/2, Step 510/1794] LR: 0.000098, Loss: 0.1519
2026-05-15 21:41:21 [INFO] [Epoch 1/2, Step 520/1794] LR: 0.000098, Loss: 0.8324
2026-05-15 21:41:21 [INFO] [Epoch 1/2, Step 530/1794] LR: 0.000098, Loss: 1.0496
2026-05-15 21:41:22 [INFO] [Epoch 1/2, Step 540/1794] LR: 0.000098, Loss: 1.1442
2026-05-15 21:41:22 [INFO] [Epoch 1/2, Step 550/1794] LR: 0.000097, Loss: 0.2789
2026-05-15 21:41:23 [INFO] [Epoch 1/2, Step 560/1794] LR: 0.000097, Loss: 0.7488
2026-05-15 21:41:23 [INFO] [Epoch 1/2, Step 570/1794] LR: 0.000097, Loss: 1.3129
2026-05-15 21:41:24 [INFO] [Epoch 1/2, Step 580/1794] LR: 0.000097, Loss: 2.4026
2026-05-15 21:41:24 [INFO] [Epoch 1/2, Step 590/1794] LR: 0.000097, Loss: 1.3990
2026-05-15 21:41:25 [INFO] [Epoch 1/2, Step 600/1794] LR: 0.000096, Loss: 0.0489
2026-05-15 21:41:25 [INFO] [Epoch 1/2, Step 610/1794] LR: 0.000096, Loss: 0.4102
2026-05-15 21:41:26 [INFO] [Epoch 1/2, Step 620/1794] LR: 0.000096, Loss: 0.1486
2026-05-15 21:41:26 [INFO] [Epoch 1/2, Step 630/1794] LR: 0.000096, Loss: 0.3594
2026-05-15 21:41:27 [INFO] [Epoch 1/2, Step 640/1794] LR: 0.000096, Loss: 0.5644
2026-05-15 21:41:27 [INFO] [Epoch 1/2, Step 650/1794] LR: 0.000095, Loss: 1.8278
2026-05-15 21:41:28 [INFO] [Epoch 1/2, Step 660/1794] LR: 0.000095, Loss: 1.0655
2026-05-15 21:41:28 [INFO] [Epoch 1/2, Step 670/1794] LR: 0.000095, Loss: 0.6607
2026-05-15 21:41:29 [INFO] [Epoch 1/2, Step 680/1794] LR: 0.000095, Loss: 0.2191
2026-05-15 21:41:29 [INFO] [Epoch 1/2, Step 690/1794] LR: 0.000094, Loss: 0.0685
2026-05-15 21:41:30 [INFO] [Epoch 1/2, Step 700/1794] LR: 0.000094, Loss: 0.3370
2026-05-15 21:41:30 [INFO] [Epoch 1/2, Step 710/1794] LR: 0.000094, Loss: 0.6745
2026-05-15 21:41:31 [INFO] [Epoch 1/2, Step 720/1794] LR: 0.000094, Loss: 1.2939
2026-05-15 21:41:31 [INFO] [Epoch 1/2, Step 730/1794] LR: 0.000093, Loss: 0.7149
2026-05-15 21:41:32 [INFO] [Epoch 1/2, Step 740/1794] LR: 0.000093, Loss: 0.1756
2026-05-15 21:41:33 [INFO] [Epoch 1/2, Step 750/1794] LR: 0.000093, Loss: 0.5963
2026-05-15 21:41:33 [INFO] [Epoch 1/2, Step 760/1794] LR: 0.000093, Loss: 0.7046
2026-05-15 21:41:34 [INFO] [Epoch 1/2, Step 770/1794] LR: 0.000092, Loss: 1.6956
2026-05-15 21:41:34 [INFO] [Epoch 1/2, Step 780/1794] LR: 0.000092, Loss: 0.5007
2026-05-15 21:41:35 [INFO] [Epoch 1/2, Step 790/1794] LR: 0.000092, Loss: 0.3058
2026-05-15 21:41:36 [INFO] [Epoch 1/2, Step 800/1794] LR: 0.000091, Loss: 0.2088
2026-05-15 21:41:36 [INFO] [Epoch 1/2, Step 810/1794] LR: 0.000091, Loss: 0.5786
2026-05-15 21:41:37 [INFO] [Epoch 1/2, Step 820/1794] LR: 0.000091, Loss: 0.8601
2026-05-15 21:41:37 [INFO] [Epoch 1/2, Step 830/1794] LR: 0.000091, Loss: 0.8821
2026-05-15 21:41:38 [INFO] [Epoch 1/2, Step 840/1794] LR: 0.000090, Loss: 0.2951
2026-05-15 21:41:38 [INFO] [Epoch 1/2, Step 850/1794] LR: 0.000090, Loss: 0.1388
2026-05-15 21:41:39 [INFO] [Epoch 1/2, Step 860/1794] LR: 0.000090, Loss: 1.1992
2026-05-15 21:41:39 [INFO] [Epoch 1/2, Step 870/1794] LR: 0.000089, Loss: 1.5692
2026-05-15 21:41:40 [INFO] [Epoch 1/2, Step 880/1794] LR: 0.000089, Loss: 0.6004
2026-05-15 21:41:41 [INFO] [Epoch 1/2, Step 890/1794] LR: 0.000089, Loss: 0.2031
2026-05-15 21:41:41 [INFO] [Epoch 1/2, Step 900/1794] LR: 0.000088, Loss: 0.5946
2026-05-15 21:41:42 [INFO] [Epoch 1/2, Step 910/1794] LR: 0.000088, Loss: 0.2468
2026-05-15 21:41:42 [INFO] [Epoch 1/2, Step 920/1794] LR: 0.000088, Loss: 1.1475
2026-05-15 21:41:43 [INFO] [Epoch 1/2, Step 930/1794] LR: 0.000088, Loss: 0.1896
2026-05-15 21:41:43 [INFO] [Epoch 1/2, Step 940/1794] LR: 0.000087, Loss: 0.5529
2026-05-15 21:41:44 [INFO] [Epoch 1/2, Step 950/1794] LR: 0.000087, Loss: 0.8473
2026-05-15 21:41:44 [INFO] [Epoch 1/2, Step 960/1794] LR: 0.000087, Loss: 0.4131
2026-05-15 21:41:45 [INFO] [Epoch 1/2, Step 970/1794] LR: 0.000086, Loss: 0.9431
2026-05-15 21:41:45 [INFO] [Epoch 1/2, Step 980/1794] LR: 0.000086, Loss: 0.8457
2026-05-15 21:41:46 [INFO] [Epoch 1/2, Step 990/1794] LR: 0.000086, Loss: 0.5924
2026-05-15 21:41:46 [INFO] [Epoch 1/2, Step 1000/1794] LR: 0.000085, Loss: 0.2735
2026-05-15 21:41:47 [INFO] [Epoch 1/2, Step 1010/1794] LR: 0.000085, Loss: 0.1044
2026-05-15 21:41:47 [INFO] [Epoch 1/2, Step 1020/1794] LR: 0.000085, Loss: 0.8780
2026-05-15 21:41:48 [INFO] [Epoch 1/2, Step 1030/1794] LR: 0.000084, Loss: 0.9956
2026-05-15 21:41:48 [INFO] [Epoch 1/2, Step 1040/1794] LR: 0.000084, Loss: 0.5777
2026-05-15 21:41:49 [INFO] [Epoch 1/2, Step 1050/1794] LR: 0.000083, Loss: 0.3414
2026-05-15 21:41:49 [INFO] [Epoch 1/2, Step 1060/1794] LR: 0.000083, Loss: 0.6770
2026-05-15 21:41:50 [INFO] [Epoch 1/2, Step 1070/1794] LR: 0.000083, Loss: 0.3499
2026-05-15 21:41:50 [INFO] [Epoch 1/2, Step 1080/1794] LR: 0.000082, Loss: 0.2350
2026-05-15 21:41:51 [INFO] [Epoch 1/2, Step 1090/1794] LR: 0.000082, Loss: 0.3666
2026-05-15 21:41:51 [INFO] [Epoch 1/2, Step 1100/1794] LR: 0.000082, Loss: 0.5840
2026-05-15 21:41:52 [INFO] [Epoch 1/2, Step 1110/1794] LR: 0.000081, Loss: 0.6245
2026-05-15 21:41:52 [INFO] [Epoch 1/2, Step 1120/1794] LR: 0.000081, Loss: 0.5956
2026-05-15 21:41:53 [INFO] [Epoch 1/2, Step 1130/1794] LR: 0.000081, Loss: 0.1235
2026-05-15 21:41:54 [INFO] [Epoch 1/2, Step 1140/1794] LR: 0.000080, Loss: 0.1891
2026-05-15 21:41:54 [INFO] [Epoch 1/2, Step 1150/1794] LR: 0.000080, Loss: 0.4188
2026-05-15 21:41:55 [INFO] [Epoch 1/2, Step 1160/1794] LR: 0.000079, Loss: 0.0751
2026-05-15 21:41:55 [INFO] [Epoch 1/2, Step 1170/1794] LR: 0.000079, Loss: 0.4550
2026-05-15 21:41:56 [INFO] [Epoch 1/2, Step 1180/1794] LR: 0.000079, Loss: 0.5279
2026-05-15 21:41:56 [INFO] [Epoch 1/2, Step 1190/1794] LR: 0.000078, Loss: 0.2572
2026-05-15 21:41:57 [INFO] [Epoch 1/2, Step 1200/1794] LR: 0.000078, Loss: 0.9624
2026-05-15 21:41:57 [INFO] [Epoch 1/2, Step 1210/1794] LR: 0.000078, Loss: 0.4356
2026-05-15 21:41:58 [INFO] [Epoch 1/2, Step 1220/1794] LR: 0.000077, Loss: 0.1305
2026-05-15 21:41:58 [INFO] [Epoch 1/2, Step 1230/1794] LR: 0.000077, Loss: 0.0690
2026-05-15 21:41:59 [INFO] [Epoch 1/2, Step 1240/1794] LR: 0.000076, Loss: 0.2010
2026-05-15 21:41:59 [INFO] [Epoch 1/2, Step 1250/1794] LR: 0.000076, Loss: 0.8517
2026-05-15 21:42:00 [INFO] [Epoch 1/2, Step 1260/1794] LR: 0.000076, Loss: 0.1705
2026-05-15 21:42:00 [INFO] [Epoch 1/2, Step 1270/1794] LR: 0.000075, Loss: 0.8618
2026-05-15 21:42:01 [INFO] [Epoch 1/2, Step 1280/1794] LR: 0.000075, Loss: 0.3596
2026-05-15 21:42:01 [INFO] [Epoch 1/2, Step 1290/1794] LR: 0.000074, Loss: 0.2135
2026-05-15 21:42:02 [INFO] [Epoch 1/2, Step 1300/1794] LR: 0.000074, Loss: 0.1013
2026-05-15 21:42:02 [INFO] [Epoch 1/2, Step 1310/1794] LR: 0.000074, Loss: 0.2392
2026-05-15 21:42:03 [INFO] [Epoch 1/2, Step 1320/1794] LR: 0.000073, Loss: 0.2125
2026-05-15 21:42:03 [INFO] [Epoch 1/2, Step 1330/1794] LR: 0.000073, Loss: 0.6525
2026-05-15 21:42:04 [INFO] [Epoch 1/2, Step 1340/1794] LR: 0.000072, Loss: 0.7641
2026-05-15 21:42:04 [INFO] [Epoch 1/2, Step 1350/1794] LR: 0.000072, Loss: 0.3308
2026-05-15 21:42:05 [INFO] [Epoch 1/2, Step 1360/1794] LR: 0.000071, Loss: 0.6004
2026-05-15 21:42:05 [INFO] [Epoch 1/2, Step 1370/1794] LR: 0.000071, Loss: 0.6063
2026-05-15 21:42:06 [INFO] [Epoch 1/2, Step 1380/1794] LR: 0.000071, Loss: 0.5619
2026-05-15 21:42:06 [INFO] [Epoch 1/2, Step 1390/1794] LR: 0.000070, Loss: 0.1106
2026-05-15 21:42:07 [INFO] [Epoch 1/2, Step 1400/1794] LR: 0.000070, Loss: 0.7706
2026-05-15 21:42:07 [INFO] [Epoch 1/2, Step 1410/1794] LR: 0.000069, Loss: 0.8052
2026-05-15 21:42:08 [INFO] [Epoch 1/2, Step 1420/1794] LR: 0.000069, Loss: 0.0512
2026-05-15 21:42:08 [INFO] [Epoch 1/2, Step 1430/1794] LR: 0.000068, Loss: 0.3990
2026-05-15 21:42:09 [INFO] [Epoch 1/2, Step 1440/1794] LR: 0.000068, Loss: 0.8056
2026-05-15 21:42:09 [INFO] [Epoch 1/2, Step 1450/1794] LR: 0.000068, Loss: 0.3479
2026-05-15 21:42:10 [INFO] [Epoch 1/2, Step 1460/1794] LR: 0.000067, Loss: 0.3589
2026-05-15 21:42:10 [INFO] [Epoch 1/2, Step 1470/1794] LR: 0.000067, Loss: 1.6948
2026-05-15 21:42:11 [INFO] [Epoch 1/2, Step 1480/1794] LR: 0.000066, Loss: 0.7456
2026-05-15 21:42:11 [INFO] [Epoch 1/2, Step 1490/1794] LR: 0.000066, Loss: 0.4041
2026-05-15 21:42:12 [INFO] [Epoch 1/2, Step 1500/1794] LR: 0.000065, Loss: 0.2764
2026-05-15 21:42:12 [INFO] [Epoch 1/2, Step 1510/1794] LR: 0.000065, Loss: 0.0675
2026-05-15 21:42:13 [INFO] [Epoch 1/2, Step 1520/1794] LR: 0.000065, Loss: 0.1951
2026-05-15 21:42:13 [INFO] [Epoch 1/2, Step 1530/1794] LR: 0.000064, Loss: 0.3156
2026-05-15 21:42:14 [INFO] [Epoch 1/2, Step 1540/1794] LR: 0.000064, Loss: 0.6625
2026-05-15 21:42:15 [INFO] [Epoch 1/2, Step 1550/1794] LR: 0.000063, Loss: 0.0712
2026-05-15 21:42:15 [INFO] [Epoch 1/2, Step 1560/1794] LR: 0.000063, Loss: 0.1853
2026-05-15 21:42:16 [INFO] [Epoch 1/2, Step 1570/1794] LR: 0.000062, Loss: 0.1329
2026-05-15 21:42:16 [INFO] [Epoch 1/2, Step 1580/1794] LR: 0.000062, Loss: 1.9947
2026-05-15 21:42:17 [INFO] [Epoch 1/2, Step 1590/1794] LR: 0.000061, Loss: 1.1370
2026-05-15 21:42:17 [INFO] [Epoch 1/2, Step 1600/1794] LR: 0.000061, Loss: 0.3340
2026-05-15 21:42:18 [INFO] [Epoch 1/2, Step 1610/1794] LR: 0.000061, Loss: 0.9324
2026-05-15 21:42:18 [INFO] [Epoch 1/2, Step 1620/1794] LR: 0.000060, Loss: 0.2699
2026-05-15 21:42:19 [INFO] [Epoch 1/2, Step 1630/1794] LR: 0.000060, Loss: 0.3760
2026-05-15 21:42:19 [INFO] [Epoch 1/2, Step 1640/1794] LR: 0.000059, Loss: 0.4571
2026-05-15 21:42:20 [INFO] [Epoch 1/2, Step 1650/1794] LR: 0.000059, Loss: 0.9502
2026-05-15 21:42:20 [INFO] [Epoch 1/2, Step 1660/1794] LR: 0.000058, Loss: 0.6017
2026-05-15 21:42:21 [INFO] [Epoch 1/2, Step 1670/1794] LR: 0.000058, Loss: 0.2305
2026-05-15 21:42:21 [INFO] [Epoch 1/2, Step 1680/1794] LR: 0.000057, Loss: 2.7793
2026-05-15 21:42:22 [INFO] [Epoch 1/2, Step 1690/1794] LR: 0.000057, Loss: 0.3284
2026-05-15 21:42:22 [INFO] [Epoch 1/2, Step 1700/1794] LR: 0.000057, Loss: 0.3836
2026-05-15 21:42:23 [INFO] [Epoch 1/2, Step 1710/1794] LR: 0.000056, Loss: 0.3774
2026-05-15 21:42:23 [INFO] [Epoch 1/2, Step 1720/1794] LR: 0.000056, Loss: 0.2475
2026-05-15 21:42:24 [INFO] [Epoch 1/2, Step 1730/1794] LR: 0.000055, Loss: 2.0099
2026-05-15 21:42:24 [INFO] [Epoch 1/2, Step 1740/1794] LR: 0.000055, Loss: 2.0065
2026-05-15 21:42:25 [INFO] [Epoch 1/2, Step 1750/1794] LR: 0.000054, Loss: 0.7834
2026-05-15 21:42:25 [INFO] [Epoch 1/2, Step 1760/1794] LR: 0.000054, Loss: 0.1383
2026-05-15 21:42:26 [INFO] [Epoch 1/2, Step 1770/1794] LR: 0.000053, Loss: 0.6199
2026-05-15 21:42:26 [INFO] [Epoch 1/2, Step 1780/1794] LR: 0.000053, Loss: 0.4410
2026-05-15 21:42:27 [INFO] [Epoch 1/2, Step 1790/1794] LR: 0.000052, Loss: 0.7222
2026-05-15 21:42:40 [INFO] 
--- Epoch 1/2 Summary ---
Validation Loss: 0.416737
Epoch Time: 0:01:55

2026-05-15 21:42:40 [INFO] Best model saved: finetuned/smoke_test_full_hpo\tokenizer\best_model (val loss: 0.416737)
2026-05-15 21:42:40 [INFO] [Epoch 2/2, Step 1800/1794] LR: 0.000052, Loss: 0.2318
2026-05-15 21:42:41 [INFO] [Epoch 2/2, Step 1810/1794] LR: 0.000052, Loss: 1.1494
2026-05-15 21:42:41 [INFO] [Epoch 2/2, Step 1820/1794] LR: 0.000051, Loss: 0.5388
2026-05-15 21:42:42 [INFO] [Epoch 2/2, Step 1830/1794] LR: 0.000051, Loss: 0.2261
2026-05-15 21:42:42 [INFO] [Epoch 2/2, Step 1840/1794] LR: 0.000050, Loss: 0.1919
2026-05-15 21:42:43 [INFO] [Epoch 2/2, Step 1850/1794] LR: 0.000050, Loss: 0.0320
2026-05-15 21:42:43 [INFO] [Epoch 2/2, Step 1860/1794] LR: 0.000049, Loss: 0.0634
2026-05-15 21:42:44 [INFO] [Epoch 2/2, Step 1870/1794] LR: 0.000049, Loss: 0.0711
2026-05-15 21:42:44 [INFO] [Epoch 2/2, Step 1880/1794] LR: 0.000048, Loss: 0.3310
2026-05-15 21:42:45 [INFO] [Epoch 2/2, Step 1890/1794] LR: 0.000048, Loss: 0.2815
2026-05-15 21:42:45 [INFO] [Epoch 2/2, Step 1900/1794] LR: 0.000047, Loss: 0.0814
2026-05-15 21:42:46 [INFO] [Epoch 2/2, Step 1910/1794] LR: 0.000047, Loss: 1.9446
2026-05-15 21:42:46 [INFO] [Epoch 2/2, Step 1920/1794] LR: 0.000047, Loss: 0.4770
2026-05-15 21:42:47 [INFO] [Epoch 2/2, Step 1930/1794] LR: 0.000046, Loss: 0.1961
2026-05-15 21:42:47 [INFO] [Epoch 2/2, Step 1940/1794] LR: 0.000046, Loss: 0.7419
2026-05-15 21:42:48 [INFO] [Epoch 2/2, Step 1950/1794] LR: 0.000045, Loss: 0.4748
2026-05-15 21:42:48 [INFO] [Epoch 2/2, Step 1960/1794] LR: 0.000045, Loss: 0.5297
2026-05-15 21:42:49 [INFO] [Epoch 2/2, Step 1970/1794] LR: 0.000044, Loss: 0.7692
2026-05-15 21:42:49 [INFO] [Epoch 2/2, Step 1980/1794] LR: 0.000044, Loss: 0.4067
2026-05-15 21:42:50 [INFO] [Epoch 2/2, Step 1990/1794] LR: 0.000043, Loss: 0.5611
2026-05-15 21:42:50 [INFO] [Epoch 2/2, Step 2000/1794] LR: 0.000043, Loss: 1.0086
2026-05-15 21:42:51 [INFO] [Epoch 2/2, Step 2010/1794] LR: 0.000042, Loss: 0.4568
2026-05-15 21:42:51 [INFO] [Epoch 2/2, Step 2020/1794] LR: 0.000042, Loss: 0.4126
2026-05-15 21:42:52 [INFO] [Epoch 2/2, Step 2030/1794] LR: 0.000042, Loss: 0.2150
2026-05-15 21:42:52 [INFO] [Epoch 2/2, Step 2040/1794] LR: 0.000041, Loss: 0.3326
2026-05-15 21:42:53 [INFO] [Epoch 2/2, Step 2050/1794] LR: 0.000041, Loss: 0.1433
2026-05-15 21:42:53 [INFO] [Epoch 2/2, Step 2060/1794] LR: 0.000040, Loss: 0.7554
2026-05-15 21:42:54 [INFO] [Epoch 2/2, Step 2070/1794] LR: 0.000040, Loss: 0.5539
2026-05-15 21:42:54 [INFO] [Epoch 2/2, Step 2080/1794] LR: 0.000039, Loss: 0.0283
2026-05-15 21:42:55 [INFO] [Epoch 2/2, Step 2090/1794] LR: 0.000039, Loss: 0.5014
2026-05-15 21:42:55 [INFO] [Epoch 2/2, Step 2100/1794] LR: 0.000038, Loss: 1.1932
2026-05-15 21:42:56 [INFO] [Epoch 2/2, Step 2110/1794] LR: 0.000038, Loss: 0.2107
2026-05-15 21:42:57 [INFO] [Epoch 2/2, Step 2120/1794] LR: 0.000038, Loss: 1.0380
2026-05-15 21:42:57 [INFO] [Epoch 2/2, Step 2130/1794] LR: 0.000037, Loss: 1.1829
2026-05-15 21:42:58 [INFO] [Epoch 2/2, Step 2140/1794] LR: 0.000037, Loss: 0.1310
2026-05-15 21:42:58 [INFO] [Epoch 2/2, Step 2150/1794] LR: 0.000036, Loss: 0.9582
2026-05-15 21:42:59 [INFO] [Epoch 2/2, Step 2160/1794] LR: 0.000036, Loss: 0.2160
2026-05-15 21:42:59 [INFO] [Epoch 2/2, Step 2170/1794] LR: 0.000035, Loss: 0.3350
2026-05-15 21:43:00 [INFO] [Epoch 2/2, Step 2180/1794] LR: 0.000035, Loss: 0.1850
2026-05-15 21:43:00 [INFO] [Epoch 2/2, Step 2190/1794] LR: 0.000035, Loss: 0.2518
2026-05-15 21:43:01 [INFO] [Epoch 2/2, Step 2200/1794] LR: 0.000034, Loss: 0.1467
2026-05-15 21:43:01 [INFO] [Epoch 2/2, Step 2210/1794] LR: 0.000034, Loss: 0.5146
2026-05-15 21:43:02 [INFO] [Epoch 2/2, Step 2220/1794] LR: 0.000033, Loss: 0.3299
2026-05-15 21:43:02 [INFO] [Epoch 2/2, Step 2230/1794] LR: 0.000033, Loss: 0.1376
2026-05-15 21:43:03 [INFO] [Epoch 2/2, Step 2240/1794] LR: 0.000032, Loss: 0.3024
2026-05-15 21:43:03 [INFO] [Epoch 2/2, Step 2250/1794] LR: 0.000032, Loss: 0.6334
2026-05-15 21:43:04 [INFO] [Epoch 2/2, Step 2260/1794] LR: 0.000032, Loss: 0.1876
2026-05-15 21:43:04 [INFO] [Epoch 2/2, Step 2270/1794] LR: 0.000031, Loss: 0.6306
2026-05-15 21:43:05 [INFO] [Epoch 2/2, Step 2280/1794] LR: 0.000031, Loss: 0.4843
2026-05-15 21:43:06 [INFO] [Epoch 2/2, Step 2290/1794] LR: 0.000030, Loss: 0.3089
2026-05-15 21:43:06 [INFO] [Epoch 2/2, Step 2300/1794] LR: 0.000030, Loss: 0.1768
2026-05-15 21:43:07 [INFO] [Epoch 2/2, Step 2310/1794] LR: 0.000030, Loss: 0.6731
2026-05-15 21:43:07 [INFO] [Epoch 2/2, Step 2320/1794] LR: 0.000029, Loss: 0.4076
2026-05-15 21:43:08 [INFO] [Epoch 2/2, Step 2330/1794] LR: 0.000029, Loss: 0.5024
2026-05-15 21:43:08 [INFO] [Epoch 2/2, Step 2340/1794] LR: 0.000028, Loss: 0.1444
2026-05-15 21:43:09 [INFO] [Epoch 2/2, Step 2350/1794] LR: 0.000028, Loss: 0.4710
2026-05-15 21:43:09 [INFO] [Epoch 2/2, Step 2360/1794] LR: 0.000027, Loss: 0.1398
2026-05-15 21:43:10 [INFO] [Epoch 2/2, Step 2370/1794] LR: 0.000027, Loss: 0.1770
2026-05-15 21:43:10 [INFO] [Epoch 2/2, Step 2380/1794] LR: 0.000027, Loss: 0.2325
2026-05-15 21:43:11 [INFO] [Epoch 2/2, Step 2390/1794] LR: 0.000026, Loss: 0.1824
2026-05-15 21:43:11 [INFO] [Epoch 2/2, Step 2400/1794] LR: 0.000026, Loss: 0.4243
2026-05-15 21:43:12 [INFO] [Epoch 2/2, Step 2410/1794] LR: 0.000025, Loss: 0.1152
2026-05-15 21:43:12 [INFO] [Epoch 2/2, Step 2420/1794] LR: 0.000025, Loss: 0.2339
2026-05-15 21:43:13 [INFO] [Epoch 2/2, Step 2430/1794] LR: 0.000025, Loss: 0.6044
2026-05-15 21:43:13 [INFO] [Epoch 2/2, Step 2440/1794] LR: 0.000024, Loss: 0.5946
2026-05-15 21:43:14 [INFO] [Epoch 2/2, Step 2450/1794] LR: 0.000024, Loss: 1.1153
2026-05-15 21:43:14 [INFO] [Epoch 2/2, Step 2460/1794] LR: 0.000024, Loss: 0.0373
2026-05-15 21:43:15 [INFO] [Epoch 2/2, Step 2470/1794] LR: 0.000023, Loss: 0.7118
2026-05-15 21:43:15 [INFO] [Epoch 2/2, Step 2480/1794] LR: 0.000023, Loss: 0.3600
2026-05-15 21:43:16 [INFO] [Epoch 2/2, Step 2490/1794] LR: 0.000022, Loss: 0.4604
2026-05-15 21:43:16 [INFO] [Epoch 2/2, Step 2500/1794] LR: 0.000022, Loss: 1.3866
2026-05-15 21:43:17 [INFO] [Epoch 2/2, Step 2510/1794] LR: 0.000022, Loss: 0.5305
2026-05-15 21:43:18 [INFO] [Epoch 2/2, Step 2520/1794] LR: 0.000021, Loss: 0.2977
2026-05-15 21:43:18 [INFO] [Epoch 2/2, Step 2530/1794] LR: 0.000021, Loss: 0.0576
2026-05-15 21:43:19 [INFO] [Epoch 2/2, Step 2540/1794] LR: 0.000021, Loss: 0.2242
2026-05-15 21:43:20 [INFO] [Epoch 2/2, Step 2550/1794] LR: 0.000020, Loss: 0.3800
2026-05-15 21:43:20 [INFO] [Epoch 2/2, Step 2560/1794] LR: 0.000020, Loss: 0.0525
2026-05-15 21:43:21 [INFO] [Epoch 2/2, Step 2570/1794] LR: 0.000019, Loss: 0.1171
2026-05-15 21:43:21 [INFO] [Epoch 2/2, Step 2580/1794] LR: 0.000019, Loss: 0.9768
2026-05-15 21:43:22 [INFO] [Epoch 2/2, Step 2590/1794] LR: 0.000019, Loss: 0.3202
2026-05-15 21:43:22 [INFO] [Epoch 2/2, Step 2600/1794] LR: 0.000018, Loss: 2.2012
2026-05-15 21:43:23 [INFO] [Epoch 2/2, Step 2610/1794] LR: 0.000018, Loss: 0.4861
2026-05-15 21:43:23 [INFO] [Epoch 2/2, Step 2620/1794] LR: 0.000018, Loss: 1.1671
2026-05-15 21:43:24 [INFO] [Epoch 2/2, Step 2630/1794] LR: 0.000017, Loss: 0.1209
2026-05-15 21:43:24 [INFO] [Epoch 2/2, Step 2640/1794] LR: 0.000017, Loss: 0.0558
2026-05-15 21:43:25 [INFO] [Epoch 2/2, Step 2650/1794] LR: 0.000017, Loss: 0.3452
2026-05-15 21:43:25 [INFO] [Epoch 2/2, Step 2660/1794] LR: 0.000016, Loss: 0.4389
2026-05-15 21:43:26 [INFO] [Epoch 2/2, Step 2670/1794] LR: 0.000016, Loss: 0.1484
2026-05-15 21:43:26 [INFO] [Epoch 2/2, Step 2680/1794] LR: 0.000016, Loss: 0.2772
2026-05-15 21:43:27 [INFO] [Epoch 2/2, Step 2690/1794] LR: 0.000015, Loss: 0.0829
2026-05-15 21:43:27 [INFO] [Epoch 2/2, Step 2700/1794] LR: 0.000015, Loss: 0.4272
2026-05-15 21:43:28 [INFO] [Epoch 2/2, Step 2710/1794] LR: 0.000015, Loss: 0.9020
2026-05-15 21:43:28 [INFO] [Epoch 2/2, Step 2720/1794] LR: 0.000014, Loss: 0.3542
2026-05-15 21:43:29 [INFO] [Epoch 2/2, Step 2730/1794] LR: 0.000014, Loss: 1.9191
2026-05-15 21:43:30 [INFO] [Epoch 2/2, Step 2740/1794] LR: 0.000014, Loss: 1.6821
2026-05-15 21:43:30 [INFO] [Epoch 2/2, Step 2750/1794] LR: 0.000013, Loss: 0.1932
2026-05-15 21:43:31 [INFO] [Epoch 2/2, Step 2760/1794] LR: 0.000013, Loss: 0.1664
2026-05-15 21:43:31 [INFO] [Epoch 2/2, Step 2770/1794] LR: 0.000013, Loss: 0.5260
2026-05-15 21:43:32 [INFO] [Epoch 2/2, Step 2780/1794] LR: 0.000013, Loss: 0.3718
2026-05-15 21:43:32 [INFO] [Epoch 2/2, Step 2790/1794] LR: 0.000012, Loss: 0.3926
2026-05-15 21:43:33 [INFO] [Epoch 2/2, Step 2800/1794] LR: 0.000012, Loss: 0.1289
2026-05-15 21:43:33 [INFO] [Epoch 2/2, Step 2810/1794] LR: 0.000012, Loss: 0.1799
2026-05-15 21:43:34 [INFO] [Epoch 2/2, Step 2820/1794] LR: 0.000011, Loss: 0.6663
2026-05-15 21:43:34 [INFO] [Epoch 2/2, Step 2830/1794] LR: 0.000011, Loss: 0.1717
2026-05-15 21:43:35 [INFO] [Epoch 2/2, Step 2840/1794] LR: 0.000011, Loss: 0.4602
2026-05-15 21:43:35 [INFO] [Epoch 2/2, Step 2850/1794] LR: 0.000011, Loss: 0.6861
2026-05-15 21:43:36 [INFO] [Epoch 2/2, Step 2860/1794] LR: 0.000010, Loss: 0.2463
2026-05-15 21:43:36 [INFO] [Epoch 2/2, Step 2870/1794] LR: 0.000010, Loss: 0.1948
2026-05-15 21:43:37 [INFO] [Epoch 2/2, Step 2880/1794] LR: 0.000010, Loss: 0.8150
2026-05-15 21:43:37 [INFO] [Epoch 2/2, Step 2890/1794] LR: 0.000009, Loss: 0.0595
2026-05-15 21:43:38 [INFO] [Epoch 2/2, Step 2900/1794] LR: 0.000009, Loss: 0.7488
2026-05-15 21:43:38 [INFO] [Epoch 2/2, Step 2910/1794] LR: 0.000009, Loss: 0.4902
2026-05-15 21:43:39 [INFO] [Epoch 2/2, Step 2920/1794] LR: 0.000009, Loss: 0.0558
2026-05-15 21:43:39 [INFO] [Epoch 2/2, Step 2930/1794] LR: 0.000008, Loss: 1.2533
2026-05-15 21:43:40 [INFO] [Epoch 2/2, Step 2940/1794] LR: 0.000008, Loss: 1.2327
2026-05-15 21:43:40 [INFO] [Epoch 2/2, Step 2950/1794] LR: 0.000008, Loss: 0.7448
2026-05-15 21:43:41 [INFO] [Epoch 2/2, Step 2960/1794] LR: 0.000008, Loss: 0.7312
2026-05-15 21:43:41 [INFO] [Epoch 2/2, Step 2970/1794] LR: 0.000007, Loss: 0.3513
2026-05-15 21:43:42 [INFO] [Epoch 2/2, Step 2980/1794] LR: 0.000007, Loss: 0.3915
2026-05-15 21:43:42 [INFO] [Epoch 2/2, Step 2990/1794] LR: 0.000007, Loss: 0.3130
2026-05-15 21:43:43 [INFO] [Epoch 2/2, Step 3000/1794] LR: 0.000007, Loss: 0.2521
2026-05-15 21:43:43 [INFO] [Epoch 2/2, Step 3010/1794] LR: 0.000007, Loss: 0.6075
2026-05-15 21:43:44 [INFO] [Epoch 2/2, Step 3020/1794] LR: 0.000006, Loss: 0.7391
2026-05-15 21:43:45 [INFO] [Epoch 2/2, Step 3030/1794] LR: 0.000006, Loss: 0.5915
2026-05-15 21:43:45 [INFO] [Epoch 2/2, Step 3040/1794] LR: 0.000006, Loss: 0.4996
2026-05-15 21:43:46 [INFO] [Epoch 2/2, Step 3050/1794] LR: 0.000006, Loss: 0.0864
2026-05-15 21:43:46 [INFO] [Epoch 2/2, Step 3060/1794] LR: 0.000006, Loss: 0.3878
2026-05-15 21:43:47 [INFO] [Epoch 2/2, Step 3070/1794] LR: 0.000005, Loss: 0.4578
2026-05-15 21:43:47 [INFO] [Epoch 2/2, Step 3080/1794] LR: 0.000005, Loss: 0.3525
2026-05-15 21:43:48 [INFO] [Epoch 2/2, Step 3090/1794] LR: 0.000005, Loss: 0.5483
2026-05-15 21:43:48 [INFO] [Epoch 2/2, Step 3100/1794] LR: 0.000005, Loss: 0.1889
2026-05-15 21:43:49 [INFO] [Epoch 2/2, Step 3110/1794] LR: 0.000005, Loss: 0.7412
2026-05-15 21:43:49 [INFO] [Epoch 2/2, Step 3120/1794] LR: 0.000004, Loss: 0.7623
2026-05-15 21:43:50 [INFO] [Epoch 2/2, Step 3130/1794] LR: 0.000004, Loss: 0.6740
2026-05-15 21:43:50 [INFO] [Epoch 2/2, Step 3140/1794] LR: 0.000004, Loss: 0.3147
2026-05-15 21:43:51 [INFO] [Epoch 2/2, Step 3150/1794] LR: 0.000004, Loss: 0.6153
2026-05-15 21:43:51 [INFO] [Epoch 2/2, Step 3160/1794] LR: 0.000004, Loss: 0.2456
2026-05-15 21:43:52 [INFO] [Epoch 2/2, Step 3170/1794] LR: 0.000003, Loss: 0.1917
2026-05-15 21:43:52 [INFO] [Epoch 2/2, Step 3180/1794] LR: 0.000003, Loss: 0.0368
2026-05-15 21:43:53 [INFO] [Epoch 2/2, Step 3190/1794] LR: 0.000003, Loss: 0.4084
2026-05-15 21:43:53 [INFO] [Epoch 2/2, Step 3200/1794] LR: 0.000003, Loss: 0.0562
2026-05-15 21:43:54 [INFO] [Epoch 2/2, Step 3210/1794] LR: 0.000003, Loss: 0.2346
2026-05-15 21:43:54 [INFO] [Epoch 2/2, Step 3220/1794] LR: 0.000003, Loss: 0.9501
2026-05-15 21:43:55 [INFO] [Epoch 2/2, Step 3230/1794] LR: 0.000003, Loss: 0.2933
2026-05-15 21:43:56 [INFO] [Epoch 2/2, Step 3240/1794] LR: 0.000002, Loss: 0.2081
2026-05-15 21:43:56 [INFO] [Epoch 2/2, Step 3250/1794] LR: 0.000002, Loss: 0.1389
2026-05-15 21:43:57 [INFO] [Epoch 2/2, Step 3260/1794] LR: 0.000002, Loss: 0.0633
2026-05-15 21:43:57 [INFO] [Epoch 2/2, Step 3270/1794] LR: 0.000002, Loss: 0.1165
2026-05-15 21:43:58 [INFO] [Epoch 2/2, Step 3280/1794] LR: 0.000002, Loss: 0.1004
2026-05-15 21:43:58 [INFO] [Epoch 2/2, Step 3290/1794] LR: 0.000002, Loss: 0.7955
2026-05-15 21:43:59 [INFO] [Epoch 2/2, Step 3300/1794] LR: 0.000002, Loss: 0.4110
2026-05-15 21:43:59 [INFO] [Epoch 2/2, Step 3310/1794] LR: 0.000002, Loss: 0.3762
2026-05-15 21:44:00 [INFO] [Epoch 2/2, Step 3320/1794] LR: 0.000001, Loss: 0.0938
2026-05-15 21:44:00 [INFO] [Epoch 2/2, Step 3330/1794] LR: 0.000001, Loss: 0.1018
2026-05-15 21:44:01 [INFO] [Epoch 2/2, Step 3340/1794] LR: 0.000001, Loss: 0.3443
2026-05-15 21:44:01 [INFO] [Epoch 2/2, Step 3350/1794] LR: 0.000001, Loss: 0.4096
2026-05-15 21:44:02 [INFO] [Epoch 2/2, Step 3360/1794] LR: 0.000001, Loss: 0.3938
2026-05-15 21:44:02 [INFO] [Epoch 2/2, Step 3370/1794] LR: 0.000001, Loss: 4.3789
2026-05-15 21:44:03 [INFO] [Epoch 2/2, Step 3380/1794] LR: 0.000001, Loss: 0.2775
2026-05-15 21:44:03 [INFO] [Epoch 2/2, Step 3390/1794] LR: 0.000001, Loss: 0.1829
2026-05-15 21:44:04 [INFO] [Epoch 2/2, Step 3400/1794] LR: 0.000001, Loss: 4.5831
2026-05-15 21:44:04 [INFO] [Epoch 2/2, Step 3410/1794] LR: 0.000001, Loss: 0.3400
2026-05-15 21:44:05 [INFO] [Epoch 2/2, Step 3420/1794] LR: 0.000001, Loss: 0.4461
2026-05-15 21:44:05 [INFO] [Epoch 2/2, Step 3430/1794] LR: 0.000000, Loss: 0.2138
2026-05-15 21:44:06 [INFO] [Epoch 2/2, Step 3440/1794] LR: 0.000000, Loss: 0.4916
2026-05-15 21:44:06 [INFO] [Epoch 2/2, Step 3450/1794] LR: 0.000000, Loss: 0.9511
2026-05-15 21:44:07 [INFO] [Epoch 2/2, Step 3460/1794] LR: 0.000000, Loss: 0.0515
2026-05-15 21:44:07 [INFO] [Epoch 2/2, Step 3470/1794] LR: 0.000000, Loss: 0.2127
2026-05-15 21:44:08 [INFO] [Epoch 2/2, Step 3480/1794] LR: 0.000000, Loss: 0.5905
2026-05-15 21:44:08 [INFO] [Epoch 2/2, Step 3490/1794] LR: 0.000000, Loss: 0.2647
2026-05-15 21:44:09 [INFO] [Epoch 2/2, Step 3500/1794] LR: 0.000000, Loss: 0.1821
2026-05-15 21:44:09 [INFO] [Epoch 2/2, Step 3510/1794] LR: 0.000000, Loss: 0.1969
2026-05-15 21:44:10 [INFO] [Epoch 2/2, Step 3520/1794] LR: 0.000000, Loss: 0.8621
2026-05-15 21:44:11 [INFO] [Epoch 2/2, Step 3530/1794] LR: 0.000000, Loss: 0.4778
2026-05-15 21:44:11 [INFO] [Epoch 2/2, Step 3540/1794] LR: 0.000000, Loss: 0.3459
2026-05-15 21:44:12 [INFO] [Epoch 2/2, Step 3550/1794] LR: 0.000000, Loss: 0.8007
2026-05-15 21:44:12 [INFO] [Epoch 2/2, Step 3560/1794] LR: 0.000000, Loss: 0.5036
2026-05-15 21:44:13 [INFO] [Epoch 2/2, Step 3570/1794] LR: 0.000000, Loss: 0.9126
2026-05-15 21:44:13 [INFO] [Epoch 2/2, Step 3580/1794] LR: 0.000000, Loss: 0.4537
2026-05-15 21:44:17 [INFO] 
--- Epoch 2/2 Summary ---
Validation Loss: 0.390815
Epoch Time: 0:01:37

2026-05-15 21:44:17 [INFO] Best model saved: finetuned/smoke_test_full_hpo\tokenizer\best_model (val loss: 0.390815)

```

---

### `finetuned/smoke_test_full_hpo/tokenizer/best_model/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "beta": 0.05,
  "d_in": 6,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.0,
  "gamma": 1.1,
  "gamma0": 1.0,
  "group_size": 5,
  "n_dec_layers": 4,
  "n_enc_layers": 4,
  "n_heads": 4,
  "resid_dropout_p": 0.0,
  "s1_bits": 10,
  "s2_bits": 10,
  "zeta": 0.05
}
```

---

### `finetuned/smoke_test_full_hpo/tokenizer/best_model/model.safetensors`

```text
[File too large to display: 15.1 MB]
```

---

### `finetuned/smoke_test_full_hpo/tokenizer/best_model/README.md`

```markdown
---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: [More Information Needed]
- Paper: [More Information Needed]
- Docs: [More Information Needed]
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

### `model/module.py`

```python
import math

from einops import rearrange, reduce
from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        # FORCE FP32: Prevents eps=1e-8 from rounding to 0.0 in autocast (FP16),
        # which would otherwise cause torch.log(0.0) -> NaNs.
        zq_32 = zq.float()
        zb = (zq_32 + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)

        # Initialize counts and compute probabilities entirely in float32
        cnt = torch.scatter_reduce(
            torch.zeros(2 ** K, device=zq.device, dtype=torch.float32),
            0,
            zi.flatten(),
            torch.ones_like(zi.flatten(), dtype=torch.float32),
            'sum'
        )
        
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        
        # Save tensors needed for backward. prob is already float32.
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        
        # Return loss cast back to the original mixed-precision dtype
        return H.to(zq.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        
        # Force incoming gradients to FP32 for numerical stability
        grad_output_32 = grad_output.float()
        
        grad_array = -grad_output_32 * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        
        # Compute final input gradient and cast back to original dtype
        grad_input = (reord_grad.unsqueeze(-1) * zq.float()).to(zq.dtype)
        
        # Return gradients for (zq, basis, K, eps)
        return grad_input, None, None, None


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

        return (
            zq,
            commit_loss + self.zeta * entropy_penalty / self.inv_temperature,
            {"H": cb_entropy, "used_codes": used_codes, "indices": indices, "group_indices": group_indices,
             "avg_prob": avg_prob}
        )

    def soft_entropy_loss(self, z):
        group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1)
        divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)

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

        avg_prob = reduce(prob, '... g d ->g d', 'mean')
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample):
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = - probs_per_dim * torch.log(probs_per_dim + 1e-8) - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
        persample_entropy = persample_entropy.sum(-1)
        return persample_entropy.mean()

    def codes_to_indexes(self, zhat):
        assert zhat.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}"
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat):
        zhat_in_group = rearrange(zhat, 'b ... (g c) -> b ... g c', c=self.group_size)
        return ((zhat_in_group + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(indices, self.basis), 2
        )
        return codes_non_centered * 2 - 1

    def group_indexes_to_codes(self, group_indices):
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
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"RotaryPositionalEmbedding requires an even head dimension. "
                f"Got dim={dim}."
            )
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        self.register_buffer(
            "cos_cached", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sin_cached", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self._seq_len_cached: int = 0

    def _update_cos_sin_cache(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len

            t = torch.arange(
                seq_len,
                device=self.inv_freq.device,
                dtype=self.inv_freq.dtype,
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            self.cos_cached = emb.cos()[None, None, :, :].to(dtype=x.dtype)
            self.sin_cached = emb.sin()[None, None, :, :].to(dtype=x.dtype)

        return self.cos_cached, self.sin_cached

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_offset: int = 0  # <--- NEW: Offset parameter for AR generation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        
        # Cache must be large enough to cover the absolute maximum position needed
        max_len = max(q_len + q_offset, k_len)
        cos, sin = self._update_cos_sin_cache(q, max_len)

        # Slice cache for Query based on its precise temporal offset
        q_cos = cos[:, :, q_offset : q_offset + q_len, :]
        q_sin = sin[:, :, q_offset : q_offset + q_len, :]
        
        # Slice cache for Key based on its full length (always starts at t=0)
        k_cos = cos[:, :, :k_len, :]
        k_sin = sin[:, :, :k_len, :]

        rotated_q = (q * q_cos) + (self._rotate_half(q) * q_sin)
        rotated_k = (k * k_cos) + (self._rotate_half(k) * k_sin)
        return rotated_q, rotated_k


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    training: bool = True
) -> torch.Tensor:
    """
    Delegates directly to PyTorch 2.0+ optimized C++ kernels.
    Resolves O(N^2) memory scaling by avoiding explicit attention matrix materialization.
    """
    effective_dropout = dropout_p if training else 0.0

    # Fast-path for causal attention without custom masks
    if is_causal and attn_mask is None:
        return F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=effective_dropout,
            is_causal=True,
            scale=scale
        )

    # Handle custom masks
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # F.sdpa expects True for elements that *are* allowed to attend.
            # Standard Transformer convention usually uses True to mask *out*.
            # Invert the boolean mask to match F.sdpa's expectation.
            attn_mask = ~attn_mask

        # FIXED: Prevent Causal Annihilation
        # Combine the padding mask with a causal mask if the layer is autoregressive.
        if is_causal:
            q_len, k_len = query.size(-2), key.size(-2)
            # Create a causal mask (True where attention is allowed)
            causal_mask = torch.tril(
                torch.ones((q_len, k_len), dtype=torch.bool, device=query.device)
            )

            if attn_mask.dtype == torch.bool:
                # Logical AND: A token can be attended to only if BOTH masks allow it
                attn_mask = attn_mask & causal_mask
            else:
                # Mixed-precision safe: use lowest finite value for active dtype
                # instead of hard -inf to prevent NaN in FP16/BF16 softmax backward
                mask_value = torch.finfo(query.dtype).min
                attn_mask = attn_mask.to(query.dtype).masked_fill(~causal_mask, mask_value)

    # Determine final is_causal flag:
    # - If attn_mask is None and is_causal=True, use PyTorch's built-in causal handling (fast path at line 368 already handled this)
    # - If attn_mask was provided, we combined it with causal mask above, so use is_causal=False since causal is baked into attn_mask
    # - If is_causal=False, ensure causal is disabled
    final_is_causal = is_causal and attn_mask is None

    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=effective_dropout,
        is_causal=final_is_causal,
        scale=scale
    )

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

        # Self-attention always processes aligned sequences, so q_offset remains 0
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

        # <--- NEW: Dynamically infer the offset based on sequence lengths
        q_offset = seq_len - q_len if q_len < seq_len else 0
        q, k = self.rotary(q, k, q_offset=q_offset)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        is_causal_flag = q_len > 1

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
        attn_out = self.cross_attn(
            query=sibling_embed,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        # <--- FIXED: Slice residual to match query length dynamically
        q_len = sibling_embed.shape[1]
        residual = hidden_states[:, -q_len:, :]
        
        return self.norm(residual + attn_out)


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
            mask1 = 2 ** torch.arange(self.s1_bits, device=x1.device, dtype=torch.long)
            mask2 = 2 ** torch.arange(self.s2_bits, device=x2.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask1) != 0 
            x2 = (x2.unsqueeze(-1) & mask2) != 0 
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
        self.dep_layer = DependencyAwareLayer(self.d_model, n_heads=self.n_heads) # <--- FIXED
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)

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

        if use_teacher_forcing and s1_targets is not None:
            # State 1: Explicit Teacher Forcing (Standard for early training/finetuning)
            sibling_embed = self.embedding.emb_s1(s1_targets)
        elif self.training:
            # State 2: Autoregressive Training (Maintains gradient graph)
            # Uses Straight-Through Gumbel-Softmax so S2 can backpropagate into S1
            s1_probs = F.gumbel_softmax(s1_logits, tau=1.0, hard=True)
            sibling_embed = torch.matmul(s1_probs, self.embedding.emb_s1.weight)
        else:
            # State 3: Standard Inference (Detached, memory-efficient)
            s1_probs = F.softmax(s1_logits.detach() / 1.0, dim=-1) # Can inject temp here later
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

        # FIX: Compute scaling statistics strictly over the trailing context window
        # to prevent normalization shift and match CustomKlineDataset training logic.
        recent_x = x[-self.max_context:] if x.shape[0] > self.max_context else x
        x_mean, x_std = np.mean(recent_x, axis=0), np.std(recent_x, axis=0)

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

            # FIX: Bound the scaling calculation to the active context window per series
            recent_x = x[-self.max_context:] if x.shape[0] > self.max_context else x
            x_mean, x_std = np.mean(recent_x, axis=0), np.std(recent_x, axis=0)

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

### `models/nos_base/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "d_model": 832,
  "ff_dim": 2048,
  "ffn_dropout_p": 0.2,
  "learn_te": true,
  "n_heads": 16,
  "n_layers": 12,
  "resid_dropout_p": 0.2,
  "s1_bits": 10,
  "s2_bits": 10,
  "token_dropout_p": 0.0
}
```

---

### `models/nos_base/model.safetensors`

```text
[File too large to display: 390.3 MB]
```

---

### `models/nos_mini/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.2,
  "learn_te": true,
  "n_heads": 4,
  "n_layers": 4,
  "resid_dropout_p": 0.2,
  "s1_bits": 10,
  "s2_bits": 10,
  "token_dropout_p": 0.0
}
```

---

### `models/nos_mini/model.safetensors`

```text
[File too large to display: 15.7 MB]
```

---

### `models/nos_small/config.json`

```json
{
  "attn_dropout_p": 0.1,
  "d_model": 512,
  "ff_dim": 1024,
  "ffn_dropout_p": 0.25,
  "learn_te": true,
  "n_heads": 8,
  "n_layers": 8,
  "resid_dropout_p": 0.25,
  "s1_bits": 10,
  "s2_bits": 10,
  "token_dropout_p": 0.1
}
```

---

### `models/nos_small/model.safetensors`

```text
[File too large to display: 94.4 MB]
```

---

### `models/nos_tokenizer_2k/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "beta": 0.05,
  "d_in": 6,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.0,
  "gamma": 1.1,
  "gamma0": 1.0,
  "group_size": 5,
  "n_dec_layers": 4,
  "n_enc_layers": 4,
  "n_heads": 4,
  "resid_dropout_p": 0.0,
  "s1_bits": 10,
  "s2_bits": 10,
  "zeta": 0.05
}
```

---

### `models/nos_tokenizer_2k/model.safetensors`

```text
[File too large to display: 15.1 MB]
```

---

### `models/nos_tokenizer_base/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "beta": 0.05,
  "d_in": 6,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.0,
  "gamma": 1.1,
  "gamma0": 1.0,
  "group_size": 4,
  "n_dec_layers": 4,
  "n_enc_layers": 4,
  "n_heads": 4,
  "resid_dropout_p": 0.0,
  "s1_bits": 10,
  "s2_bits": 10,
  "zeta": 0.05
}
```

---

### `models/nos_tokenizer_base/model.safetensors`

```text
[File too large to display: 15.1 MB]
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
        self.persistent_workers = training_config.get('persistent_workers', False)
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
```

---

### `hpo_tuner.py`

```python
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
            release_gpu_memory(tokenizer)

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
            release_gpu_memory(model, tokenizer)
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
                 cursor.execute("PRAGMA busy_timeout=60000")
                 cursor.close()

             self._logger.info(
                 "SQLite WAL mode, NullPool, and busy_timeout=60s applied via connection hook."
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
            os.replace(tmp_path, final_path)  # Atomic on POSIX
            self._logger.info(f"Results atomically written to: {final_path}")
            print(f"📊 Results saved to: {final_path}")
        except Exception as exc:
            self._logger.error(
                f"Failed to save results to {final_path}: {exc}", exc_info=True
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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
```

---

### `launch_hpo.sh`

```bash
#!/usr/bin/env bash
# =============================================================================
# launch_hpo.sh
# Asynchronous Multi-GPU HPO Launcher for Nos Model Finetuning
#
# Architecture:
#   Spawns one independent hpo_tuner.py process per GPU. Each process gets
#   its own CUDA context via CUDA_VISIBLE_DEVICES, with no shared memory,
#   no DDP, and no synchronisation barriers. All workers share only the
#   Optuna SQLite/PostgreSQL study database.
#
# Compatibility:
#   - Linux        : bash 4.0+  (Ubuntu, Debian, CentOS, RHEL, Amazon Linux)
#   - Windows      : Git Bash, WSL2, Cygwin (all ship bash 4+)
#   - macOS        : bash 3.2+ via /bin/bash (limited testing)
#   - Cloud VMs    : AWS, GCP, Azure GPU instances (all Linux-based)
#   - Windows VMs  : Azure NV-series, AWS G-series with Git Bash / WSL2
#
# Prerequisites:
#   - nvidia-smi   accessible in PATH
#   - python       accessible in PATH (or set PYTHON_BIN below)
#   - hpo_tuner.py in the same directory as this script (or set SCRIPT_DIR)
#
# Usage:
#   chmod +x launch_hpo.sh
#
#   # Tune both phases across all GPUs (30 trials per worker):
#   ./launch_hpo.sh --config configs/config.yaml --phase both --n-trials 30
#
#   # Tune tokenizer only on GPUs 0,1,2 (GPU subset):
#   ./launch_hpo.sh --gpus 0,1,2 --config configs/config.yaml --phase tokenizer
#
#   # Dry run — print what would be launched without executing:
#   ./launch_hpo.sh --dry-run --config configs/config.yaml --phase both
#
#   # Resume a previously interrupted study:
#   ./launch_hpo.sh --config configs/config.yaml --phase both --n-trials 30
#   (Optuna's load_if_exists=True means workers automatically resume)
#
#   # Use a custom Python binary (e.g., conda env):
#   PYTHON_BIN=/opt/conda/envs/nos/bin/python ./launch_hpo.sh --config ...
#
#   # Limit VRAM per GPU (useful on shared machines):
#   GPU_MEMORY_FRACTION=0.8 ./launch_hpo.sh --config configs/config.yaml
#
# Environment Variables (all optional):
#   PYTHON_BIN              Path to python executable. Default: auto-detected.
#   GPU_MEMORY_FRACTION     CUDA memory fraction [0.1-1.0]. Default: unset.
#   HPO_TIMEOUT_SECONDS     Kill workers after N seconds. Default: unset (no limit).
#   HPO_WORKER_NICE         nice(1) priority for workers [−20 to 19]. Default: 0.
#   LOG_DIR                 Override log directory. Default: logs/hpo_workers.
#
# Exit Codes:
#   0   All workers completed successfully.
#   1   One or more workers failed (details in log files).
#   2   Configuration or environment error (bad args, missing deps).
#   3   Interrupted by user (SIGINT/SIGTERM). Partial results are preserved.
# =============================================================================

# ── Bash version guard ────────────────────────────────────────────────────────
# Arrays with negative indexing and associative arrays require bash 4.0+.
# macOS ships bash 3.2 at /bin/bash but Git Bash / Homebrew bash is 5.x.
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "ERROR: bash 4.0 or later is required." >&2
    echo "       Detected: bash ${BASH_VERSION}" >&2
    echo "       macOS users: brew install bash && use /usr/local/bin/bash" >&2
    echo "       Windows users: use Git Bash 4+ or WSL2." >&2
    exit 2
fi

set -euo pipefail
# -e  : Exit on any unhandled error
# -u  : Treat unset variables as errors
# -o pipefail : Propagate pipe failures (not just last command)

# ── Script self-location (works with symlinks and spaces in paths) ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# ── Colour output (disabled automatically when not a terminal) ────────────────
if [ -t 1 ] && command -v tput &>/dev/null && tput colors &>/dev/null; then
    C_RESET="\033[0m"
    C_BOLD="\033[1m"
    C_RED="\033[31m"
    C_GREEN="\033[32m"
    C_YELLOW="\033[33m"
    C_CYAN="\033[36m"
    C_WHITE="\033[37m"
else
    C_RESET="" C_BOLD="" C_RED="" C_GREEN="" C_YELLOW="" C_CYAN="" C_WHITE=""
fi

# ── Logging helpers ───────────────────────────────────────────────────────────
_ts()    { date "+%Y-%m-%d %H:%M:%S"; }
_info()  { echo -e "${C_CYAN}[$(_ts)] [INFO]${C_RESET}  $*"; }
_ok()    { echo -e "${C_GREEN}[$(_ts)] [ OK ]${C_RESET}  $*"; }
_warn()  { echo -e "${C_YELLOW}[$(_ts)] [WARN]${C_RESET}  $*"; }
_error() { echo -e "${C_RED}[$(_ts)] [ERR ]${C_RESET}  $*" >&2; }
_die()   { _error "$*"; exit 2; }

# ── Default configuration ─────────────────────────────────────────────────────
GPU_SUBSET=""                          # Empty = use all GPUs
DRY_RUN=false
PYTHON_BIN="${PYTHON_BIN:-}"           # Auto-detected below if empty
GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-}"
HPO_TIMEOUT_SECONDS="${HPO_TIMEOUT_SECONDS:-}"
HPO_WORKER_NICE="${HPO_WORKER_NICE:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/hpo_workers}"
HPO_TUNER_SCRIPT="${SCRIPT_DIR}/hpo_tuner.py"
HPO_ARGS=()                            # Arguments forwarded to hpo_tuner.py

# ── Argument parser ───────────────────────────────────────────────────────────
# Separates launcher-specific flags from hpo_tuner.py pass-through arguments.
_usage() {
    cat <<EOF

${C_BOLD}Usage:${C_RESET}
  ${SCRIPT_NAME} [LAUNCHER OPTIONS] [HPO_TUNER OPTIONS]

${C_BOLD}Launcher Options:${C_RESET}
  --gpus <0,1,2,...>      Comma-separated GPU IDs to use. Default: all GPUs.
  --dry-run               Print launch commands without executing them.
  --log-dir <path>        Override log directory. Default: logs/hpo_workers/
  --help, -h              Show this help message.

${C_BOLD}HPO Tuner Options (passed through to hpo_tuner.py):${C_RESET}
  --config <path>         Path to YAML config. Default: configs/config.yaml
  --phase <phase>         tokenizer | basemodel | both. Default: both
  --n-trials <int>        Number of Optuna trials per worker.
  --apply-best            Write best params to a new config file.
  --tokenizer-path <path> Tokenizer path (for --phase basemodel).
  --output-config <path>  Output path for best config YAML.
  --show-importance       Print hyperparameter importance after tuning.

${C_BOLD}Environment Variables:${C_RESET}
  PYTHON_BIN              Python executable. Default: auto-detected.
  GPU_MEMORY_FRACTION     CUDA memory fraction [0.1-1.0].
  HPO_TIMEOUT_SECONDS     Worker timeout in seconds.
  HPO_WORKER_NICE         Process nice priority [-20 to 19]. Default: 0.
  LOG_DIR                 Log directory override.

${C_BOLD}Examples:${C_RESET}
  ${SCRIPT_NAME} --config configs/config.yaml --phase both --n-trials 30
  ${SCRIPT_NAME} --gpus 0,1,2 --config configs/config.yaml --phase tokenizer
  ${SCRIPT_NAME} --dry-run --config configs/config.yaml --phase both
  PYTHON_BIN=/opt/conda/bin/python ${SCRIPT_NAME} --config configs/config.yaml

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            [[ $# -lt 2 ]] && _die "--gpus requires an argument."
            GPU_SUBSET="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --log-dir)
            [[ $# -lt 2 ]] && _die "--log-dir requires an argument."
            LOG_DIR="$2"; shift 2 ;;
        --help|-h)
            _usage; exit 0 ;;
        *)
            # Everything else is forwarded to hpo_tuner.py
            HPO_ARGS+=("$1"); shift ;;
    esac
done

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Environment Detection & Validation
# ══════════════════════════════════════════════════════════════════════════════

# ── OS detection ──────────────────────────────────────────────────────────────
_detect_os() {
    local os_name
    os_name="$(uname -s 2>/dev/null || echo 'Unknown')"
    case "${os_name}" in
        Linux*)   echo "linux"   ;;
        Darwin*)  echo "macos"   ;;
        CYGWIN*)  echo "windows" ;;
        MINGW*)   echo "windows" ;;
        MSYS*)    echo "windows" ;;
        *)
            # Final fallback: check for Windows-style paths
            if [[ "${OSTYPE:-}" == "msys" ]] || [[ "${OSTYPE:-}" == "cygwin" ]]; then
                echo "windows"
            else
                echo "unknown"
            fi
            ;;
    esac
}

OS_TYPE="$(_detect_os)"
_info "Detected OS: ${OS_TYPE}"

# ── Python detection ──────────────────────────────────────────────────────────
_detect_python() {
    # If user explicitly set PYTHON_BIN, validate and use it
    if [[ -n "${PYTHON_BIN}" ]]; then
        if command -v "${PYTHON_BIN}" &>/dev/null; then
            echo "${PYTHON_BIN}"
            return 0
        else
            _die "PYTHON_BIN='${PYTHON_BIN}' is not executable or not in PATH."
        fi
    fi

    # Auto-detection priority:
    # 1. python3   (preferred on Linux/macOS/WSL2)
    # 2. python    (Windows native, conda base envs)
    # 3. py -3     (Windows Python Launcher — py.exe)
    local candidates=("python3" "python" "py")
    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" &>/dev/null; then
            # Verify it is Python 3 (not Python 2)
            local version
            version="$("${candidate}" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo '0')"
            if [[ "${version}" == "3" ]]; then
                echo "${candidate}"
                return 0
            fi
        fi
    done

    # Windows py.exe launcher
    if command -v py &>/dev/null; then
        local version
        version="$(py -3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo '0')"
        if [[ "${version}" == "3" ]]; then
            echo "py -3"
            return 0
        fi
    fi

    _die "No Python 3 interpreter found in PATH. " \
         "Set PYTHON_BIN=/path/to/python3 or activate your conda/venv."
}

PYTHON_BIN="$(_detect_python)"
_info "Python interpreter: ${PYTHON_BIN}"

# Validate Python version is 3.8+
PY_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo '0.0')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"
if [[ "${PY_MAJOR}" -lt 3 ]] || { [[ "${PY_MAJOR}" -eq 3 ]] && [[ "${PY_MINOR}" -lt 8 ]]; }; then
    _die "Python 3.8 or later required. Found: Python ${PY_VERSION}"
fi
_info "Python version: ${PY_VERSION} ✓"

# ── nvidia-smi detection ──────────────────────────────────────────────────────
_detect_nvidia_smi() {
    # Standard PATH check
    if command -v nvidia-smi &>/dev/null; then
        echo "nvidia-smi"
        return 0
    fi

    # Windows-specific install paths (not always in PATH inside Git Bash)
    local win_paths=(
        "/c/Windows/System32/nvidia-smi.exe"
        "/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"
        "/c/Windows/System32/nvidia-smi"
    )
    for path in "${win_paths[@]}"; do
        if [[ -x "${path}" ]]; then
            echo "${path}"
            return 0
        fi
    done

    return 1
}

NVIDIA_SMI=""
if ! NVIDIA_SMI="$(_detect_nvidia_smi)"; then
    _die "nvidia-smi not found in PATH or standard install locations." \
         "Ensure NVIDIA drivers are installed and nvidia-smi is accessible."
fi
_info "nvidia-smi: ${NVIDIA_SMI} ✓"

# ── hpo_tuner.py location check ───────────────────────────────────────────────
if [[ ! -f "${HPO_TUNER_SCRIPT}" ]]; then
    _die "hpo_tuner.py not found at: ${HPO_TUNER_SCRIPT}" \
         "Place launch_hpo.sh in the same directory as hpo_tuner.py."
fi
_info "HPO script: ${HPO_TUNER_SCRIPT} ✓"

# ── Optuna installation check ─────────────────────────────────────────────────
if ! ${PYTHON_BIN} -c "import optuna" &>/dev/null; then
    _die "Optuna is not installed in the detected Python environment." \
         "Run: ${PYTHON_BIN} -m pip install optuna plotly"
fi
OPTUNA_VERSION="$(${PYTHON_BIN} -c 'import optuna; print(optuna.__version__)' 2>/dev/null || echo 'unknown')"
_info "Optuna version: ${OPTUNA_VERSION} ✓"

# ── PyTorch + CUDA check ──────────────────────────────────────────────────────
TORCH_CUDA_AVAILABLE="$(${PYTHON_BIN} -c \
    'import torch; print("yes" if torch.cuda.is_available() else "no")' \
    2>/dev/null || echo 'no')"
if [[ "${TORCH_CUDA_AVAILABLE}" != "yes" ]]; then
    _warn "torch.cuda.is_available() returned False." \
          "Workers will run on CPU. Performance will be severely degraded."
fi

# ── nice availability (non-critical on Windows) ───────────────────────────────
NICE_CMD=""
if command -v nice &>/dev/null && [[ "${OS_TYPE}" != "windows" ]]; then
    NICE_CMD="nice -n ${HPO_WORKER_NICE}"
fi

# ── timeout availability ──────────────────────────────────────────────────────
TIMEOUT_CMD=""
if [[ -n "${HPO_TIMEOUT_SECONDS}" ]]; then
    if command -v timeout &>/dev/null; then
        TIMEOUT_CMD="timeout ${HPO_TIMEOUT_SECONDS}"
    elif command -v gtimeout &>/dev/null; then
        # macOS via coreutils: brew install coreutils
        TIMEOUT_CMD="gtimeout ${HPO_TIMEOUT_SECONDS}"
    else
        _warn "HPO_TIMEOUT_SECONDS=${HPO_TIMEOUT_SECONDS} set but 'timeout' not found." \
              "Workers will run without a time limit."
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: GPU Enumeration
# ══════════════════════════════════════════════════════════════════════════════

# ── Enumerate all available GPUs ──────────────────────────────────────────────
_enumerate_gpus() {
    local raw_output
    # --query-gpu=index gives numeric IDs; csv,noheader strips the header row
    raw_output="$(${NVIDIA_SMI} --query-gpu=index,name,memory.total \
        --format=csv,noheader,nounits 2>/dev/null)" || {
        _die "nvidia-smi failed to enumerate GPUs. " \
             "Check driver installation: ${NVIDIA_SMI} --version"
    }

    if [[ -z "${raw_output}" ]]; then
        _die "nvidia-smi returned no GPU information."
    fi
    echo "${raw_output}"
}

# Build GPU arrays
declare -a ALL_GPU_IDS=()
declare -a ALL_GPU_NAMES=()
declare -a ALL_GPU_VRAM=()

while IFS=',' read -r gpu_id gpu_name gpu_vram; do
    # Trim whitespace (critical for Windows where nvidia-smi adds extra spaces)
    gpu_id="$(echo "${gpu_id}"   | tr -d '[:space:]')"
    gpu_name="$(echo "${gpu_name}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    gpu_vram="$(echo "${gpu_vram}" | tr -d '[:space:]')"

    ALL_GPU_IDS+=("${gpu_id}")
    ALL_GPU_NAMES+=("${gpu_name}")
    ALL_GPU_VRAM+=("${gpu_vram}")
done < <(_enumerate_gpus)

TOTAL_GPUS="${#ALL_GPU_IDS[@]}"
if [[ "${TOTAL_GPUS}" -eq 0 ]]; then
    _die "No GPUs detected. Check nvidia-smi output manually: ${NVIDIA_SMI}"
fi

# ── Apply GPU subset filter ───────────────────────────────────────────────────
declare -a SELECTED_GPU_IDS=()
declare -a SELECTED_GPU_NAMES=()
declare -a SELECTED_GPU_VRAM=()

if [[ -n "${GPU_SUBSET}" ]]; then
    # Parse comma-separated list: "0,1,2" → array
    IFS=',' read -ra REQUESTED_IDS <<< "${GPU_SUBSET}"
    for req_id in "${REQUESTED_IDS[@]}"; do
        req_id="$(echo "${req_id}" | tr -d '[:space:]')"
        local_found=false
        for i in "${!ALL_GPU_IDS[@]}"; do
            if [[ "${ALL_GPU_IDS[$i]}" == "${req_id}" ]]; then
                SELECTED_GPU_IDS+=("${ALL_GPU_IDS[$i]}")
                SELECTED_GPU_NAMES+=("${ALL_GPU_NAMES[$i]}")
                SELECTED_GPU_VRAM+=("${ALL_GPU_VRAM[$i]}")
                local_found=true
                break
            fi
        done
        if [[ "${local_found}" == false ]]; then
            _warn "Requested GPU ${req_id} not found in available GPUs [${ALL_GPU_IDS[*]}]. Skipping."
        fi
    done

    if [[ "${#SELECTED_GPU_IDS[@]}" -eq 0 ]]; then
        _die "No valid GPUs remain after applying --gpus '${GPU_SUBSET}'."
    fi
else
    # Use all available GPUs
    SELECTED_GPU_IDS=("${ALL_GPU_IDS[@]}")
    SELECTED_GPU_NAMES=("${ALL_GPU_NAMES[@]}")
    SELECTED_GPU_VRAM=("${ALL_GPU_VRAM[@]}")
fi

NUM_WORKERS="${#SELECTED_GPU_IDS[@]}"

# ── Check for existing HPO DB and warn if starting fresh ──────────────────────
_check_existing_db() {
    local storage_uri=""
    # Extract storage URI from HPO_ARGS if --config was passed
    local config_path=""
    for i in "${!HPO_ARGS[@]}"; do
        if [[ "${HPO_ARGS[$i]}" == "--config" ]]; then
            config_path="${HPO_ARGS[$((i+1))]:-}"
            break
        fi
    done

    if [[ -n "${config_path}" ]] && [[ -f "${config_path}" ]]; then
        # Simple grep — avoids requiring PyYAML at bash level
        storage_uri="$(grep -E '^\s*storage:' "${config_path}" \
            | sed 's/.*storage:[[:space:]]*//' \
            | tr -d '"'"'" \
            | tr -d '[:space:]' || echo '')"
        if [[ "${storage_uri}" == "null" ]] || [[ -z "${storage_uri}" ]]; then
            _warn "hpo.storage is null in config. Using in-memory Optuna storage." \
                  "Results will NOT be shared between workers or persisted across runs." \
                  "Set hpo.storage to a SQLite or PostgreSQL URI for multi-GPU HPO."
        else
            # Extract file path from sqlite:///path?timeout=60
            local db_path
            db_path="$(echo "${storage_uri}" \
                | sed 's|sqlite:///||' \
                | sed 's|?.*||')"
            if [[ -f "${db_path}" ]]; then
                _info "Existing Optuna DB found: ${db_path}"
                _info "Workers will resume the existing study (load_if_exists=True)."
            fi
        fi
    fi
}
_check_existing_db

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Signal Handling & Cleanup
# ══════════════════════════════════════════════════════════════════════════════

declare -a WORKER_PIDS=()
INTERRUPTED=false

_cleanup() {
    local signal="${1:-SIGTERM}"
    INTERRUPTED=true

    echo ""
    _warn "Received ${signal}. Sending SIGTERM to all worker processes..."

    local kill_failed=0
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || kill_failed=$((kill_failed + 1))
            _warn "  Sent SIGTERM to PID ${pid}"
        fi
    done

    # Give workers 10 seconds to shut down gracefully
    local grace_seconds=10
    _info "Waiting up to ${grace_seconds}s for graceful shutdown..."
    local elapsed=0
    while [[ ${elapsed} -lt ${grace_seconds} ]]; do
        local still_running=0
        for pid in "${WORKER_PIDS[@]}"; do
            kill -0 "${pid}" 2>/dev/null && still_running=$((still_running + 1))
        done
        [[ ${still_running} -eq 0 ]] && break
        sleep 1
        elapsed=$((elapsed + 1))
    done

    # Force kill any remaining workers
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            _warn "  Force killing PID ${pid} (SIGKILL)"
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done

    _warn "Interrupted. Completed trials have been saved to the Optuna DB."
    _warn "Re-run the same command to resume from where you left off."
    exit 3
}

# Trap SIGINT (Ctrl+C), SIGTERM (kill), and SIGHUP (terminal close)
# Note: SIGHUP is not available on Windows but the trap is harmless there
trap '_cleanup SIGINT'  INT
trap '_cleanup SIGTERM' TERM
trap '_cleanup SIGHUP'  HUP 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Print Launch Plan
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${C_BOLD}╔══════════════════════════════════════════════════════════════╗${C_RESET}"
echo -e "${C_BOLD}║       Nos HPO — Asynchronous Multi-GPU Launcher              ║${C_RESET}"
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
printf  "║  %-20s : %-37s║\n" "OS"           "${OS_TYPE}"
printf  "║  %-20s : %-37s║\n" "Python"       "${PYTHON_BIN} (${PY_VERSION})"
printf  "║  %-20s : %-37s║\n" "Optuna"       "${OPTUNA_VERSION}"
printf  "║  %-20s : %-37s║\n" "Total GPUs"   "${TOTAL_GPUS} detected"
printf  "║  %-20s : %-37s║\n" "Worker GPUs"  "${NUM_WORKERS} selected"
printf  "║  %-20s : %-37s║\n" "Log dir"      "${LOG_DIR}"
[[ -n "${HPO_TIMEOUT_SECONDS}" ]] && \
printf  "║  %-20s : %-37s║\n" "Worker timeout"  "${HPO_TIMEOUT_SECONDS}s"
[[ "${DRY_RUN}" == true ]] && \
printf  "║  %-20s : %-37s║\n" "Mode"         "DRY RUN — no processes launched"
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
echo    "║  Selected GPUs:                                              ║"
for i in "${!SELECTED_GPU_IDS[@]}"; do
    printf "║    GPU %-3s : %-20s  %6s MiB VRAM          ║\n" \
        "${SELECTED_GPU_IDS[$i]}" \
        "${SELECTED_GPU_NAMES[$i]:0:20}" \
        "${SELECTED_GPU_VRAM[$i]}"
done
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
echo    "║  HPO arguments forwarded to hpo_tuner.py:                    ║"
printf  "║    %-58s║\n" "${HPO_ARGS[*]:0:58}"
echo -e "${C_BOLD}╚══════════════════════════════════════════════════════════════╝${C_RESET}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Worker Launch
# ══════════════════════════════════════════════════════════════════════════════

mkdir -p "${LOG_DIR}"

# Per-run timestamp for log file naming (shared across all workers in this run)
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

declare -a LOG_FILES=()

for i in "${!SELECTED_GPU_IDS[@]}"; do
    GPU_ID="${SELECTED_GPU_IDS[$i]}"
    GPU_NAME="${SELECTED_GPU_NAMES[$i]}"
    LOG_FILE="${LOG_DIR}/worker_gpu${GPU_ID}_${RUN_TIMESTAMP}.log"
    LOG_FILES+=("${LOG_FILE}")

    # ── Build the full command ─────────────────────────────────────────────
    # Construct as an array to handle paths with spaces correctly
    CMD=()

    # Process priority (Linux/macOS only)
    if [[ -n "${NICE_CMD}" ]]; then
        CMD+=($NICE_CMD)
    fi

    # Timeout wrapper (if configured)
    if [[ -n "${TIMEOUT_CMD}" ]]; then
        CMD+=($TIMEOUT_CMD)
    fi

    # Python interpreter
    # Split PYTHON_BIN in case it contains flags (e.g., "py -3")
    read -ra PY_PARTS <<< "${PYTHON_BIN}"
    CMD+=("${PY_PARTS[@]}")

    # Unbuffered output: critical so log files get real-time writes,
    # not buffered writes that only flush when the process exits.
    CMD+=("-u")

    # The HPO script
    CMD+=("${HPO_TUNER_SCRIPT}")

    # Forward all HPO arguments
    CMD+=("${HPO_ARGS[@]}")

    # ── Environment for this worker ────────────────────────────────────────
    # CUDA_VISIBLE_DEVICES: Restricts this process to exactly one physical GPU.
    # CUDA remaps the assigned GPU to logical index cuda:0 inside the process.
    # PYTHONUNBUFFERED: Belt-and-suspenders for -u flag (some launchers ignore -u).
    # PYTHONFAULTHANDLER: Dumps C-level stack traces on segfaults (invaluable for debugging).
    declare -a WORKER_ENV=(
        "CUDA_VISIBLE_DEVICES=${GPU_ID}"
        "PYTHONUNBUFFERED=1"
        "PYTHONFAULTHANDLER=1"
    )

    # Optional VRAM fraction limiter
    if [[ -n "${GPU_MEMORY_FRACTION}" ]]; then
        WORKER_ENV+=("PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        # Note: actual fraction limiting requires PyTorch code changes;
        # this env var limits the allocator's split size as a proxy.
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        _info "[DRY RUN] GPU ${GPU_ID} (${GPU_NAME}) would run:"
        echo  "         env ${WORKER_ENV[*]} ${CMD[*]}"
        echo  "         >> ${LOG_FILE} 2>&1 &"
    else
        # Write a header into the log file before the process starts
        {
            echo "============================================================"
            echo "  Nos HPO Worker Log"
            echo "  Run timestamp : ${RUN_TIMESTAMP}"
            echo "  GPU ID        : ${GPU_ID} (${GPU_NAME})"
            echo "  GPU VRAM      : ${SELECTED_GPU_VRAM[$i]} MiB"
            echo "  Command       : ${CMD[*]}"
            echo "  Environment   : ${WORKER_ENV[*]}"
            echo "  Started at    : $(date)"
            echo "============================================================"
        } > "${LOG_FILE}"

        # Launch the worker process
        # env sets per-process environment without polluting the parent shell
        env "${WORKER_ENV[@]}" "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
        WORKER_PID=$!

        WORKER_PIDS+=("${WORKER_PID}")
        _ok "GPU ${GPU_ID} (${GPU_NAME}) → PID ${WORKER_PID} → ${LOG_FILE}"
    fi
done

# ── Dry run exits here ────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == true ]]; then
    echo ""
    _info "Dry run complete. No processes were launched."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Live Monitoring
# ══════════════════════════════════════════════════════════════════════════════

echo ""
_info "All ${NUM_WORKERS} worker(s) launched."
echo ""
echo -e "  ${C_CYAN}Monitor logs (all workers):${C_RESET}"
echo    "    tail -f ${LOG_DIR}/worker_gpu*_${RUN_TIMESTAMP}.log"
echo ""
echo -e "  ${C_CYAN}Monitor individual worker:${C_RESET}"
for i in "${!SELECTED_GPU_IDS[@]}"; do
    echo "    tail -f ${LOG_FILES[$i]}"
done
echo ""
echo -e "  ${C_CYAN}Monitor GPU utilisation:${C_RESET}"
echo    "    watch -n 2 nvidia-smi"
echo ""
echo -e "  ${C_CYAN}Stop all workers:${C_RESET}"
echo    "    Ctrl+C  (graceful SIGTERM → waits 10s → SIGKILL)"
echo ""

# ── Optional: background VRAM monitor ─────────────────────────────────────────
# Logs nvidia-smi output every 60 seconds into a separate file for post-run
# VRAM analysis. Killed automatically when the script exits.
VRAM_LOG="${LOG_DIR}/vram_monitor_${RUN_TIMESTAMP}.log"
(
    while true; do
        {
            echo "--- $(date) ---"
            "${NVIDIA_SMI}" \
                --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
                --format=csv,noheader,nounits 2>/dev/null || true
            echo ""
        } >> "${VRAM_LOG}" 2>/dev/null
        sleep 60
    done
) &
VRAM_MONITOR_PID=$!
_info "VRAM monitor started (PID ${VRAM_MONITOR_PID}) → ${VRAM_LOG}"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Wait for All Workers & Collect Results
# ══════════════════════════════════════════════════════════════════════════════

declare -a EXIT_CODES=()
declare -a FAILED_GPUS=()
declare -a SUCCEEDED_GPUS=()
FAILED_COUNT=0
SUCCEEDED_COUNT=0

_info "Waiting for all ${NUM_WORKERS} worker(s) to complete..."
echo  "  (This may take several hours for large HPO runs)"
echo ""

for i in "${!WORKER_PIDS[@]}"; do
    PID="${WORKER_PIDS[$i]}"
    GPU_ID="${SELECTED_GPU_IDS[$i]}"
    GPU_NAME="${SELECTED_GPU_NAMES[$i]}"
    LOG_FILE="${LOG_FILES[$i]}"

    EXIT_CODE=0
    wait "${PID}" || EXIT_CODE=$?
    EXIT_CODES+=("${EXIT_CODE}")

    # Append completion footer to the worker's log
    {
        echo ""
        echo "============================================================"
        echo "  Worker exited at: $(date)"
        echo "  Exit code: ${EXIT_CODE}"
        echo "============================================================"
    } >> "${LOG_FILE}" 2>/dev/null || true

    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        _ok  "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT 0 | SUCCESS"
        SUCCEEDED_GPUS+=("${GPU_ID}")
        SUCCEEDED_COUNT=$((SUCCEEDED_COUNT + 1))
    elif [[ "${EXIT_CODE}" -eq 3 ]]; then
        # Exit code 3 = interrupted by user (our own convention from hpo_tuner.py)
        _warn "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT 3 | INTERRUPTED"
        _warn "  Partial results are preserved in the Optuna DB."
        SUCCEEDED_GPUS+=("${GPU_ID}")  # Treat interruption as non-failure
        SUCCEEDED_COUNT=$((SUCCEEDED_COUNT + 1))
    else
        _error "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT ${EXIT_CODE} | FAILED"
        _error "  Last 20 lines of log:"
        tail -n 20 "${LOG_FILE}" | sed 's/^/    /' >&2
        FAILED_GPUS+=("${GPU_ID}")
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# ── Kill the VRAM monitor ─────────────────────────────────────────────────────
kill "${VRAM_MONITOR_PID}" 2>/dev/null || true
wait "${VRAM_MONITOR_PID}" 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Final Report
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${C_BOLD}══════════════════════════════════════════════════════════════${C_RESET}"
echo -e "${C_BOLD}  HPO LAUNCHER — FINAL REPORT${C_RESET}"
echo    "══════════════════════════════════════════════════════════════"
printf  "  %-24s : %s\n" "Run timestamp"    "${RUN_TIMESTAMP}"
printf  "  %-24s : %s\n" "Total workers"    "${NUM_WORKERS}"
printf  "  %-24s : %s\n" "Succeeded"        "${SUCCEEDED_COUNT}"
printf  "  %-24s : %s\n" "Failed"           "${FAILED_COUNT}"
printf  "  %-24s : %s\n" "Log directory"    "${LOG_DIR}"
printf  "  %-24s : %s\n" "VRAM log"         "${VRAM_LOG}"
echo    "══════════════════════════════════════════════════════════════"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
    echo ""
    _error "Failed GPU workers: ${FAILED_GPUS[*]}"
    _error "Inspect their logs:"
    for fail_gpu in "${FAILED_GPUS[@]}"; do
        echo  "  ${LOG_DIR}/worker_gpu${fail_gpu}_${RUN_TIMESTAMP}.log"
    done
    echo ""
    _info "Completed trials from successful workers are preserved in the Optuna DB."
    _info "You can re-run the same command to launch replacement workers for the failed GPUs:"
    echo  "  ./launch_hpo.sh --gpus $(IFS=','; echo "${FAILED_GPUS[*]}") ${HPO_ARGS[*]}"
    echo ""
    exit 1
fi

echo ""
_ok "All ${NUM_WORKERS} worker(s) completed successfully."
echo ""
_info "Next steps:"
echo  "  1. Apply best params to config:"
echo  "     ${PYTHON_BIN} hpo_tuner.py ${HPO_ARGS[*]} --apply-best"
echo  "  2. View hyperparameter importance:"
echo  "     ${PYTHON_BIN} hpo_tuner.py ${HPO_ARGS[*]} --show-importance"
echo  "  3. Review VRAM usage over time:"
echo  "     cat ${VRAM_LOG}"
echo ""
exit 0
```

---

### `finetune_base_model.py`

```python
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
    Memory-mapped Kline dataset.

    The first call for a given (data_path, data_type) pair builds numpy cache
    files alongside the CSV.  Subsequent calls load them via mmap — zero-copy
    and safe across multiple DataLoader workers and DDP ranks.
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

        self.py_rng      = random.Random(seed)
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

    # ------------------------------------------------------------------
    def _load_or_build_cache(self):
        cache_x     = f"{self.data_path}.{self.data_type}.x.npy"
        cache_stamp = f"{self.data_path}.{self.data_type}.stamp.npy"
        lock_path   = f"{self.data_path}.{self.data_type}.lock"

        # Cross-process file locking to prevent race conditions in HPO/distributed
        from filelock import FileLock
        with FileLock(lock_path, timeout=600):
            # Double-check inside lock - another worker may have built it while we waited
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

            # Atomic writes: write to PID-stamped temp files, then os.replace
            tmp_x     = f"{cache_x}.tmp.{os.getpid()}.npy"
            tmp_stamp = f"{cache_stamp}.tmp.{os.getpid()}.npy"

            np.save(tmp_x, x_arr)
            np.save(tmp_stamp, stamp_arr)

            os.replace(tmp_x, cache_x)
            os.replace(tmp_stamp, cache_stamp)

        # Outside the lock, memory-map the built files
        self.x_data     = np.load(cache_x,     mmap_mode='r')
        self.stamp_data = np.load(cache_stamp, mmap_mode='r')

    # ------------------------------------------------------------------
    def set_epoch_seed(self, epoch: int):
        self.py_rng.seed(self.seed + epoch)
        self.current_epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        max_start = len(self.x_data) - self.window
        
        # PyTorch handles robust index shuffling internally.
        start_idx = idx % (max_start + 1)

        end_idx = start_idx + self.window

        # Copy out of mmap so workers get writable arrays
        x       = self.x_data[start_idx:end_idx].copy()
        x_stamp = self.stamp_data[start_idx:end_idx].copy()

        # ---------------------------------------------------------
        # 🛠️ FIX: Calculate mean and std ONLY on the lookback window
        # ---------------------------------------------------------
        lookback_x = x[:self.lookback_window]
        x_mean = lookback_x.mean(axis=0)
        x_std  = lookback_x.std(axis=0)
    
        # Apply historical stats to the ENTIRE window (history + future)
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
    # FIX: Use drop_last=True to prevent DDP padding from duplicating edge-case
    # samples and artificially skewing the validation loss during HPO.
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False, drop_last=True) if use_ddp else None
    )

    loader_kw = dict(
        num_workers=config.num_workers, 
        pin_memory=True,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle    = (train_sampler is None),
        drop_last  = True,
        sampler    = train_sampler,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = config.batch_size,
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
        max_lr          = config.predictor_learning_rate,
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

            # ── Apply Scheduled Sampling ──
            use_tf = random.random() < tf_prob

            logits = raw_model()(
                s1_ids=token_in[0],
                s2_ids=token_in[1],
                stamp=batch_x_stamp[:, :-1, :],
                use_teacher_forcing=use_tf,
                s1_targets=token_out[0] if use_tf else None
            )

            loss, s1_loss, s2_loss = raw_model().head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            # Scale loss by accumulation steps for accurate mean gradient
            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()

            current_accum_loss += loss.item()

            # ── Optimizer step (once per accumulation cycle) ────────────────
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    raw_model().parameters(),
                    max_norm=max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                avg_loss = current_accum_loss / accumulation_steps
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

        # ── Optuna Pruning Hook ───────────────────────────────────────────────
        if trial is not None:
            import optuna 
            trial.report(avg_val_loss, epoch)
            
            if trial.should_prune():
                prune_msg = f"Trial {trial.number} pruned at epoch {epoch} (val_loss: {avg_val_loss:.6f})"
                logger.info(prune_msg)
                if rank == 0:
                    print(prune_msg)
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
```

---

### `finetune_tokenizer.py`

```python
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
    # FIX: Use drop_last=True to prevent DDP padding from duplicating edge-case
    # samples and artificially skewing the validation loss during HPO.
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=True) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
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
        max_lr=config.tokenizer_learning_rate,
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

            # FIXED: Apply dynamic loss weights from configuration
            weighted_recon = (
                (config.tokenizer_recon_pre_weight * recon_loss_pre) +
                (config.tokenizer_recon_all_weight * recon_loss_all)
            )

            loss = (config.tokenizer_recon_weight * weighted_recon) + (config.tokenizer_bsq_weight * bsq_loss)

            # Scale loss by accumulation steps for accurate mean gradient
            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()

            current_accum_loss += loss.item()

            # ── Optimizer step (once per accumulation cycle) ────────────────
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Grad clip from config
                torch.nn.utils.clip_grad_norm_(
                    (model.module if use_ddp else model).parameters(),
                    max_norm=max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                avg_loss = current_accum_loss / accumulation_steps

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

        # ── Optuna Pruning Hook ───────────────────────────────────────────────
        if trial is not None:
            import optuna 
            trial.report(avg_val_loss, epoch)
            
            if trial.should_prune():
                prune_msg = f"Trial {trial.number} pruned at epoch {epoch} (val_loss: {avg_val_loss:.6f})"
                logger.info(prune_msg)
                if rank == 0:
                    print(prune_msg)
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
    

```

---

### `train_sequential.py`

```python
"""
train_sequential.py

Sequential fine-tuning pipeline for the Nos model.
Runs tokenizer and basemodel training phases in order, with optional
HPO parameter injection via apply_bsq_overrides and apply_dropout_overrides.

Launch (single GPU):
    python train_sequential.py --config config.yaml

Launch (multi-GPU via torchrun):
    torchrun --nproc_per_node=8 train_sequential.py --config config.yaml
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

sys.path.append('../')
from model import Nos, NosTokenizer, NosPredictor

from config_loader import CustomFinetuneConfig
from finetune_tokenizer import train_tokenizer, set_seed, setup_logging as setup_tokenizer_logging
from finetune_base_model import train_model, create_dataloaders, setup_logging as setup_basemodel_logging

# ── HPO override helpers ───────────────────────────────────────────────────────
# Reusing these functions from hpo_tuner.py guarantees that the dropout and BSQ
# parameter injection logic is identical during HPO search and final training.
from hpo_tuner import apply_dropout_overrides, apply_bsq_overrides


class SequentialTrainer:
    """
    Orchestrates sequential fine-tuning of the Nos tokenizer and basemodel.

    Handles device setup, distributed training initialisation, directory
    creation, and delegates to the phase-specific training functions.
    """

    def __init__(self, config_path: str = None):
        self.config = CustomFinetuneConfig(config_path)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                str(self.config.device_id if hasattr(self.config, 'device_id') else 0)
            )
        )
        self.device = self._setup_device()

        self.config.print_config_summary()

    # ── Device / Distributed Setup ─────────────────────────────────────────────

    def _setup_device(self):
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")

        if self.rank == 0:
            print(
                f"Using device: {device} "
                f"(rank={self.rank}, world_size={self.world_size}, "
                f"local_rank={self.local_rank})"
            )
        return device

    def _setup_distributed(self):
        if self.world_size > 1 and torch.cuda.is_available():
            backend = os.environ.get("DIST_BACKEND", "nccl").lower()
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            if self.rank == 0:
                print(
                    f"Distributed training initialised: "
                    f"backend={backend}, world_size={self.world_size}"
                )
        else:
            if self.rank == 0:
                print(
                    "Distributed training not enabled, "
                    "using single GPU/CPU training"
                )

    # ── Directory / Model Existence Helpers ────────────────────────────────────

    def _check_existing_models(self):
        tokenizer_exists = os.path.exists(self.config.tokenizer_best_model_path)
        basemodel_exists = os.path.exists(self.config.basemodel_best_model_path)

        print(f"Tokenizer model exists: {tokenizer_exists}")
        print(f"Basemodel model exists: {basemodel_exists}")

        return tokenizer_exists, basemodel_exists

    def _create_directories(self):
        os.makedirs(self.config.tokenizer_save_path, exist_ok=True)
        os.makedirs(self.config.basemodel_save_path, exist_ok=True)
        print(f"Created directory: {self.config.tokenizer_save_path}")
        print(f"Created directory: {self.config.basemodel_save_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Tokenizer Fine-tuning
    # ══════════════════════════════════════════════════════════════════════════

    def train_tokenizer_phase(self):
        """
        Fine-tunes the NosTokenizer.

        Model is loaded (pretrained or randomly initialised), HPO-derived
        architecture overrides are injected *before* the model is moved to
        the GPU, then standard tokenizer training proceeds.
        """
        print("\n" + "=" * 60)
        print("Starting Tokenizer Fine-tuning Phase")
        print("=" * 60)

        tokenizer_exists, _ = self._check_existing_models()
        if tokenizer_exists and self.config.skip_existing:
            print("Tokenizer model already exists, skipping training")
            return True

        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_tokenizer_logging(self.config.exp_name, log_dir, self.rank)

        set_seed(self.config.seed)

        # ── Model instantiation ────────────────────────────────────────────
        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading pretrained tokenizer...")
            if self.rank == 0:
                print("Loading pretrained tokenizer...")
            tokenizer = NosTokenizer.from_pretrained(
                self.config.pretrained_tokenizer_path
            )
        else:
            if self.rank == 0:
                print(
                    "pre_trained_tokenizer=False, "
                    "randomly initialising Tokenizer architecture"
                )
            import json
            cfg_path = os.path.join(
                self.config.pretrained_tokenizer_path, 'config.json'
            )
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
            )

        # ── HPO parameter injection (must happen BEFORE .to(device)) ──────
        # Applying overrides on CPU avoids device-mismatch errors that can
        # occur when modifying module attributes after GPU placement,
        # particularly inside DDP wrappers.
        tokenizer = apply_bsq_overrides(tokenizer, self.config)
        tokenizer = apply_dropout_overrides(tokenizer, self.config)

        # ── Move to device ─────────────────────────────────────────────────
        tokenizer = tokenizer.to(self.device)

        # ── Diagnostics ───────────────────────────────────────────────────
        model_size = sum(p.numel() for p in tokenizer.parameters())
        logger.info(f"Tokenizer parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Tokenizer parameters: {model_size:,}")

        logger.info("=== Training Configuration ===")
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"Lookback window: {self.config.lookback_window}")
        logger.info(f"Predict window: {self.config.predict_window}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.tokenizer_learning_rate}")
        logger.info(f"Training epochs: {self.config.tokenizer_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Distributed training: False")

        # ── Training ───────────────────────────────────────────────────────
        logger.info("Starting tokenizer fine-tuning training...")
        if self.rank == 0:
            print("Starting tokenizer fine-tuning training...")

        start_time = time.time()
        best_val_loss = train_tokenizer(
            tokenizer,
            self.device,
            self.config,
            self.config.tokenizer_save_path,
            logger,
        )
        training_time = time.time() - start_time

        final_msg = (
            f"Tokenizer training completed! "
            f"Best validation loss: {best_val_loss:.4f}\n"
            f"Training time: {training_time / 60:.2f} minutes\n"
            f"Model saved to: {self.config.tokenizer_save_path}"
        )
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")

        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Basemodel (Predictor) Fine-tuning
    # ══════════════════════════════════════════════════════════════════════════

    def train_basemodel_phase(self):
        """
        Fine-tunes the Nos predictor with a (optionally frozen) tokenizer.

        Both the tokenizer and predictor receive HPO-derived overrides before
        being moved to the GPU.  The tokenizer receives both BSQ and dropout
        overrides; the predictor receives only dropout overrides (BSQ lives
        exclusively in the tokenizer architecture).
        """
        print("\n" + "=" * 60)
        print("Starting Basemodel Fine-tuning Phase")
        print("=" * 60)

        if getattr(self.config, 'pre_trained_tokenizer', True):
            if not os.path.exists(self.config.finetuned_tokenizer_path):
                raise FileNotFoundError(
                    f"Fine-tuned tokenizer does not exist: "
                    f"{self.config.finetuned_tokenizer_path}"
                )

        _, basemodel_exists = self._check_existing_models()
        if basemodel_exists and self.config.skip_existing:
            print("Basemodel model already exists, skipping training")
            return True

        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_basemodel_logging(self.config.exp_name, log_dir, self.rank)

        set_seed(self.config.seed)

        # ── Tokenizer instantiation ────────────────────────────────────────
        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading fine-tuned tokenizer...")
            if self.rank == 0:
                print("Loading fine-tuned tokenizer...")
            tokenizer = NosTokenizer.from_pretrained(
                self.config.finetuned_tokenizer_path
            )
        else:
            if self.rank == 0:
                print(
                    "pre_trained_tokenizer=False, "
                    "randomly initialising Tokenizer architecture "
                    "for Predictor training"
                )
            import json
            cfg_path = os.path.join(
                self.config.pretrained_tokenizer_path, 'config.json'
            )
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
            )

        # ── HPO injection into the frozen tokenizer (CPU, before .to()) ───
        tokenizer = apply_bsq_overrides(tokenizer, self.config)
        tokenizer = apply_dropout_overrides(tokenizer, self.config)

        # ── Move tokenizer to device ───────────────────────────────────────
        tokenizer = tokenizer.to(self.device)

        # ── Predictor instantiation ────────────────────────────────────────
        if getattr(self.config, 'pre_trained_predictor', True):
            logger.info("Loading pretrained predictor...")
            if self.rank == 0:
                print("Loading pretrained predictor...")
            model = Nos.from_pretrained(self.config.pretrained_predictor_path)
        else:
            if self.rank == 0:
                print(
                    "pre_trained_predictor=False, "
                    "randomly initialising Predictor architecture"
                )
            import json
            cfg_path = os.path.join(
                self.config.pretrained_predictor_path, 'config.json'
            )
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            print("model_config: ", arch)
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
            )

        # ── HPO dropout injection into predictor (CPU, before .to()) ──────
        # BSQ overrides are intentionally omitted here — the BSQ quantizer
        # lives inside the tokenizer, not the predictor.
        model = apply_dropout_overrides(model, self.config)

        # ── Move predictor to device ───────────────────────────────────────
        model = model.to(self.device)

        # ── Diagnostics ───────────────────────────────────────────────────
        model_size = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Model parameters: {model_size:,}")

        logger.info("=== Training Configuration ===")
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"Lookback window: {self.config.lookback_window}")
        logger.info(f"Predict window: {self.config.predict_window}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.predictor_learning_rate}")
        logger.info(f"Training epochs: {self.config.basemodel_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Tokenizer path: {self.config.finetuned_tokenizer_path}")
        logger.info(f"Pretrained model path: {self.config.pretrained_predictor_path}")

        # ── Training ───────────────────────────────────────────────────────
        logger.info("Starting fine-tuning training...")
        if self.rank == 0:
            print("Starting fine-tuning training...")

        start_time = time.time()
        best_val_loss = train_model(
            model,
            tokenizer,
            self.device,
            self.config,
            self.config.basemodel_save_path,
            logger,
        )
        training_time = time.time() - start_time

        final_msg = (
            f"Basemodel training completed! "
            f"Best validation loss: {best_val_loss:.4f}\n"
            f"Training time: {training_time / 60:.2f} minutes\n"
            f"Model saved to: {self.config.basemodel_save_path}"
        )
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")

        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Top-level Orchestration
    # ══════════════════════════════════════════════════════════════════════════

    def run_training(self):
        if self.rank == 0:
            print("Starting Nos model sequential fine-tuning training")
            print(f"Experiment name: {self.config.experiment_name}")
            print(f"Experiment description: {self.config.experiment_description}")

        self._setup_distributed()
        self._create_directories()

        total_start_time = time.time()

        try:
            if self.config.train_tokenizer:
                success = self.train_tokenizer_phase()
                if not success:
                    print("Tokenizer training failed, terminating training")
                    return False
            else:
                print("Skipping Tokenizer training phase")

            if self.config.train_basemodel:
                success = self.train_basemodel_phase()
                if not success:
                    print("Basemodel training failed, terminating training")
                    return False
            else:
                print("Skipping Basemodel training phase")

            total_time = time.time() - total_start_time

            if self.rank == 0:
                print("\n" + "=" * 60)
                print("Training completed!")
                print("=" * 60)
                print(f"Total training time: {total_time / 60:.2f} minutes")
                print(f"Tokenizer model: {self.config.tokenizer_best_model_path}")
                print(f"Basemodel model: {self.config.basemodel_best_model_path}")
                print("=" * 60)

            return True

        except Exception as e:
            if self.rank == 0:
                print(f"Error occurred during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Nos Model Sequential Fine-tuning Training'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path (default: config.yaml)',
    )
    parser.add_argument(
        '--skip-tokenizer',
        action='store_true',
        help='Skip tokenizer training phase',
    )
    parser.add_argument(
        '--skip-basemodel',
        action='store_true',
        help='Skip basemodel training phase',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip training for existing models',
    )

    args = parser.parse_args()

    trainer = SequentialTrainer(args.config)

    if args.skip_tokenizer:
        trainer.config.train_tokenizer = False
    if args.skip_basemodel:
        trainer.config.train_basemodel = False
    if args.skip_existing:
        trainer.config.skip_existing = True

    success = trainer.run_training()

    if success:
        print("Training completed successfully!")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)
    else:
        print("Training failed!")
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

### `requirements.txt`

```text
# Core Deep Learning & Math
torch>=2.0.0
numpy
pandas
einops

# Model Hub & Progress Tracking
huggingface-hub
tqdm

# Configuration Parsing
PyYAML

# Hyperparameter Optimization (HPO)
optuna
plotly
SQLAlchemy
```

---

### `CLAUDE.md`

```markdown
# CLAUDE.md — Project Context for Claude Code

## Project Overview

This is **uNOS** (micro Neural Operational System), a financial time-series prediction system built on a custom transformer architecture. The project finetunes pretrained NOS models on OHLCV (Open-High-Low-Close-Volume) market data for next-step price prediction.

**Project Type**: Python ML/Deep Learning project  
**Domain**: Financial time-series forecasting  
**Primary Use**: Training and running inference on custom finetuned models for cryptocurrency/stock price prediction

---

## Architecture Summary

### The Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE FLOW                           │
│                                                                     │
│  Raw CSV Data (OHLCV time series)                                   │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  PHASE 1: Tokenizer Finetuning                           │       │
│  │  (NosTokenizer)                                          │       │
│  │  Input: Raw OHLCV → learns domain-specific codes        │       │
│  │  Loss: MSE reconstruction + BSQ entropy                 │       │
│  │  Output: Finetuned tokenizer → finetuned/tokenizer/      │       │
│  └─────────────────────────────────────────────────────────┘       │
│      │                                                              │
│      ▼  (tokenizer frozen, used for encoding)                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  PHASE 2: Basemodel (Predictor) Finetuning              │       │
│  │  (Nos / NosPredictor)                                   │       │
│  │  Input: Tokenized sequences → learns temporal patterns  │       │
│  │  Loss: CrossEntropy(s1_tokens) + CrossEntropy(s2_tokens) │       │
│  │  Output: Finetuned predictor → finetuned/basemodel/      │       │
│  └─────────────────────────────────────────────────────────┘       │
│      │                                                              │
│      ▼                                                              │
│  Inference: predictor.predict(df, timestamps, pred_len)            │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | File | Purpose |
|-----------|------|---------|
| **NosTokenizer** | `model/nos.py:13-179` | Encodes OHLCV into discrete tokens using BSQ |
| **Nos** | `model/nos.py:181-336` | Main transformer with hierarchical embedding |
| **NosPredictor** | `model/nos.py:463-633` | High-level inference wrapper |
| **BSQuantizer** | `model/module.py` | Binary Spherical Quantization |
| **HierarchicalEmbedding** | `model/module.py` | S1+S2 token embeddings |
| **DependencyAwareLayer** | `model/module.py` | S2 conditioned on S1 |

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_sequential.py` | Main training orchestrator — runs tokenizer then basemodel |
| `hpo_tuner.py` | Optuna-based hyperparameter optimization |
| `config_loader.py` | YAML config parsing with defaults |
| `finetune_tokenizer.py` | Tokenizer-specific training loop |
| `finetune_base_model.py` | Basemodel-specific training loop |
| `codebase.py` | Generates CODE.md documentation |
| `train.md` | Comprehensive training guide (281KB+) |

---

## Configuration

### Default Config Location
`configs/config.yaml`

### Key Config Sections

```yaml
data:
  data_path: "data/1h.csv"        # CSV with timestamps, OHLCV columns
  lookback_window: 512            # Historical context length
  predict_window: 48              # Prediction horizon
  clip: 5.0                       # Normalization clipping

training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
  batch_size: 32
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001   # Keep very low to avoid forgetting

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"

hpo:
  enabled: false                  # Set true to activate Optuna
  n_trials: 30
  storage: null                   # Use null for in-memory (dev), SQLite for persistence
```

### Required CSV Format
```csv
timestamps,open,high,low,close,volume,amount
2024-01-01 00:00:00,45000.0,45100.0,44900.0,45050.0,100.0,4500000.0
...
```

---

## Common Commands

### Quick Start (Default Config)
```bash
# Train both tokenizer and basemodel
python train_sequential.py --config configs/config.yaml

# Train only tokenizer (Phase 1)
python train_sequential.py --config configs/config.yaml --skip-basemodel

# Train only basemodel (Phase 2, requires finetuned tokenizer)
python train_sequential.py --config configs/config.yaml --skip-tokenizer

# Skip models that already exist
python train_sequential.py --config configs/config.yaml --skip-existing
```

### Hyperparameter Optimization
```bash
# Run HPO for tokenizer only
python hpo_tuner.py --config configs/config.yaml --phase tokenizer --n-trials 40

# Run HPO for basemodel only (requires finetuned tokenizer)
python hpo_tuner.py --config configs/config.yaml --phase basemodel --tokenizer-path finetuned/test_1h/tokenizer/best_model --n-trials 40

# Apply best HPO params to new config
python hpo_tuner.py --config configs/config.yaml --phase tokenizer --apply-best --output-config configs/config_best.yaml
```

### Inference Example
```python
import pandas as pd
from model import NosPredictor, Nos, NosTokenizer
from transformers import AutoModel

# Load finetuned models
tokenizer = NosTokenizer.from_pretrained("finetuned/test_1h/tokenizer/best_model")
model = Nos.from_pretrained("finetuned/test_1h/basemodel/best_model")

# Create predictor
predictor = NosPredictor(model, tokenizer, device="cuda:0", max_context=512, clip=5.0)

# Load data
df = pd.read_csv("data/1h.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])
df = df.sort_values('timestamps').reset_index(drop=True)

# Prepare timestamps
x_timestamp = df['timestamps'].iloc[-512:]
y_timestamp = pd.date_range(start=x_timestamp.iloc[-1], periods=49, freq='h')[1:]

# Predict
preds = predictor.predict(
    df.iloc[-512:],
    x_timestamp,
    y_timestamp,
    pred_len=48,
    T=1.0,           # temperature
    top_k=0,
    top_p=0.9,
    sample_count=5   # number of samples to average
)
print(preds)
```

---

## Training Tips & Pitfalls

### Critical Rules

1. **Predictor LR must be very low** (1e-6 to 5e-6)
   - Pretrained model is valuable — too high causes catastrophic forgetting
   - If val loss increases sharply during training, reduce LR 10x

2. **Always use pretrained models** (`pre_trained: true`)
   - Never train from scratch — the pretrained tokenizerlearns domain-agnostic representations
   - Path: `models/nos_tokenizer_2k` and `models/nos_mini`

3. **Tokenize first, then predict** (never skip tokenizer)
   - Tokenizer quality directly limits predictor ceiling
   - Run tokenizer quality check after Phase 1:
   ```python
   # Reconstruction MSE should be < 0.05
   ```

4. **Data validation before training**
   - Check for gaps in time series (>2h for 1h data)
   - Verify no zero/negative prices
   - Ensure sufficient validation samples (>200 usable windows)

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| CUDA OOM | Batch too large | Reduce `batch_size` or increase `accumulation_steps` |
| Val loss flat | LR too low | Increase `tokenizer_learning_rate` to 5e-4 |
| Val loss spikes | Catastrophic forgetting | Reduce `predictor_learning_rate` to 1e-7 |
| Codebook collapse | BSQ params wrong | Increase `bsq_gamma` to 1.5 |
| HPO all pruned | Too few epochs | Increase `hpo_tokenizer_epochs` to 15 |

---

## Development Patterns

### Adding New Data Sources
1. Ensure CSV has: `timestamps`, `open`, `high`, `low`, `close`, `volume`, `amount`
2. Update `data.data_path` in config
3. Validate with data quality check script (see train.md Part 1)

### Modifying Model Architecture
- Tokenizer changes: Edit `model/nos.py` class `NosTokenizer`
- Predictor changes: Edit `model/nos.py` class `Nos`
- Core modules: Edit `model/module.py`

### Adding New Training Scenarios
1. Copy `configs/config.yaml` to new file
2. Modify `model_paths.exp_name` for isolation
3. Update hyperparameters as needed

---

## File Structure

```
uNOS/
├── model/
│   ├── __init__.py      # Exports: NosTokenizer, Nos, NosPredictor
│   ├── nos.py           # Core model classes (634 lines)
│   └── module.py        # Building blocks: TransformerBlock, BSQuantizer, etc.
├── models/              # Pretrained weights
│   ├── nos_tokenizer_2k/
│   ├── nos_tokenizer_base/
│   ├── nos_tokenizer_2k/
│   ├── nos_mini/
│   └── nos_base/
├── configs/             # YAML configs
│   ├── config.yaml     # Default config
│   └── ...
├── data/                # Training data (user-provided CSVs)
├── finetuned/           # Output: trained models
│   └── {exp_name}/
│       ├── tokenizer/best_model/
│       └── basemodel/best_model/
├── train_sequential.py  # Main training entry point
├── hpo_tuner.py        # HPO with Optuna
├── config_loader.py    # Config parsing
├── finetune_tokenizer.py
├── finetune_base_model.py
├── train.md            # Comprehensive guide (281KB)
├── CODE.md             # Auto-generated code docs
└── CLAUDE.md           # This file
```

---

## Dependencies

```
torch
pandas
numpy
pyyaml
huggingface_hub
optuna
plotly
kaleido
tqdm
```

Install: `pip install torch pandas numpy pyyaml huggingface_hub optuna plotly kaleido tqdm`

---

## User Preferences

- **Working Directory**: `c:\Users\kashy\OneDrive\Documents\uNOS`
- **Git User**: Mayank Kashyap
- **Platform**: Windows 11

---

## Notes

- The `train.md` file contains an extremely detailed 950+ line guide covering every aspect of the pipeline
- Use `python codebase.py` to regenerate CODE.md if files change
- HPO results persist in SQLite (when storage is configured) — can resume interrupted runs
- Batch inference supported via `predictor.predict_batch()` for parallel prediction
```

---
