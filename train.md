# Deep Research: Maximum Accuracy Pipeline Guide

## Understanding the System Architecture First

Before commands, you must understand **what each component does** and **why order matters**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE FLOW                           │
│                                                                     │
│  Raw CSV Data                                                       │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  PHASE 1: NosTokenizer Finetuning                       │       │
│  │                                                         │       │
│  │  Input: OHLCV time series                               │       │
│  │  Goal:  Learn domain-specific binary codes              │       │
│  │  Loss:  MSE(reconstruction) + BSQ entropy loss          │       │
│  │  Output: Finetuned tokenizer (best_model/)              │       │
│  └─────────────────────────────────────────────────────────┘       │
│      │                                                              │
│      ▼  (tokenizer is FROZEN here)                                 │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  PHASE 2: Nos Predictor (Basemodel) Finetuning          │       │
│  │                                                         │       │
│  │  Input: Token sequences from finetuned tokenizer        │       │
│  │  Goal:  Learn temporal prediction patterns              │       │
│  │  Loss:  CrossEntropy(s1_tokens) + CrossEntropy(s2_tokens│       │
│  │  Output: Finetuned predictor (best_model/)              │       │
│  └─────────────────────────────────────────────────────────┘       │
│      │                                                              │
│      ▼                                                              │
│  Inference: NosPredictor.predict()                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Complete Strategy Decision Tree

```
START HERE
    │
    ▼
Do you have a finetuned tokenizer already?
    │
    ├── NO ──► Run Phase 1 (Tokenizer) first
    │              │
    │              ▼
    │          Do you want to find optimal hyperparameters?
    │              ├── YES ──► HPO tokenizer first (5-10 epochs/trial)
    │              └── NO  ──► Direct training with defaults
    │
    └── YES ──► Skip to Phase 2 (Basemodel)
                   │
                   ▼
               Do you want to find optimal hyperparameters?
                   ├── YES ──► HPO basemodel (3-5 epochs/trial)
                   └── NO  ──► Direct training with defaults
```

---

## Part 2: Step-by-Step Maximum Accuracy Pipeline

### STEP 0: Environment Setup & Data Validation

```bash
# Install all dependencies
pip install optuna plotly kaleido torch pandas numpy pyyaml

# Verify your data quality first (run this Python snippet)
python -c "
import pandas as pd
import numpy as np

df = pd.read_csv('data/1h.csv')
df['timestamps'] = pd.to_datetime(df['timestamps'])
df = df.sort_values('timestamps')

print('=== DATA QUALITY REPORT ===')
print(f'Total rows: {len(df)}')
print(f'Date range: {df.timestamps.min()} to {df.timestamps.max()}')
print(f'Missing values:\n{df.isnull().sum()}')

# Check for gaps in hourly data
df['time_diff'] = df['timestamps'].diff().dt.total_seconds() / 3600
gaps = df[df['time_diff'] > 2]
print(f'Data gaps (>2h): {len(gaps)}')
if len(gaps) > 0:
    print(gaps[['timestamps','time_diff']].head(10))

# Check for zero/negative prices
for col in ['open','high','low','close']:
    bad = df[df[col] <= 0]
    print(f'{col} <= 0: {len(bad)} rows')

# Recommended split sizes
total = len(df)
train_n = int(total * 0.9)
val_n = total - train_n
print(f'\nTrain samples: {train_n}')
print(f'Val samples: {val_n}')
print(f'With lookback=512+predict=48=560 window:')
print(f'  Train usable: {max(0, train_n - 560 + 1)}')
print(f'  Val usable: {max(0, val_n - 560 + 1)}')
"
```

**When to use**: Always run this before anything else. If val usable samples < 200, reduce `lookback_window`.

---

### STEP 1: Configure Your YAML Properly

Create `configs/config_test_1h_full.yaml`:

```yaml
# configs/config_test_1h_full.yaml
# OPTIMIZED FOR MAXIMUM ACCURACY

data:
  data_path: "data/1h.csv"
  
  # CRITICAL: lookback_window should be 4-8x predict_window
  # For 1h data predicting 48h ahead: 512 is good (10.6 days context)
  lookback_window: 512
  predict_window: 48
  max_context: 512
  
  # clip: 5.0 works for most financial data
  # Increase to 8.0 if your data has extreme spikes
  # Decrease to 3.0 if data is stable
  clip: 5.0
  
  # 90/10 split: use more train data when dataset < 50k rows
  # Use 80/20 when dataset > 100k rows for better validation signal
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

training:
  # TOKENIZER: Higher LR is safe, tokenizer is more robust
  tokenizer_epochs: 50        # More epochs = better reconstruction
  
  # BASEMODEL: Fewer epochs prevents catastrophic forgetting
  basemodel_epochs: 30
  
  # batch_size: 32 for <8GB VRAM, 64 for 16GB+, 128 for 24GB+
  batch_size: 32
  
  log_interval: 20            # More frequent logging for monitoring
  num_workers: 4              # Set to CPU cores - 2
  seed: 42

  # TOKENIZER LR: 1e-4 to 3e-4 is the sweet spot
  tokenizer_learning_rate: 0.0002
  
  # PREDICTOR LR: KEEP VERY LOW to prevent catastrophic forgetting
  # Pretrained model is valuable — don't destroy it
  predictor_learning_rate: 0.000005   # 5e-6: safer than 1e-6 for speed

  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1

  accumulation_steps: 1

  # Scheduler params (now configurable)
  tokenizer_pct_start: 0.05    # 5% warmup
  tokenizer_div_factor: 10.0
  basemodel_pct_start: 0.05
  basemodel_div_factor: 10.0

  # Gradient clipping
  tokenizer_max_grad_norm: 2.0
  basemodel_max_grad_norm: 1.0   # Lower for predictor = safer

  # Dropout overrides (null = use pretrained values)
  # Set during finetuning if you see overfitting
  ffn_dropout_p: null
  attn_dropout_p: null
  resid_dropout_p: null
  token_dropout_p: null

  # BSQ overrides (null = use pretrained values)
  bsq_beta: null
  bsq_gamma0: null
  bsq_gamma: null
  bsq_zeta: null

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"
  
  exp_name: "run_1h_v1"
  base_path: "finetuned"

  base_save_path: ""
  finetuned_tokenizer: ""

  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "Nos_1h_finetune"
  description: "1h OHLCV finetuning for maximum accuracy"
  use_comet: false

  train_tokenizer: true
  train_basemodel: true
  skip_existing: false

  pre_trained: true    # ALWAYS true — never train from scratch

device:
  use_cuda: true
  device_id: 0

# HPO Configuration
hpo:
  enabled: true
  n_trials: 40           # 40 trials finds good params reliably
  direction: "minimize"
  sampler: "tpe"         # TPE is best for < 100 trials
  pruner: "median"       # Prune bad trials early
  
  # Use SQLite to persist results (can resume if interrupted)
  storage: "sqlite:///finetuned/run_1h_v1/hpo_results.db"
  study_name: "nos_1h_hpo_v1"

  optimize_tokenizer: true
  optimize_basemodel: true

  # IMPORTANT: Use small epochs for HPO speed
  # Rule: hpo_epochs = max(3, total_epochs // 6)
  hpo_tokenizer_epochs: 8    # Fast proxy evaluation
  hpo_basemodel_epochs: 4

  search_space:
    # ── TOKENIZER ──────────────────────────────────────────────
    tokenizer_learning_rate:
      type: float
      low: 5.0e-5
      high: 8.0e-4
      log: true

    tokenizer_pct_start:
      type: float
      low: 0.02
      high: 0.10
      log: false

    tokenizer_div_factor:
      type: categorical
      choices: [5.0, 10.0, 25.0]

    tokenizer_max_grad_norm:
      type: float
      low: 0.5
      high: 4.0
      log: false

    bsq_beta:
      type: float
      low: 0.01
      high: 0.15
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
      high: 0.15
      log: true

    # ── BASEMODEL ──────────────────────────────────────────────
    predictor_learning_rate:
      type: float
      low: 5.0e-7
      high: 2.0e-5
      log: true

    basemodel_pct_start:
      type: float
      low: 0.02
      high: 0.10
      log: false

    basemodel_div_factor:
      type: categorical
      choices: [5.0, 10.0, 25.0]

    basemodel_max_grad_norm:
      type: float
      low: 0.3
      high: 3.0
      log: false

    ffn_dropout_p:
      type: float
      low: 0.0
      high: 0.35
      log: false

    resid_dropout_p:
      type: float
      low: 0.0
      high: 0.25
      log: false

    token_dropout_p:
      type: float
      low: 0.0
      high: 0.15
      log: false

    # ── SHARED ─────────────────────────────────────────────────
    adam_weight_decay:
      type: float
      low: 0.005
      high: 0.3
      log: true

    adam_beta1:
      type: float
      low: 0.87
      high: 0.95
      log: false

    adam_beta2:
      type: float
      low: 0.92
      high: 0.999
      log: false

    batch_size:
      type: categorical
      choices: [16, 32, 64]

    accumulation_steps:
      type: categorical
      choices: [1, 2, 4]

    clip:
      type: float
      low: 3.0
      high: 8.0
      log: false
```

---

## Part 3: All Commands with Full Explanation

### SCENARIO A: First Time (No Finetuned Models Exist)

This is the complete workflow from scratch to maximum accuracy:

```bash
# ════════════════════════════════════════════════════════════
# STAGE 1: HPO for Tokenizer
# ════════════════════════════════════════════════════════════
# WHEN: Always run this first — tokenizer quality determines
#       the ceiling of predictor accuracy
# WHY:  Finding optimal tokenizer LR and BSQ weights is critical
#       A bad tokenizer = bad token quality = predictor can never be good
# TIME: ~2-4 hours for 40 trials on GPU

python hpo_tuner.py \
  --config configs/config_test_1h_full.yaml \
  --phase tokenizer \
  --n-trials 40

# What this does:
# 1. Runs 40 short trials (8 epochs each = fast proxy)
# 2. Each trial: loads pretrained tokenizer → applies new HPs → trains → measures val MSE
# 3. TPE sampler learns which params matter most
# 4. Median pruner kills bad trials after epoch 2
# 5. Saves all results to SQLite DB (resumable)
# 6. Reports best params at the end
```

```bash
# ════════════════════════════════════════════════════════════
# STAGE 2: Full Tokenizer Training with Best Params
# ════════════════════════════════════════════════════════════
# WHEN: After HPO tokenizer completes
# WHY:  HPO used only 8 epochs — now train FULL 50 epochs
#       with the best hyperparameters found
# TIME: ~30-60 min for 50 epochs on GPU

# First, apply best params to a new config file
python hpo_tuner.py \
  --config configs/config_test_1h_full.yaml \
  --phase tokenizer \
  --apply-best \
  --output-config configs/config_1h_best_tok.yaml

# Verify the output config has best params applied
cat configs/config_1h_best_tok.yaml

# Now train full tokenizer with best hyperparameters
# (skip basemodel for now)
python train_sequential.py \
  --config configs/config_1h_best_tok.yaml \
  --skip-basemodel

# Output: finetuned/run_1h_v1/tokenizer/best_model/
```

```bash
# ════════════════════════════════════════════════════════════
# STAGE 3: Verify Tokenizer Quality
# ════════════════════════════════════════════════════════════
# WHEN: After tokenizer training completes
# WHY:  Confirm tokenizer learned your data domain properly
#       before investing time in predictor training

python -c "
import torch
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from model import NosTokenizer

# Load finetuned tokenizer
tok = NosTokenizer.from_pretrained('finetuned/run_1h_v1/tokenizer/best_model')
tok.eval()

# Load a sample of validation data
df = pd.read_csv('data/1h.csv')
df = df.sort_values('timestamps').reset_index(drop=True)
val_start = int(len(df) * 0.9)
val_df = df.iloc[val_start:val_start+600]  # 600 rows

feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
x = val_df[feature_cols].values.astype(np.float32)

# Normalize same way as training
x_mean, x_std = x.mean(axis=0), x.std(axis=0)
x_norm = (x - x_mean) / (x_std + 1e-5)
x_norm = np.clip(x_norm, -5.0, 5.0)

# Test reconstruction quality
x_tensor = torch.from_numpy(x_norm).unsqueeze(0)  # (1, T, 6)

with torch.no_grad():
    (z_pre, z_full), bsq_loss, quantized, indices = tok(x_tensor)
    
    recon_mse = ((z_full - x_tensor) ** 2).mean().item()
    recon_mse_pre = ((z_pre - x_tensor) ** 2).mean().item()

print(f'Reconstruction MSE (full): {recon_mse:.6f}')
print(f'Reconstruction MSE (pre):  {recon_mse_pre:.6f}')
print(f'BSQ Loss: {bsq_loss.item():.6f}')
print(f'Unique tokens used: {indices[0].unique().numel()} / {2**10}')

# Good tokenizer: recon_mse < 0.05
# Bad tokenizer: recon_mse > 0.2 (retrain with different HPs)
if recon_mse < 0.05:
    print('✅ Tokenizer quality: GOOD')
elif recon_mse < 0.15:
    print('⚠️  Tokenizer quality: ACCEPTABLE (consider more epochs)')
else:
    print('❌ Tokenizer quality: POOR (retrain with different HPs)')
"
```

```bash
# ════════════════════════════════════════════════════════════
# STAGE 4: HPO for Basemodel (Predictor)
# ════════════════════════════════════════════════════════════
# WHEN: After tokenizer is verified as good quality
# WHY:  Find optimal predictor LR and regularization
#       that prevents catastrophic forgetting while adapting
# TIME: ~1-3 hours for 40 trials on GPU

python hpo_tuner.py \
  --config configs/config_1h_best_tok.yaml \
  --phase basemodel \
  --tokenizer-path finetuned/run_1h_v1/tokenizer/best_model \
  --n-trials 40

# What this does:
# 1. Loads your FINETUNED tokenizer (frozen, not trained)
# 2. Loads PRETRAINED predictor (nos_mini)
# 3. Tries different LRs, dropouts, grad norms
# 4. Runs 4 epochs per trial (fast proxy)
# 5. Reports best predictor hyperparameters
```

```bash
# ════════════════════════════════════════════════════════════
# STAGE 5: Apply Best Basemodel Params & Train Full
# ════════════════════════════════════════════════════════════
# WHEN: After basemodel HPO completes
# WHY:  Now train FULL 30 epochs with best found params

# Apply both tokenizer and basemodel best params
python hpo_tuner.py \
  --config configs/config_1h_best_tok.yaml \
  --phase basemodel \
  --tokenizer-path finetuned/run_1h_v1/tokenizer/best_model \
  --apply-best \
  --output-config configs/config_1h_final.yaml

# Train ONLY basemodel (tokenizer already done)
python train_sequential.py \
  --config configs/config_1h_final.yaml \
  --skip-tokenizer

# Output: finetuned/run_1h_v1/basemodel/best_model/
```

---

### SCENARIO B: One-Shot Full Pipeline (Simplest Path)

```bash
# ════════════════════════════════════════════════════════════
# WHEN: You want maximum automation with one command
# WHY:  Let the system find everything automatically
# TIME: ~4-8 hours total on GPU
# TRADEOFF: Less control, but finds good params automatically
# ════════════════════════════════════════════════════════════

python hpo_tuner.py \
  --config configs/config_test_1h_full.yaml \
  --phase both \
  --apply-best \
  --output-config configs/config_1h_hpo_best.yaml

# After HPO completes, run FULL training with best params
# The HPO uses short epochs — now use full epoch counts
python train_sequential.py --config configs/config_1h_hpo_best.yaml

# Monitor training in real-time
tail -f finetuned/run_1h_v1/logs/tokenizer_training_rank_0.log
tail -f finetuned/run_1h_v1/logs/basemodel_training_rank_0.log
```

---

### SCENARIO C: You Already Have a Finetuned Tokenizer

```bash
# ════════════════════════════════════════════════════════════
# WHEN: Tokenizer trained previously, only need to tune predictor
# WHY:  Save time by reusing existing tokenizer work
# ════════════════════════════════════════════════════════════

# Verify the tokenizer exists
ls finetuned/run_1h_v1/tokenizer/best_model/

# Run ONLY basemodel HPO
python hpo_tuner.py \
  --config configs/config_test_1h_full.yaml \
  --phase basemodel \
  --tokenizer-path finetuned/run_1h_v1/tokenizer/best_model \
  --n-trials 40 \
  --apply-best \
  --output-config configs/config_1h_base_best.yaml

# Train full basemodel with best params
python train_sequential.py \
  --config configs/config_1h_base_best.yaml \
  --skip-tokenizer
```

---

### SCENARIO D: Resume Interrupted HPO

```bash
# ════════════════════════════════════════════════════════════
# WHEN: HPO was interrupted (crash, timeout, etc.)
# WHY:  SQLite storage saves all completed trials
#       Optuna automatically resumes from where it stopped
# ════════════════════════════════════════════════════════════

# Just run the same command again — it will continue automatically
python hpo_tuner.py \
  --config configs/config_test_1h_full.yaml \
  --phase tokenizer \
  --n-trials 40

# To check current HPO progress before resuming
python -c "
import optuna
storage = 'sqlite:///finetuned/run_1h_v1/hpo_results.db'
study = optuna.load_study(
    study_name='nos_1h_hpo_v1_tokenizer', 
    storage=storage
)
completed = [t for t in study.trials if t.value is not None]
print(f'Completed trials: {len(completed)}')
print(f'Best val loss so far: {study.best_value:.6f}')
print(f'Best params so far: {study.best_params}')
"
```

---

### SCENARIO E: Quick Training Without HPO (Baseline)

```bash
# ════════════════════════════════════════════════════════════
# WHEN: You want a quick baseline or have limited compute
# WHY:  HPO takes hours — sometimes default params are good enough
#       Use this to establish a baseline first
# ════════════════════════════════════════════════════════════

# Train full pipeline with default params
python train_sequential.py \
  --config configs/config_test_1h.yaml

# OR train only tokenizer first
python train_sequential.py \
  --config configs/config_test_1h.yaml \
  --skip-basemodel

# OR train only basemodel (if tokenizer already done)
python train_sequential.py \
  --config configs/config_test_1h.yaml \
  --skip-tokenizer

# OR skip existing models (only train what's missing)
python train_sequential.py \
  --config configs/config_test_1h.yaml \
  --skip-existing
```

---

### SCENARIO F: Iterative Refinement (Best Results Strategy)

```bash
# ════════════════════════════════════════════════════════════
# WHEN: You have time and want absolute maximum accuracy
# WHY:  Each round builds on the previous best
# ════════════════════════════════════════════════════════════

# ROUND 1: Quick HPO with fewer trials to get ballpark
python hpo_tuner.py \
  --config configs/config_test_1h_full.yaml \
  --phase tokenizer \
  --n-trials 20 \
  --apply-best \
  --output-config configs/config_round1.yaml

# Train with round 1 best
python train_sequential.py \
  --config configs/config_round1.yaml \
  --skip-basemodel

# ROUND 2: Narrow search space around round 1 best
# Edit configs/config_round2.yaml to use tighter search ranges
# based on what round 1 found, then run more trials
python hpo_tuner.py \
  --config configs/config_round2.yaml \
  --phase tokenizer \
  --n-trials 20 \
  --apply-best \
  --output-config configs/config_round2_best.yaml
```

---

## Part 4: Monitoring & Debugging Commands

### Real-time Training Monitoring

```bash
# Watch tokenizer training loss in real-time
tail -f finetuned/run_1h_v1/logs/tokenizer_training_rank_0.log

# Watch basemodel training loss in real-time  
tail -f finetuned/run_1h_v1/logs/basemodel_training_rank_0.log

# Watch both simultaneously (requires tmux or two terminals)
# Terminal 1:
tail -f finetuned/run_1h_v1/logs/tokenizer_training_rank_0.log

# Terminal 2:
tail -f finetuned/run_1h_v1/logs/basemodel_training_rank_0.log
```

### Check GPU Usage

```bash
# Monitor GPU memory and utilization
watch -n 1 nvidia-smi

# Python GPU check
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM Total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'VRAM Used: {torch.cuda.memory_allocated(0)/1e9:.3f} GB')
    print(f'VRAM Reserved: {torch.cuda.memory_reserved(0)/1e9:.3f} GB')
else:
    print('No GPU available — using CPU (will be slow)')
"
```

### Analyze HPO Results

```bash
# Full HPO analysis report
python -c "
import optuna
import json

storage = 'sqlite:///finetuned/run_1h_v1/hpo_results.db'

for phase in ['tokenizer', 'basemodel']:
    study_name = f'nos_1h_hpo_v1_{phase}'
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials if t.value is not None]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        print(f'\n=== {phase.upper()} HPO Results ===')
        print(f'Total trials: {len(study.trials)}')
        print(f'Completed: {len(completed)}')
        print(f'Pruned (saved time): {len(pruned)}')
        print(f'Best val loss: {study.best_value:.6f}')
        print(f'\nBest hyperparameters:')
        for k, v in sorted(study.best_params.items()):
            if isinstance(v, float):
                print(f'  {k}: {v:.6f}')
            else:
                print(f'  {k}: {v}')
                
        # Show top 5 trials
        sorted_trials = sorted(completed, key=lambda t: t.value)
        print(f'\nTop 5 trials:')
        for i, t in enumerate(sorted_trials[:5]):
            print(f'  #{i+1}: val_loss={t.value:.6f} (trial {t.number})')
    except Exception as e:
        print(f'{phase}: No results yet ({e})')
"

# Generate HTML visualization plots
python -c "
import optuna

storage = 'sqlite:///finetuned/run_1h_v1/hpo_results.db'
study_name = 'nos_1h_hpo_v1_tokenizer'

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Parameter importance plot
    fig1 = optuna.visualization.plot_param_importances(study)
    fig1.write_html('hpo_tokenizer_importance.html')
    
    # Optimization history
    fig2 = optuna.visualization.plot_optimization_history(study)
    fig2.write_html('hpo_tokenizer_history.html')
    
    # Parallel coordinate plot
    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    fig3.write_html('hpo_tokenizer_parallel.html')
    
    print('Plots saved: hpo_tokenizer_*.html')
    print('Open these in a browser to analyze results')
except Exception as e:
    print(f'Error: {e}')
"
```

### Verify Trained Models

```bash
# Check what models exist
python -c "
import os

paths = {
    'Pretrained Tokenizer': 'models/nos_tokenizer_2k',
    'Pretrained Predictor': 'models/nos_mini',
    'Finetuned Tokenizer': 'finetuned/run_1h_v1/tokenizer/best_model',
    'Finetuned Predictor': 'finetuned/run_1h_v1/basemodel/best_model',
}

for name, path in paths.items():
    exists = os.path.exists(path)
    if exists:
        files = os.listdir(path) if os.path.isdir(path) else []
        print(f'✅ {name}: {path}')
        print(f'   Files: {files}')
    else:
        print(f'❌ {name}: NOT FOUND at {path}')
"
```

---

## Part 5: Troubleshooting Decision Table

```bash
# ═══════════════════════════════════════════════════
# PROBLEM: CUDA Out of Memory
# ═══════════════════════════════════════════════════
# Symptom: RuntimeError: CUDA out of memory

# Solution 1: Reduce batch size in config
sed -i 's/batch_size: 32/batch_size: 16/' configs/config_test_1h_full.yaml

# Solution 2: Increase accumulation steps (same effective batch size)
sed -i 's/accumulation_steps: 1/accumulation_steps: 2/' configs/config_test_1h_full.yaml

# Solution 3: Check for memory leaks
python -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
print(f'Available: {torch.cuda.get_device_properties(0).total_memory/1e9 - torch.cuda.memory_reserved(0)/1e9:.2f} GB')
"
```

```bash
# ═══════════════════════════════════════════════════
# PROBLEM: Validation loss not improving (tokenizer)
# ═══════════════════════════════════════════════════
# Symptom: Val loss stays at same value for many epochs

# Diagnosis: Check if learning rate is too low
python -c "
# If the tokenizer LR is too low:
# tokenizer_learning_rate: 0.0002 -> try 0.0005

# If BSQ codebook collapse (all tokens same):
# Increase bsq_gamma (higher = more codebook diversity)
# bsq_gamma: 1.1 -> try 1.5

print('Suggested fixes:')
print('1. Increase tokenizer_learning_rate to 5e-4')
print('2. Increase bsq_gamma to 1.5')
print('3. Decrease bsq_beta to 0.02')
"
```

```bash
# ═══════════════════════════════════════════════════
# PROBLEM: Basemodel validation loss increases (forgetting)
# ═══════════════════════════════════════════════════
# Symptom: Val loss goes down then sharply up

# Solution: Dramatically reduce predictor LR
python -c "
print('Catastrophic forgetting detected!')
print('Fix: Reduce predictor_learning_rate')
print('  From: 0.000005')
print('  To:   0.0000005  (10x smaller)')
print()
print('Also try:')
print('  basemodel_max_grad_norm: 0.5  (tighter clipping)')
print('  ffn_dropout_p: 0.1  (add regularization)')
"

# Update config and retrain
python train_sequential.py \
  --config configs/config_test_1h.yaml \
  --skip-tokenizer
```

```bash
# ═══════════════════════════════════════════════════
# PROBLEM: HPO trials all getting pruned
# ═══════════════════════════════════════════════════
# Symptom: Most trials show PRUNED, very few complete

# Solution: Increase hpo_*_epochs or disable pruner
python -c "
print('Fix options:')
print('1. Increase hpo_tokenizer_epochs from 8 to 15')
print('2. Change pruner from median to none')
print('3. Increase n_startup_trials in MedianPruner')
"
```

```bash
# ═══════════════════════════════════════════════════
# PROBLEM: optuna not installed
# ═══════════════════════════════════════════════════
pip install optuna plotly kaleido

# Verify installation
python -c "
import optuna
print(f'Optuna version: {optuna.__version__}')
optuna.logging.set_verbosity(optuna.logging.WARNING)
print('Optuna working correctly')
"
```

---

## Part 6: The Optimal Execution Order Summary

```
┌─────────────────────────────────────────────────────────────┐
│            OPTIMAL EXECUTION ORDER FOR MAX ACCURACY         │
├────┬────────────────────────────────────────────────────────┤
│ #  │ Command                                                │
├────┼────────────────────────────────────────────────────────┤
│ 1  │ python -c "...data validation..."                      │
│    │ → Verify data quality before investing compute         │
├────┼────────────────────────────────────────────────────────┤
│ 2  │ python hpo_tuner.py --phase tokenizer --n-trials 40   │
│    │ → Find optimal tokenizer hyperparameters (2-4h)        │
├────┼────────────────────────────────────────────────────────┤
│ 3  │ python hpo_tuner.py --apply-best (tokenizer output)   │
│    │ → Write best tokenizer params to new config            │
├────┼────────────────────────────────────────────────────────┤
│ 4  │ python train_sequential.py --skip-basemodel           │
│    │ → Full tokenizer training with best params (1h)        │
├────┼────────────────────────────────────────────────────────┤
│ 5  │ python -c "...tokenizer quality check..."              │
│    │ → Verify reconstruction MSE < 0.05 before continuing   │
├────┼────────────────────────────────────────────────────────┤
│ 6  │ python hpo_tuner.py --phase basemodel --n-trials 40   │
│    │ → Find optimal predictor hyperparameters (1-3h)        │
├────┼────────────────────────────────────────────────────────┤
│ 7  │ python hpo_tuner.py --apply-best (basemodel output)   │
│    │ → Write best predictor params to final config          │
├────┼────────────────────────────────────────────────────────┤
│ 8  │ python train_sequential.py --skip-tokenizer           │
│    │ → Full predictor training with best params (30-60min)  │
├────┼────────────────────────────────────────────────────────┤
│ 9  │ tail -f .../logs/basemodel_training_rank_0.log        │
│    │ → Monitor that val loss decreases consistently         │
└────┴────────────────────────────────────────────────────────┘

Total time estimate (with GPU):
  HPO Phase 1 (tokenizer): 2-4 hours
  Full tokenizer training: 30-60 min
  HPO Phase 2 (basemodel): 1-3 hours
  Full basemodel training: 30-60 min
  ─────────────────────────────────
  Total: 4-9 hours for maximum accuracy
```