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