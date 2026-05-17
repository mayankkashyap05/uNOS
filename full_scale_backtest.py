import os
import logging
from typing import List, Dict, Any, Optional, Union
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from finetune_base_model import CustomFinetuneConfig
from model import Nos, NosTokenizer, NosPredictor

# =============================================================================
# Constants & Configuration
# =============================================================================

CONFIG_PATH = "configs/config_t40_hpo_winner.yaml"  # Make sure this points to your active config
RESULTS_DIR = "backtest_results"

PRICE_COLS = frozenset(['open', 'high', 'low', 'close'])
VOL_AMT_COLS = frozenset(['volume', 'amount'])

class VerdictThresholds:
    """Constants for signal verdict thresholds."""
    ALPHA_EXCEPTIONAL = 57.0
    ALPHA_STRONG = 55.0
    ALPHA_BASE = 53.0
    ALPHA_WEAK = 51.0
    ALPHA_MARGINAL = 50.0

    CORR_STRONG = 0.5
    CORR_MODERATE = 0.3
    CORR_WEAK = 0.1

    EPSILON = 1e-10
    DIR_MOVE_THRESHOLD = 0.0001

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("nos_backtest_engine")

# =============================================================================
# Utility Functions
# =============================================================================

def _sort_extra_cols(cols: List[str]) -> List[str]:
    """Sort extra_* columns numerically when possible, alphabetically otherwise."""
    def _sort_key(c: str) -> tuple:
        try:
            return (0, int(c.split('_')[1]))
        except (ValueError, IndexError):
            return (1, c)
    return sorted(cols, key=_sort_key)


def get_all_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the ordered list of all feature columns exactly as the model sees them.
    """
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    extra_cols = [c for c in df.columns if c.startswith('extra_')]
    extra_cols = _sort_extra_cols(extra_cols)
    return base_cols + extra_cols


def _get_column_category(col: str) -> str:
    """Determine the logical category of a feature column."""
    if col in PRICE_COLS:
        return "PRICE"
    elif col in VOL_AMT_COLS:
        return "VOL/AMT"
    elif col.startswith('extra_'):
        return "EXTRA"
    return "OTHER"


def _get_column_verdict(acc: float, corr: float, category: str) -> str:
    """Generate a categorical verdict string based on calculated metrics."""
    if category in ("PRICE", "VOL/AMT"):
        if np.isnan(acc):
            if corr > VerdictThresholds.CORR_MODERATE:
                return "🟡 CORRELATION SIGNAL"
            return "⚪ INSUFFICIENT DATA"
            
        if acc > VerdictThresholds.ALPHA_EXCEPTIONAL:
            return "🟢🟢 EXCEPTIONAL ALPHA"
        elif acc > VerdictThresholds.ALPHA_STRONG:
            return "🟢 STRONG ALPHA"
        elif acc > VerdictThresholds.ALPHA_BASE:
            return "🟢 MODEL HAS ALPHA"
        elif acc > VerdictThresholds.ALPHA_WEAK:
            return "🟡 WEAK SIGNAL"
        elif acc > VerdictThresholds.ALPHA_MARGINAL:
            return "🟡 MARGINAL"
        return "🔴 NO SIGNAL"
    else:
        if np.isnan(acc):
            if corr > VerdictThresholds.CORR_STRONG:
                return "🟢 STRONG CORRELATION"
            elif corr > VerdictThresholds.CORR_MODERATE:
                return "🟡 MODERATE CORRELATION"
            elif corr > VerdictThresholds.CORR_WEAK:
                return "🟡 WEAK CORRELATION"
            return "🔴 NO CORRELATION"
            
        if acc > VerdictThresholds.ALPHA_STRONG and corr > VerdictThresholds.CORR_MODERATE:
            return "🟢🟢 EXCEPTIONAL (DIR+CORR)"
        elif acc > VerdictThresholds.ALPHA_BASE:
            return "🟢 STRONG SIGNAL"
        elif acc > VerdictThresholds.ALPHA_WEAK:
            return "🟡 WEAK SIGNAL"
        elif acc > VerdictThresholds.ALPHA_MARGINAL:
            return "🟡 MARGINAL"
        return "🔴 NO SIGNAL"

# =============================================================================
# Mathematical & Metrics Engine
# =============================================================================

class MetricsCalculator:
    """Encapsulates statistical operations for evaluating model predictions."""

    @staticmethod
    def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    @staticmethod
    def compute_mape(pred: np.ndarray, true: np.ndarray) -> float:
        denom = np.abs(true) + VerdictThresholds.EPSILON
        return float(np.mean(np.abs((pred - true) / denom)) * 100)

    @staticmethod
    def compute_r2(pred: np.ndarray, true: np.ndarray) -> float:
        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + VerdictThresholds.EPSILON))) if ss_tot > VerdictThresholds.EPSILON else 0.0

    @staticmethod
    def compute_vectorized_path_accuracy(pred: np.ndarray, true: np.ndarray, initial_price: float) -> float:
        """Vectorized approach to calculate full-path directional accuracy."""
        if len(pred) < 1:
            return np.nan

        shifted_true = np.roll(true, 1)
        shifted_true[0] = initial_price

        t_moves = true - shifted_true
        p_moves = pred - shifted_true

        valid_mask = (np.abs(t_moves) / (np.abs(shifted_true) + VerdictThresholds.EPSILON)) > VerdictThresholds.DIR_MOVE_THRESHOLD
        
        if not np.any(valid_mask):
            return np.nan

        correct_dir = np.sign(p_moves[valid_mask]) == np.sign(t_moves[valid_mask])
        return float(np.mean(correct_dir))

def compute_metrics_all_columns(task: Dict[str, Any], pred_df: pd.DataFrame, feature_cols: List[str]) -> Optional[Dict[str, Any]]:
    """
    Compute directional accuracy, correlation, RMSE, MAPE for EVERY column.
    """
    truth_df = task['truth']
    min_len = min(len(pred_df), len(truth_df))
    
    if min_len == 0:
        return None

    available_cols = [c for c in feature_cols if c in pred_df.columns and c in truth_df.columns]
    if not available_cols:
        return None

    result = {
        'start_time': task['y_ts'].iloc[0] if not task['y_ts'].empty else None,
        'end_time': task['y_ts'].iloc[-1] if not task['y_ts'].empty else None,
    }

    col_corrs, col_rmses, col_mapes, col_accs = [], [], [], []

    for col in available_cols:
        pred_vals = pred_df[col].values[:min_len]
        true_vals = truth_df[col].values[:min_len]
        last_price = task['context'][col].iloc[-1] if col in task['context'].columns else None

        # Base error metrics
        rmse = MetricsCalculator.compute_rmse(pred_vals, true_vals)
        mape = MetricsCalculator.compute_mape(pred_vals, true_vals)
        mae = float(np.mean(np.abs(pred_vals - true_vals)))
        r_squared = MetricsCalculator.compute_r2(pred_vals, true_vals)
        
        mean_true = np.mean(np.abs(true_vals)) + VerdictThresholds.EPSILON
        nrmse = (rmse / mean_true) * 100

        # Correlation
        if min_len < 2 or np.std(pred_vals) < VerdictThresholds.EPSILON or np.std(true_vals) < VerdictThresholds.EPSILON:
            corr = 0.0
        else:
            corr = np.corrcoef(pred_vals, true_vals)[0, 1]
            if np.isnan(corr): corr = 0.0

        # Directional Accuracy (T+1 and Path)
        correct_dir = None
        path_acc = np.nan
        
        if last_price is not None and not np.isnan(last_price):
            pred_move = pred_vals[0] - last_price
            true_move = true_vals[0] - last_price
            
            if abs(true_move) / (abs(last_price) + VerdictThresholds.EPSILON) > VerdictThresholds.DIR_MOVE_THRESHOLD:
                correct_dir = 1 if (np.sign(pred_move) == np.sign(true_move)) else 0

            if min_len > 1:
                path_acc = MetricsCalculator.compute_vectorized_path_accuracy(pred_vals, true_vals, last_price)

        # Build column results
        result.update({
            f'{col}_corr': corr,
            f'{col}_rmse': rmse,
            f'{col}_nrmse': nrmse,
            f'{col}_mape': mape,
            f'{col}_mae': mae,
            f'{col}_r2': r_squared,
            f'{col}_path_acc': path_acc,
            f'{col}_last_price': float(last_price) if last_price is not None else np.nan,
            f'{col}_pred_t1': float(pred_vals[0]),
            f'{col}_true_t1': float(true_vals[0])
        })

        if correct_dir is not None:
            result[f'{col}_acc'] = correct_dir
            col_accs.append(correct_dir)

        col_corrs.append(corr)
        col_rmses.append(rmse)
        col_mapes.append(mape)

    # Global Aggregates
    result['avg_corr_all'] = np.mean(col_corrs) if col_corrs else 0.0
    result['avg_rmse_all'] = np.mean(col_rmses) if col_rmses else 0.0
    result['avg_mape_all'] = np.mean(col_mapes) if col_mapes else 0.0
    result['avg_acc_all'] = np.mean(col_accs) if col_accs else np.nan

    # Group Aggregates (PRICE vs VOL/AMT)
    price_accs = [result[f'{c}_acc'] for c in PRICE_COLS if f'{c}_acc' in result]
    result['avg_acc_price'] = np.mean(price_accs) if price_accs else np.nan

    va_accs = [result[f'{c}_acc'] for c in VOL_AMT_COLS if f'{c}_acc' in result]
    result['avg_acc_va'] = np.mean(va_accs) if va_accs else np.nan

    return result

# =============================================================================
# Reporting & I/O Engine
# =============================================================================

class DataExporter:
    """Handles writing backtest results to disk efficiently."""
    
    @staticmethod
    def save_results(df_res: pd.DataFrame, col_summary_rows: List[Dict[str, Any]], feature_cols: List[str]) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        try:
            full_path = os.path.join(RESULTS_DIR, "backtest_results_full.csv")
            df_res.to_csv(full_path, index=False)
            logger.info(f"Saved full results: {full_path}")

            if col_summary_rows:
                summary_path = os.path.join(RESULTS_DIR, "backtest_column_summary.csv")
                pd.DataFrame(col_summary_rows).to_csv(summary_path, index=False)
                logger.info(f"Saved column summary: {summary_path}")

            if len(df_res) > 20:
                rolling_path = os.path.join(RESULTS_DIR, "backtest_rolling_accuracy.csv")
                DataExporter._save_rolling_metrics(df_res, feature_cols, rolling_path)

            deepdive_dir = os.path.join(RESULTS_DIR, "column_deep_dives")
            os.makedirs(deepdive_dir, exist_ok=True)
            for col in feature_cols:
                col_keys = [k for k in df_res.columns if k.startswith(f'{col}_')]
                if col_keys:
                    col_df = df_res[['start_time', 'end_time'] + col_keys]
                    col_df.to_csv(os.path.join(deepdive_dir, f"deepdive_{col}.csv"), index=False)
            
        except OSError as e:
            logger.exception(f"I/O Error while saving results: {e}")

    @staticmethod
    def _save_rolling_metrics(df_res: pd.DataFrame, feature_cols: List[str], target_path: str) -> None:
        rolling_df = pd.DataFrame()
        rolling_df['start_time'] = df_res['start_time']
        window_size = min(50, len(df_res) // 4)
        min_periods = max(10, window_size // 2)

        for col in feature_cols:
            acc_key, corr_key = f'{col}_acc', f'{col}_corr'
            if acc_key in df_res.columns:
                rolling_df[f'{col}_rolling_acc'] = df_res[acc_key].rolling(
                    window=window_size, min_periods=min_periods).mean() * 100
            if corr_key in df_res.columns:
                rolling_df[f'{col}_rolling_corr'] = df_res[corr_key].rolling(
                    window=window_size, min_periods=min_periods).mean()

        rolling_df.to_csv(target_path, index=False)


class ReportingEngine:
    """Manages the generation of terminal output and metrics summarization."""
    
    @staticmethod
    def print_detailed_report(results: List[Dict[str, Any]], feature_cols: List[str]) -> None:
        if not results:
            logger.warning("No results generated to report.")
            return

        df_res = pd.DataFrame(results)
        logger.info(f"Generated {len(results)} samples across {len(feature_cols)} feature columns.")
        
        col_summary_rows = []

        # ── 1. Print Detailed Column-by-Column Deep Dives ──
        for idx, col in enumerate(feature_cols):
            corr_key, acc_key = f'{col}_corr', f'{col}_acc'
            if corr_key not in df_res.columns:
                continue
                
            cat = _get_column_category(col)
            corr_vals = df_res[corr_key].dropna()
            acc_vals = df_res[acc_key].dropna() if acc_key in df_res.columns else pd.Series(dtype=float)

            mean_corr = corr_vals.mean() if not corr_vals.empty else 0.0
            mean_acc = acc_vals.mean() * 100 if not acc_vals.empty else np.nan
            verdict = _get_column_verdict(mean_acc, mean_corr, cat)
            
            # Print the deep dive log
            logger.info(f"[{idx + 1}/{len(feature_cols)}] DEEP DIVE: {col.upper():<8} ({cat:<8}) - Accuracy: {mean_acc:>6.2f}% | Verdict: {verdict}")

            col_summary_rows.append({
                'column': col,
                'category': cat,
                'accuracy_%': mean_acc,
                'mean_corr': mean_corr,
                'n_samples': len(corr_vals),
                'n_directional': len(acc_vals),
                'verdict': verdict,
            })

        # ── 2. Print Summary Table ──
        logger.info("=" * 85)
        logger.info(f"{'COLUMN':<15} | {'CATEGORY':<10} | {'DIR. ACCURACY':<15} | {'CORRELATION':<12} | {'VERDICT'}")
        logger.info("-" * 85)
        
        for row in col_summary_rows:
            acc_str = f"{row['accuracy_%']:.2f}%" if not np.isnan(row['accuracy_%']) else "N/A"
            corr_str = f"{row['mean_corr']:.4f}"
            
            logger.info(f"{row['column'].upper():<15} | {row['category']:<10} | {acc_str:<15} | {corr_str:<12} | {row['verdict']}")
        logger.info("=" * 85)

        # ── 3. Print Category Aggregates ──
        overall_acc = df_res['avg_acc_all'].mean() * 100 if 'avg_acc_all' in df_res.columns else np.nan
        price_acc = df_res['avg_acc_price'].mean() * 100 if 'avg_acc_price' in df_res.columns else np.nan
        va_acc = df_res['avg_acc_va'].mean() * 100 if 'avg_acc_va' in df_res.columns else np.nan

        logger.info("--- CATEGORY AGGREGATES ---")
        logger.info(f"Avg Accuracy [ALL COLUMNS]:   {overall_acc:.2f}%" if not np.isnan(overall_acc) else "Avg Accuracy [ALL COLUMNS]:   N/A")
        logger.info(f"Avg Accuracy [PRICE ONLY]:    {price_acc:.2f}%" if not np.isnan(price_acc) else "Avg Accuracy [PRICE ONLY]:    N/A")
        logger.info(f"Avg Accuracy [VOL/AMT ONLY]:  {va_acc:.2f}%" if not np.isnan(va_acc) else "Avg Accuracy [VOL/AMT ONLY]:  N/A")
        logger.info("=====================================================================================")

        DataExporter.save_results(df_res, col_summary_rows, feature_cols)

# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    logger.info("Initializing Nos FULL-SCALE INSTITUTIONAL MULTI-COLUMN BACKTEST")

    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config not found: {CONFIG_PATH}")
        return

    config = CustomFinetuneConfig(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loaded config parameters: Lookback={config.lookback_window}, Forecast={config.predict_window}, BatchSize={config.batch_size}, Device={device}")

    try:
        tokenizer_path = config.finetuned_tokenizer_path if os.path.exists(config.finetuned_tokenizer_path) else config.tokenizer_best_model_path
        tokenizer = NosTokenizer.from_pretrained(tokenizer_path)
        model = Nos.from_pretrained(config.basemodel_best_model_path).to(device)

        predictor = NosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=str(device),
            max_context=config.max_context,
            clip=config.clip
        )
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load models.")
        return

    logger.info(f"Loading dataset from: {config.data_path}")
    full_df = pd.read_csv(config.data_path)
    full_df['timestamps'] = pd.to_datetime(full_df['timestamps'])

    if 'volume' not in full_df.columns:
        full_df['volume'] = 0.0
    if 'amount' not in full_df.columns:
        full_df['amount'] = full_df['volume'] * full_df[['open', 'high', 'low', 'close']].mean(axis=1) if 'volume' in full_df.columns else 0.0

    feature_cols = get_all_feature_cols(full_df)
    
    if len(feature_cols) != tokenizer.d_in:
        logger.error(f"FATAL: Tokenizer expects {tokenizer.d_in} features, found {len(feature_cols)}. Exiting.")
        return

    train_end = int(len(full_df) * config.train_ratio)
    val_df = full_df.iloc[train_end:].reset_index(drop=True)
    val_df[feature_cols] = val_df[feature_cols].ffill().fillna(0.0)

    logger.info("Generating backtest windows...")
    tasks = []
    max_start = len(val_df) - config.lookback_window - config.predict_window

    for start_idx in range(0, max_start + 1, config.predict_window):
        hist_end = start_idx + config.lookback_window
        future_end = hist_end + config.predict_window

        ctx = val_df.iloc[start_idx:hist_end].copy().reset_index(drop=True)
        tgt = val_df.iloc[hist_end:future_end].copy().reset_index(drop=True)

        if len(ctx) == config.lookback_window and len(tgt) == config.predict_window:
            tasks.append({
                'context': ctx,
                'x_ts': ctx['timestamps'].reset_index(drop=True),
                'y_ts': tgt['timestamps'].reset_index(drop=True),
                'truth': tgt,
            })

    if not tasks:
        logger.warning("No scenarios generated. Exiting.")
        return

    logger.info(f"Running inference on {len(tasks)} tasks.")
    results, errors = [], 0

    for i in tqdm(range(0, len(tasks), config.batch_size), desc="Backtesting"):
        batch_tasks = tasks[i: i + config.batch_size]
        
        try:
            preds_batch = predictor.predict_batch(
                df_list=[t['context'] for t in batch_tasks],
                x_timestamp_list=[t['x_ts'] for t in batch_tasks],
                y_timestamp_list=[t['y_ts'] for t in batch_tasks],
                pred_len=config.predict_window,
                T=0.01, top_k=0, top_p=0.99, sample_count=1, verbose=False
            )

            for j, pred_df in enumerate(preds_batch):
                try:
                    res = compute_metrics_all_columns(batch_tasks[j], pred_df, feature_cols)
                    if res: results.append(res)
                except ValueError as ve:
                    errors += 1
                    if errors <= 5: logger.exception(f"Metric calculation error in task {i + j}: {ve}")
                    
        except RuntimeError as re:
            errors += 1
            if errors <= 5: logger.exception(f"Batch prediction error at index {i}: {re}")

    if errors > 0:
        logger.warning(f"Encountered {errors} total errors during prediction pipeline.")

    logger.info("Generating reports...")
    ReportingEngine.print_detailed_report(results, feature_cols)

if __name__ == "__main__":
    main()