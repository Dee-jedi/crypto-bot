"""
Diagnostic script to understand why the AI model rejects every trade.
Examines: label distribution, prediction distribution, confidence values.
"""
import numpy as np
import pandas as pd
from collections import Counter

from config import TF_SIGNAL, TF_BIAS, SEQ_LEN
from features import build_features, FEAT_COLS
from labels import build_labels
from models import EnsembleModel

# 1. Load data
print("=" * 60)
print("  AI MODEL DIAGNOSTIC")
print("=" * 60)

df_15m = pd.read_csv('data/cache/BTCUSDT_15m_60m.csv', index_col=0, parse_dates=True)
print(f"\nData range: {df_15m.index.min()} to {df_15m.index.max()}")
print(f"Total rows: {len(df_15m):,}")

# 2. Build features
df_15m = build_features(df_15m, funding_rate=0.0)
print(f"Features built: {len(FEAT_COLS)} columns")

# 3. Check label distribution from training data
labels = build_labels(df_15m)
print(f"\n--- LABEL DISTRIBUTION (full dataset, lookahead=96) ---")
counts = Counter(labels)
total = len(labels)
for label_val in sorted(counts.keys()):
    name = {0: 'SHORT', 1: 'LONG', 2: 'NEUTRAL'}[label_val]
    print(f"  {name} ({label_val}): {counts[label_val]:>7,} ({counts[label_val]/total*100:.1f}%)")

# 4. Load the trained model
model = EnsembleModel(FEAT_COLS, symbol_tag='BTCUSDT')
loaded = model.load()
print(f"\nModel loaded: {loaded}")

if not loaded:
    print("ERROR: No saved model found! Cannot diagnose predictions.")
    exit(1)

# 5. Sample predictions on 2025 data
df_2025 = df_15m[df_15m.index >= '2025-01-01'].copy()
print(f"\n--- PREDICTION ANALYSIS ON 2025 DATA ---")
print(f"2025 candles: {len(df_2025):,}")

# Take 200 evenly spaced samples
sample_indices = np.linspace(SEQ_LEN, len(df_2025) - 1, min(200, len(df_2025) - SEQ_LEN), dtype=int)

preds = []
confs = []
class_counts = Counter()

for idx in sample_indices:
    try:
        window = df_2025.iloc[max(0, idx - SEQ_LEN + 1):idx + 1]
        if len(window) >= SEQ_LEN:
            pred, conf, _ = model.predict(window)
            preds.append(pred)
            confs.append(conf)
            class_counts[pred] += 1
    except Exception as e:
        print(f"  Error at idx {idx}: {e}")

print(f"\nPredictions sampled: {len(preds)}")
print(f"\n--- PREDICTION CLASS DISTRIBUTION ---")
for cls in sorted(class_counts.keys()):
    name = {0: 'SHORT', 1: 'LONG', 2: 'NEUTRAL'}[cls]
    print(f"  {name} ({cls}): {class_counts[cls]:>5} ({class_counts[cls]/len(preds)*100:.1f}%)")

print(f"\n--- CONFIDENCE STATISTICS ---")
confs_arr = np.array(confs)
print(f"  Mean confidence : {confs_arr.mean():.4f}")
print(f"  Min  confidence : {confs_arr.min():.4f}")
print(f"  Max  confidence : {confs_arr.max():.4f}")
print(f"  Median          : {np.median(confs_arr):.4f}")
print(f"  Std             : {confs_arr.std():.4f}")

# Threshold analysis
for thresh in [0.40, 0.45, 0.50, 0.51, 0.55, 0.60, 0.68]:
    above = sum(1 for c in confs if c >= thresh)
    print(f"  Above {thresh:.2f}: {above}/{len(confs)} ({above/len(confs)*100:.1f}%)")

# 6. Check what the model predicts vs what the backtest needs
print(f"\n--- CRITICAL CHECK: PRED==1 (LONG) with conf >= 0.51 ---")
long_high_conf = sum(1 for p, c in zip(preds, confs) if p == 1 and c >= 0.51)
print(f"  Count: {long_high_conf}/{len(preds)} ({long_high_conf/len(preds)*100:.1f}%)")

print(f"\n--- CRITICAL CHECK: PRED==0 (SHORT) with conf >= 0.51 ---")
short_high_conf = sum(1 for p, c in zip(preds, confs) if p == 0 and c >= 0.51)
print(f"  Count: {short_high_conf}/{len(preds)} ({short_high_conf/len(preds)*100:.1f}%)")

print(f"\n--- CRITICAL CHECK: PRED==2 (NEUTRAL) ---")
neutral = sum(1 for p in preds if p == 2)
print(f"  Count: {neutral}/{len(preds)} ({neutral/len(preds)*100:.1f}%)")

# 7. Show first 20 raw predictions
print(f"\n--- FIRST 20 RAW PREDICTIONS ---")
print(f"  {'idx':>5} | {'pred':>5} | {'conf':>8} | {'class':>8}")
print(f"  {'-'*5} | {'-'*5} | {'-'*8} | {'-'*8}")
for j in range(min(20, len(preds))):
    name = {0: 'SHORT', 1: 'LONG', 2: 'NEUTRAL'}[preds[j]]
    print(f"  {sample_indices[j]:>5} | {preds[j]:>5} | {confs[j]:>8.4f} | {name:>8}")

print("\n" + "=" * 60)
print("  DIAGNOSTIC COMPLETE")
print("=" * 60)
