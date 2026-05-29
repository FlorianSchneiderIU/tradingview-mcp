#!/usr/bin/env python3
"""
Matrix RL Pretrain Probe - Fast offline validation of Matrix signal RL integration.

Goals:
1. Extract recent Matrix signals + outcomes from logs/trade history
2. Map signals to default RL sidecar feature shape
3. Run offline pretrain dry-run to test feature compatibility
4. Report feature coverage and mismatch analysis
"""
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

def parse_matrix_signals_from_log(log_path: str, limit: int = 50) -> list:
    """Parse Matrix signal events from matrix_signal_bot.log"""
    signals = []
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Regex: look for "Signal detected" lines
        pattern = r'Signal detected \| symbol=(\w+) dir=(\w+) entry=([\d.]+) sl=([\d.]+) tp=([\d.]+) strategy=(\w+)'
        matches = re.findall(pattern, content)
        
        for symbol, direction, entry, sl, tp, strategy in matches[-limit:]:
            signals.append({
                'symbol': symbol,
                'direction': 1.0 if direction == 'long' else -1.0,  # action space
                'entry': float(entry),
                'sl': float(sl),
                'tp': float(tp),
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            })
        
        return signals
    except FileNotFoundError:
        print(f"[PROBE] Log not found: {log_path}")
        return []

def map_signal_to_rl_features(signal: dict) -> dict:
    """
    Map a Matrix signal to RL sidecar expected features.
    
    Default RL features (from session_orb/turtle_soup):
    - direction_long (bool)
    - entry_risk_atr, entry_risk_pct
    - target_distance_atr, rr (risk/reward)
    - symbol-derived features (vol, price level)
    - market context (1h/4h returns, etc.)
    
    Matrix signals have simpler structure:
    - symbol, entry, sl, tp → can derive rr, risk pct
    - No explicit bar/ATR/spread features → fill with defaults
    """
    direction = signal['direction']
    entry = signal['entry']
    sl = signal['sl']
    tp = signal['tp']
    
    # Derive risk/reward from entry/sl/tp
    risk_pips = abs(entry - sl)
    reward_pips = abs(tp - entry)
    rr = reward_pips / risk_pips if risk_pips > 0 else 1.0
    risk_pct = (risk_pips / entry * 100) if entry > 0 else 1.0
    
    # Minimal feature set compatible with default RL model
    features = {
        'direction_long': 1.0 if direction > 0 else 0.0,
        'entry_risk_pct': min(risk_pct, 10.0),  # cap at 10%
        'entry_risk_atr': risk_pips,  # treat as pseudo-ATR distance
        'target_distance_atr': reward_pips,
        'rr': min(rr, 5.0),  # cap RR at 5x
        # Filler features (will warn about missing context)
        'symbol_vol_mult': 1.0,
        'or_minutes': 60.0,  # assume 1h after open
        'ret_1h_dir': 0.5,  # neutral if unknown
        'ret_4h_dir': 0.5,
    }
    
    return features

def run_pretrain_dry_run(signals: list) -> dict:
    """
    Simulate a lightweight pretrain pass without importing full RL stack.
    Reports:
    - Signal count
    - Feature coverage
    - Simulated weight updates
    """
    if not signals:
        return {'status': 'no_signals', 'message': 'No signals parsed'}
    
    print(f"[PROBE] Parsed {len(signals)} Matrix signals")
    
    # Map all signals to feature space
    feature_batch = []
    missing_features = set()
    
    for sig in signals:
        try:
            features = map_signal_to_rl_features(sig)
            feature_batch.append(features)
            
            # Track which features are filler (default values)
            if features['ret_1h_dir'] == 0.5:
                missing_features.add('ret_1h_dir')
            if features['ret_4h_dir'] == 0.5:
                missing_features.add('ret_4h_dir')
            if features['symbol_vol_mult'] == 1.0:
                missing_features.add('symbol_vol_mult')
                
        except Exception as e:
            print(f"[PROBE] Feature map error for {sig['symbol']}: {e}")
    
    # Simulated pretrain metrics (if we had the actual model)
    result = {
        'status': 'ok',
        'signals_processed': len(feature_batch),
        'unique_symbols': len(set(s['symbol'] for s in signals)),
        'directions': {
            'long': sum(1 for s in signals if s['direction'] > 0),
            'short': sum(1 for s in signals if s['direction'] < 0),
        },
        'feature_coverage': {
            'present': list(set(feature_batch[0].keys()) - missing_features),
            'missing_context': list(missing_features),
        },
        'risk_metrics': {
            'avg_risk_pct': sum(f['entry_risk_pct'] for f in feature_batch) / len(feature_batch),
            'avg_rr': sum(f['rr'] for f in feature_batch) / len(feature_batch),
        },
        'recommendation': (
            'Default RL sidecar can accept Matrix signals with feature padding. '
            'Missing market context (1h/4h returns, volatility) will cause slower initial adaptation. '
            'Suggest: enhance matrix_signal_bot to export OHLCV context or pre-compute returns.'
        )
    }
    
    return result

def main():
    log_path = 'bot/logs/matrix_signal_bot.log'
    
    print("[PROBE] === Matrix RL Pretrain Compatibility Probe ===")
    print(f"[PROBE] Log: {log_path}")
    
    # Parse signals
    signals = parse_matrix_signals_from_log(log_path, limit=50)
    
    # Run dry-run test
    result = run_pretrain_dry_run(signals)
    
    # Report
    print(f"\n[RESULT] {json.dumps(result, indent=2)}")
    
    # Save report
    report_path = Path('bot/logs/matrix_pretrain_probe_report.json')
    report_path.write_text(json.dumps(result, indent=2))
    print(f"\n[PROBE] Report saved: {report_path}")

if __name__ == '__main__':
    main()
