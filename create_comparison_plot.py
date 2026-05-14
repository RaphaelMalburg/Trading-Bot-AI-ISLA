"""
Comparison visualization between 4H and 1H timeframe backtests.
Creates a side-by-side comparison of key metrics.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load results from both backtests
def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Load both backtest results
results_4h = load_results("models/backtest_results_eval.json")
results_1h = load_results("models/backtest_results_lowtf.json")

# Metrics to compare
metrics = ['total_return', 'buy_hold_return', 'sharpe', 'max_drawdown', 'win_rate', 'accuracy']
metric_labels = ['Total Return (%)', 'Buy & Hold Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Accuracy']

# Extract values
values_4h = [results_4h[m] for m in metrics]
values_1h = [results_1h[m] for m in metrics]

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Bar chart comparison
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, values_4h, width, label='4H Timeframe', color='#ff6b6b', alpha=0.8)
bars2 = ax1.bar(x + width/2, values_1h, width, label='1H Timeframe', color='#4ecdc4', alpha=0.8)

ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.set_title('Backtest Performance Comparison: 4H vs 1H Timeframe', fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels, rotation=15, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
def add_value_labels(bars, values, ax):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(values_4h + values_1h) * 0.01),
                f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1, values_4h, ax1)
add_value_labels(bars2, values_1h, ax1)

# Plot 2: Equity curves comparison (if data available)
# Since we don't have the equity curves in the JSON, we'll show a text summary
ax2.axis('off')
summary_text = f"""
BACKTEST SUMMARY COMPARISON
{'='*50}

4H TIMEFRAME (4-hour candles):
• Total Return: {results_4h['total_return']}%
• Buy & Hold: {results_4h['buy_hold_return']}%
• Outperformance: {results_4h['total_return'] - results_4h['buy_hold_return']:+.1f}%
• Sharpe Ratio: {results_4h['sharpe']}
• Max Drawdown: {results_4h['max_drawdown']}%
• Win Rate: {results_4h['win_rate']}%
• Total Trades: {results_4h['total_trades']}
• Model Accuracy: {results_4h['accuracy']}

1H TIMEFRAME (1-hour candles):
• Total Return: {results_1h['total_return']}%
• Buy & Hold: {results_1h['buy_hold_return']}%
• Outperformance: {results_1h['total_return'] - results_1h['buy_hold_return']:+.1f}%
• Sharpe Ratio: {results_1h['sharpe']}
• Max Drawdown: {results_1h['max_drawdown']}%
• Win Rate: {results_1h['win_rate']}%
• Total Trades: {results_1h['total_trades']}
• Model Accuracy: {results_1h['accuracy']}

KEY INSIGHTS:
{'='*50}
• The 1H timeframe shows significantly better risk-adjusted returns
  (Sharpe 1.29 vs -0.26) despite slightly lower absolute returns
  
• Much lower drawdown in 1H timeframe (3.62% vs 26.92%)
  
• Higher win rate in 1H timeframe (55.56% vs 32.35%)
  
• Better model accuracy in 1H timeframe (79.3% vs 65.7%)
  
• Fewer trades in 1H timeframe (9 vs 34) suggests more selective,
  higher-quality signals

TOP FEATURES (1H Timeframe):
• macd_diff: 0.0774
• log_return: 0.0756
• bb_width: 0.0741
• volume_sma_ratio: 0.0740
• atr_14: 0.0729

These features suggest momentum (MACD), price action (returns),
volatility (BB width, ATR), and volume are most predictive.
"""

ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

plt.tight_layout()
plt.savefig('models/backtest_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Comparison visualization saved to: models/backtest_comparison.png")