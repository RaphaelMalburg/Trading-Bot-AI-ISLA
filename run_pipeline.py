"""
Full ML Pipeline Runner — one-command validation.

Executes:
1. Model training (with scaler)
2. ML metrics generation (SHAP, calibration)
3. Backtest (single-split + walk-forward)
4. Model comparison (if xgboost/torch available)

Usage: python run_pipeline.py
Outputs: models/*.pkl, models/*.png, models/*.json
"""

import subprocess
import sys
import time
from pathlib import Path


def run_step(name, command):
    print(f"\n{'=' * 50}")
    print(f"[{name}]")
    print(f"{'=' * 50}")
    start = time.time()
    result = subprocess.run(command, shell=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"❌ {name} failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"✅ {name} completed in {elapsed:.1f}s")


def main():
    print("🚀 ML Trading Bot — Full Pipeline Runner")
    print("This will train models, generate diagnostics, and run backtests.\n")

    steps = [
        ("TRAIN MODEL", "python src/model_training.py"),
        ("ML METRICS", "python src/plot_ml_metrics.py"),
        ("BACKTEST (Single)", "python src/backtest_dynamic.py"),
        ("BACKTEST (Walk-Forward)", "python src/backtest_dynamic.py --walkforward"),
        ("MODEL COMPARISON", "python src/compare_models.py"),
    ]

    for name, cmd in steps:
        try:
            run_step(name, cmd)
        except KeyboardInterrupt:
            print("\n⚠️  Pipeline interrupted by user.")
            sys.exit(130)

    print("\n" + "=" * 50)
    print("🎉 Pipeline completed successfully!")
    print("=" * 50)
    print("\n📊 Artifacts generated in models/:")
    print("  - rf_model.pkl (trained Random Forest)")
    print("  - scaler.pkl (StandardScaler)")
    print("  - confusion_matrix.png, shap_summary.png")
    print("  - backtest_results.json")
    print("  - comparison_metrics.json (if xgboost/torch installed)")
    print("\n📈 View dashboard: python src/main.py → http://localhost:5000")


if __name__ == "__main__":
    main()
