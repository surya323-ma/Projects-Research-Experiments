# AlphaScape Engine
(formerly "Multi-Factor Alpha Landscape Explorer" — renamed after a rework pass; see Notes below)

# Overview
An experimental quantitative research platform for studying synthetic alpha factors, predictive relationships, and factor interactions across a panel of synthetic assets, using machine learning and institutional-style visualization dashboards.

# Features
Synthetic financial factor generation, with a genuine weak momentum/mean-reversion signal baked into the price process
Momentum analysis
Volatility factor modeling
Liquidity imbalance factors
Synthetic sentiment proxies (kept as pure-noise control factors)
Time-aware, walk-forward regression prediction (RandomForest, RidgeCV — no train/test leakage)
PCA and t-SNE factor analysis (t-SNE is actually run now, not disabled)
Alpha decay simulation (now wired into the dashboard)
3D factor interaction landscapes
Factor-factor and cross-asset correlation heatmaps
Permutation/impurity feature importance diagnostics
Out-of-sample PnL and block-bootstrap Monte Carlo simulation
Technologies

# Python • NumPy • Pandas • Scikit-Learn • Matplotlib • Seaborn • SciPy

Run
bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
# python alphascape_engine.py

Outputs (saved to dashboard_outputs/): feature importance, predicted-vs-actual, residuals, factor-factor correlation, cross-asset correlation, 3D factor landscape, t-SNE signal density, alpha decay simulation, out-of-sample PnL, Monte Carlo simulation, statistical summary.
