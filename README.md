# Project-Research-Experiments

A collection of experimental machine learning and quantitative finance research projects built using synthetic data, statistical modeling, and advanced visualization techniques in Python.

These projects are designed for:

* quantitative research
* machine learning experimentation
* market simulation
* financial visualization
* educational exploration

All systems use fully synthetic/generated data and are intended strictly for educational and research purposes only.

---

# Projects Included

## 1. LiquidStress Engine
*(formerly "Liquidity Shock Simulator" — renamed after a rework pass; see Notes below)*

### Overview
A quantitative market microstructure simulation project focused on liquidity crises, spread expansion, execution pressure, and volatility stress events.

### Features
* Synthetic order book generation with volatility clustering (GARCH-style)
* Bid/ask liquidity depth modeling
* Spread widening simulations
* Liquidity imbalance analysis
* Time-validated machine learning regression models (Ridge, Lasso, Gradient Boosting)
* Block-bootstrap Monte Carlo stress scenarios with VaR
* 3D liquidity surface visualization
* Correlation heatmaps
* Permutation-based feature importance
* Predicted vs actual ML diagnostics
* Simulated PnL analysis

### Technologies
Python • NumPy • Pandas • Scikit-Learn • Matplotlib • Seaborn • SciPy

### Run
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
python liquidstress_engine.py
```

Outputs: `liquidstress_dashboard.png`, `liquidstress_3d_surface.png`, `liquidstress_correlation.png`

---

## 2. RegimeLens Engine
*(formerly "Neural Regime Detection Engine" — renamed after a rework pass; see Notes below)*

### Overview
An AI + quantitative finance project that detects synthetic market regimes such as trends, crashes, panic states, and volatility transitions using unsupervised machine learning and probabilistic modeling — and validates the detections against ground truth.

### Features
* Synthetic market regime generation (trend, mean-reversion, panic, crash, high-volatility)
* Regime clustering models (Gaussian Mixture, KMeans)
* PCA dimensionality reduction
* Volatility-state transitions
* Confidence score mapping
* Ground-truth validation: majority-vote accuracy, Adjusted Rand Index, Normalized Mutual Info
* BIC-based model-order selection
* 3D volatility-regime visualizations
* Transition heatmaps
* Regime-conditional Monte Carlo simulations

### Technologies
Python • NumPy • Pandas • Scikit-Learn • Matplotlib • Seaborn • SciPy

### Run
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
python regimelens_engine.py
```

Outputs: `regimelens_dashboard.png`, `regimelens_bic_sweep.png`

---

## 3. Multi-Factor Alpha Landscape Explorer

### Overview
An experimental quantitative research platform for studying synthetic alpha factors, predictive relationships, and factor interactions using machine learning and institutional-style visualization dashboards.

### Features
* Synthetic financial factor generation
* Momentum analysis
* Volatility factor modeling
* Liquidity imbalance factors
* Synthetic sentiment proxies
* Regression-based prediction systems
* PCA/t-SNE factor analysis
* Alpha decay simulation
* 3D factor interaction landscapes
* Signal heatmaps
* Feature importance diagnostics
* Simulated strategy analytics

### Technologies
Python • NumPy • Pandas • Scikit-Learn • Matplotlib • Seaborn • SciPy

### Run
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
python multi_factor_alpha_landscape_explorer.py
```

---

# Visualization Style

These projects focus heavily on institutional-style quantitative research visualization, including:

* dark-themed analytics dashboards
* 3D financial surfaces
* correlation matrices
* volatility landscapes
* residual distributions
* feature importance charts
* Monte Carlo simulation outputs
* quantitative heatmaps

---

# Notes

* All data is synthetically generated
* No broker APIs or real trading execution
* CPU-friendly projects
* Built for experimentation and learning
* Suitable for ML + quant portfolio projects
* **Projects 1 and 2** were renamed and reworked from their original versions to fix real correctness bugs (unscaled clustering features, a silently constant spread column, a broadcast bug that zeroed out depth imbalance, a missing regime class, dead dashboard code, and a random — leakage-prone — train/test split on time series data), and to add honest, walk-forward-validated result reporting instead of unvalidated single-split metrics. Their old filenames (`liquidity_shock_simulator.py`, `neural_regime_detection_engine.py`) are no longer used — use the filenames under each project's Run section above.

---

# Disclaimer

These projects are strictly for educational and research purposes only.

They are NOT:

* financial advice
* trading systems for live markets
* investment recommendations
* production-grade trading infrastructure

Use responsibly for learning, experimentation, and visualization purposes only.

---

# License

MIT License
