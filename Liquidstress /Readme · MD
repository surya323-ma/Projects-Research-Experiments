# LiquidStress Engine
### Synthetic Market Liquidity Crisis Simulator & ML Stress-Testing Dashboard

*(formerly "Liquidity Shock Simulator" — renamed and substantially reworked for correctness and result quality)*

A quantitative market microstructure research tool that generates a synthetic limit order book with realistic volatility clustering and liquidity-shock regimes, engineers stress-aware features, properly time-validates regression models, runs Monte Carlo stress scenarios, and renders a full multi-panel diagnostics dashboard.

All data is fully synthetic. For educational / research use only — not investment advice.

## What changed from the original version

**Bugs fixed (these were silently corrupting the results):**
- `spread` in the output DataFrame was built from only the *final* loop value, so every row had an identical, constant spread — depth/spread dynamics were never actually reaching the model. Now tracked as a proper per-step series.
- Liquidity depth updates used a single scalar random draw broadcast across all 20 bid/ask columns, so bid and ask depth were always identical and `depth_imbalance` was silently always 0. Now updated elementwise.
- `run_ml_models` used a **random** 75/25 train/test split on time-series data — this leaks future information into training and inflates apparent accuracy. Replaced with a proper walk-forward (`TimeSeriesSplit`) evaluation.
- The "XGBoost-style" model was actually just `Ridge(alpha=10)` relabeled. Replaced with a real `GradientBoostingRegressor`.
- `plot_feature_importance` read `.coef_` directly, which crashes on tree models and is meaningless on unscaled features. Replaced with `permutation_importance`, and Ridge/Lasso are now wrapped in a `StandardScaler` pipeline (the old unscaled versions were numerically ill-conditioned).
- Every plotting function in the original file was defined but **never called** — `main()` only ever showed one histogram. All panels are now wired into one dashboard.
- Price/spread/liquidity used memoryless multiplicative random walks, so over a few thousand steps they could drift to unrealistic scales (spread compounding into the hundreds, price into the thousands). Added gentle mean-reversion so the simulation reads as a stationary stress regime instead of a runaway trend.

**Result quality:**
- Predicting **future volatility** (real signal, since the generator has genuine GARCH-style clustering) now gets a stable, honest out-of-sample **R² ≈ 0.11–0.18** across a 6-fold walk-forward evaluation.
- Predicting **future return** stays near-zero/negative R², which is reported deliberately as the expected, realistic result — returns on a near-random-walk price are supposed to be hard to predict, and pretending otherwise would be the wrong kind of "good result."
- Monte Carlo now uses a block bootstrap (preserves stress-event clustering) instead of independent row resampling, and reports a 5% VaR line.

## Dashboard output

Running the script produces three PNGs:
- `liquidstress_dashboard.png` — price path with shock markers, feature importance, predicted-vs-actual, residuals, execution pressure heatmap, stress event cloud, Monte Carlo PnL distribution
- `liquidstress_3d_surface.png` — 3D liquidity stress surface
- `liquidstress_correlation.png` — full feature correlation matrix

## Run

```
pip install numpy pandas matplotlib seaborn scipy scikit-learn
python liquidstress_engine.py
```
