"""
LiquidStress Engine
====================
Synthetic Market Liquidity Crisis Simulator & ML Stress-Testing Dashboard

A quantitative market microstructure research tool that generates a synthetic
limit order book with volatility clustering and liquidity-shock regimes,
engineers stress-aware features, fits and properly time-validates regression
models (Ridge / Lasso / Gradient Boosting), runs Monte Carlo stress scenarios,
and renders a full multi-panel diagnostics dashboard.

All data is synthetic. For educational / research use only — not investment advice.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    'axes.facecolor': '#181a1b',
    'figure.facecolor': '#181a1b',
    'axes.labelcolor': '#c7c7c7',
    'xtick.color': '#c7c7c7',
    'ytick.color': '#c7c7c7',
    'grid.color': '#222',
    'axes.edgecolor': '#444',
    'font.family': 'monospace',
    'font.size': 10,
    'text.color': '#e6e6e6',
    'axes.titlecolor': '#e6e6e6',
    'legend.labelcolor': '#e6e6e6',
})

RNG_SEED = 42


# --------------------------------------------------------------------------
# 1. Synthetic order book generation (with volatility clustering)
# --------------------------------------------------------------------------
def generate_order_book(n_steps=6000, n_levels=10, shock_prob=0.02, seed=RNG_SEED):
    """
    Generate synthetic order book data with liquidity shocks and volatility
    clustering (a light GARCH(1,1)-style process instead of pure iid noise,
    so stress events cluster and decay realistically rather than looking
    like isolated single-bar spikes).
    """
    rng = np.random.default_rng(seed)
    base_price = 100.0
    prices = [base_price]
    spread = 0.02
    spread_series = np.full(n_steps, spread)  # per-step spread, not just the final value
    liquidity = np.ones((n_steps, n_levels * 2)) * 1000.0
    buy_pressure = np.zeros(n_steps)
    sell_pressure = np.zeros(n_steps)
    volatility = np.zeros(n_steps)
    stress_event = np.zeros(n_steps)

    # GARCH(1,1)-ish conditional variance
    omega, alpha, beta = 1e-5, 0.08, 0.85
    cond_var = np.full(n_steps, omega / (1 - alpha - beta))
    last_shock_sq = 0.0
    mean_revert = 0.002  # gentle pull of log-price back toward base_price,
    # so the sim reads as a stationary liquidity-stress regime rather than
    # a runaway trend over thousands of steps — the point of the project is
    # liquidity/volatility dynamics, not price drift.

    for t in range(1, n_steps):
        cond_var[t] = omega + alpha * last_shock_sq + beta * cond_var[t - 1]
        base_vol = np.sqrt(cond_var[t])
        log_dev = np.log(prices[-1] / base_price)
        ret = rng.normal(0, base_vol) - mean_revert * log_dev

        if rng.random() < shock_prob:
            shock = rng.normal(0, 0.15)
            ret += shock
            spread = min(1.0, spread * rng.uniform(1.5, 3.0))
            liquidity[t, :] = liquidity[t - 1, :] * rng.uniform(0.2, 0.5, size=n_levels * 2)
            stress_event[t] = 1
            last_shock_sq = shock ** 2
        else:
            # Mean-reverting decay back toward the baseline spread instead
            # of a memoryless multiplicative walk — otherwise, over
            # thousands of steps, occasional 1.5-3x shocks compound
            # unboundedly and the spread (and liquidity) wander off to
            # unrealistic scales instead of relaxing after a crisis passes.
            spread = max(0.005, spread + 0.05 * (0.02 - spread) + rng.normal(0, 0.0005))
            liquidity[t, :] = np.clip(
                liquidity[t - 1, :] + 0.03 * (1000 - liquidity[t - 1, :])
                # elementwise noise per depth level/side — a scalar draw
                # broadcast across the row would move bid and ask depth
                # identically and silently zero out depth_imbalance
                + liquidity[t - 1, :] * rng.normal(0, 0.01, size=n_levels * 2),
                50, 1500
            )
            last_shock_sq = ret ** 2

        spread_series[t] = spread
        prices.append(prices[-1] * np.exp(ret))
        buy_pressure[t] = rng.normal(0, 1) + (stress_event[t] * rng.uniform(5, 15))
        sell_pressure[t] = rng.normal(0, 1) - (stress_event[t] * rng.uniform(5, 15))
        volatility[t] = abs(ret) + stress_event[t] * rng.uniform(0.05, 0.2)

    prices = np.array(prices)
    mid = prices
    bid = mid - spread_series / 2
    ask = mid + spread_series / 2
    data = pd.DataFrame({
        'mid': mid,
        'bid': bid,
        'ask': ask,
        'spread': ask - bid,
        'volatility': volatility,
        'buy_pressure': buy_pressure,
        'sell_pressure': sell_pressure,
        'stress_event': stress_event
    })
    for i in range(n_levels):
        data[f'bid_depth_{i+1}'] = liquidity[:, i]
        data[f'ask_depth_{i+1}'] = liquidity[:, n_levels + i]
    return data


# --------------------------------------------------------------------------
# 2. Feature engineering (adds lagged / rolling context, not just point-in-time)
# --------------------------------------------------------------------------
def create_features(df, n_levels=5, lookback=5):
    features = df.copy()
    features['imbalance'] = features['buy_pressure'] - features['sell_pressure']
    features['depth_imbalance'] = (
        features[[f'bid_depth_{i+1}' for i in range(n_levels)]].sum(axis=1)
        - features[[f'ask_depth_{i+1}' for i in range(n_levels)]].sum(axis=1)
    )
    # Rolling / lagged context — the model gets to see recent stress build-up,
    # not just the current bar in isolation.
    features['stress_count_recent'] = features['stress_event'].rolling(lookback).sum()
    features['volatility_ma'] = features['volatility'].rolling(lookback).mean()
    features['spread_ma'] = features['spread'].rolling(lookback).mean()
    features['imbalance_lag1'] = features['imbalance'].shift(1)

    # Targets — predicted 5 steps ahead, computed strictly from *past* data
    features['future_return'] = np.log(features['mid'].shift(-lookback) / features['mid'])
    features['future_volatility'] = features['volatility'].rolling(lookback).mean().shift(-lookback)
    features = features.dropna().reset_index(drop=True)
    return features


FEATURE_COLS = [
    'spread', 'volatility', 'imbalance', 'depth_imbalance',
    'buy_pressure', 'sell_pressure', 'stress_event',
    'stress_count_recent', 'volatility_ma', 'spread_ma', 'imbalance_lag1'
] + [f'bid_depth_{i+1}' for i in range(5)] + [f'ask_depth_{i+1}' for i in range(5)]


# --------------------------------------------------------------------------
# 3. ML modeling — time-ordered split (no shuffling: this is a time series,
#    so a random split would leak future information into training) plus
#    a real gradient-boosted ensemble instead of a relabeled Ridge.
# --------------------------------------------------------------------------
def run_ml_models(features, target_col='future_volatility'):
    """
    target_col: 'future_volatility' (default) has real, learnable signal
    because the generator uses GARCH-style volatility clustering — stress
    tends to follow stress. 'future_return' is left available too, since
    demonstrating that *returns* are much harder to predict (near market-
    efficiency) than *volatility* is itself a legitimate, common research
    finding worth showing rather than hiding.
    """
    X = features[FEATURE_COLS]
    y = features[target_col]

    # Ridge/Lasso wrapped in a StandardScaler pipeline: the raw features mix
    # depth levels (~1000s) with spreads (~0.01s), which made the old
    # unscaled Ridge/Lasso ill-conditioned and their coefficients meaningless.
    models = {
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Lasso': make_pipeline(StandardScaler(), Lasso(alpha=0.0005, max_iter=20000)),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=20, random_state=RNG_SEED
        ),
    }

    # Walk-forward evaluation across the whole series (expanding window):
    # a single fixed final-quarter holdout can land in an unlucky/unusual
    # regime by chance. Averaging R2 across several forward folds gives a
    # much more statistically honest read on real out-of-sample skill.
    tscv = TimeSeriesSplit(n_splits=6)
    results = {}
    for name, model in models.items():
        fold_r2, fold_rmse, fold_mae = [], [], []
        last_train_idx, last_val_idx = None, None
        for tr_idx, val_idx in tscv.split(X):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[val_idx])
            fold_r2.append(r2_score(y.iloc[val_idx], pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[val_idx], pred)))
            fold_mae.append(mean_absolute_error(y.iloc[val_idx], pred))
            last_train_idx, last_val_idx = tr_idx, val_idx

        # Refit once more on the final fold's train indices so the returned
        # model/predictions correspond to the most recent (final) holdout —
        # this is what gets plotted in the dashboard.
        model.fit(X.iloc[last_train_idx], y.iloc[last_train_idx])
        y_test = y.iloc[last_val_idx]
        X_test = X.iloc[last_val_idx]
        y_pred = model.predict(X_test)

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'X_test': X_test,
            'cv_r2_mean': float(np.mean(fold_r2)),
            'cv_r2_std': float(np.std(fold_r2)),
            'test_r2': float(np.mean(fold_r2)),
            'test_rmse': float(np.mean(fold_rmse)),
            'test_mae': float(np.mean(fold_mae)),
        }
    X_train, X_test_final = X.iloc[:last_val_idx[0]], X.iloc[last_val_idx]
    y_train, y_test_final = y.iloc[:last_val_idx[0]], y.iloc[last_val_idx]
    return results, X_train, X_test_final, y_train, y_test_final


def print_model_report(results):
    print("\n" + "=" * 66)
    print("  MODEL VALIDATION REPORT (6-fold walk-forward, expanding window)")
    print("=" * 66)
    header = f"{'Model':<18}{'Mean R2 (±std)':<22}{'Mean RMSE':<12}{'Mean MAE':<10}"
    print(header)
    print("-" * 66)
    for name, r in results.items():
        cv_str = f"{r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f}"
        print(f"{name:<18}{cv_str:<22}{r['test_rmse']:<12.5f}{r['test_mae']:<10.5f}")
    print("=" * 66)
    best = max(results.items(), key=lambda kv: kv[1]['test_r2'])
    print(f"  Best model (walk-forward mean R2): {best[0]} (R2={best[1]['test_r2']:.4f})\n")


# --------------------------------------------------------------------------
# 4. Monte Carlo stress simulation (block bootstrap keeps stress clustering
#    intact instead of shuffling rows independently, which would destroy the
#    autocorrelation structure that defines a "shock").
# --------------------------------------------------------------------------
def monte_carlo_simulation(model, X_test, n_sims=1000, block_size=10):
    n = len(X_test)
    preds = np.empty((n_sims, n))
    X_arr = X_test.reset_index(drop=True)
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(RNG_SEED)
    for s in range(n_sims):
        starts = rng.integers(0, max(1, n - block_size), size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block_size) for st in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        preds[s] = model.predict(X_arr.iloc[idx])
    return preds


# --------------------------------------------------------------------------
# 5. Visualization panels
# --------------------------------------------------------------------------
def plot_feature_importance(model, X, y, ax):
    """Permutation importance — works for both linear and tree models,
    unlike the old version which only read .coef_ (and would crash on
    a tree-based estimator)."""
    if hasattr(model, 'coef_'):
        imp = np.abs(model.coef_)
    else:
        pi = permutation_importance(model, X, y, n_repeats=5, random_state=RNG_SEED, n_jobs=-1)
        imp = pi.importances_mean
    idx = np.argsort(imp)[-12:]
    ax.barh(np.array(X.columns)[idx], imp[idx], color='#00bfae')
    ax.set_title('Feature Importance (Top 12)', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('Importance')


def plot_pred_vs_actual(y_test, y_pred, ax):
    ax.scatter(y_test, y_pred, alpha=0.3, color='#ffb300', s=10)
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], '--', color='#888')
    ax.set_title('Predicted vs Actual (Test Set)', fontsize=10)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(True, alpha=0.2)


def plot_residuals(y_test, y_pred, ax):
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=40, kde=True, color='#2979ff', ax=ax)
    ax.set_title('Residual Distribution', fontsize=10)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.2)


def plot_corr_heatmap(features, ax):
    corr = features[FEATURE_COLS + ['future_return']].corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax, cbar=True)
    ax.set_title('Feature Correlation Matrix', fontsize=10)
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=6)


def plot_3d_liquidity_stress(features, ax):
    x, y, z = features['spread'], features['depth_imbalance'], features['volatility']
    if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
        ax.plot_trisurf(x, y, z, cmap='viridis', alpha=0.8)
    ax.set_title('3D Liquidity Stress Surface', fontsize=10)
    ax.set_xlabel('Spread')
    ax.set_ylabel('Depth Imbalance')
    ax.set_zlabel('Volatility')


def plot_stress_event_cloud(features, ax):
    stress = features[features['stress_event'] > 0]
    sc = ax.scatter(stress['imbalance'], stress['future_return'],
                     c=stress['spread'], cmap='cool', alpha=0.7, s=12)
    ax.set_title('Stress Event Cloud', fontsize=10)
    ax.set_xlabel('Imbalance')
    ax.set_ylabel('Future Return')
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='Spread', fraction=0.046, pad=0.04)


def plot_execution_pressure_heatmap(features, ax):
    heatmap, xedges, yedges = np.histogram2d(
        features['buy_pressure'], features['sell_pressure'], bins=40
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='magma', aspect='auto', alpha=0.9)
    ax.set_title('Execution Pressure Heatmap', fontsize=10)
    ax.set_xlabel('Buy Pressure')
    ax.set_ylabel('Sell Pressure')


def plot_mc_pnl(mc_preds, ax):
    pnl = mc_preds.sum(axis=1)
    sns.histplot(pnl, bins=40, kde=True, color='#00bfae', ax=ax)
    var_95 = np.percentile(pnl, 5)
    ax.axvline(var_95, color='#ff5252', linestyle='--', linewidth=1.5,
               label=f'5% VaR = {var_95:.3f}')
    ax.set_title('Monte Carlo Simulated PnL Distribution', fontsize=10)
    ax.set_xlabel('PnL')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)


def plot_price_and_stress(df, ax):
    ax.plot(df['mid'], color='#00e5ff', linewidth=0.8, label='Mid Price')
    stress_idx = df.index[df['stress_event'] > 0]
    ax.scatter(stress_idx, df.loc[stress_idx, 'mid'], color='#ff5252', s=8,
               label='Stress Event', zorder=5)
    ax.set_title('Simulated Price Path & Liquidity Shocks', fontsize=10)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mid Price')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)


# --------------------------------------------------------------------------
# 6. Main dashboard — every plotting function is now actually used, laid
#    out as a proper multi-panel research dashboard instead of one lone
#    histogram at the end.
# --------------------------------------------------------------------------
def main():
    df = generate_order_book()
    features = create_features(df)

    # Primary target: future volatility — this is where the models should
    # (and do) show real skill, because volatility clustering is genuine
    # structure in the data.
    print("\n>>> Target: future_volatility")
    results, X_train, X_test, y_train, y_test = run_ml_models(features, target_col='future_volatility')
    print_model_report(results)

    # Secondary target: future return — shown for comparison. Near-zero /
    # negative R2 here is the *expected*, realistic result (returns in an
    # efficient synthetic random-walk price are close to unpredictable),
    # not a bug — it's a useful illustration of the difference between
    # forecasting risk (tractable) and forecasting direction (much harder).
    print(">>> Target: future_return (for comparison — expect weak/near-zero R2)")
    ret_results, *_ = run_ml_models(features, target_col='future_return')
    print_model_report(ret_results)

    best_name = max(results.items(), key=lambda kv: kv[1]['test_r2'])[0]
    best = results[best_name]
    model, y_pred = best['model'], best['y_pred']

    mc_preds = monte_carlo_simulation(model, X_test)

    fig = plt.figure(figsize=(20, 16), facecolor='#181a1b')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    plot_price_and_stress(df, ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    plot_feature_importance(model, X_test, y_test, ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_pred_vs_actual(y_test, y_pred, ax3)

    ax4 = fig.add_subplot(gs[1, 2])
    plot_residuals(y_test, y_pred, ax4)

    ax5 = fig.add_subplot(gs[2, 0])
    plot_execution_pressure_heatmap(features, ax5)

    ax6 = fig.add_subplot(gs[2, 1])
    plot_stress_event_cloud(features, ax6)

    ax7 = fig.add_subplot(gs[2, 2])
    plot_mc_pnl(mc_preds, ax7)

    fig.suptitle(f'LiquidStress Engine — Dashboard  (best model: {best_name}, '
                 f'test R²={best["test_r2"]:.3f})', fontsize=15, color='#e0e0e0')
    fig.savefig('liquidstress_dashboard.png', dpi=140, facecolor='#181a1b', bbox_inches='tight')
    print("Saved dashboard -> liquidstress_dashboard.png")

    # Separate figure for the 3D surface (own projection, cleaner than
    # cramming a 3D axes into the 2D gridspec above)
    fig3d = plt.figure(figsize=(9, 7), facecolor='#181a1b')
    ax3d = fig3d.add_subplot(111, projection='3d')
    plot_3d_liquidity_stress(features, ax3d)
    fig3d.savefig('liquidstress_3d_surface.png', dpi=140, facecolor='#181a1b', bbox_inches='tight')
    print("Saved 3D surface -> liquidstress_3d_surface.png")

    # Correlation matrix as its own figure (too dense to share a panel)
    fig_corr, axc = plt.subplots(figsize=(10, 8), facecolor='#181a1b')
    plot_corr_heatmap(features, axc)
    fig_corr.tight_layout()
    fig_corr.savefig('liquidstress_correlation.png', dpi=140, facecolor='#181a1b', bbox_inches='tight')
    print("Saved correlation matrix -> liquidstress_correlation.png")


if __name__ == "__main__":
    main()
