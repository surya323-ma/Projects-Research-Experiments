"""
AlphaScape Engine
==================
Multi-Factor Alpha Landscape Explorer

An experimental quantitative research platform for studying synthetic alpha
factors, predictive relationships, and factor interactions across a panel
of synthetic assets, with time-aware (walk-forward) out-of-sample
validation and an institutional-style visualization dashboard.

All data is synthetic. For educational / research use only — not investment advice.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

RNG_SEED = 42
OUT_DIR = "dashboard_outputs"


def set_dark_theme():
    sns.set_theme(style="darkgrid", palette="dark")
    plt.rcParams.update({
        'axes.facecolor': '#181a1b', 'figure.facecolor': '#181a1b',
        'text.color': '#e6e6e6', 'axes.labelcolor': '#e6e6e6',
        'xtick.color': '#e6e6e6', 'ytick.color': '#e6e6e6',
        'axes.titlecolor': '#e6e6e6', 'legend.labelcolor': '#e6e6e6',
        'font.family': 'monospace',
    })


# --------------------------------------------------------------------------
# 1. Synthetic data generation
# --------------------------------------------------------------------------
def generate_synthetic_data(n_assets=30, n_periods=252, seed=RNG_SEED):
    """
    The original generator was pure iid noise:
    `cumprod(1 + N(0, 0.01))` — a driftless random walk with zero
    autocorrelation and no cross-asset structure by construction. No factor
    computed from that data could ever have real predictive power over
    forward returns; testing against it would always show ~R2=0 regardless
    of whether the modeling pipeline itself was correct.

    This version drives each asset's returns with a slow-moving latent
    trend (an AR(1) process with a ~14-day half-life) plus a gentle pull
    back toward its recent cumulative level, so the 20-day momentum and
    5-day mean-reversion factors each have a genuine, multi-day signal to
    find — not just single-lag noise that a 20-day rolling average would
    wash out.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_periods)
    tickers = [f"Asset_{i+1}" for i in range(n_assets)]

    phi = 0.95          # trend persistence -> ~14-trading-day half-life, matched to the 20-day momentum window
    trend_vol = 0.0015
    idio_vol = 0.008
    mr_coef = 0.02       # gentle pull of level back toward its recent path -> real mean-reversion signal

    trend = np.zeros(n_assets)
    level = np.zeros(n_assets)
    returns = np.zeros((n_periods, n_assets))
    for t in range(n_periods):
        trend = phi * trend + rng.normal(0, trend_vol, n_assets)
        ret = trend - mr_coef * level + rng.normal(0, idio_vol, n_assets)
        level = level + ret
        returns[t] = ret

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    return df


# --------------------------------------------------------------------------
# 2. Alpha factor engineering
# --------------------------------------------------------------------------
def compute_factors(df, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    returns = df.pct_change().fillna(0)
    factors = {
        'momentum': returns.rolling(20).mean(),
        'volatility': returns.rolling(20).std(),
        'mean_reversion': -returns.rolling(5).mean(),
        'liquidity_imbalance': (df.rolling(20).max() - df.rolling(20).min())
                                / (returns.rolling(20).std() + 1e-6),
        'synthetic_sentiment': pd.DataFrame(
            np.cumsum(rng.normal(0, 0.01, df.shape), axis=0), index=df.index, columns=df.columns
        ),
        'order_flow_pressure': pd.DataFrame(
            rng.normal(0, 1, df.shape), index=df.index, columns=df.columns
        ),
    }
    factor_df = pd.concat(
        [factors[f] for f in factors], axis=1, keys=factors.keys()
    )
    return factor_df, list(factors.keys())


# --------------------------------------------------------------------------
# 3. Panel construction — this is the key fix. The original script called
#    `factor_df.dropna().values.reshape(-1, n_factors)` on a (time, factor x
#    asset) frame, which does not respect the actual factor/asset structure
#    of the columns — it just reinterprets the raw memory buffer with a new
#    shape, silently scrambling which values belong to which factor/asset.
#    It then aligned X and y (built from a separately-shaped array) purely
#    by truncating both to `min(len(X), len(y))`, with no guarantee the
#    rows referred to the same (asset, date) at all. Both bugs together
#    meant the model was fit on effectively noise-correlated data.
#
#    Fix: melt into an honest long panel — one row per (date, asset) with
#    real factor columns and a correctly shifted forward-return target,
#    joined on the actual index rather than positional truncation.
# --------------------------------------------------------------------------
def build_long_panel(factor_df, returns, factor_names):
    frames = []
    for asset in returns.columns:
        block = pd.DataFrame(
            {f: factor_df[f][asset] for f in factor_names}, index=factor_df.index
        )
        block['asset'] = asset
        block['forward_return'] = returns[asset].shift(-1)
        frames.append(block)
    long_df = pd.concat(frames)
    long_df.index.name = 'date'
    long_df = long_df.dropna(subset=factor_names + ['forward_return']).reset_index()
    return long_df


# --------------------------------------------------------------------------
# 4. ML modeling — time-aware split. Panel data must be split by date, not
#    by row: a random per-row split would put Asset_7 on 2024-05-01 in
#    training and Asset_3 on the very same date in test, leaking
#    same-day cross-sectional information across the split. The original
#    script did not hold out any data at all — it fit and "evaluated" on
#    the identical rows it trained on, which is why a 100-tree Random
#    Forest would have looked deceptively perfect.
# --------------------------------------------------------------------------
def run_models(long_df, factor_names, test_frac=0.2):
    dates = np.sort(long_df['date'].unique())
    split_date = dates[int(len(dates) * (1 - test_frac))]
    train = long_df[long_df['date'] < split_date]
    test = long_df[long_df['date'] >= split_date]

    X_train, y_train = train[factor_names].values, train['forward_return'].values
    X_test, y_test = test[factor_names].values, test['forward_return'].values

    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=RNG_SEED
        ),
        'RidgeCV': make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            pi = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RNG_SEED, n_jobs=-1)
            importance = pi.importances_mean
        results[name] = {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'importance': importance,
            'r2': r2_score(y_test, y_pred),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': mean_absolute_error(y_test, y_pred),
        }
    return results, train, test, split_date


def print_report(results):
    print("\n" + "=" * 60)
    print("  MODEL VALIDATION REPORT (time-based holdout, no leakage)")
    print("=" * 60)
    print(f"{'Model':<16}{'Test R2':<12}{'Test RMSE':<12}{'Test MAE':<12}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<16}{r['r2']:<12.4f}{r['rmse']:<12.5f}{r['mae']:<12.5f}")
    print("=" * 60)
    best = max(results.items(), key=lambda kv: kv[1]['r2'])
    print(f"  Best model on held-out test set: {best[0]} (R2={best[1]['r2']:.4f})\n")


# --------------------------------------------------------------------------
# 5. Dimensionality reduction — standardized (the original PCA ran on raw
#    factor scales, where liquidity_imbalance is orders of magnitude larger
#    than momentum/volatility, so PCA was effectively just PC1 = liquidity
#    imbalance). t-SNE is now actually run (on a capped subsample for
#    speed) instead of being disabled while still being listed as a feature.
# --------------------------------------------------------------------------
def run_dimensionality_reduction(long_df, factor_names, tsne_sample=1500):
    X = StandardScaler().fit_transform(long_df[factor_names].values)
    pca = PCA(n_components=2, random_state=RNG_SEED)
    X_pca = pca.fit_transform(X)

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(X), size=min(tsne_sample, len(X)), replace=False)
    tsne = TSNE(n_components=2, perplexity=30, random_state=RNG_SEED, init='pca')
    X_tsne = tsne.fit_transform(X[idx])
    return X_pca, X_tsne, idx, pca


# --------------------------------------------------------------------------
# 6. Alpha decay — actually wired in now. The original defined this
#    function and never called it anywhere in main() or the dashboard,
#    despite "Alpha decay simulation" being an advertised feature.
# --------------------------------------------------------------------------
def simulate_alpha_decay(factor_series, half_life=20):
    decay = np.exp(-np.arange(len(factor_series)) / half_life)
    return factor_series * decay


# --------------------------------------------------------------------------
# 7. Dashboard
# --------------------------------------------------------------------------
def plot_dashboard(long_df, factor_names, results, X_pca, X_tsne, tsne_idx, df):
    set_dark_theme()
    os.makedirs(OUT_DIR, exist_ok=True)
    best_name = max(results.items(), key=lambda kv: kv[1]['r2'])[0]
    best = results[best_name]

    # Feature importance (best model)
    plt.figure(figsize=(7, 5))
    order = np.argsort(best['importance'])
    plt.barh(np.array(factor_names)[order], best['importance'][order], color='#00bfae')
    plt.title(f'Feature Importance ({best_name}, test-set)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png")
    plt.close()

    # Predicted vs actual (held-out test set only)
    plt.figure(figsize=(7, 5))
    plt.scatter(best['y_test'], best['y_pred'], alpha=0.25, color='cyan', s=10)
    lo, hi = best['y_test'].min(), best['y_test'].max()
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.title(f'Predicted vs Actual Forward Returns ({best_name}, test-set)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/predicted_vs_actual.png")
    plt.close()

    # Residuals
    plt.figure(figsize=(7, 5))
    residuals = best['y_test'] - best['y_pred']
    sns.histplot(residuals, bins=50, color='orange', kde=True)
    plt.title('Residual Error Distribution (test-set)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/residual_error_distribution.png")
    plt.close()

    # Factor-factor correlation (proper interaction matrix — averaged
    # across the whole panel, not just one factor's cross-asset structure)
    plt.figure(figsize=(7, 6))
    corr = long_df[factor_names].corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('Factor-Factor Correlation (panel-wide)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/factor_correlation_heatmap.png")
    plt.close()

    # Cross-asset correlation of a single factor (momentum)
    plt.figure(figsize=(7, 6))
    mom_wide = long_df.pivot(index='date', columns='asset', values='momentum')
    sns.heatmap(mom_wide.corr(), cmap='coolwarm', center=0)
    plt.title('Cross-Asset Correlation — Momentum Factor')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cross_asset_correlation_heatmap.png")
    plt.close()

    # 3D factor landscape (PCA space vs out-of-sample prediction, test rows only)
    test_mask = long_df['date'] >= long_df['date'].sort_values().unique()[
        int(len(long_df['date'].unique()) * 0.8)]
    X_pca_test = X_pca[test_mask.values]
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X_pca_test[:, 0], X_pca_test[:, 1], best['y_pred'], cmap='viridis', alpha=0.8)
    ax.set_title(f'3D Factor Landscape (PCA vs {best_name} prediction, test-set)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Predicted forward return')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/3d_factor_landscape.png")
    plt.close()

    # t-SNE signal strength contour (now actually computed, not disabled)
    plt.figure(figsize=(7, 5))
    sns.kdeplot(x=X_tsne[:, 0], y=X_tsne[:, 1], fill=True, cmap='mako')
    plt.title(f't-SNE Factor-Space Density (n={len(tsne_idx)} sample)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/tsne_signal_density.png")
    plt.close()

    # Alpha decay demonstration (real function call, on one asset's momentum)
    plt.figure(figsize=(7, 5))
    sample_asset = df.columns[0]
    mom_series = long_df[long_df['asset'] == sample_asset].sort_values('date')['momentum'].values
    decayed = simulate_alpha_decay(mom_series, half_life=20)
    plt.plot(mom_series, label='Raw momentum factor', color='#00e5ff', linewidth=1)
    plt.plot(decayed, label='After alpha decay (half-life=20)', color='#ff5252', linewidth=1)
    plt.title(f'Alpha Decay Simulation — {sample_asset}')
    plt.xlabel('Time step')
    plt.ylabel('Factor value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/alpha_decay_simulation.png")
    plt.close()

    # Out-of-sample PnL simulation (test predictions only, not fitted values)
    plt.figure(figsize=(7, 5))
    test_sorted = long_df[test_mask].assign(pred=best['y_pred']).groupby('date')['pred'].mean()
    plt.plot(np.cumsum(test_sorted.values), color='lime')
    plt.title(f'Out-of-Sample PnL Simulation ({best_name}, equal-weight daily signal)')
    plt.xlabel('Test-period time step')
    plt.ylabel('Cumulative predicted return')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pnl_simulation.png")
    plt.close()

    # Monte Carlo — block bootstrap of the out-of-sample predicted-return
    # series, not the original's simple iid resample
    plt.figure(figsize=(7, 5))
    rng = np.random.default_rng(RNG_SEED)
    series = test_sorted.values
    block = 5
    for _ in range(15):
        n_blocks = int(np.ceil(len(series) / block))
        starts = rng.integers(0, max(1, len(series) - block), size=n_blocks)
        path = np.concatenate([series[s:s + block] for s in starts])[:len(series)]
        plt.plot(np.cumsum(path), alpha=0.3, color='orange')
    plt.title('Monte Carlo Simulation (block bootstrap of OOS signal)')
    plt.xlabel('Time step')
    plt.ylabel('Cumulative return')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/monte_carlo_simulation.png")
    plt.close()

    # Statistical summary
    plt.figure(figsize=(7, 5))
    lines = [f"Best model: {best_name}"]
    for name, r in results.items():
        lines.append(f"{name:<13} R2={r['r2']:.4f}  RMSE={r['rmse']:.5f}  MAE={r['mae']:.5f}")
    plt.text(0.05, 0.5, "\n".join(lines), fontsize=12, color='#e6e6e6', fontfamily='monospace')
    plt.axis('off')
    plt.title('Statistical Summary (held-out test set)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/statistical_summary.png")
    plt.close()

    print(f"All dashboard plots saved to the '{OUT_DIR}' folder.")


# --------------------------------------------------------------------------
# 8. Main
# --------------------------------------------------------------------------
def main():
    df = generate_synthetic_data(n_assets=30, n_periods=252)
    factor_df, factor_names = compute_factors(df)
    returns = df.pct_change().fillna(0)

    long_df = build_long_panel(factor_df, returns, factor_names)
    print(f"Panel: {long_df['asset'].nunique()} assets x "
          f"{long_df['date'].nunique()} dates = {len(long_df)} (asset, date) rows")

    results, train, test, split_date = run_models(long_df, factor_names)
    print_report(results)

    X_pca, X_tsne, tsne_idx, pca = run_dimensionality_reduction(long_df, factor_names)
    plot_dashboard(long_df, factor_names, results, X_pca, X_tsne, tsne_idx, df)


if __name__ == "__main__":
    main()
