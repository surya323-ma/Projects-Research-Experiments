"""
RegimeLens Engine
==================
Unsupervised Market Regime Detection & Validation Dashboard

A quantitative research tool that generates synthetic market data covering
five regimes (trend, mean-reversion, panic, crash, high-volatility), fits
unsupervised regime-detection models (Gaussian Mixture, KMeans), validates
detected regimes against the known ground truth, and renders a full
multi-panel diagnostics dashboard including a regime-conditional Monte
Carlo simulation.

All data is synthetic. For educational / research use only — not investment advice.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RNG_SEED = 42
REGIMES = ['trend', 'mean_revert', 'panic', 'crash', 'high_vol']
REGIME_PARAMS = {
    'trend':       (0.05, 0.8),
    'mean_revert': (0.0, 0.5),
    'panic':       (-0.1, 2.0),
    'crash':       (-0.2, 3.0),
    'high_vol':    (0.0, 2.5),
}


# --------------------------------------------------------------------------
# 1. Synthetic market data generation
# --------------------------------------------------------------------------
def generate_synthetic_market_data(n_steps=3000, seed=RNG_SEED):
    """
    Builds a synthetic price series that cycles through all five regimes.

    The original generator picked a random sequence of ~10 segments and then
    truncated to n_steps — with the fixed seed, that truncation happened to
    drop the 'trend' regime entirely (0 samples), so every model downstream
    was silently being trained/evaluated on only 4 of the 5 advertised
    regimes. This version tiles each regime multiple times before shuffling
    and generates enough total length to guarantee every regime survives
    the truncation, with a runtime check that raises if it somehow doesn't.
    """
    rng = np.random.default_rng(seed)
    reps_per_regime = 4
    regime_sequence = list(REGIMES) * reps_per_regime
    rng.shuffle(regime_sequence)

    data = [100.0]  # start at an index level of 100, read as a price series
    regime_labels = []
    for reg in regime_sequence:
        length = int(rng.choice([150, 200, 250, 300]))
        mu, sigma = REGIME_PARAMS[reg]
        segment = np.cumsum(rng.normal(mu, sigma, length)) + data[-1]
        data.extend(segment.tolist())
        regime_labels.extend([reg] * length)
        if len(regime_labels) >= n_steps:
            break

    data = np.array(data[1:n_steps + 1])
    regime_labels = np.array(regime_labels[:n_steps])
    missing = set(REGIMES) - set(regime_labels)
    if missing:
        raise RuntimeError(f"Regime coverage bug: {missing} never appeared in {n_steps} steps")

    returns = np.diff(data, prepend=data[0])
    volatility = pd.Series(returns).rolling(20).std().bfill().values
    df = pd.DataFrame({
        'price': data, 'returns': returns,
        'volatility': volatility, 'regime_true': regime_labels
    })
    return df


# --------------------------------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------------------------------
def create_features(df):
    df = df.copy()
    df['ma_10'] = df['price'].rolling(10).mean()
    df['ma_50'] = df['price'].rolling(50).mean()
    df['momentum'] = df['price'] - df['price'].shift(10)
    df['vol_rolling'] = df['returns'].rolling(20).std()
    df = df.bfill().fillna(0)
    return df


FEATURE_COLS = ['returns', 'volatility', 'ma_10', 'ma_50', 'momentum', 'vol_rolling']


# --------------------------------------------------------------------------
# 3. Regime detection models
# --------------------------------------------------------------------------
def fit_gmm(X, n_components=5):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                           random_state=RNG_SEED, n_init=5)
    gmm.fit(X)
    return gmm


def fit_kmeans(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=RNG_SEED, n_init=10)
    kmeans.fit(X)
    return kmeans


def fit_pca(X, n_components=3):
    pca = PCA(n_components=n_components, random_state=RNG_SEED)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def select_gmm_components(X, max_k=8):
    """BIC-based model-order selection — the original script just hardcoded
    n_components=5 because that happens to match the number of simulated
    regimes, which a real detector wouldn't know in advance. This reports
    what an unsupervised BIC sweep would actually pick."""
    bics = []
    for k in range(2, max_k + 1):
        g = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=RNG_SEED, n_init=3).fit(X)
        bics.append((k, g.bic(X)))
    best_k = min(bics, key=lambda kv: kv[1])[0]
    return bics, best_k


def classify_regimes_gmm(gmm, X):
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    confidence = probs.max(axis=1)
    return labels, confidence, probs


def classify_regimes_kmeans(kmeans, X):
    labels = kmeans.predict(X)
    distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
    confidence = 1 - (distances / (distances.max() + 1e-6))
    return labels, confidence


# --------------------------------------------------------------------------
# 4. Validation against ground truth — the original script generated
#    `regime_true` and then never used it anywhere. Cluster IDs are
#    arbitrary, so we align each cluster to its majority true regime
#    before scoring — this is the only honest way to report accuracy for
#    an unsupervised method.
# --------------------------------------------------------------------------
def align_clusters_to_labels(cluster_labels, true_labels):
    mapping = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        majority = pd.Series(true_labels[mask]).mode().iloc[0]
        mapping[c] = majority
    aligned = np.array([mapping[c] for c in cluster_labels])
    return aligned, mapping


def validate_clustering(name, cluster_labels, true_labels):
    aligned, mapping = align_clusters_to_labels(cluster_labels, true_labels)
    accuracy = float(np.mean(aligned == true_labels))
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    print(f"\n{name} validation vs. ground-truth regimes:")
    print(f"  Cluster -> majority-regime mapping: {mapping}")
    print(f"  Majority-vote accuracy : {accuracy:.3f}")
    print(f"  Adjusted Rand Index    : {ari:.3f}")
    print(f"  Normalized Mutual Info : {nmi:.3f}")
    return {'accuracy': accuracy, 'ari': ari, 'nmi': nmi, 'mapping': mapping}


# --------------------------------------------------------------------------
# 5. Regime-conditional Monte Carlo — the original panel just drew ten
#    unrelated plain random walks with fixed N(0,1) noise, disconnected
#    from anything the models fit. This version bootstraps from the
#    detector's own fitted per-regime transition matrix and per-regime
#    return statistics, so the simulation actually reflects what the model
#    learned about regime persistence and regime-conditional volatility.
# --------------------------------------------------------------------------
def regime_conditional_monte_carlo(gmm_labels, returns, n_sims=15, horizon=None):
    horizon = horizon or len(returns)
    n_states = gmm_labels.max() + 1

    # Empirical transition matrix between detected states
    trans = np.zeros((n_states, n_states))
    for i in range(1, len(gmm_labels)):
        trans[gmm_labels[i - 1], gmm_labels[i]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans = np.divide(trans, row_sums, out=np.full_like(trans, 1.0 / n_states), where=row_sums != 0)

    # Empirical return distribution per detected state (bootstrap pool)
    state_returns = {s: returns[gmm_labels == s] for s in range(n_states)
                      if np.any(gmm_labels == s)}

    rng = np.random.default_rng(RNG_SEED)
    sims = np.zeros((n_sims, horizon))
    for s in range(n_sims):
        state = rng.choice(list(state_returns.keys()))
        path = np.zeros(horizon)
        level = 0.0
        for t in range(horizon):
            pool = state_returns.get(state)
            level += rng.choice(pool) if pool is not None and len(pool) else 0.0
            path[t] = level
            if row_sums[state] > 0:
                state = rng.choice(n_states, p=trans[state])
        sims[s] = path
    return sims


# --------------------------------------------------------------------------
# 6. Visualization
# --------------------------------------------------------------------------
def set_dark_theme():
    plt.style.use('dark_background')
    sns.set_theme(style='darkgrid', rc={'axes.facecolor': '#181818', 'figure.facecolor': '#181818'})
    plt.rcParams.update({
        'axes.labelcolor': '#e6e6e6', 'xtick.color': '#e6e6e6', 'ytick.color': '#e6e6e6',
        'axes.edgecolor': '#555', 'figure.facecolor': '#181818', 'axes.facecolor': '#181818',
        'text.color': '#e6e6e6', 'axes.titlecolor': '#e6e6e6', 'legend.labelcolor': '#e6e6e6',
    })


def plot_dashboard(df, X_pca, gmm, gmm_labels, gmm_conf, gmm_probs, mc_sims, save_path):
    set_dark_theme()
    fig = plt.figure(figsize=(22, 16), facecolor='#181818')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    n_states = gmm_probs.shape[1]

    ax1 = fig.add_subplot(gs[0, 0])
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(1, len(gmm_labels)):
        transition_matrix[gmm_labels[i - 1], gmm_labels[i]] += 1
    sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='mako', ax=ax1, cbar=False)
    ax1.set_title('Regime Transition Counts')
    ax1.set_xlabel('To State')
    ax1.set_ylabel('From State')

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot_trisurf(np.arange(len(df)), df['volatility'], gmm_labels, cmap=cm.viridis, linewidth=0.2)
    ax2.set_title('3D Volatility-Regime Surface')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax2.set_zlabel('State')

    ax3 = fig.add_subplot(gs[0, 2])
    sns.lineplot(x=np.arange(len(gmm_conf)), y=gmm_conf, ax=ax3, color='cyan', linewidth=0.8)
    ax3.set_title('GMM Regime Confidence')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Confidence')

    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='Spectral', alpha=0.7, s=10)
    ax4.set_title('PCA Projection (Detected States)')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax4, label='State')

    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(gmm.means_, annot=True, fmt='.2f', cmap='rocket', ax=ax5,
                xticklabels=FEATURE_COLS, cbar=False)
    ax5.set_title('GMM State Means (Standardized Features)')
    ax5.set_xlabel('Feature')
    ax5.set_ylabel('State')
    ax5.tick_params(axis='x', labelsize=7, rotation=30)

    ax6 = fig.add_subplot(gs[1, 2])
    for i in range(n_states):
        sns.kdeplot(gmm_probs[:, i], label=f'State {i}', ax=ax6)
    ax6.set_title('Regime Probability Distributions')
    ax6.set_xlabel('Probability')
    ax6.set_ylabel('Density')
    ax6.legend(fontsize=7)

    ax7 = fig.add_subplot(gs[2, 0], projection='3d')
    ax7.scatter(np.arange(len(df)), df['volatility'], gmm_labels, c=gmm_labels, cmap='Spectral', alpha=0.7, s=6)
    ax7.set_title('Time vs Volatility vs State')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Volatility')
    ax7.set_zlabel('State')

    ax8 = fig.add_subplot(gs[2, 1])
    for sim in mc_sims:
        ax8.plot(sim, alpha=0.35, color='orange', linewidth=0.8)
    ax8.set_title('Regime-Conditional Monte Carlo Paths')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Simulated Level')

    ax9 = fig.add_subplot(gs[2, 2])
    sns.histplot(df['volatility'], bins=30, color='magenta', ax=ax9, kde=True)
    ax9.set_title('Volatility Distribution')
    ax9.set_xlabel('Volatility')
    ax9.set_ylabel('Frequency')

    fig.suptitle('RegimeLens Engine — Market Regime Detection Dashboard', fontsize=18, color='#e6e6e6', y=0.99)
    fig.savefig(save_path, dpi=140, facecolor='#181818', bbox_inches='tight')
    print(f"Saved dashboard -> {save_path}")


def plot_bic_sweep(bics, best_k, save_path):
    set_dark_theme()
    ks, vals = zip(*bics)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#181818')
    ax.plot(ks, vals, marker='o', color='#00e5ff')
    ax.axvline(best_k, color='#ff5252', linestyle='--', label=f'BIC-selected k={best_k}')
    ax.set_title('GMM Model-Order Selection (BIC)')
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel('BIC (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.savefig(save_path, dpi=140, facecolor='#181818', bbox_inches='tight')
    print(f"Saved BIC sweep -> {save_path}")


# --------------------------------------------------------------------------
# 7. Main pipeline
# --------------------------------------------------------------------------
def main():
    df = generate_synthetic_market_data(n_steps=3000)
    df = create_features(df)
    X_raw = df[FEATURE_COLS].values

    # Standardize before clustering/PCA — the original script fed raw,
    # unscaled features (ma_10/ma_50/momentum are price-scale, ~10-100x
    # larger than returns/volatility), so GMM and KMeans were effectively
    # clustering on price level alone, drowning out the return/volatility
    # signal that actually distinguishes regimes.
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    bics, best_k = select_gmm_components(X, max_k=8)
    print(f"BIC-optimal component count: {best_k} (5 regimes were simulated)")

    gmm = fit_gmm(X, n_components=5)
    kmeans = fit_kmeans(X, n_clusters=5)
    pca, X_pca = fit_pca(X, n_components=3)

    gmm_labels, gmm_conf, gmm_probs = classify_regimes_gmm(gmm, X)
    kmeans_labels, kmeans_conf = classify_regimes_kmeans(kmeans, X)

    true_labels = df['regime_true'].values
    validate_clustering('GMM', gmm_labels, true_labels)
    validate_clustering('KMeans', kmeans_labels, true_labels)

    mc_sims = regime_conditional_monte_carlo(gmm_labels, df['returns'].values, n_sims=15)

    plot_dashboard(df, X_pca, gmm, gmm_labels, gmm_conf, gmm_probs, mc_sims,
                   save_path='regimelens_dashboard.png')
    plot_bic_sweep(bics, best_k, save_path='regimelens_bic_sweep.png')


if __name__ == '__main__':
    main()
