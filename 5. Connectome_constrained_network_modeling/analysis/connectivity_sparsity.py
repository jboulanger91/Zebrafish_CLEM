import pickle
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Paths
path_models_sparse = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data\results\freepop\mask_traces_freepop16_loadtop0\RNNFreePop_neurons102_tau0.1_input2step_softplus\top_5")
path_models_dense = Path(r"C:\Users\Roberto\Desktop\highlights\clem_rnns\data\results\mask_traces_freeneurons_2_nosparsity\RNNFreeNeurons_neurons102_tau0.1_input2step_softplus\top_5")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_W_numpy(model, which="full", apply_mask=False):
    """
    Extract W as numpy array.
    If apply_mask=True, entries where mask_W == 0 are set to NaN,
    so downstream functions can use np.nanmean / np.nansum etc.
    """
    with torch.no_grad():
        if which == "fast":
            W = model.W_fast().detach().cpu().numpy()
        elif which == "full":
            W = model.W().detach().cpu().numpy()
        elif which == "mask":
            W = model.mask_W.detach().cpu().numpy() * model.signs.view(1, -1).detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown which='{which}'")

        if apply_mask:
            mask = model.mask_W.detach().cpu().numpy()
            W = W.copy()
            W[mask == 0] = np.nan  # flag non-existing synapses as NaN

    return W



def effective_rank(sv):
    """Participation ratio of singular values (effective rank)."""
    sv = np.array(sv)
    sv2 = sv ** 2
    s = sv2 / sv2.sum()
    return 1.0 / (s ** 2).sum()


def principal_angles(A, B, k=4):
    """
    Principal angles between column space of A[:,0:k] and B[:,0:k].
    Returns cosines of principal angles (1 = perfectly aligned).
    """
    Qa, _ = np.linalg.qr(A[:, :k])
    Qb, _ = np.linalg.qr(B[:, :k])
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(sv, 0, 1)


def linear_cka(W1, W2):
    """
    Linear CKA between two matrices treated as sets of features.
    CKA(X,Y) = ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    """
    K1 = W1 @ W1.T
    K2 = W2 @ W2.T
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H
    num = np.sum(K1c * K2c)
    denom = np.sqrt(np.sum(K1c * K1c) * np.sum(K2c * K2c))
    return float(num / (denom + 1e-12))


def _to_dense_for_matrix_ops(W):
    """Replace NaN (absent synapses) with 0 for matrix-level operations."""
    W = W.copy()
    W[np.isnan(W)] = 0.0
    return W


# ──────────────────────────────────────────────
# Analysis 1: Block-average heatmap
# ──────────────────────────────────────────────

def plot_block_heatmap(
    model,
    which="full",
    pop_names=None,
    title="Block-average |W|",
    signed=False,
    ax=None,
    show=True,
    apply_mask=False,
    vrange=None
):
    W = get_W_numpy(model, which=which, apply_mask=apply_mask)
    population_indices = model.population_indices
    n_pops = len(population_indices)

    B = np.zeros((n_pops, n_pops))
    for i, idx_i in enumerate(population_indices):
        for j, idx_j in enumerate(population_indices):
            sub = W[np.ix_(idx_i, idx_j)]
            # nanmean/nansum ignores absent synapses when apply_mask=True
            B[i, j] = np.nanmean(sub) if signed else np.nanmean(np.abs(sub))

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    if vrange is None:
        vmin = None; vmax = None
    else:
        vmin = vrange[0]
        vmax = vrange[1]
    im = ax.imshow(B, cmap="RdBu_r" if signed else "hot_r", aspect="auto", vmin=vmin, vmax=vmax)
    if pop_names:
        ax.set_xticks(range(n_pops)); ax.set_xticklabels(pop_names, rotation=45, ha="right")
        ax.set_yticks(range(n_pops)); ax.set_yticklabels(pop_names)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    if created_fig and show:
        plt.show()

    return B, fig


# ──────────────────────────────────────────────
# Analysis 2: Singular value spectrum
# ──────────────────────────────────────────────

def plot_sv_spectrum(
    models,
    labels=None,
    k_show=30,
    title="Singular value spectrum",
    which="fast",
    show=True,
    apply_mask=False,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, model in enumerate(models):
        W = get_W_numpy(model, which=which, apply_mask=apply_mask)
        W = _to_dense_for_matrix_ops(W)  # NaN -> 0 for SVD
        sv = np.linalg.svd(W, compute_uv=False)
        er = effective_rank(sv)
        label = (labels[i] if labels else f"model_{i}") + f" (eff.rank={er:.1f})"
        ax.plot(sv[:k_show], marker='o', markersize=3, label=label)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig


# ──────────────────────────────────────────────
# Analysis 3: SVD mode alignment across solutions
# ──────────────────────────────────────────────

def sv_alignment_matrix(models, k=4, which="fast", apply_mask=False):
    n = len(models)
    Us = []
    for m in models:
        W = get_W_numpy(m, which=which, apply_mask=apply_mask)
        W = _to_dense_for_matrix_ops(W)  # NaN -> 0 for SVD
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        Us.append(U)

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_angles = principal_angles(Us[i], Us[j], k=k)
            A[i, j] = cos_angles.mean()

    return A

def plot_alignment_matrix(
    A,
    labels=None,
    title="SVD subspace alignment",
    show=True,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, vmin=0, vmax=1, cmap="viridis")
    if labels:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()

    if show:
        plt.show()

    return fig

# ──────────────────────────────────────────────
# Analysis 4: Block E/I signed weight pattern
# ──────────────────────────────────────────────

def plot_block_EI(
    model,
    which="full",
    pop_names=None,
    title="Net signed weight per block",
    show=True,
    apply_mask=False,
):
    W = get_W_numpy(model, which=which, apply_mask=apply_mask)
    signs_np = model.signs.detach().cpu().numpy()
    population_indices = model.population_indices
    n_pops = len(population_indices)

    B_exc = np.zeros((n_pops, n_pops))
    B_inh = np.zeros((n_pops, n_pops))

    for i, idx_i in enumerate(population_indices):
        for j, idx_j in enumerate(population_indices):
            sub = W[np.ix_(idx_i, idx_j)]        # may contain NaN where mask==0
            signs_j = signs_np[idx_j]
            exc_mask = signs_j > 0
            inh_mask = signs_j < 0
            B_exc[i, j] = np.nansum(sub[:, exc_mask])
            B_inh[i, j] = np.nansum(np.abs(sub[:, inh_mask]))

    net = B_exc - B_inh
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, name in zip(axes, [B_exc, B_inh, net], ["Excitatory input", "Inhibitory input", "Net (E - I)"]):
        im = ax.imshow(data, cmap="RdBu_r", aspect="auto")
        ax.set_title(name)
        if pop_names:
            ax.set_xticks(range(n_pops)); ax.set_xticklabels(pop_names, rotation=45, ha="right")
            ax.set_yticks(range(n_pops)); ax.set_yticklabels(pop_names)
        plt.colorbar(im, ax=ax)
    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, net


# ──────────────────────────────────────────────
# Analysis 5: CKA comparison sparse vs dense
# ──────────────────────────────────────────────

def compare_sparse_vs_dense_cka(
    sparse_models,
    dense_models,
    which="fast",
    show=True,
    apply_mask=False,
):
    C = np.zeros((len(sparse_models), len(dense_models)))
    for i, ms in enumerate(sparse_models):
        Ws = _to_dense_for_matrix_ops(get_W_numpy(ms, which=which, apply_mask=apply_mask))
        for j, md in enumerate(dense_models):
            Wd = _to_dense_for_matrix_ops(get_W_numpy(md, which=which, apply_mask=apply_mask))
            C[i, j] = linear_cka(Ws, Wd)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("Dense models"); ax.set_ylabel("Sparse models")
    ax.set_title("CKA: sparse vs dense")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()

    if show:
        plt.show()

    return C, fig


apply_mask = True
pop_names = ["LA","LB","LC","LD","RA","RB","RC","RD", "X"]

# sparse_models: list of best sparse-trained RNNFreeNeurons instances
# dense_models:  list of best dense-trained (same architecture, no sparsity mask)
sparse_models = []
for path_model in path_models_sparse.glob(f"model_*.pkl"):
    # Load model instance
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    sparse_models.append(model)

dense_models = []
for path_model in path_models_dense.glob(f"model_*.pkl"):
    # Load model instance
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    dense_models.append(model)

# Analysis 1: block heatmap for each model
for i_top in range(len(sparse_models)):
    print(f"Analysis 1 | Evaluating top model {i_top}")
    model = sparse_models[i_top]

    plot_block_heatmap(model, pop_names=pop_names, signed=False, apply_mask=apply_mask, vrange=(0, 1))
    plot_block_heatmap(model, pop_names=pop_names, signed=True, title=f"Signed block-mean W\nmodel {i_top}", vrange=(-1, 1), apply_mask=apply_mask)

# Analysis 2: SV spectrum across all best sparse models
plot_sv_spectrum(sparse_models, labels=[f"sparse_{i}" for i in range(len(sparse_models))], which="full",
                title="SV spectrum – sparse solutions", apply_mask=False)

# Analysis 3: subspace alignment within sparse, within dense, and across
A_sparse = sv_alignment_matrix(sparse_models, k=4, apply_mask=False)
A_dense  = sv_alignment_matrix(dense_models,  k=4, apply_mask=False)
A_cross  = np.array([[principal_angles(
                        np.linalg.svd(get_W_numpy(ms, "fast", apply_mask=False), full_matrices=False)[0],
                        np.linalg.svd(get_W_numpy(md, "fast", apply_mask=False), full_matrices=False)[0], k=4).mean()
                       for md in dense_models] for ms in sparse_models])
plot_alignment_matrix(A_sparse, title="Subspace alignment – sparse vs sparse")
plot_alignment_matrix(A_dense,  title="Subspace alignment – dense vs dense")
plot_alignment_matrix(A_cross,  title="Subspace alignment – sparse vs dense")

# Analysis 4: E/I block structure
signs_np = sparse_models[0].signs.cpu().numpy()
plot_block_EI(model, pop_names=pop_names, apply_mask=apply_mask)

# Analysis 5: CKA sparse vs dense
C = compare_sparse_vs_dense_cka(sparse_models, dense_models, apply_mask=apply_mask)
print("CKA sparse vs dense:\n", C)
