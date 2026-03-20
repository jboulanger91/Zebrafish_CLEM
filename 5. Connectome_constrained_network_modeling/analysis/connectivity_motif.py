import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# ------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------
THR = 0.05           # nonzero threshold for determining edges
N_RANDOM = 100       # number of random null-networks per model

top_model_id_list = [5, 23, 12, 56, 64, 50, 51]
path_dir = Path(r"C:\Users\Roberto\Academics\data\models\rnn_ds\dales_law\integrator_neurons10_tau0.5_input2step")

i_model = -1
W_list = []
for path_model in path_dir.glob("model_*.pkl"):
    i_model += 1
    if i_model not in top_model_id_list:
        continue
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    W = model.W().detach().numpy()
    W_list.append(W)

W_array = np.array(W_list)
# W_top shape = (M, N, N)
M, N, _ = W_array.shape


# ------------------------------------------------------------------------
# Convert weights to signed adjacency matrix
# ------------------------------------------------------------------------
def adjacency_from_weight(W, thr=THR):
    A = np.zeros_like(W, dtype=int)
    A[W > thr]  = 1
    A[W < -thr] = -1
    return A


# ------------------------------------------------------------------------
# Randomization for signed networks
# ------------------------------------------------------------------------
def generate_random_signed_adjacency(n_neurons=10, max_value=1, E_frac=0.7):
    W = np.random.rand(n_neurons, n_neurons) * max_value
    n_neurons_E = int(n_neurons * E_frac)
    W[n_neurons_E:] *= -1
    A = adjacency_from_weight(W)
    return A


# ------------------------------------------------------------------------
# Triplet enumeration
# ------------------------------------------------------------------------
triplets = list(permutations(range(N), 3))


# ------------------------------------------------------------------------
# Motif detection
# ------------------------------------------------------------------------
def detect_motif(A, i, j, k):
    eij = A[i, j]
    ejk = A[j, k]
    eik = A[i, k]
    eji = A[j, i]
    eki = A[k, i]
    ekj = A[k, j]

    # chain: i → j → k
    if eij != 0 and ejk != 0 and eik == 0:
        return "chain"

    # divergent: i → j and i → k
    if eij != 0 and eik != 0 and ejk == 0:
        return "divergent"

    # convergent: i ← j → k  OR  i ← k → j
    if eji != 0 and ejk != 0 and eik == 0:
        return "convergent"
    if eki != 0 and ekj != 0 and eij == 0:
        return "convergent"

    return None


motif_types = ["chain", "divergent", "convergent"]


# ------------------------------------------------------------------------
# Count motifs in one adjacency matrix
# ------------------------------------------------------------------------
def count_motifs(A):
    counts = {"chain": 0, "divergent": 0, "convergent": 0}
    for i, j, k in triplets:
        m = detect_motif(A, i, j, k)
        if m:
            counts[m] += 1
    return np.array([counts[m] for m in motif_types])


# ------------------------------------------------------------------------
# Compute motif counts + Z-scores
# ------------------------------------------------------------------------
real_counts_all = []
rand_mean_all = []
rand_std_all = []

for W in W_list:
    A_real = adjacency_from_weight(W)
    real_counts = count_motifs(A_real)
    real_counts_all.append(real_counts)

    # Random ensemble
    rand_counts = []
    for _ in range(N_RANDOM):
        A_rand = generate_random_signed_adjacency(n_neurons=10, max_value=3, E_frac=0.7)
        rand_counts.append(count_motifs(A_rand))

    rand_counts = np.stack(rand_counts)
    rand_mean = rand_counts.mean(axis=0)
    rand_std  = rand_counts.std(axis=0) + 1e-12

    rand_mean_all.append(rand_mean)
    rand_std_all.append(rand_std)

real_counts_all = np.stack(real_counts_all)
rand_mean_all = np.stack(rand_mean_all)
rand_std_all  = np.stack(rand_std_all)

# Z-scores
Z_all = (real_counts_all - rand_mean_all) / rand_std_all
Z_mean = Z_all.mean(axis=0)
Z_std  = Z_all.std(axis=0)


# ------------------------------------------------------------------------
# Plot Z-scores
# ------------------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.bar(motif_types, Z_mean, yerr=Z_std, capsize=4)
plt.ylabel("Z-score (motif enrichment)")
plt.title("Motif significance relative to random networks")
plt.tight_layout()
plt.show()

print("Mean Z-scores for motifs:", dict(zip(motif_types, Z_mean)))
