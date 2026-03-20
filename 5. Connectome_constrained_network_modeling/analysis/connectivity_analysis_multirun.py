import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

from utils.train_batch import TrainSignal
from utils.operators import integrate

# ------------------------------------------------
# Configuration
# ------------------------------------------------
path_dir = Path(r"C:\Users\Roberto\Academics\data\models\rnn_ds\dales_fit_U\integrator_neurons10_tau0.1_input2step")   # directory with model_X.pt
TOP_FRAC  = 0.05
device    = "cpu"

n_signal_test = 50
tau = 5
dt = 0.1
duration_rest_start = 1
duration_stimulus = 14
duration_rest_end = 5


# Example: define some test input signals and target functions
def generate_test_integration_signals(n_signal_test, dt=dt, tau=tau):
    amplitude_input_signal_list = np.linspace(0, 1, n_signal_test)
    input_signal_step = np.concatenate((np.zeros(int(duration_rest_start / dt)), np.ones(int(duration_stimulus / dt)),
                                   np.zeros(int(duration_rest_end / dt))))
    sine = lambda t: 0.5 * np.sin(t - np.pi / 2) + 0.5
    input_signal_sine = np.concatenate(
        (np.zeros(int(duration_rest_start / dt)), sine(np.arange(0, duration_stimulus, dt)),
         np.zeros(int(duration_rest_end / dt))))

    input_signal_list = []
    for i in range(n_signal_test):
        input_signal_list.append(input_signal_step * amplitude_input_signal_list[i])
        input_signal_list.append(input_signal_sine * amplitude_input_signal_list[i])

    output_signal_list = []
    for input_signal in input_signal_list:
        output_signal = integrate(input_signal, tau=tau, dt=dt)
        output_signal_list.append(output_signal)

    return input_signal_list, output_signal_list

test_inputs, test_targets = generate_test_integration_signals(n_signal_test=n_signal_test)
test_inputs = torch.tensor(np.array(test_inputs))  # convert to np first to speed up conversion
test_targets = torch.tensor(np.array(test_targets))

# ------------------------------------------------
# Storage
# ------------------------------------------------
performances = []
W_mats = []

# ------------------------------------------------
# Loop over all trained models
# ------------------------------------------------
i_model = 0
for path_model in path_dir.glob(f"model_*.pkl"):
    print(f"Evaluating model {i_model}")
    i_model += 1

    # Load model instance
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    # model = torch.load(path_model, map_location=device, weights_only=False)
    model.eval()

    # Evaluate on all test signals and compute average MSE
    mse_list = []

    for u, y_true in zip(test_inputs, test_targets):
        x0 = torch.zeros(model.n_units)
        _, y_pred = model(x0, u)
        mse = torch.mean((y_pred - y_true)**2).item()
        mse_list.append(mse)

    avg_mse = np.mean(mse_list)
    performances.append(avg_mse)

    # Extract connectivity matrix
    W_mats.append(model.W().detach().cpu().numpy())

N_MODELS = len(performances)
performances = np.array(performances)
W_mats = np.stack(W_mats, axis=0)   # shape: (N_MODELS, N, N)

# Plot distribution of performance
plt.figure(figsize=(6,4))
plt.hist(performances, bins=20)
plt.xlabel("Average test MSE")
plt.ylabel("Count")
plt.title("Performance distribution across 100 models")
plt.tight_layout()
plt.show()

# Select top-performant models
num_top = int(TOP_FRAC * N_MODELS)
top_indices = np.argsort(performances)[:num_top]
W_top = W_mats[top_indices]    # shape: (num_top, N, N)

# Population mean and standard deviation of weights
W_mean = W_top.mean(axis=0)
W_std  = W_top.std(axis=0)

plt.figure(figsize=(5,4))
W_value_lim = np.max(np.abs(W_mean))
plt.imshow(W_mean, cmap="seismic", vmin=-W_value_lim, vmax=W_value_lim)
plt.colorbar()
plt.title(f"Mean connectivity among top {int(TOP_FRAC*100)}% models")
plt.tight_layout()
plt.show()

# PCA on flattened connectivity matrices
W_flat = W_top.reshape(num_top, -1)   # shape (num_top, N*N)
pca = PCA(n_components=2)
Z = pca.fit_transform(W_flat)

plt.figure(figsize=(5,4))
plt.scatter(Z[:,0], Z[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of connectivity (top 10%)")
plt.tight_layout()
plt.show()

# Hierarchical clustering of connectivity matrices
Z_link = linkage(W_flat, method='ward')

plt.figure(figsize=(6,4))
dendrogram(Z_link)
plt.title("Clustering of top-performing connectivity matrices")
plt.tight_layout()
plt.show()
