import torch
import numpy as np
from dotenv import dotenv_values
from pathlib import Path

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from analysis.load_synapse_matrix import get_W
from utils.configuration_rnn import ConfigurationRNN

n_eig = 1

env_path = "../.env"
env = dotenv_values(env_path)

path_W_csv = Path(env["PATH_W_CSV"])
W, dict_neurons = get_W(path_W_csv, do_symmetry_transform=False)

_eig_W = torch.linalg.eigvals(torch.tensor(W))
eig_W = torch.sort(_eig_W[_eig_W.nonzero()].real)[0:n_eig]
W_LiMI = W[np.ix_(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["iMI"]["idx_list"], dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["iMI"]["idx_list"])]
W_LcMI = W[np.ix_(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["cMI"]["idx_list"], dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["cMI"]["idx_list"])]
W_LMON = W[np.ix_(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["MON"]["idx_list"], dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["MON"]["idx_list"])]
W_LsMI = W[np.ix_(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["sMI"]["idx_list"], dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["sMI"]["idx_list"])]
W_RiMI = W[np.ix_(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["iMI"]["idx_list"], dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["iMI"]["idx_list"])]
W_RcMI = W[np.ix_(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["cMI"]["idx_list"], dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["cMI"]["idx_list"])]
W_RMON = W[np.ix_(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["MON"]["idx_list"], dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["MON"]["idx_list"])]
W_RsMI = W[np.ix_(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["sMI"]["idx_list"], dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["sMI"]["idx_list"])]

_eig_W_LiMI = torch.linalg.eigvals(torch.tensor(W_LiMI))
eig_W_LiMI = torch.sort(_eig_W_LiMI[_eig_W_LiMI.nonzero()].real)[0:n_eig]
_eig_W_LcMI = torch.linalg.eigvals(torch.tensor(W_LcMI))
eig_W_LcMI = torch.sort(_eig_W_LcMI[_eig_W_LcMI.nonzero()].real)[0:n_eig]
_eig_W_LMON = torch.linalg.eigvals(torch.tensor(W_LMON))
eig_W_LMON = torch.sort(_eig_W_LMON[_eig_W_LMON.nonzero()].real)[0:n_eig]
_eig_W_LsMI = torch.linalg.eigvals(torch.tensor(W_LsMI))
eig_W_LsMI = torch.sort(_eig_W_LsMI[_eig_W_LsMI.nonzero()].real)[0:n_eig]
_eig_W_RiMI = torch.linalg.eigvals(torch.tensor(W_RiMI))
eig_W_RiMI = torch.sort(_eig_W_RiMI[_eig_W_RiMI.nonzero()].real)[0:n_eig]
_eig_W_RcMI = torch.linalg.eigvals(torch.tensor(W_RcMI))
eig_W_RcMI = torch.sort(_eig_W_RcMI[_eig_W_RcMI.nonzero()].real)[0:n_eig]
_eig_W_RMON = torch.linalg.eigvals(torch.tensor(W_RMON))
eig_W_RMON = torch.sort(_eig_W_RMON[_eig_W_RMON.nonzero()].real)[0:n_eig]
_eig_W_RsMI = torch.linalg.eigvals(torch.tensor(W_RsMI))
eig_W_RsMI = torch.sort(_eig_W_RsMI[_eig_W_RsMI.nonzero()].real)[0:n_eig]
