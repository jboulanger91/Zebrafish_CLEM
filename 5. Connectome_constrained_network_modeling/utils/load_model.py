import torch
from pathlib import Path

# Manually add root path for imports to improve interoperability
import sys;

from dotenv import dotenv_values

sys.path.insert(0, "..")

from model.core.RNNFreePop import RNNFreePop
from model.core.RNNClem import RNNClem
from model.core.RNNConnectome import RNNConnectome
from model.core.RNNFixedConnectivity import RNNFixedConnectivity
from analysis.load_synapse_matrix import get_W

def process_checkpoint_rnnfixedconnectivity(checkpoint):
    if "mode_U" not in checkpoint["custom_attrs"].keys():
        checkpoint["custom_attrs"]["mode_U"] = None

    if checkpoint["state_dict"]["U_raw"].dim() == 1:
        checkpoint["state_dict"]["U_raw"] = checkpoint["state_dict"]["U_raw"].unsqueeze(-1)

def load_model(pt_path, verbose=False, skip_if_error=True):
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    if verbose:
        # Sanity check
        print(f"Checkpoint was saved from class: {checkpoint['class_name']}")
        print(f"Custom attrs: {list(checkpoint['custom_attrs'].keys())}")

    # ── Instantiate the model ───────────────────────────────────────────────
    # Option A: if your class takes hyperparams as constructor args
    attrs = checkpoint["custom_attrs"]
    if checkpoint["class_name"] == "RNNFreePop":
        model = RNNFreePop(nA=attrs["nA"], nB=attrs["nB"], nC=attrs["nC"], nD=attrs["nD"], nX=attrs["nX"], dt=attrs["dt"])
    elif checkpoint["class_name"] == "RNNClem":
        model = RNNClem(nA=attrs["nA"], nB=attrs["nB"], nC=attrs["nC"], nD=attrs["nD"], dt=attrs["dt"])
    elif checkpoint["class_name"] == "RNNFixedConnectivity":
        if "dict_neurons" not in checkpoint.keys():  # ##### DEBUG
            env = dotenv_values()
            path_W_csv = Path(env["PATH_W_CSV"])
            W_norm, checkpoint["dict_neurons"] = get_W(path_W_csv)
        process_checkpoint_rnnfixedconnectivity(checkpoint)  # ##### DEBUG
        model = RNNFixedConnectivity(checkpoint["state_dict"]["W_norm"], checkpoint["dict_neurons"],
                                     n_beta=torch.numel(checkpoint["state_dict"]["beta"]),
                                     fixed_U=checkpoint["custom_attrs"]["mode_U"],
                                     sparsity_U=1,
                                     tau=checkpoint["custom_attrs"]["dt"] / checkpoint["custom_attrs"]["alpha"],
                                     dt=checkpoint["custom_attrs"]["dt"],
                                     clamp_weights_min=checkpoint["custom_attrs"]["clamp_weights_min"])
    elif checkpoint["class_name"] == "RNNConnectome":
        model = RNNConnectome(checkpoint["dict_neurons"],
                             tau=checkpoint["custom_attrs"]["dt"] / checkpoint["custom_attrs"]["alpha"],
                             dt=checkpoint["custom_attrs"]["dt"],
                             clamp_weights_min=checkpoint["custom_attrs"]["clamp_weights_min"])
    else:
        raise NotImplementedError

    # ── Restore parameters ──────────────────────────────────────────────────
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.eval()  # set to eval mode if running inference
    except RuntimeError:
        if skip_if_error:
            print(f"Error in loading model {pt_path}")
            return None
        else:
            raise Exception(f"Error in loading model {pt_path}")

    # ── Restore custom attributes ───────────────────────────────────────────
    for k, v in checkpoint["custom_attrs"].items():
        setattr(model, k, v)

    if verbose:
        print("Model loaded successfully.")
        print(model)

    return model