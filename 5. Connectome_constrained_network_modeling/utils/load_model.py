import torch
from pathlib import Path

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

from model.core.RNNFreePop import RNNFreePop



def load_model(pt_path, verbose=False, skip_if_error=True):
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    if verbose:
        # Sanity check
        print(f"Checkpoint was saved from class: {checkpoint['class_name']}")
        print(f"Custom attrs: {list(checkpoint['custom_attrs'].keys())}")

    # ── Instantiate the model ───────────────────────────────────────────────
    # Option A: if your class takes hyperparams as constructor args
    attrs = checkpoint["custom_attrs"]
    model = RNNFreePop(nA=attrs["nA"], nB=attrs["nB"], nC=attrs["nC"], nD=attrs["nD"], nX=attrs["nX"], dt=attrs["dt"])

    # ── Restore parameters ──────────────────────────────────────────────────
    try:
        model.load_state_dict(checkpoint["state_dict"])
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