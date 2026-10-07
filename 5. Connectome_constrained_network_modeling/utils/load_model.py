"""
Overview
--------
This module provides a robust deserialization utility (`load_model`) for loading trained
recurrent neural network models (specifically `RNNConnectome`) from PyTorch checkpoint
files (.pt) onto the CPU, restoring state dictionaries, architecture parameters, and
custom metadata attributes.

No environment variables are directly required by this utility module.

Core Pipeline & Workflow:
1. Checkpoint Deserialization:
   - Loads serialized PyTorch checkpoint dictionaries from disk onto CPU memory using
     `torch.load(map_location="cpu")`.
   - Optionally logs checkpoint class names and available metadata keys for debugging.

2. Architecture Instantiation:
   - Inspects the saved `class_name` attribute (currently targeting `RNNConnectome`).
   - Reconstructs network hyperparameters (computing timescale tau from dt / alpha, restoring
     slow populations, integration step size, Dale's law clamping bounds) and instantiates
     the PyTorch recurrent model with its anatomical neuron metadata (`dict_neurons`).

3. Weight Restoration:
   - Injects saved weights into the newly initialized module using `load_state_dict`
     with non-strict alignment (`strict=False`).
   - Sets the network to inference/evaluation mode (`model.eval()`).
   - Handles `RuntimeError` mismatch exceptions gracefully by either skipping or raising,
     depending on the `skip_if_error` flag.

4. Custom Attribute Reconstitution:
   - Dynamically reattaches custom attributes (e.g., loss values, connectivity masks,
     time constants, population indices) directly onto the instantiated model object.
"""

import torch

# Manually add root path for imports to improve interoperability across parent modules
import sys; sys.path.insert(0, "..")

from model.core.RNNConnectome import RNNConnectome
from utils.load_connectome import get_W


def load_model(pt_path, verbose=False, skip_if_error=True):
    # Load serialized checkpoint dictionary onto CPU memory
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    if verbose:
        # Sanity check: display architectural class and custom metadata keys
        print(f"Checkpoint was saved from class: {checkpoint['class_name']}")
        print(f"Custom attrs: {list(checkpoint['custom_attrs'].keys())}")

    # -------------------------------------------------------------------------
    # Instantiate the model
    # -------------------------------------------------------------------------
    # Reconstruct the model instance using serialized configuration hyperparameters
    if checkpoint["class_name"] == "RNNConnectome":
        model = RNNConnectome(checkpoint["dict_neurons"],
                              tau=checkpoint["custom_attrs"]["dt"] / checkpoint["custom_attrs"]["alpha"],
                              slow_populations=checkpoint["custom_attrs"]["slow_populations"],
                             dt=checkpoint["custom_attrs"]["dt"], pack_parameters=checkpoint["custom_attrs"]["pack_parameters"],
                             clamp_weights_min=checkpoint["custom_attrs"]["clamp_weights_min"])
    else:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Restore parameters
    # -------------------------------------------------------------------------
    # Load learned state dictionary weights and set model to evaluation mode
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.eval()  # set to eval mode if running inference
    except RuntimeError as e:
        # Handle state dict mismatch or corrupted checkpoint files
        if skip_if_error:
            print(f"Error in loading model {pt_path}")
            return None
        else:
            raise Exception(f"Error in loading model {pt_path}")

    # -------------------------------------------------------------------------
    # Restore custom attributes
    # -------------------------------------------------------------------------
    # Dynamically bind non-tensor metadata attributes (e.g. loss values, time parameters) to model
    for k, v in checkpoint["custom_attrs"].items():
        setattr(model, k, v)

    if verbose:
        print("Model loaded successfully.")
        print(model)

    return model