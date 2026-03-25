"""Utility functions for the cell classification pipeline.

This module contains helper functions that support the main pipeline
but are not core classification logic.

Functions
---------
check_swc_validity : Validate and fix SWC neuron files

Author: Florian Kaempf
REFACTORED: 2026-01-30 - Extracted from class_predictor.py
"""

import pandas as pd

# navis imported conditionally since it may not be available in all environments
try:
    import navis
except ImportError:
    navis = None


def check_swc_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Check and fix invalid SWC neuron files.

    Validates that node IDs are properly ordered (node_id > parent_id for
    non-root nodes) and reloads any invalid SWC files from their origin.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'swc' column containing navis neuron objects.

    Returns
    -------
    pd.DataFrame
        DataFrame with fixed SWC neurons.

    Example:
    -------
    >>> from core.utilities import check_swc_validity
    >>> cells_df = check_swc_validity(cells_df)
    """
    if navis is None:
        raise ImportError("navis is required for checking SWC validity")

    for i, cell in df.iterrows():
        # Check if any node_id is less than its parent_id (invalid ordering)
        if (cell.swc.nodes.node_id < cell.swc.nodes.parent_id).any():
            # Reload the SWC from origin to fix the ordering
            df.loc[i, "swc"].nodes = navis.read_swc(cell.swc.origin).nodes

    return df
