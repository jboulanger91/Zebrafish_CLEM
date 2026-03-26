"""Load cell morphology skeletons.

This module provides functions to load SWC skeletons for cells across
different imaging modalities: photoactivation, CLEM, and EM.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import navis
import numpy as np

from src.util.cell_paths import get_cell_file_prefix

warnings.filterwarnings("ignore")


def load_mesh(cell, swc=False):
    """Load SWC skeleton data for a single cell."""
    cell_name = get_cell_file_prefix(cell['imaging_modality'], cell['cell_name'])
    cell_dir = Path(cell['cell_data_dir'])

    is_pa = cell['imaging_modality'] == 'photoactivation'
    file_suffix = '.swc' if is_pa else '_mapped.swc'

    path = cell_dir / (cell_name + file_suffix)

    if path.exists():
        try:
            cell['swc'] = navis.read_swc(
                path, units="um", read_meta=False,
            )
        except Exception as e:
            print(f"Error loading SWC from {path.name}: {e}")
            cell['swc'] = np.nan
    else:
        print(f"No SWC found at {path}")
        cell['swc'] = np.nan

    if not isinstance(cell['swc'], float):
        cell['swc'].nodes.loc[:, 'radius'] = 0.5
        cell['swc'].nodes.loc[0, 'radius'] = 2
    return cell
