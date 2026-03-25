"""Load cell morphology meshes and skeletons.

This module provides functions to load 3D morphology data (OBJ meshes and
SWC skeletons) for cells across different imaging modalities: photoactivation,
CLEM, and EM.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import navis
import numpy as np

from src.util.cell_paths import get_cell_file_prefix

warnings.filterwarnings("ignore")


def load_mesh(
    cell, swc=False,
    load_both=False, load_repaired=False,
):
    """Load mesh and SWC data for a single cell."""
    def load_file(file_path, file_type, is_swc=False):
        if file_path.exists():
            try:
                if is_swc:
                    return navis.read_swc(
                        file_path, units="um", read_meta=False,
                    )
                return navis.read_mesh(file_path, units="um")
            except Exception as e:
                print(f"Error loading {file_type} from {file_path.name}: {e}")
                return np.nan
        else:
            print(f"No {file_type} found at {file_path}")
            return np.nan

    if load_both:
        swc = True
    is_pa = cell['imaging_modality'] == 'photoactivation'
    file_suffix = '.swc' if is_pa else '_mapped.swc'
    if swc and load_repaired and not is_pa:
        file_suffix = "_repaired" + file_suffix

    cell_name = get_cell_file_prefix(cell['imaging_modality'], cell['cell_name'])
    cell_dir = Path(cell['cell_data_dir'])

    if cell['imaging_modality'] == 'photoactivation':
        path = cell_dir / (cell_name + file_suffix)
    else:
        path = cell_dir / 'mapped' / (cell_name + file_suffix)

    if swc:
        cell['swc'] = load_file(path, 'SWC', is_swc=True)
    if not swc or load_both:
        if cell['imaging_modality'] == 'clem':
            cell['axon_mesh'] = load_file(path.parent / f'{cell_name}_axon_mapped.obj', 'axon')
            if 'axon' not in cell_name:
                cell['dendrite_mesh'] = load_file(
                    path.parent / f'{cell_name}_dendrite_mapped.obj',
                    'dendrite',
                )
                cell['soma_mesh'] = load_file(
                    path.parent / f'{cell_name}_soma_mapped.obj',
                    'soma',
                )
        elif cell['imaging_modality'] == 'EM':
            cell['axon_mesh'] = load_file(
                path.parent / f'{cell_name}_axon_mapped.obj', 'axon',
            )

            cell['dendrite_mesh'] = load_file(
                path.parent / f'{cell_name}_dendrite_mapped.obj',
                'dendrite',
            )
            cell['soma_mesh'] = load_file(path.parent / f'{cell_name}_soma_mapped.obj', 'soma')

    if not isinstance(cell['swc'], float):
        cell['swc'].nodes.loc[:, 'radius'] = 0.5
        cell['swc'].nodes.loc[0, 'radius'] = 2
    if isinstance(cell['swc'], float):
        pass
    return cell
