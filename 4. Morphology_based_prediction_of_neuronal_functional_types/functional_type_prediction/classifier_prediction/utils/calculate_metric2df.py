"""Calculate morphological metrics and save to HDF5 feature files.

This module computes morphological features for neuron classification,
including cable length, branching statistics, spatial coordinates,
Sholl analysis, and contralateral crossing metrics. Results are saved
as HDF5 files for downstream use in the classification pipeline.

Parallelized with joblib for multi-core speedup on recalculation.
"""

import contextlib
import os
import sys
from pathlib import Path

import h5py
import joblib
import navis
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into a tqdm progress bar."""
    class TqdmCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# Add src directory to path for morphology imports
_REPO_ROOT = Path(__file__).resolve().parents[3]  # hbsf_new/
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))  # Needed for bare morphology.* imports

try:
    from src.util.output_paths import get_output_dir  # noqa: E402
except ModuleNotFoundError:
    from util.output_paths import get_output_dir  # noqa: E402

from morphology.find_branches import find_branches  # noqa: E402
from morphology.fragment_neurite import (  # noqa: E402
    direct_angle_and_crossing_extraction,
)
from myio.load_cells2df import load_cells_predictor_pipeline  # noqa: E402

WIDTH_BRAIN = 495.56
HALF_BRAIN = WIDTH_BRAIN / 2


def _ic_index(x_coords):
    """Compute ipsilateral-contralateral index from x coordinates."""
    distances = ((HALF_BRAIN - x_coords) / HALF_BRAIN)
    return float(distances.mean())


def _compute_simple_features(swc):
    """Extract all simple morphological features from a single SWC neuron.

    Replaces ~20 individual .apply() calls with a single pass.
    Returns a dict of feature_name -> value.
    """
    nodes = swc.nodes
    extents = swc.extents

    # Strahler index (modifies swc in-place)
    navis.strahler_index(swc)

    # Sholl analysis
    sholl = navis.sholl_analysis(swc, radii=np.arange(10, 200, 10), center='root')
    sholl_max_idx = sholl.branch_points.idxmax()

    return {
        "contralateral_branches": len(
            nodes.loc[(nodes.x > HALF_BRAIN) & (nodes.type == 'branch'), 'type']
        ),
        "ipsilateral_branches": len(
            nodes.loc[(nodes.x < HALF_BRAIN) & (nodes.type == 'branch'), 'type']
        ),
        "cable_length": swc.cable_length,
        "bbox_volume": extents[0] * extents[1] * extents[2],
        "x_extent": extents[0],
        "y_extent": extents[1],
        "z_extent": extents[2],
        "x_avg": float(np.mean(nodes.x)),
        "y_avg": float(np.mean(nodes.y)),
        "z_avg": float(np.mean(nodes.z)),
        "soma_x": float(np.mean(nodes.loc[0, "x"])),
        "soma_y": float(np.mean(nodes.loc[0, "y"])),
        "soma_z": float(np.mean(nodes.loc[0, "z"])),
        "tortuosity": navis.tortuosity(swc),
        "n_leafs": swc.n_leafs,
        "n_branches": swc.n_branches,
        "n_ends": swc.n_ends,
        "n_edges": swc.n_edges,
        "main_branchpoint": navis.find_main_branchpoint(swc),
        "n_persistence_points": len(navis.persistence_points(swc)),
        "max_strahler_index": nodes.strahler_index.max(),
        "sholl_distance_max_branches": sholl_max_idx,
        "sholl_distance_max_branches_cable_length": sholl.cable_length[sholl_max_idx],
        "sholl_distance_max_branches_geosidic": sholl_max_idx,
        "sholl_distance_max_branches_geosidic_cable_length": sholl.cable_length[sholl_max_idx],
    }


def _compute_branch_features(swc, cell_name, cell_branches_df):
    """Extract branch-based and spatial features for a single cell.

    Args:
        swc: The resampled SWC neuron.
        cell_name: Cell identifier.
        cell_branches_df: Pre-filtered branches DataFrame for this cell.

    Returns dict of feature_name -> value.
    """
    result = {}
    nodes = swc.nodes

    # Main path features
    main_path = cell_branches_df[
        (cell_branches_df['main_path']) & (cell_branches_df['end_type'] != 'end')
    ]
    result["main_path_longest_neurite"] = main_path['longest_neurite_in_branch'].iloc[0]
    result["main_path_total_branch_length"] = main_path['total_branch_length'].iloc[0]

    # First major branch (>= 50 um, non-main)
    major = cell_branches_df[
        (~cell_branches_df['main_path'])
        & (cell_branches_df['end_type'] != 'end')
        & (cell_branches_df['total_branch_length'] >= 50)
    ]
    if len(major) > 0:
        result["first_major_branch_longest_neurite"] = major['longest_neurite_in_branch'].iloc[0]
        result["first_major_branch_total_branch_length"] = major['total_branch_length'].iloc[0]
    else:
        result["first_major_branch_longest_neurite"] = 0
        result["first_major_branch_total_branch_length"] = 0

    # Fragment features
    fragmented_neuron = navis.split_into_fragments(swc, swc.n_leafs)
    result["first_branch_longest_neurite"] = navis.longest_neurite(
        fragmented_neuron[1]
    ).cable_length
    result["first_branch_total_branch_length"] = fragmented_neuron[1].cable_length

    # Cable to first branch
    temp = navis.prune_twigs(swc, 5, recursive=True)
    temp_node_id = temp.nodes.loc[temp.nodes.type == 'branch', 'node_id'].iloc[0]
    temp = navis.cut_skeleton(temp, temp_node_id)
    result["cable_length_2_first_branch"] = temp[1].cable_length
    result["z_distance_first_2_first_branch"] = (
        temp[1].nodes.iloc[0].z - temp[1].nodes.iloc[-1].z
    )

    # Biggest non-main branch
    non_main = cell_branches_df[
        (~cell_branches_df['main_path']) & (cell_branches_df['end_type'] != 'end')
    ]
    sorted_non_main = non_main.sort_values('total_branch_length', ascending=False)
    result["biggest_branch_longest_neurite"] = sorted_non_main['longest_neurite_in_branch'].iloc[0]
    result["biggest_branch_total_branch_length"] = non_main['total_branch_length'].iloc[0]

    # Longest connected path
    result["longest_connected_path"] = cell_branches_df['longest_connected_path'].iloc[0]

    # Hemisphere features
    ipsi_mask = nodes.x < HALF_BRAIN
    contra_mask = nodes.x > HALF_BRAIN
    n_nodes = len(nodes.x)

    result["n_nodes_ipsi_hemisphere"] = int(ipsi_mask.sum())
    result["n_nodes_contra_hemisphere"] = int(contra_mask.sum())
    result["n_nodes_ipsi_hemisphere_fraction"] = ipsi_mask.sum() / n_nodes
    result["n_nodes_contra_hemisphere_fraction"] = contra_mask.sum() / n_nodes
    result["x_location_index"] = _ic_index(nodes.x)
    result["fraction_contra"] = contra_mask.sum() / n_nodes

    ipsi_nodes = nodes.loc[ipsi_mask]
    contra_nodes = nodes.loc[contra_mask]

    result["y_extent_ipsi"] = ipsi_nodes["y"].max() - ipsi_nodes["y"].min()
    result["z_extent_ipsi"] = ipsi_nodes["z"].max() - ipsi_nodes["z"].min()

    result["max_x_ipsi"] = ipsi_nodes["x"].max()
    result["max_y_ipsi"] = ipsi_nodes["y"].max()
    result["max_z_ipsi"] = ipsi_nodes["z"].max()
    result["min_x_ipsi"] = ipsi_nodes["x"].min()
    result["min_y_ipsi"] = ipsi_nodes["y"].min()
    result["min_z_ipsi"] = ipsi_nodes["z"].min()

    max_x_contra = contra_nodes["x"].max()
    if np.isnan(max_x_contra):
        for key in ["max_x_contra", "max_y_contra", "max_z_contra",
                     "min_x_contra", "min_y_contra", "min_z_contra"]:
            result[key] = 0
    else:
        result["max_x_contra"] = max_x_contra
        result["max_y_contra"] = contra_nodes["y"].max()
        result["max_z_contra"] = contra_nodes["z"].max()
        result["min_x_contra"] = contra_nodes["x"].min()
        result["min_y_contra"] = contra_nodes["y"].min()
        result["min_z_contra"] = contra_nodes["z"].min()

    # Persistence features
    persist_pts = navis.persistence_points(swc)
    deltas = (persist_pts['death'] - persist_pts['birth']).to_numpy()
    result["avg_delta_death_birth_persitence"] = float(np.mean(deltas))
    result["median_delta_death_birth_persitence"] = float(np.mean(deltas))  # Note: original uses mean
    result["std_delta_death_birth_persitence"] = float(np.std(deltas))

    y_extent_contra = contra_nodes["y"].max() - contra_nodes["y"].min()
    z_extent_contra = contra_nodes["z"].max() - contra_nodes["z"].min()
    if np.isnan(y_extent_contra):
        result["z_extent_contra"] = 0
        result["y_extent_contra"] = 0
    else:
        result["z_extent_contra"] = z_extent_contra
        result["y_extent_contra"] = y_extent_contra

    return result


def _compute_angle_features(swc_nodes, morphology):
    """Extract angle/crossing features for a single cell.

    Returns dict with angle, angle2d, x_cross, y_cross, z_cross.
    """
    if morphology == 'contralateral':
        try:
            angle, crossing_coords, _ = direct_angle_and_crossing_extraction(
                swc_nodes, projection="3d"
            )
            angle2d, crossing_coords, _ = direct_angle_and_crossing_extraction(
                swc_nodes, projection="2d"
            )
            return {
                "angle": angle,
                "angle2d": angle2d,
                "x_cross": crossing_coords[0],
                "y_cross": crossing_coords[1],
                "z_cross": crossing_coords[2],
            }
        except Exception:
            pass
    return {
        "angle": np.nan, "angle2d": np.nan,
        "x_cross": np.nan, "y_cross": np.nan, "z_cross": np.nan,
    }


def calculate_metric2df(
    cell_df, file_name, path_to_data, force_new=False, n_jobs=None
):
    """Calculate morphological metrics for cells and save to HDF5.

    Args:
        cell_df: DataFrame with cell data including 'swc' column.
        file_name: Name for the output HDF5 file.
        path_to_data: Base path to data directory.
        force_new: If True, force recalculation. Default False.
        n_jobs: Number of parallel jobs. Default uses all cores minus one.
    """
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)

    def check_skip_condition(
        path_to_data, file_name, key, cell_df, force_new
    ):
        """Check whether metric calculation can be skipped."""
        file_path = get_output_dir(
            "classifier_pipeline", "features"
        ) / f'{file_name}_features.hdf5'
        if file_path.exists():
            with h5py.File(file_path, 'r') as h5file:
                if (
                    key in h5file
                    and h5file['angle_cross/axis1'].shape[0]
                    == cell_df.shape[0]
                    and not force_new
                ):
                    return True
        return False

    # If force_new, delete any existing incomplete HDF5 first
    if force_new:
        features_path = get_output_dir(
            "classifier_pipeline", "features"
        ) / f'{file_name}_features.hdf5'
        if features_path.exists():
            features_path.unlink()
            print(f"   Deleted existing HDF5 for regeneration: {features_path.name}")

    if not check_skip_condition(
        path_to_data,
        file_name,
        'predictor_pipeline_features',
        cell_df,
        force_new,
    ):
        # resample
        cell_df.loc[:, 'not_resampled_swc'] = cell_df['swc']
        cell_df.loc[:, 'swc'] = cell_df.swc.apply(
            lambda x: x.resample("0.5 micron")
        )
        cell_df = cell_df.reset_index(drop=True)

        # Phase 1: Simple features (parallelized)
        n_cells = len(cell_df)
        with _tqdm_joblib(tqdm(desc="Simple features", total=n_cells)):
            simple_results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_simple_features)(swc) for swc in cell_df["swc"]
            )
        simple_df = pd.DataFrame(simple_results, index=cell_df.index)
        for col in simple_df.columns:
            cell_df.loc[:, col] = simple_df[col]

        # Phase 2: Build branches (parallelized)
        with _tqdm_joblib(tqdm(desc="Branch structures", total=n_cells)):
            branch_results = Parallel(n_jobs=n_jobs)(
                delayed(find_branches)(row['swc'].nodes, row.cell_name)
                for _, row in cell_df.iterrows()
            )
        branches_df = pd.concat(branch_results, ignore_index=True)

        # Phase 3: Branch-based + spatial features (parallelized)
        # Pre-group branches by cell_name for fast lookup
        branches_by_cell = {
            name: group for name, group in branches_df.groupby('cell_name')
        }
        with _tqdm_joblib(tqdm(desc="Branch & spatial features", total=n_cells)):
            branch_feat_results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_branch_features)(
                    row['swc'], row.cell_name, branches_by_cell[row.cell_name]
                )
                for _, row in cell_df.iterrows()
            )
        branch_feat_df = pd.DataFrame(branch_feat_results, index=cell_df.index)
        for col in branch_feat_df.columns:
            cell_df.loc[:, col] = branch_feat_df[col]

        # Save predictor pipeline features
        temp1_index = list(cell_df.columns).index(
            'contralateral_branches'
        )
        temp1 = cell_df.loc[:, cell_df.columns[temp1_index:]]

        features_dir = get_output_dir(
            "classifier_pipeline", "features"
        )
        temp1.to_hdf(
            features_dir / f'{file_name}_features.hdf5',
            'predictor_pipeline_features',
        )

        temp2 = cell_df.loc[
            :,
            [
                'cell_name',
                'imaging_modality',
                'function',
                'morphology',
                'neurotransmitter',
            ],
        ].copy()
        # Overwrite non-canonical labels to "neg_control" for HDF5 storage
        # (runtime uses is_neg_control boolean instead)
        if "is_neg_control" in cell_df.columns:
            temp2.loc[cell_df["is_neg_control"], "function"] = "neg_control"
        temp2.to_hdf(
            features_dir / f'{file_name}_features.hdf5',
            'function_morphology_neurotransmitter',
        )

    if not check_skip_condition(
        path_to_data, file_name, 'angle_cross', cell_df, force_new
    ):
        # Phase 4: Angle/crossing features (parallelized)
        with _tqdm_joblib(tqdm(desc="Angle/crossing features", total=len(cell_df))):
            angle_results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_angle_features)(
                    row['not_resampled_swc'].nodes, row.morphology
                )
                for _, row in cell_df.iterrows()
            )
        angle_df = pd.DataFrame(angle_results, index=cell_df.index)
        for col in angle_df.columns:
            cell_df.loc[:, col] = angle_df[col]

        temp3 = cell_df.loc[
            :,
            ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross'],
        ]
        features_dir = get_output_dir(
            "classifier_pipeline", "features"
        )
        temp3.to_hdf(
            features_dir / f'{file_name}_features.hdf5',
            'angle_cross',
        )

        complete_df = pd.concat([temp2, temp1, temp3], axis=1)
        complete_df.to_hdf(
            features_dir / f'{file_name}_features.hdf5',
            'complete_df',
        )


