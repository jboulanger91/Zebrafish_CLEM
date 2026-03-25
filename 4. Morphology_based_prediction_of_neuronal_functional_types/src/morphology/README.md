# src/morphology

Morphological analysis module for the zebrafish hindbrain structure-function project. This module provides tools for repairing neuron skeleton (SWC) files, fragmenting neurite trees into segments for structural analysis, identifying branch structures and main paths, computing branching angles and midline-crossing coordinates, and performing NBLAST morphological similarity comparisons between neuron groups. It operates on neurons from all imaging modalities (photoactivation, CLEM, EM) after their skeletons have been loaded.

## File Listing

| File | Description |
|---|---|
| `__init__.py` | Package init. Exports `find_branches`, `find_end_neurites`, `fragment_neuron_into_segments`, `find_crossing_neurite`, `angle_between_vectors`, `direct_angle_and_crossing_extraction`, `repair_neuron`, `nblast_one_group`, `nblast_two_groups`, `nblast_two_groups_custom_matrix`, `compute_nblast_within_and_between`. |
| `repair_swc.py` | Repair and clean SWC neuron morphology files. Re-roots neurons at the soma, reassigns sequential node IDs, enforces valid parent-before-child ordering via topological sort, and writes corrected SWC files to disk. Includes optional 3D Plotly visualization for repair validation. |
| `find_branches.py` | Branch identification and characterization. Decomposes a neuron skeleton into branch segments, classifies segments by node count and cable length quintiles, identifies the longest connected path (main path), and computes per-branch metrics (longest neurite in branch, total branch length). |
| `fragment_neurite.py` | Neurite fragmentation, midline-crossing detection, and branching angle computation. Fragments neurons into segments between branch/end nodes, locates the neurite fragment that crosses the brain midline, traces paths between fragments via DFS, and computes 3D or 2D branching angles between the main branch and the first diverging branch. |
| `nblast.py` | NBLAST morphological similarity analysis. Computes within-group and between-group NBLAST scores using navis dotprops, with optional root-centering (translating neurons so the root node is at the origin) and custom scoring matrices. |

## Key Functions

### `repair_swc.py`

| Function / Class | Description |
|---|---|
| `repair_indices(nodes_df)` | Reassign sequential node IDs starting from 0; update all parent references accordingly. |
| `nodes2array(nodes_df)` | Convert a nodes DataFrame to a stacked numpy array (node_id, label, x, y, z, radius, parent_id, type). |
| `dfs_Solution` | Helper class implementing recursive DFS and topological sort for DAG node ordering. |
| `addEdge(adj, u, v)` | Add a directed edge in an adjacency list. |
| `repair_hierarchy(df)` | Reorder node IDs via topological sort so parents always precede children in the SWC table. |
| `check_neuron_by_viz(df_fixed, df_original, ...)` | Interactive 3D Plotly comparison of repaired vs original neuron, with fragment coloring. |
| `repair_neuron(navis_element, path, viz_check)` | Top-level repair entry point. Re-roots at soma (radius=2 node), repairs indices and hierarchy, writes SWC with standard header. |

### `find_branches.py`

| Function | Description |
|---|---|
| `find_branches(df, cell_name)` | Main entry point. Takes a nodes DataFrame and cell name; returns a DataFrame of branch segments with columns: `branch_id`, `branch_type_nodes`, `start_node`, `end_node`, `cable_length`, `n_nodes`, `nodes`, `connected2`, `longest_connected_path`, `longest_neurite_in_branch`, `total_branch_length`, `main_path`, `cell_name`. |

### `fragment_neurite.py`

| Function | Description |
|---|---|
| `find_end_neurites(nodes_df)` | Extract segments from each end node back to the nearest branch point. Returns dict mapping end-node IDs to node-ID lists. |
| `fragment_neuron_into_segments(nodes_df)` | Fragment a neuron at every branch and end node. Returns dict mapping terminal/branch node IDs to node-ID lists per segment. |
| `find_crossing_neurite(fragmented_dict, nodes_df)` | Find the fragment that crosses the brain midline (using `BRAIN_WIDTH_UM` constant). Returns (fragment key, crossing coordinates) or (nan, nan). |
| `find_fragment_main_branching(fragmented_dict, current, target, ...)` | DFS path-finding between two fragment keys through the fragment adjacency graph. |
| `find_main_branch(fragmented_dict, path)` | Identify the primary branch fragment containing the root node. |
| `find_first_branch(fragmented_dict, main_key, path)` | Find the first branch fragment that diverges from the main branch. |
| `calculate_vector(nodes_df)` | Compute the 3D direction vector between first and last nodes of a segment. |
| `angle_between_vectors(branch1, branch2, against_z)` | Compute the 3D angle (degrees, supplementary) between two branch direction vectors. |
| `angle_between_vectors2d(branch1, branch2, against_z)` | Compute the 2D angle projected onto the XZ plane. |
| `direct_angle_and_crossing_extraction(nodes_df, ...)` | High-level function that combines fragmentation, crossing detection, path finding, and angle computation into a single call. Returns (angle, crossing_coords, fragments_list). |

### `nblast.py`

| Function | Description |
|---|---|
| `nblast_one_group(df, k, resample_size)` | Compute within-group NBLAST similarity matrix. Input DataFrame must have `swc` (navis TreeNeuron) and `cell_name` columns. Returns a labeled similarity DataFrame. |
| `nblast_two_groups(df1, df2, k, resample_size, shift_neurons)` | Compute between-group NBLAST similarity. Optionally root-centers neurons before comparison. Returns labeled DataFrame with df1 names as index and df2 names as columns. |
| `nblast_two_groups_custom_matrix(df1, df2, custom_matrix, k, resample_size, shift_neurons)` | Same as `nblast_two_groups` but uses a custom NBLAST scoring matrix and 10-core parallelism. |
| `compute_nblast_within_and_between(df, query_keys_input)` | Partition a DataFrame by category keys (morphology: ipsilateral/contralateral; neurotransmitter: excitatory/inhibitory; function: motion_integrator/motion_onset/slow_motion_integrator) and return (within_values, between_values) as flattened NBLAST score arrays. |

## Dependencies on Other `src/` Modules

| Module | Usage |
|---|---|
| `src.morphology.fragment_neurite` | Used by `repair_swc.py` for visualization (fragment-based coloring in `check_neuron_by_viz`). |
| `src.myio.load_cells2df` | Used by `repair_swc.py` (main block) and `fragment_neurite.py` (main block) to load cell data. |
| `src.constants.BRAIN_WIDTH_UM` | Used by `fragment_neurite.py` for midline-crossing detection threshold. |

## External Dependencies

`navis`, `numpy`, `pandas`, `plotly`, `matplotlib`, `tqdm`

## Output Location

`repair_swc.py` writes repaired SWC files alongside the original data. For CLEM/EM cells, output goes to `<cell_metadata_path.parent>/mapped/<cell_name>_repaired_mapped.swc`. For photoactivation cells, output goes to `<cell_metadata_path.parent>/<cell_name>_repaired.swc`. The NBLAST and branch analysis functions return in-memory DataFrames and arrays rather than writing to disk.
