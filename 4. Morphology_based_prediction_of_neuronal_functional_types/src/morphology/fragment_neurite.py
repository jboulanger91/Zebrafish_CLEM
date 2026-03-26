"""Fragment neurite morphology into segments for analysis."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import navis
import numpy as np

# Path setup for local imports
_SRC = Path(__file__).resolve().parents[1]  # src/ directory
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Local imports - use absolute imports compatible with sys.path setup
from myio.load_cells2df import BRAIN_WIDTH_UM, load_cells_predictor_pipeline  # noqa: E402


def symmetric_log_transform(x, linthresh=1):
    """Apply symmetric log transform preserving sign."""
    return np.sign(x) * np.log1p(np.abs(x / linthresh))
def find_end_neurites(nodes_df):
    """Extract segments from end nodes back to the nearest branch point.

    Returns
    -------
        Dict mapping end-node IDs to lists of node IDs along each segment.
    """
    all_segments_dict = {}
    for _i, node in nodes_df.loc[
        nodes_df['type'] == 'end', :
    ].iterrows():
        all_segments_dict[node['node_id']] = []
        exit_var = False
        work_cell = node
        all_segments_dict[node['node_id']].append(
            int(work_cell['node_id'])
        )
        while exit_var != 'branch' and exit_var != 'root':

            try:
                work_cell = nodes_df.loc[
                    nodes_df['node_id']
                    == work_cell['parent_id'].iloc[0], :
                ]
            except (AttributeError, IndexError):
                work_cell = nodes_df.loc[
                    nodes_df['node_id']
                    == work_cell['parent_id'], :
                ]
            exit_var = work_cell['type'].iloc[0]
            all_segments_dict[node['node_id']].append(
                int(work_cell['node_id'])
            )

    return all_segments_dict
def fragment_neuron_into_segments(nodes_df):
    """Fragment a neuron into segments between branch and end nodes.

    Returns
    -------
        Dict mapping terminal/branch node IDs to lists of node IDs per segment.
    """
    all_segments_dict = {}
    for _i, node in nodes_df.loc[
        (nodes_df['type'] == 'end')
        | (nodes_df['type'] == 'branch'), :
    ].iterrows():
        all_segments_dict[node['node_id']] = []
        exit_var = False
        work_cell = node
        all_segments_dict[node['node_id']].append(
            int(work_cell['node_id'])
        )
        while exit_var != 'branch' and exit_var != 'root':

            try:
                work_cell = nodes_df.loc[
                    nodes_df['node_id']
                    == work_cell['parent_id'].iloc[0], :
                ]
            except (AttributeError, IndexError):
                work_cell = nodes_df.loc[
                    nodes_df['node_id']
                    == work_cell['parent_id'], :
                ]
            exit_var = work_cell['type'].iloc[0]
            all_segments_dict[node['node_id']].append(
                int(work_cell['node_id'])
            )

    return all_segments_dict
def find_crossing_neurite(fragmented_dict,nodes_df):
    """Find the fragment that crosses the brain midline.

    Returns
    -------
        Tuple of (fragment key, crossing coordinates) or (nan, nan) if none found.
    """
    key_crossing = None
    for key in fragmented_dict:
        temp = nodes_df.loc[
            nodes_df['node_id'].isin(fragmented_dict[key]),
            'x',
        ]
        #check if any branch has a node close to the midline
        if (
            (temp > (BRAIN_WIDTH_UM / 2) + 2).any()
            and (temp < (BRAIN_WIDTH_UM / 2) - 2).any()
        ):
            key_crossing=key
            #exact crossing
            idx_crossing = abs(
                nodes_df.loc[
                    nodes_df['node_id'].isin(
                        fragmented_dict[key]
                    ), 'x'
                ] - (BRAIN_WIDTH_UM / 2)
            ).argmin()
            node_id_crossing = nodes_df.loc[
                nodes_df['node_id'].isin(
                    fragmented_dict[key]
                ), :
            ].iloc[idx_crossing].node_id
            coords_crossing = np.array(
                nodes_df.loc[
                    nodes_df['node_id'] == node_id_crossing,
                    ['x', 'y', 'z']
                ]
            )[0]

            #median crossing
            # x_cross = nodes_df.loc[(nodes_df['node_id'].isin(fragmented_dict[key]))&
            #                        ((nodes_df['x']>(width_brain/2)-2)|
            #                         (nodes_df['x']<(width_brain/2)+2)), 'x'].median()
            # y_cross = nodes_df.loc[(nodes_df['node_id'].isin(fragmented_dict[key])) &
            #                        ((nodes_df['x'] >  (width_brain / 2) - 2) |
            #                         (nodes_df['x'] <  (width_brain / 2) + 2)), 'y'].median()
            #
            # z_cross = nodes_df.loc[(nodes_df['node_id'].isin(fragmented_dict[key])) &
            #                        ((nodes_df['x'] >  (width_brain / 2) - 2) |
            #                         (nodes_df['x']  < (width_brain / 2) + 5)), 'z'].median()
            # coords_crossing = np.array([x_cross, y_cross, z_cross])

            #print('Crossing neurite found!')
    if key_crossing is None:
        coords_crossing = np.nan
        key_crossing = np.nan
    return key_crossing,coords_crossing
def find_fragment_main_branching(
    fragmented_dict, current_fragment_key,
    target_fragment_key, visited=None, reverse=False,
):
    """Find the path of fragment keys from current to target via DFS.

    Args:
        fragmented_dict: Dict of fragment key to node-ID lists.
        current_fragment_key: Starting fragment key.
        target_fragment_key: Target fragment key to reach.
        visited: Set of already-visited keys (used in recursion).
        reverse: If True, reverse all fragment node lists before searching.

    Returns
    -------
        List of fragment keys forming the path, or None if unreachable.
    """
    if reverse:
        for key in fragmented_dict:
            fragmented_dict[key] = fragmented_dict[key][::-1]

    if visited is None:
        visited = set()

    if current_fragment_key in visited:
        return None

    visited.add(current_fragment_key)

    next_steps = []
    for key in fragmented_dict:
        if key != current_fragment_key and (
            fragmented_dict[current_fragment_key][0]
            in fragmented_dict[key]
        ):
                if key == target_fragment_key:
                    return [current_fragment_key, key]
                else:
                    next_steps.append(key)

    if not next_steps:
        return None
    else:
        for key in next_steps:
            result = find_fragment_main_branching(
                fragmented_dict, key,
                target_fragment_key, visited,
            )
            if result is not None:
                return [current_fragment_key] + result

    return None
def alternative_find_fragment_main_branching(
    fragmented_dict, current_fragment_key,
    target_fragment_key, visited=None,
):
    """Find path between fragments using an alternative traversal strategy.

    Args:
        fragmented_dict: Dict of fragment key to node-ID lists.
        current_fragment_key: Starting fragment key.
        target_fragment_key: Target fragment key to reach.
        visited: Set of already-visited keys (used in recursion).

    Returns
    -------
        List of fragment keys forming the path, or None if unreachable.
    """
    if visited is None:
        visited = set()

    if current_fragment_key in visited:
        return None

    visited.add(current_fragment_key)

    next_steps = []
    for key in fragmented_dict:
        if key != current_fragment_key and (
            fragmented_dict[current_fragment_key][0]
            in fragmented_dict[key]
        ):
                if key == target_fragment_key:
                    return [current_fragment_key, key]
                else:
                    next_steps.append(key)

    if not next_steps:
        return None
    else:
        for key in next_steps:
            result = find_fragment_main_branching(
                fragmented_dict, key,
                target_fragment_key, visited,
            )
            if result is not None:
                return [current_fragment_key] + result

    return None
def find_main_branch(fragmented_dict,path):
    """Identify the primary branch fragment containing the root node.

    Returns
    -------
        Fragment key of the main branch.
    """
    for _i,key in enumerate(path[::-1]):


            if min(fragmented_dict[key])< 10:
                primary = key
                break


    return primary

def find_first_branch(fragmented_dict,main_branch_key,path):
    """Find the first branch fragment that diverges from the main branch.

    Returns
    -------
        Fragment key of the first branch off the main branch.
    """
    for key in path[::-1]:

        if (
            sum(
                x in fragmented_dict[main_branch_key]
                for x in fragmented_dict[key]
            ) >= 1
            and key != main_branch_key
        ):
            key_first_branch = key
            break

    return key_first_branch


def calculate_vector(nodes_df,segment_length=10):
    """Calculate the direction vector between the first and last node.

    Returns
    -------
        3D numpy array representing the direction vector.
    """
    a1 = np.array(nodes_df.iloc[0].loc[['x','y','z']])
    a2 = np.array(
        nodes_df.iloc[-1].loc[['x','y','z']]
    )

    vector = a1 - a2
    return vector
def angle_between_vectors(
    branch1, branch2, against_z=False,
):
    """Compute the 3D angle in degrees between two branch direction vectors.

    Args:
        branch1: Nodes DataFrame for the first branch.
        branch2: Nodes DataFrame for the second branch.
        against_z: If True, use the z-axis unit vector instead of branch1.

    Returns
    -------
        Angle in degrees (supplementary).
    """
    v1 = calculate_vector(branch1)
    if against_z:
        v1 = np.array([0,0,1])
    v2 = calculate_vector(branch2)
    # Calculate the dot product of the vectors
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes of the vectors
    magnitude_a = np.linalg.norm(v1)
    magnitude_b = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_a * magnitude_b)

    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)

    # Convert the angle to degrees (optional)
    angle_degrees = np.degrees(angle_radians)

    return 180 - angle_degrees

def angle_between_vectors2d(
    branch1, branch2, against_z=False,
):
    """Compute the 2D angle in degrees between two branches projected onto the XZ plane.

    Args:
        branch1: Nodes DataFrame for the first branch.
        branch2: Nodes DataFrame for the second branch.
        against_z: If True, use the z-axis unit vector instead of branch1.

    Returns
    -------
        Angle in degrees (supplementary).
    """
    v1 = calculate_vector(branch1)
    if against_z:
        v1 = np.array([0,0,1])
    v2 = calculate_vector(branch2)

    v1_xz = v1[[0,2]]
    v2_xz = v2[[0,2]]

    # Calculate the dot product of the vectors
    dot_product = np.dot(v1_xz, v2_xz)

    # Calculate the magnitudes of the vectors
    magnitude_a = np.linalg.norm(v1_xz)
    magnitude_b = np.linalg.norm(v2_xz)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_a * magnitude_b)

    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)

    # Convert the angle to degrees (optional)
    angle_degrees = np.degrees(angle_radians)

    return 180-angle_degrees

def repair_neuron(nodes_df):
    """Reassign sequential node IDs and update parent references accordingly.

    Returns
    -------
        DataFrame with repaired node and parent IDs.
    """
    for it, (i, cell) in enumerate(nodes_df.iterrows(), start=1):
        nodes_df.loc[
            nodes_df["parent_id"] == cell['node_id'],
            'parent_id',
        ] = it
        nodes_df.loc[i,'node_id'] =it
    return nodes_df
def fragmented_to_plot(df,fragments):
    """Convert fragment dicts into a list of navis TreeNeuron objects for plotting.

    Returns
    -------
        List of TreeNeuron objects, one per fragment.
    """
    all_fragments = []
    for key in fragments:

        temp_df = df.loc[
            df['node_id'].isin(fragments[key]), :
        ]
        temp_df.loc[
            ~temp_df['parent_id'].isin(
                list(temp_df['node_id'])
            ), 'parent_id'
        ] = -1


        all_fragments.append(
            navis.TreeNeuron(temp_df, name=key)
        )
        if -1 in list(
            all_fragments[-1].nodes['parent_id']
        ):
            temp_node_id = all_fragments[-1].nodes.loc[
                all_fragments[-1].nodes['parent_id']
                == -1, 'node_id',
            ].iloc[0]
            all_fragments[-1].soma = temp_node_id
        if len(all_fragments) == 1:
            all_fragments[0].nodes.loc[
                all_fragments[0].nodes['parent_id']
                == -1, 'radius',
            ] = 2

    return all_fragments
def direct_angle_and_crossing_extraction(
    nodes_df, angle2zaxis=False, projection="3d",
):
    """Compute branching angle and midline crossing coordinates for a neuron.

    Args:
        nodes_df: DataFrame of neuron nodes with x/y/z and topology columns.
        angle2zaxis: If True, measure the angle against the z-axis.
        projection: '3d' or '2d' angle computation.

    Returns
    -------
        Tuple of (angle, crossing_coords, fragments_list).
    """
    fragments = fragment_neuron_into_segments(
        nodes_df.sort_values('node_id')
    )
    fragments_list = fragmented_to_plot(
        nodes_df, fragments,
    )
    #nodes_df = repair_neuron(nodes_df)
    # if not (nodes_df.loc[:, "x"] > (width_brain / 2)+10).any():
    #     return np.nan, np.nan,fragments_list

    #extract all fragments

    #write into dict


    #extract crossing neurite
    crossing_key, crossing_coords = (
        find_crossing_neurite(fragments, nodes_df)
    )
    if crossing_key is None or np.isnan(crossing_key):

        return np.nan, np.nan,fragments_list
    else:
        possible_path = find_fragment_main_branching(
            fragments,
            list(fragments.keys())[0],
            crossing_key,
        )
        if possible_path is None:
            possible_path = find_fragment_main_branching(
                fragments,
                list(fragments.keys())[0],
                crossing_key, reverse=True,
            )

        if possible_path is None:
            return np.nan, crossing_coords,fragments_list

        main_branch_key = find_main_branch(
            fragments, possible_path,
        )
        father_of_crossing_key = find_first_branch(
            fragments, main_branch_key, possible_path,
        )


        limit = 2

        main_branch = nodes_df.loc[
            nodes_df['node_id'].isin(
                fragments[main_branch_key]
            ), :
        ]
        main_branch.loc[
            ~main_branch['parent_id'].isin(
                list(main_branch['node_id'])
            ), 'parent_id'
        ] = -1
        main_branch_tree = navis.TreeNeuron(main_branch)
        index_lessthanlimit = navis.geodesic_matrix(
            main_branch_tree,
            max(main_branch_tree.nodes.node_id),
            limit=limit,
        ).T
        close_nodes = (
            index_lessthanlimit[
                index_lessthanlimit <= limit
            ].dropna().index
        )
        main_branch_vector = main_branch.loc[
            main_branch['node_id'].isin(close_nodes), :
        ]
        main_branch_vector = (
            main_branch_vector.iloc[[0,-1],:]
        )
        main_branch_vector.loc[
            :, 'parent_id'
        ].iloc[-1] = (
            main_branch_vector.iloc[0].loc['node_id']
        )
        main_branch_vector.loc[
            :, 'parent_id'
        ].iloc[0] = -1

        first_branch = nodes_df.loc[
            nodes_df['node_id'].isin(
                fragments[father_of_crossing_key]
            ), :
        ]
        first_branch.loc[
            ~first_branch['parent_id'].isin(
                list(first_branch['node_id'])
            ), 'parent_id'
        ] = -1
        first_branch_tree = navis.TreeNeuron(
            first_branch,
        )
        index_lessthanlimit = navis.geodesic_matrix(
            first_branch_tree,
            min(first_branch_tree.nodes.node_id),
            limit=limit,
        ).T
        close_nodes = (
            index_lessthanlimit[
                index_lessthanlimit <= limit
            ].dropna().index
        )
        first_branch_vector = first_branch.loc[
            first_branch['node_id'].isin(close_nodes),
            :,
        ]
        first_branch_vector = (
            first_branch_vector.iloc[[0, -1], :]
        )
        first_branch_vector.loc[
            :, 'parent_id'
        ].iloc[-1] = (
            first_branch_vector.iloc[0].loc['node_id']
        )
        first_branch_vector.loc[
            :, 'parent_id'
        ].iloc[0] = -1


        if projection == '3d':
            angle = angle_between_vectors(
                main_branch_vector,
                first_branch_vector, angle2zaxis,
            )
        first_branch.iloc[0].loc["parent_id"] = -1
        if projection == '2d':
            angle = angle_between_vectors2d(
                main_branch_vector,
                first_branch_vector, angle2zaxis,
            )
        angle = round(angle, 2)

        # VALIDATION PLOTTING
        calculate_vector(main_branch)


        if False:
            fig, ax = navis.plot2d(
                navis.TreeNeuron(main_branch),
                view='xz', color='blue',
                figsize=(20, 20),
            )
            navis.plot2d(
                navis.TreeNeuron(first_branch),
                view='xz', color='red', ax=ax,
            )
            plt.show()
            plt.plot(
                [first_branch_vector.iloc[0].loc['x'],
                 first_branch_vector.iloc[1].loc['x']],
                [first_branch_vector.iloc[0].loc['z'],
                 first_branch_vector.iloc[1].loc['z']],
                color='green', alpha=0.5,
            )
            plt.plot(
                [main_branch_vector.iloc[0].loc['x'],
                 main_branch_vector.iloc[1].loc['x']],
                [main_branch_vector.iloc[0].loc['z'],
                 main_branch_vector.iloc[1].loc['z']],
                color='red', alpha=0.5,
            )
            plt.axis('equal')
            plt.show()

            #validation 3d
            import plotly
            fig = navis.plot3d(
                fragments_list, backend='plotly',
                width=1920, height=1080,
                hover_name=True,
            )


            fig = navis.plot3d(
                navis.TreeNeuron(first_branch),
                backend='plotly', fig=fig,
                width=1920, height=1080,
                hover_name=True,
            )
            fig = navis.plot3d(
                navis.TreeNeuron(main_branch),
                backend='plotly', fig=fig,
                width=1920, height=1080,
                hover_name=True, colors='blue',
            )

            fig = navis.plot3d(
                navis.TreeNeuron(first_branch_vector),
                backend='plotly', fig=fig,
                width=1920, height=1080,
                hover_name=True, colors='red',
            )
            fig = navis.plot3d(
                navis.TreeNeuron(main_branch_vector),
                backend='plotly', fig=fig,
                width=1920, height=1080,
                hover_name=True, colors='red',
            )




            fig.update_layout(
                scene={
                    'xaxis': {
                        'autorange': 'reversed',
                    },
                    'yaxis': {'autorange': True},

                    'zaxis': {'autorange': True},
                    'aspectmode': "data",
                    'aspectratio': {
                        "x": 1, "y": 1, "z": 1,
                    },
                }
            )

            plotly.offline.plot(
                fig, filename="test.html",
                auto_open=True, auto_play=False,
            )



        return angle, crossing_coords,fragments_list



def _pick_cells(df, modality):
    """Pick one ipsilateral and one contralateral cell from *df*.

    A cell is considered contralateral if its SWC x-range spans
    both sides of the midline by more than 5 um.
    """
    midline = BRAIN_WIDTH_UM / 2
    ipsi, contra = None, None
    for idx in df.index:
        swc = df.loc[idx, "swc"]
        if swc is None:
            continue
        x_min, x_max = swc.nodes["x"].min(), swc.nodes["x"].max()
        crosses = (x_min < midline - 5) and (x_max > midline + 5)
        if crosses and contra is None:
            contra = idx
        elif not crosses and ipsi is None:
            ipsi = idx
        if ipsi is not None and contra is not None:
            break
    results = []
    if ipsi is not None:
        results.append((f"{modality}_ipsi", ipsi))
    if contra is not None:
        results.append((f"{modality}_contra", contra))
    return results


def _process_one_cell(label, swc):
    """Run fragment_neurite pipeline on a single SWC and print results."""
    nodes_df = swc.nodes
    fragmented_dict = fragment_neuron_into_segments(nodes_df)
    crossing_key, crossing_coords = find_crossing_neurite(fragmented_dict, nodes_df)

    is_nan = isinstance(crossing_key, float) and np.isnan(crossing_key)
    if is_nan:
        print(f"  [{label}] No crossing neurite found (ipsilateral)")
        return

    possible_path = find_fragment_main_branching(
        fragmented_dict,
        list(fragmented_dict.keys())[0],
        crossing_key,
    )
    if possible_path is None:
        print(f"  [{label}] No path to crossing fragment")
        return

    father_of_crossing_key = find_main_branch(fragmented_dict, possible_path)

    main_branch = nodes_df.loc[
        nodes_df["node_id"].isin(fragmented_dict[list(fragmented_dict.keys())[0]]), :
    ]
    main_branch = main_branch.copy()
    main_branch.loc[
        ~main_branch["parent_id"].isin(main_branch["node_id"].tolist()), "parent_id"
    ] = -1

    first_branch = nodes_df.loc[
        nodes_df["node_id"].isin(fragmented_dict[father_of_crossing_key]), :
    ]
    first_branch = first_branch.copy()
    first_branch.loc[
        ~first_branch["parent_id"].isin(first_branch["node_id"].tolist()), "parent_id"
    ] = -1

    angle = angle_between_vectors(main_branch, first_branch, False)
    print(
        f"  [{label}] fragments={len(fragmented_dict)}, "
        f"crossing={crossing_key}, angle={angle:.1f} deg"
    )


if __name__ == "__main__":
    from src.util.get_base_path import get_base_path

    path_to_data = get_base_path()

    # Load all three modalities
    print("Loading EM cells...")
    cells_em = load_cells_predictor_pipeline(
        path_to_data=path_to_data, modalities=["em"],
    )
    cells_em.loc[:, "swc"] = [
        navis.prune_twigs(x, 20, recursive=True) for x in cells_em["swc"]
    ]

    print("Loading CLEM cells...")
    cells_clem = load_cells_predictor_pipeline(
        path_to_data=path_to_data, modalities=["clem"],
    )
    cells_clem.loc[:, "swc"] = [
        navis.prune_twigs(x, 20, recursive=True) for x in cells_clem["swc"]
    ]

    print("Loading PA cells...")
    cells_pa = load_cells_predictor_pipeline(
        path_to_data=path_to_data, modalities=["pa"],
    )
    cells_pa.loc[:, "swc"] = [
        navis.prune_twigs(x, 20, recursive=True) for x in cells_pa["swc"]
    ]

    # Pick 1 ipsi + 1 contra per modality (6 total)
    selected = []
    for df, mod in [(cells_em, "EM"), (cells_clem, "CLEM"), (cells_pa, "PA")]:
        picks = _pick_cells(df, mod)
        for label, idx in picks:
            name = df.loc[idx, "cell_name"]
            selected.append((label, name, df.loc[idx, "swc"]))

    print(f"\nProcessing {len(selected)} neurons:")
    for label, name, swc in selected:
        print(f"\n  {label}: cell {name} ({len(swc.nodes)} nodes)")
        _process_one_cell(label, swc)

    print("\nDone.")
