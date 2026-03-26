"""Repair and clean SWC neuron morphology files."""

import os
import sys
from datetime import datetime
from pathlib import Path

import navis
import numpy as np
import plotly
from tqdm import tqdm

# Required for cross-package bare imports (myio.*, util.*) in standalone execution
_SRC = Path(__file__).resolve().parents[1]  # src/ directory
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Sibling imports use relative paths; cross-package use bare imports via sys.path
from myio.load_cells2df import load_cells_predictor_pipeline  # noqa: E402
from util.get_base_path import get_base_path  # noqa: E402

try:
    from .fragment_neurite import fragment_neuron_into_segments  # noqa: E402
except ImportError:
    from fragment_neurite import fragment_neuron_into_segments  # noqa: E402


def repair_indices(nodes_df):
    """Reassign sequential node IDs starting from 0 and update parent references.

    Returns
    -------
        DataFrame with repaired node and parent IDs.
    """
    for it, (i, cell) in enumerate(nodes_df.iterrows()):
        nodes_df.loc[nodes_df["parent_id"]==cell['node_id'],'parent_id'] = it
        nodes_df.loc[i,'node_id'] =it
    return nodes_df

def nodes2array(nodes_df):
    """Convert a nodes DataFrame into a stacked numpy array.

    Returns
    -------
        2D numpy array with columns [node_id, label, x, y, z, radius, parent_id, type].
    """
    nodes_array = nodes_df['node_id'].to_numpy().astype(int)
    label_array = nodes_df['label'].to_numpy().astype(int)
    x_array = nodes_df['x'].to_numpy()
    y_array = nodes_df['y'].to_numpy()
    z_array = nodes_df['z'].to_numpy()
    radius_array = nodes_df['radius'].to_numpy()
    parent_array = nodes_df['parent_id'].to_numpy()

    type_array = nodes_df['type']

    a = np.stack(
        [nodes_array, label_array, x_array, y_array,
         z_array, radius_array, parent_array, type_array],
        axis=1,
    )
    return a


class dfs_Solution:
    """Depth-first search helper for topological sorting of DAGs."""

    # Topo sort only exists in DAGs i.e.
    # Direct Acyclic graph
    def dfs(self, adj, vis, node, n, stck):
        """Perform recursive DFS and append node to stack on completion."""
        vis[node] = 1
        for it in adj[node]:
            if not vis[it]:
                self.dfs(adj, vis, it, n, stck)
        stck.append(node)

    # During the traversal u must
    # be visited before v
    def topo_sort(self, adj, n):
        """Return topologically sorted node list for the adjacency graph.

        Args:
            adj: Adjacency list representation of the graph.
            n: Number of nodes.

        Returns
        -------
            List of node indices in topological order.
        """
        vis = [0] * n

        # using stack ADT
        stck = []
        for i in range(n):
            if not vis[i]:
                self.dfs(adj, vis, i, n, stck)
        return stck


# Function to add an edge
def addEdge(adj: list[int], u: int, v: int) -> None:
    """Add a directed edge from node u to node v in the adjacency list."""
    adj[u].append(v)


def repair_hierarchy(df):
    """Reorder node IDs via topological sort so parents always precede children.

    Returns
    -------
        DataFrame with reassigned node and parent IDs in valid SWC order.
    """
    #create adjency
    n = df.shape[0]

    adj = [[] for _ in range(n)]
    for unique_parent in df.loc[:,'parent_id']:
        a_with_certain_parent = df.loc[df['parent_id']==unique_parent,:]
        for _i,node in a_with_certain_parent.iterrows():
            if node['parent_id'] != -1:
                addEdge(adj, node['node_id'], node['parent_id'])


    s = dfs_Solution()
    try:
        sys.setrecursionlimit(3000)
        solution = s.topo_sort(adj, n)
    except RecursionError:
        sys.setrecursionlimit(30000)
        solution = s.topo_sort(adj, n)


    new_id_assignment = {-1:-1}
    for i,new_assign in zip(range(df.shape[0]),solution, strict=False):
        new_id_assignment[new_assign] = i

    df.loc[:,'node_id'] = df.loc[:,'node_id'].apply(lambda x: new_id_assignment[x])
    df.loc[:,'parent_id'] = df.loc[:,'parent_id'].apply(lambda x: new_id_assignment[x])
    df = df.sort_values('parent_id')

    return df

def check_neuron_by_viz(df_fixed, df_original,only_fragment=False):
    """Visualize repaired vs. original neuron in an interactive 3D Plotly plot.

    Args:
        df_fixed: Repaired neuron nodes DataFrame.
        df_original: Original neuron nodes DataFrame.
        only_fragment: If True, show only fragmented view without the full neuron.
    """
    # fragment
    fragments = fragment_neuron_into_segments(df_fixed)
    all_fragments = []
    for key in fragments:
        temp_df = df_fixed.loc[df_fixed['node_id'].isin(fragments[key]), :]
        temp_df.loc[~temp_df['parent_id'].isin(list(temp_df['node_id'])), 'parent_id'] = -1

        all_fragments.append(navis.TreeNeuron(temp_df, name=key))

    fixed_neuron = navis.TreeNeuron(df_fixed)
    fixed_neuron.nodes.loc[fixed_neuron.nodes['parent_id'] == -1, 'radius'] = 1
    fig = navis.plot3d(all_fragments, backend='plotly',
                       width=1920, height=1080, hover_name=True)
    if not only_fragment:
        fig = navis.plot3d(fixed_neuron, backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True)

        fig = navis.plot3d(navis.TreeNeuron(df_original), backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True, colors='green')


    fig.update_layout(
        scene={
            'xaxis': {'autorange': 'reversed'},  # reverse !!!
            'yaxis': {'autorange': True},

            'zaxis': {'autorange': True},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1}
        }
    )

    plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)

def repair_neuron(navis_element,path=None,viz_check=False):
    """Repair a neuron SWC by re-rooting, reindexing, and writing to disk.

    Args:
        navis_element: A navis TreeNeuron to repair.
        path: Output file path for the repaired SWC. Defaults to 'test.swc'.
        viz_check: If True, open an interactive 3D comparison plot after repair.
    """
    plot = False
    if navis_element.nodes.loc[
        (navis_element.nodes['radius'] == 2)
        & (navis_element.nodes['parent_id'] == -1), :
    ].empty:
        plot=True
        fig = navis.plot3d(navis_element, backend='plotly',
                           width=1920, height=1080, hover_name=True, alpha=1)




        temp_node_id = navis_element.nodes.loc[
            (navis_element.nodes['radius'] == 2), 'node_id'
        ].values[0]
        navis_element = navis_element.reroot(temp_node_id)
        print('REROOT',navis_element.name,'REROOT')
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},  # reverse !!!
                'yaxis': {'autorange': True},

                'zaxis': {'autorange': True},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1}
            }
        )


    df = repair_indices(navis_element.nodes)


    df = repair_hierarchy(df)

    df = df.iloc[:,:-1]
    #write neuron
    header = (
        "# SWC format file based on specifications at"
        " http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html\n"
        "# Generated by 'map_and_skeletonize_cell' of the ANTs"
        " registration helper library developed by the Bahl"
        " lab Konstanz.\n"
        "# Labels: 0 = undefined; 1 = soma; 2 = axon;"
        " 3 = dendrite; 4 = Presynapse; 5 = Postsynapse\n"
    )
    if path is None:
        with open("test.swc", 'w') as fp:
            fp.write(header)
            df.to_csv(fp, index=False, sep=' ', header=None)
    else:
        with open(path, 'w') as fp:
            fp.write(header)
            df.to_csv(fp, index=False, sep=' ', header=None)
    if plot and viz_check:
        ttt = navis.read_swc(path)
        ttt.soma = ttt.nodes.loc[
            (ttt.nodes['parent_id'] == -1), 'node_id'
        ].values[0]
        ttt.name  = ttt.name+ " repaired"
        fig = navis.plot3d(ttt, backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True, alpha=1)

        plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)

if __name__ == '__main__':
    name_time = datetime.now()
    path_to_data = get_base_path()

    all_cells = load_cells_predictor_pipeline(
        path_to_data=path_to_data,
        modalities=['em'],
        mirror=False,
    )
    all_cells = all_cells.dropna(subset='swc', axis=0)

    from src.util.cell_paths import get_cell_file_prefix

    for _i, cell in tqdm(all_cells.iterrows(), total=all_cells.shape[0]):
        cell_dir = Path(cell.cell_data_dir)
        file_prefix = get_cell_file_prefix(cell.imaging_modality, cell.cell_name)

        if cell['imaging_modality'] in ('clem', 'EM'):
            repair_neuron(
                cell['swc'],
                path=cell_dir / f"{file_prefix}_repaired_mapped.swc",
            )

        elif cell['imaging_modality'] == 'photoactivation':
            os.makedirs(cell_dir, exist_ok=True)
            repair_neuron(
                cell['swc'],
                path=cell_dir / f'{file_prefix}_repaired.swc',
            )

        print(f"{file_prefix}_repaired.swc finished")

