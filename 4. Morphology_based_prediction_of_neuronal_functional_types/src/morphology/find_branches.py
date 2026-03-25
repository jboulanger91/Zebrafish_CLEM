"""Branch identification and analysis for neuron morphology trees."""

import navis
import numpy as np
import pandas as pd


def find_branches(df, cell_name):
    """Identify and characterize branch segments in a neuron skeleton."""
    def find_branch_terminal(
        current_cell, all_branch_segments,
        temp_terminal_list=None, current_length=0,
        all_length=0,
    ):


        # Initialize temp_terminal_list only if it's the first call
        if temp_terminal_list is None:
            temp_terminal_list = []

        for _i, connection in all_branch_segments.loc[
            all_branch_segments['start_node']
            == current_cell['end_node'], :
        ].iterrows():
            all_length += connection['cable_length']
            if connection['end_type'] != 'end':
                temp_terminal_list, all_length = find_branch_terminal(
                    connection, all_branch_segments,
                    temp_terminal_list,
                    current_length + connection['cable_length'],
                    all_length,
                )
            else:
                temp_terminal_list.append([
                    connection['branch_id'],
                    current_length + connection['cable_length'],
                ])

        return temp_terminal_list,all_length






    #inpute is swc.nodes
    all_start_nodes = df[df['type'].isin(['branch',"end"]) ]



    all_start_nodes = all_start_nodes.loc[:, ['node_id', 'parent_id', "x", 'y', 'z', 'type']]

    all_branch_segments = pd.DataFrame(columns=[
        'branch_id', 'branch_type_nodes', 'branch_type_cl',
        'start_node', 'end_node', 'cable_length',
        'n_nodes', 'nodes', 'connected2',
    ])

    for mod_column in ["nodes",'connected2']:
        all_branch_segments[mod_column] = np.nan
        all_branch_segments[mod_column] = all_branch_segments[mod_column].astype(object)



    for branch_id, (_i, end) in enumerate(all_start_nodes.iterrows()):
        len_current_branch = 1
        current_node_type = None
        curent_parent_id = end['parent_id']
        node_ids_in_branch = [end['node_id']]

        while True:
            current_node_id = curent_parent_id
            current_node_type = df.loc[df['node_id'] == current_node_id,'type'].iloc[0]
            curent_parent_id = df.loc[df['node_id'] == current_node_id,'parent_id'].iloc[0]
            len_current_branch+=1
            node_ids_in_branch.append(current_node_id)
            if current_node_type in ['root','branch']:
                break

        temp_df = pd.DataFrame(columns=[
            'branch_id', 'start_node', 'end_node',
            'cable_length', 'n_nodes', 'nodes', 'connected2',
        ])
        for mod_column in ["nodes",'connected2']:
            temp_df[mod_column] = np.nan
            temp_df[mod_column] = temp_df[mod_column].astype(object)

        temp_df.loc[0,'branch_id'] = branch_id
        temp_df.loc[0, 'start_node'] = current_node_id
        temp_df.loc[0, 'end_node'] = end['node_id']
        temp_df.loc[0, 'end_type'] = end['type']
        temp_df.loc[0, 'n_nodes'] = len(node_ids_in_branch)
        temp_df.at[0, 'nodes'] = node_ids_in_branch

        temp_TreeNeuron = df.loc[df['node_id'].isin(temp_df.nodes.iloc[0]),:]
        temp_TreeNeuron.loc[temp_TreeNeuron['node_id']==current_node_id,'parent_id'] = -1
        temp_TreeNeuron = navis.TreeNeuron(temp_TreeNeuron, units='um')
        temp_df.at[0, 'cable_length'] = temp_TreeNeuron.cable_length

        all_branch_segments = pd.concat([all_branch_segments, temp_df])


    max_length = np.max(all_branch_segments.n_nodes)

    all_branch_segments.loc[
        all_branch_segments['n_nodes'] <= (max_length * (1 / 5)),
        'branch_type_nodes',
    ] = '0'
    all_branch_segments.loc[
        (all_branch_segments['n_nodes'] > (max_length * (1 / 5)))
        & (all_branch_segments['n_nodes'] <= (max_length * (2 / 5))),
        'branch_type_nodes',
    ] = '1'
    all_branch_segments.loc[
        (all_branch_segments['n_nodes'] > (max_length * (2 / 5)))
        & (all_branch_segments['n_nodes'] <= (max_length * (3 / 5))),
        'branch_type_nodes',
    ] = '2'
    all_branch_segments.loc[
        (all_branch_segments['n_nodes'] > (max_length * (3 / 5)))
        & (all_branch_segments['n_nodes'] <= (max_length * (4 / 5))),
        'branch_type_nodes',
    ] = '3'
    all_branch_segments.loc[
        all_branch_segments['n_nodes'] > (max_length * (4 / 5)),
        'branch_type_nodes',
    ] = '4'

    max_length = np.max(all_branch_segments.cable_length)




    trick_df = pd.DataFrame(columns=['connected2'])
    for mod_column in ['connected2']:
        trick_df[mod_column] = np.nan
        trick_df[mod_column] = trick_df[mod_column].astype(object)


    for _iq, segment_query in all_branch_segments.iterrows():
        connected_segments = []
        for _it, segment_target in all_branch_segments.loc[
            all_branch_segments['branch_id']
            != segment_query['branch_id'], :
        ].iterrows():
            if np.intersect1d(segment_query['nodes'],segment_target['nodes']).size != 0:
                connected_segments.append(segment_target.branch_id)





        temp_df = pd.DataFrame(columns=['connected2'])
        temp_df['connected2'].astype(object)
        temp_df.at[0, 'connected2'] = connected_segments
        trick_df = pd.concat([trick_df,temp_df])



    all_branch_segments['connected2'] = trick_df




    temp_TreeNeuron = navis.TreeNeuron(df,units='um').prune_by_longest_neurite()
    all_branch_segments.loc[:, 'longest_connected_path'] = temp_TreeNeuron.cable_length






    all_branch_segments['cell_name'] = cell_name


    all_branch_segments = all_branch_segments.reset_index(drop=True)
    for i,branch in all_branch_segments.iterrows():
        temp_longest_neurite_in_branch, temp_total_branch_length = (
            find_branch_terminal(branch, all_branch_segments)
        )
        try:
            all_branch_segments.loc[
                i, 'longest_neurite_in_branch'
            ] = np.max(
                np.array(temp_longest_neurite_in_branch)[:, 1]
            )
        except (IndexError, ValueError):
            all_branch_segments.loc[i, 'longest_neurite_in_branch'] = 0
        all_branch_segments.loc[i, 'total_branch_length'] = temp_total_branch_length

    #mark main path
    all_branch_segments = all_branch_segments.sort_values('start_node').reset_index(drop=True)
    start_min = all_branch_segments['start_node'].min()
    mask = (
        (all_branch_segments['start_node'] == start_min)
        & (all_branch_segments['end_type'] != 'end')
    )
    temp_longest_neurite_in_branch, temp_total_branch_length = (
        find_branch_terminal(
            all_branch_segments.loc[mask, :].iloc[0],
            all_branch_segments,
        )
    )
    arr = np.array(temp_longest_neurite_in_branch)
    end_longest_path = arr[np.argmax(arr[:, 1]), 0]
    item_iter = int(end_longest_path)
    all_branch_segments['main_path'] = False
    while True:
        all_branch_segments.loc[
            all_branch_segments['branch_id'] == item_iter,
            'main_path',
        ] = True
        start_node = all_branch_segments.loc[
            all_branch_segments['branch_id'] == item_iter,
            'start_node',
        ].iloc[0]


        min_start = all_branch_segments['start_node'].min()
        is_root = item_iter in all_branch_segments.loc[
            all_branch_segments['start_node'] == min_start,
            'branch_id',
        ].tolist()
        if is_root:
            all_branch_segments.loc[
                all_branch_segments['branch_id'] == item_iter,
                'main_path',
            ] = True
            break
        item_iter = all_branch_segments.loc[
            all_branch_segments['end_node'] == start_node,
            'branch_id',
        ].iloc[0]

    all_branch_segments.loc[
        all_branch_segments['total_branch_length'] > 0,
        'total_branch_length',
    ]



    return all_branch_segments







