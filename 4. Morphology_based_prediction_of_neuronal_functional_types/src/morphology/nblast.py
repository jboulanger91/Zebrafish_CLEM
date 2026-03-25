"""NBLAST morphological similarity analysis for neuron comparisons."""

import copy

import navis
import numpy as np


def nblast_one_group(df, k=5, resample_size=0.1):
    """Compute within-group NBLAST similarity matrix."""
    my_neuron_list = navis.NeuronList(df['swc'],k=k,resample = resample_size)
    dps = navis.make_dotprops(my_neuron_list,k=k,resample = resample_size,progress =False)
    nbl = navis.nblast(dps, dps, progress=False)
    np.array(nbl)
    nbl.index = df.cell_name
    nbl.columns = df.cell_name
    nbl = nbl.iloc[:int(np.floor(len(nbl)/2)),int(np.floor(len(nbl)/2)):]

    return nbl

def nblast_two_groups(
    df1, df2, k=5, resample_size=0.1, shift_neurons=True,
):
    """Compute between-group NBLAST similarity matrix for two neuron sets.

    Args:
        df1: DataFrame with 'swc' and 'cell_name' columns for query neurons.
        df2: DataFrame with 'swc' and 'cell_name' columns for target neurons.
        k: Number of nearest neighbors for dotprop generation.
        resample_size: Resampling resolution for dotprops.
        shift_neurons: If True, translate neurons so their root is at the origin.

    Returns
    -------
        DataFrame of NBLAST scores with df1 cell names as index and df2 as columns.
    """
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    if shift_neurons:
        for i,cell in df1.iterrows():
            import copy
            df1.loc[i, 'swc2'] = copy.deepcopy(df1.loc[i, 'swc'])

            temp = df1.iloc[i]['swc2'].nodes.loc[
                df1.iloc[i]['swc2'].nodes['parent_id'] == -1,
                ['x', 'y', 'z'],
            ]
            dx = 0 - temp['x'].iloc[0]
            dy = 0 - temp['y'].iloc[0]
            dz = 0 - temp['z'].iloc[0]
            df1.loc[i,'swc2'].nodes.loc[:,"x"] = cell.swc.nodes.x + dx
            df1.loc[i,'swc2'].nodes.loc[:,"y"] = cell.swc.nodes.y + dy
            df1.loc[i,'swc2'].nodes.loc[:,"z"] = cell.swc.nodes.z + dz
        for i,cell in df2.iterrows():
            df2.loc[i, 'swc2'] = copy.deepcopy(df2.loc[i, 'swc'])
            temp = df2.iloc[i]['swc2'].nodes.loc[
                df2.iloc[i]['swc'].nodes['parent_id'] == -1,
                ['x', 'y', 'z'],
            ]
            dx = 0 - temp['x'].iloc[0]
            dy = 0 - temp['y'].iloc[0]
            dz = 0 - temp['z'].iloc[0]
            df2.loc[i,'swc2'].nodes.loc[:,"x"] = cell.swc.nodes.x + dx
            df2.loc[i,'swc2'].nodes.loc[:,"y"] = cell.swc.nodes.y + dy
            df2.loc[i,'swc2'].nodes.loc[:,"z"] = cell.swc.nodes.z + dz

        my_neuron_list1 = navis.NeuronList(df1['swc2'], k=k, resample=resample_size)
        my_neuron_list2 = navis.NeuronList(df2['swc2'], k=k, resample=resample_size)
    else:

        my_neuron_list1 = navis.NeuronList(df1['swc'],k=k,resample = resample_size)
        my_neuron_list2 = navis.NeuronList(df2['swc'], k=k, resample=resample_size)


    dps1 = navis.make_dotprops(my_neuron_list1,k=k,resample = resample_size,progress =False)
    dps2 = navis.make_dotprops(my_neuron_list2, k=k, resample=resample_size,progress =False)

    nbl = navis.nblast(dps1, dps2, progress=False)
    np.array(nbl)
    nbl.index = df1.cell_name
    nbl.columns = df2.cell_name
    return nbl


def nblast_two_groups_custom_matrix(
    df1, df2, custom_matrix, k=5,
    resample_size=0.1, shift_neurons=False,
):
    """Compute between-group NBLAST similarity using a custom scoring matrix.

    Args:
        df1: DataFrame with 'swc' and 'cell_name' columns for query neurons.
        df2: DataFrame with 'swc' and 'cell_name' columns for target neurons.
        custom_matrix: Custom NBLAST scoring matrix.
        k: Number of nearest neighbors for dotprop generation.
        resample_size: Resampling resolution for dotprops.
        shift_neurons: If True, translate neurons so their root is at the origin.

    Returns
    -------
        DataFrame of NBLAST scores with df1 cell names as index and df2 as columns.
    """
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    if shift_neurons:
        for i,cell in df1.iterrows():
            import copy
            df1.loc[i, 'swc2'] = copy.deepcopy(df1.loc[i, 'swc'])

            temp = df1.iloc[i]['swc2'].nodes.loc[
                df1.iloc[i]['swc2'].nodes['parent_id'] == -1,
                ['x', 'y', 'z'],
            ]
            dx = 0 - temp['x'].iloc[0]
            dy = 0 - temp['y'].iloc[0]
            dz = 0 - temp['z'].iloc[0]
            df1.loc[i,'swc2'].nodes.loc[:,"x"] = cell.swc.nodes.x + dx
            df1.loc[i,'swc2'].nodes.loc[:,"y"] = cell.swc.nodes.y + dy
            df1.loc[i,'swc2'].nodes.loc[:,"z"] = cell.swc.nodes.z + dz
        for i,cell in df2.iterrows():
            df2.loc[i, 'swc2'] = copy.deepcopy(df2.loc[i, 'swc'])
            temp = df2.iloc[i]['swc2'].nodes.loc[
                df2.iloc[i]['swc'].nodes['parent_id'] == -1,
                ['x', 'y', 'z'],
            ]
            dx = 0 - temp['x'].iloc[0]
            dy = 0 - temp['y'].iloc[0]
            dz = 0 - temp['z'].iloc[0]
            df2.loc[i,'swc2'].nodes.loc[:,"x"] = cell.swc.nodes.x + dx
            df2.loc[i,'swc2'].nodes.loc[:,"y"] = cell.swc.nodes.y + dy
            df2.loc[i,'swc2'].nodes.loc[:,"z"] = cell.swc.nodes.z + dz

        my_neuron_list1 = navis.NeuronList(df1['swc2'], k=k, resample=resample_size)
        my_neuron_list2 = navis.NeuronList(df2['swc2'], k=k, resample=resample_size)
    else:

        my_neuron_list1 = navis.NeuronList(df1['swc'],k=k,resample = resample_size)
        my_neuron_list2 = navis.NeuronList(df2['swc'], k=k, resample=resample_size)
    if shift_neurons:
        my_neuron_list1 = navis.NeuronList(df1['swc2'], k=k, resample=resample_size)
        my_neuron_list2 = navis.NeuronList(df2['swc2'], k=k, resample=resample_size)
    else:
        my_neuron_list1 = navis.NeuronList(df1['swc'], k=k, resample=resample_size)
        my_neuron_list2 = navis.NeuronList(df2['swc'], k=k, resample=resample_size)


    dps1 = navis.make_dotprops(my_neuron_list1,k=k,resample = resample_size,progress =False)
    dps2 = navis.make_dotprops(my_neuron_list2, k=k, resample=resample_size,progress =False)

    nbl = navis.nblast(
        dps1, dps2, progress=False,
        smat=custom_matrix, n_cores=10,
    )
    # nbl = navis.nblast(dps1, dps2, progress=False, n_cores=10)
    np.array(nbl)
    nbl.index = df1.cell_name
    nbl.columns = df2.cell_name
    return nbl


def compute_nblast_within_and_between(
    df, query_keys_input=None,
):
    """Compute within-group and between-group NBLAST scores for a neuron subset.

    Args:
        df: DataFrame containing neuron data with morphology/function/neurotransmitter columns.
        query_keys_input: List of category values (e.g. 'motion_integrator', 'contralateral')
            to define the query group.

    Returns
    -------
        Tuple of (within_values, between_values) as flattened arrays of NBLAST scores.
    """
    if query_keys_input is None:
        query_keys_input = []
    given_categories = []
    query_keys = copy.deepcopy(query_keys_input)

    reverse_cell_type_categories = {
        'ipsilateral': 'morphology',
        'contralateral': 'morphology',
        'inhibitory': 'neurotransmitter',
        'excitatory': 'neurotransmitter',
        'motion_integrator': 'function',
        'motion_onset': 'function',
        'motion onset': 'function',
        'slow_motion_integrator': 'function',
        'slow motion integrator': 'function',
    }

    for key in query_keys:
        given_categories.append(reverse_cell_type_categories[key])

    query_command = ''

    for key in given_categories:
        query_command += f"(df['{key}'].isin(query_keys))&"

    query_command = query_command[:-1]
    if query_command == '':
        raise ValueError(
            'query_keys you have to enter a query_key '
            'for compute_nblast_within_and_between to work.'
        )


    df_query = df.loc[eval(query_command),:]
    df_target = df.loc[~(eval(query_command)),:]

    within_values = np.array(nblast_one_group(df_query)).flatten()
    between_values = np.array(nblast_two_groups(df_query,df_target)).flatten()

    print(f"Mean NBLAST within {query_keys_input}: {np.mean(within_values)}")
    print(f"Mean NBLAST between {query_keys_input}: {np.mean(between_values)}\n")

    return within_values,between_values
