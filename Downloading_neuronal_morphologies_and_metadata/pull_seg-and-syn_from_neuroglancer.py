"""
Neuroglancer Segment and Synapse Retrieval for Zebrafish Connectome (clem_zfish1)

Version: 0.6
Date: 30/07/2024
Author: Jonathan Boulanger-Weill

Overview:
---------
This script automates the process of retrieving and processing neuronal segments and synapses from the 
Neuroglancer platform, specifically for the clem_zfish1 dataset. The retrieved data is processed and stored 
in metadata files, with additional functionality to manage problematic synapses, generate visualizations 
of neuronal activity, and handle manual synapse annotations.

Functionality:
--------------
- **Segment and Synapse Retrieval:** 
  The script pulls neuronal segments and associated synapse data using Neuroglancer and CAVEclient APIs. 
  It ensures all valid segments and synapses are retrieved, while handling problematic axons and dendrites.
  
- **Data Processing and Metadata Generation:**
  The script generates detailed metadata files for each neuron, including structural and functional information, 
  reconstruction status, and synapse connectivity details. It also uploads these segments to a specified directory 
  and processes related functional data.

- **Visualization:**
  It visualizes neuronal activity dynamics, generating smoothed traces and individual trial data, and saves these 
  visualizations as PDF files.

Dependencies:
-------------
- Python 3.10.13
- Packages: 
  - navis
  - cloudvolume
  - numpy
  - pandas
  - h5py
  - matplotlib
  - scipy
  - caveclient

Installation:
-------------
To set up the environment, use the following command:
    conda env create --file pull_from_neuroglancer.yaml

Usage:
------
1. Set the ROOT_PATH and PATH_ALL_CELLS to the appropriate directories.
2. Ensure access to a private CloudVolume token for data retrieval.
3. Run the script to generate metadata files, retrieve synapses, and visualize activity dynamics.
4. For access or further inquiries, contact jonathan.boulanger.weill@gmail.com.

Example:
--------
seed_cell_ids = ['576460752680588674', '576460752680588675']  # Example seed cell IDs
output_neurons, input_neurons = get_missed_outputs_inputs_neurons(ROOT_CELLS, PATH_ALL_CELLS, seed_cell_ids)

This example processes the specified cells and generates metadata, synapse data, and visualizations for the selected neurons.

License:
--------
This script is released under the MIT License.
"""

import os
import datetime 
from datetime import date
from pathlib import Path

import navis
import cloudvolume as cv
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from caveclient import CAVEclient
import warnings
warnings.filterwarnings("ignore")

# Patch cloudvolume for navis
navis.patch_cloudvolume()

# Define constants and paths
CLOUD_VOLUME_URL = "graphene://https://data.proofreading.zetta.ai/segmentation/api/v1/lichtman_zebrafish_hindbrain_001"
DATASTACK_NAME = "lichtman_zebrafish_hindbrain"
SERVER_ADDRESS = "https://proofreading.zetta.ai"

EXCEL_FILE_PATH = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/collab_rgcs/rgc_axons_output_020525.csv"
MANUAL_SYNAPSES_PATH = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/synapses/manual_synapses"
ROOT_PATH = Path("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/collab_rgcs/traced_axons_neurons")
HDF5_PATH="/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/all_cells.h5"
SIZE_CUT_OFF = 44

# Initialize CloudVolume and CAVEclient
vol = cv.CloudVolume(CLOUD_VOLUME_URL, use_https=True, progress=False)
client = CAVEclient(datastack_name=DATASTACK_NAME, server_address=SERVER_ADDRESS)

# Load the entire workbook
df = pd.read_csv(EXCEL_FILE_PATH, dtype=str)
num_cells = len(df)

# Get synapse table info
synapse_table = client.info.get_datastack_info()['synapse_table']

# Function to format synapse data
def format_synapse(segment_id, position, id, size, manual, validation, date):
    segment_str = str(segment_id)
    position_str = ','.join(map(str, position))
    id_str = '' if id is None else str(id)
    size_str = '' if size is None else str(size)
    return f"{segment_str},{position_str},{id_str},{size_str},{manual},{validation},{date}"

# Function to check problematic axons and dendrites
def check_problematic_synapses(df, synapse_table):
    problematic_axons = []
    problematic_dendrites = []

    for idx in range(num_cells):
        print(f"Processed cells: {idx}") 
        axon_id = str(df.iloc[idx, 7])
        try:
            client.materialize.live_query(
                synapse_table,
                datetime.datetime.now(datetime.timezone.utc),
                filter_equal_dict={'pre_pt_root_id': int(axon_id)}
            )
        except Exception:
            problematic_axons.append(axon_id)
            print(f"Problematic axon: {axon_id}") 

        dendrites_id = str(df.iloc[idx, 8])
        if dendrites_id != '0':  # In case we are only downloading an axon
            try:
                client.materialize.live_query(
                    synapse_table,
                    datetime.datetime.now(datetime.timezone.utc),
                    filter_equal_dict={'post_pt_root_id': int(dendrites_id)}
                )
            except Exception:
                problematic_dendrites.append(dendrites_id)
                print(f"Problematic dendrite: {dendrites_id}") 

    return problematic_axons, problematic_dendrites

problematic_axons, problematic_dendrites = check_problematic_synapses(df, synapse_table)

print("Problematic axons:", problematic_axons)
print("Problematic dendrites:", problematic_dendrites)

def generate_metadata_files(df, root_path):
    num_cells = len(df)
    
    for idx in range(num_cells):
        element_type = str(df.iloc[idx, 0])
        print(f"Processing {element_type} {idx + 1} / {num_cells}, ID: {df.iloc[idx, 5]}, functional ID: {df.iloc[idx, 1]}")

        # Common fields
        functional_id = str(df.iloc[idx, 1])
        connectivity =  "na" if pd.isnull(df.iloc[idx, 2]) else str(df.iloc[idx, 2])
        comment = "na" if pd.isnull(df.iloc[idx, 3]) else str(df.iloc[idx, 3])
        reconstruction_status = str(df.iloc[idx, 4])
        neuroglancer_link = str(df.iloc[idx, 12])
        imaging_modality = str(df.iloc[idx, 13])
        date_of_tracing = str(df.iloc[idx, 14])
        tracer_names = str(df.iloc[idx, 15])
        
        if element_type == 'cell':
            nucleus_id = str(df.iloc[idx, 5])
            soma_id = str(df.iloc[idx, 6])
            axon_id = str(df.iloc[idx, 7])
            dendrites_id = str(df.iloc[idx, 8])
            segment_id = f"clem_zfish1_cell_{nucleus_id}"

            functional_classifier = str(df.iloc[idx, 9])
            neurotransmitter_classifier = str(df.iloc[idx, 10])
            projection_classifier = str(df.iloc[idx, 11])

            if axon_id == '0':
                axon_id = "na"
            if dendrites_id == '0':
                dendrites_id = "na"

            lines = [
                "type = \"" + 'cell' + "\"", 
                "cell_name = " + nucleus_id, 
                "nucleus_id = " + nucleus_id, 
                "soma_id = " + soma_id,  
                "axon_id = " + axon_id, 
                "dendrites_id = " + dendrites_id,
                "functional_id = \"" + functional_id + "\"", 
                "cell_type_labels = [" + "\"" + functional_classifier + "\"" + ", " + "\"" + neurotransmitter_classifier + "\"" + ", " + "\"" + projection_classifier + "\"" + "]",
                "imaging_modality = \"" + imaging_modality + "\"",
                "date_of_tracing = " + " " + date_of_tracing,
                "tracer_names = \"" + tracer_names + "\"",
                "neuroglancer_link = \"" + neuroglancer_link + "\"",
                "connectivity = \"" + connectivity + "\"",
                "reconstruction_status = \"" + reconstruction_status + "\"",
                "comment = \"" + comment + "\"",
            ]
        else:
            nucleus_id = "na"
            soma_id = "na"
            axon_id = str(df.iloc[idx, 7])
            dendrites_id = "na"
            segment_id = f"clem_zfish1_axon_{axon_id}"

            functional_classifier = "na"
            neurotransmitter_classifier = "na"
            projection_classifier = "na"

            lines = [
                "type = \"" + 'axon' + "\"",
                "cell_name = \"" + 'na' + "\"", 
                "nucleus_id = \"" + 'na' + "\"", 
                "soma_id = \"" + 'na' + "\"",  
                "axon_id = " + axon_id, 
                "dendrites_id = \"" + 'na' + "\"",
                "functional_id = \"" + functional_id + "\"", 
                "cell_type_labels = [" + "\"" + functional_classifier + "\"" + ", " + "\"" + neurotransmitter_classifier + "\"" + ", " + "\"" + projection_classifier + "\"" + "]",
                "imaging_modality = \"" + imaging_modality + "\"",
                "date_of_tracing = " + " " + date_of_tracing,
                "tracer_names = \"" + tracer_names + "\"",
                "neuroglancer_link = \"" + neuroglancer_link + "\"",
                "connectivity = \"" + connectivity + "\"",
                "reconstruction_status = \"" + reconstruction_status + "\"",
                "comment = \"" + comment + "\"",
            ]

        # Create directory for the element
        element_path = os.path.join(root_path, segment_id)
        if not os.path.exists(element_path):
            os.makedirs(element_path)

        # Write metadata to a text file
        path_text_file = os.path.join(element_path, f"{segment_id}_metadata.txt")
        with open(path_text_file, "w") as f:
            for line in lines:
                f.write(line)
                f.write("\n")

        upload_segments(element_type, df, idx, segment_id)
        process_functional_data(df, idx, segment_id)

def upload_segments(element_type, df, idx, segment_id):
    if element_type == 'cell':
        upload_cell_segments(df, idx, segment_id)
    else:
        upload_axon_segments(df, idx, segment_id)

def upload_cell_segments(df, idx, segment_id):
    nucleus_id = str(df.iloc[idx, 5])
    soma_id = str(df.iloc[idx, 6])
    axon_id = str(df.iloc[idx, 7])
    dendrites_id = str(df.iloc[idx, 8])
    save_mesh(segment_id, soma_id, nucleus_id, axon_id, dendrites_id)
    
    synapse_file_path = ROOT_PATH / segment_id / f"{segment_id}_synapses.txt"
    write_synapse_file(synapse_file_path, axon_id, dendrites_id, segment_id, date)

def upload_axon_segments(df, idx, segment_id):
    axon_id = str(df.iloc[idx, 7])
    save_mesh(segment_id, soma_id=None, nucleus_id=None, axon_id=axon_id, dendrites_id=None)
    
    synapse_file_path = ROOT_PATH / segment_id / f"{segment_id}_synapses.txt"
    write_synapse_file(synapse_file_path, axon_id, "0", segment_id, date)

def save_mesh(segment_id, soma_id=None, nucleus_id=None, axon_id=None, dendrites_id=None):
    segment_path = os.path.join(ROOT_PATH, segment_id)
    os.makedirs(segment_path, exist_ok=True)

    neuron_parts = []

    # Function to check if ID is valid
    def is_valid_id(mesh_id):
        return mesh_id is not None and mesh_id != '0'

    # Get and save soma and nucleus if they exist
    if is_valid_id(soma_id) and is_valid_id(nucleus_id):
        soma_parts = vol.mesh.get([soma_id, nucleus_id], as_navis=True)
        soma_path = os.path.join(segment_path, f"{segment_id}_soma.obj")
        soma_nuc = navis.combine_neurons(soma_parts)
        navis.write_mesh(soma_nuc, soma_path, filetype="obj")
        neuron_parts.extend([soma_id, nucleus_id])

    # Get and save axon if it exists
    if is_valid_id(axon_id):
        axon = vol.mesh.get([axon_id], as_navis=True)
        axon_path = os.path.join(segment_path, f"{segment_id}_axon.obj")
        navis.write_mesh(axon, axon_path, filetype="obj")
        neuron_parts.append(axon_id)

    # Get and save dendrites if they exist
    if is_valid_id(dendrites_id):
        dendrites = vol.mesh.get([dendrites_id], as_navis=True)
        dendrites_path = os.path.join(segment_path, f"{segment_id}_dendrite.obj")
        navis.write_mesh(dendrites, dendrites_path, filetype="obj")
        neuron_parts.append(dendrites_id)

    # Get and save the whole neuron if more than one part exists
    if len(neuron_parts) > 1:
        neuron_parts_data = vol.mesh.get(neuron_parts, as_navis=True)
        neuron_path = os.path.join(segment_path, f"{segment_id}.obj")
        neuron = navis.combine_neurons(neuron_parts_data)
        navis.write_mesh(neuron, neuron_path, filetype="obj")

        # Plot to double check
        # fig = neuron.plot3d()

def convert_to_int_safe(value, default=0):
        try:
            return int(value)
        except ValueError:
            return default
 
def write_synapse_file(synapse_file_path, axon_id, dendrites_id, segment_id, date):
    synapse_table = client.info.get_datastack_info()['synapse_table']

    # Initialize output_synapses and input_synapses with empty dataframes
    output_synapses = pd.DataFrame()
    input_synapses = pd.DataFrame()

    # Fetch output synapses if axon_id is not "0"
    if axon_id != "0":
        output_synapses = client.materialize.live_query(
            synapse_table,
            datetime.datetime.now(datetime.timezone.utc),
            filter_equal_dict={'pre_pt_root_id': int(axon_id)}
        )

    # Fetch input synapses if dendrites_id is not "0"
    if dendrites_id != "0":
        input_synapses = client.materialize.live_query(
            synapse_table,
            datetime.datetime.now(datetime.timezone.utc),
            filter_equal_dict={'post_pt_root_id': int(dendrites_id)}
        )

    # Apply size cut-off
    size_cut_off = SIZE_CUT_OFF

    # Extract individual parameters for output synapses if available
    if not output_synapses.empty:
        output_segment = output_synapses.post_pt_root_id
        output_position = output_synapses.ctr_pt_position.apply(lambda x: [2 * x[0], 2 * x[1], x[2]])
        output_synapse_id = output_synapses.id
        output_size = output_synapses.iloc[:, 4]
        output_prediction_list = ["predicted" for _ in range(len(output_synapses))]
        output_validation_list = ["valid" if value > size_cut_off else "below cut-off" for value in output_synapses.iloc[:, 4]]
    else:
        output_segment = []
        output_position = []
        output_synapse_id = []
        output_size = []
        output_prediction_list = []
        output_validation_list = []

    # Extract individual parameters for input synapses if available
    if not input_synapses.empty:
        input_segment = input_synapses.pre_pt_root_id
        input_position = input_synapses.ctr_pt_position.apply(lambda x: [2 * x[0], 2 * x[1], x[2]])
        input_synapse_id = input_synapses.id
        input_size = input_synapses.iloc[:, 4]
        input_prediction_list = ["predicted" for _ in range(len(input_synapses))]
        input_validation_list = ["valid" if value > size_cut_off else "below cut-off" for value in input_synapses.iloc[:, 4]]
    else:
        input_segment = []
        input_position = []
        input_synapse_id = []
        input_size = []
        input_prediction_list = []
        input_validation_list = []

    # Manually annotated synapses
    manual_synapses = {'pre': {}, 'post': {}}
    
    for syn_type in ['pre', 'post']:
        manual_excel_path = f"{MANUAL_SYNAPSES_PATH}/{segment_id}_{syn_type}synapses_manual.xlsx"
        
        if os.path.exists(manual_excel_path):
            manual_synapses_df = pd.read_excel(manual_excel_path)
            segments = manual_synapses_df['segment_id'].tolist()
            positions = manual_synapses_df[['position_x', 'position_y', 'position_z']].values.tolist()
            manual_synapses[syn_type] = {
                'segments': segments,
                'positions': positions,
                'ids': [None] * len(segments),
                'sizes': [None] * len(segments),
                'prediction_list': ["manual" for _ in range(len(segments))],
                'validation_list': ["valid" for _ in range(len(segments))],  # Assuming manually annotated synapses are always valid
                'date_list': manual_synapses_df['date'].dt.strftime('%Y-%m-%d').tolist()
            }
        else:
            manual_synapses[syn_type] = {
                'segments': [],
                'positions': [],
                'ids': [],
                'sizes': [],
                'prediction_list': [],
                'validation_list': [],
                'date_list': []
            }

    # Write formatted synapses to file
    today_date = date.today().strftime('%Y-%m-%d')
    date_list_output = [today_date for _ in range(len(output_segment))]
    date_list_input = [today_date for _ in range(len(input_segment))]

    with open(synapse_file_path, "w") as file:
        file.write("(presynaptic: [")
        for segment, position, id, size, manual, validation, date in zip(output_segment, output_position, output_synapse_id, output_size, output_prediction_list, output_validation_list, date_list_output):
            file.write("'" + format_synapse(segment, position, id, size, manual, validation, date) + "', ")
        for segment, position, id, size, manual, validation, date in zip(manual_synapses['pre']['segments'], manual_synapses['pre']['positions'], manual_synapses['pre']['ids'], manual_synapses['pre']['sizes'], manual_synapses['pre']['prediction_list'], manual_synapses['pre']['validation_list'], manual_synapses['pre']['date_list']):
            file.write("'" + format_synapse(segment, position, id, size, manual, validation, date) + "', ")
        file.write("],postsynaptic: [")
        for segment, position, id, size, manual, validation, date in zip(input_segment, input_position, input_synapse_id, input_size, input_prediction_list, input_validation_list, date_list_input):
            file.write("'" + format_synapse(segment, position, id, size, manual, validation, date) + "', ")
        for segment, position, id, size, manual, validation, date in zip(manual_synapses['post']['segments'], manual_synapses['post']['positions'], manual_synapses['post']['ids'], manual_synapses['post']['sizes'], manual_synapses['post']['prediction_list'], manual_synapses['post']['validation_list'], manual_synapses['post']['date_list']):
            file.write("'" + format_synapse(segment, position, id, size, manual, validation, date) + "', ")
        file.write("])")

    # Now split the file into pre and post synapses 
    with open(synapse_file_path, 'r') as fp:
        split_data = fp.read().split(",postsynaptic")

    presynaptic_data_str = split_data[0].replace("'", "")
    postsynaptic_data_str = split_data[1].replace("'", "")

    for i, data_str in enumerate([presynaptic_data_str, postsynaptic_data_str]):
        start_index = data_str.find("[")
        end_index = data_str.find("]")
        synaptic_data = data_str[start_index + 1:end_index]
        synaptic_list = synaptic_data.split(", ")

        table_data = []
        for entry in synaptic_list:
            if entry != "[]" and entry != "":
                values = entry.split(",")
                partner_cell_id = values[0].strip("'")
                date = values[8].strip("'")
                
                table_data.append({
                    'partner_cell_id': partner_cell_id,
                    'x': int(values[1]),
                    'y': int(values[2]),
                    'z': int(values[3]),
                    'synapse_id': convert_to_int_safe(values[4]),  # Safely convert to integer
                    'size': convert_to_int_safe(values[5]),        # Safely convert to integer
                    'prediction_status': values[6],
                    'validation_status': values[7],
                    'date': date
                })

        df = pd.DataFrame(table_data)

        #Generate output path 
        synapse_file_path = Path(synapse_file_path)
        new_stem = synapse_file_path.stem.replace('_synapses', '')
        new_synapse_file_path = synapse_file_path.with_name(new_stem + synapse_file_path.suffix)
        output_file = new_synapse_file_path.with_name(new_synapse_file_path.stem + ('_ng_res_presynapses.csv' if i == 0 else '_ng_res_postsynapses.csv'))

        df.to_csv(output_file, index=False, sep=' ', header=None, float_format='%.8f')     

def process_functional_data(df, idx, segment_id):
    functional_id = str(df.iloc[idx, 1])
    if functional_id != 'not functionally imaged':  
        with h5py.File(HDF5_PATH, "r") as hdf_file:
            neuron_group = hdf_file[f"neuron_{functional_id}"] 

            # Get average activity for left and right stimuli
            avg_activity_left = neuron_group["average_activity_left"][()]
            avg_activity_right = neuron_group["average_activity_right"][()]
            # Get individual trial data for left and right stimuli
            trials_left = neuron_group["neuronal_activity_trials_left"][()]
            trials_right = neuron_group["neuronal_activity_trials_right"][()]

            # Create a new hdf5 and save 
            destination_hdf5_path = str(ROOT_PATH) + "/" + segment_id + "/" + segment_id + "_dynamics.hdf5"
            with h5py.File(destination_hdf5_path, "w") as f: 
                f.create_dataset('dF_F/single_trial_rdms_left', data=avg_activity_left)
                f.create_dataset('dF_F/single_trial_rdms_right', data=avg_activity_right)
                f.create_dataset('dF_F/average_rdms_left_dF_F_calculated_on_single_trials', data=avg_activity_left)
                f.create_dataset('dF_F/average_rdms_right_dF_F_calculated_on_single_trials', data=avg_activity_right)

            # Plot the activity traces
            # Smooth using a Savitzky-Golay filter
            smooth_avg_activity_left = savgol_filter(avg_activity_left, 20, 3)
            smooth_avg_activity_right = savgol_filter(avg_activity_right, 20, 3)
            smooth_trials_left = savgol_filter(trials_left, 20, 3, axis=1)
            smooth_trials_right = savgol_filter(trials_right, 20, 3, axis=1)

            # Define time axis in seconds
            dt = 0.5  # Time step is 0.5 seconds
            time_axis = np.arange(len(avg_activity_left)) * dt

            # Plot smoothed average activity with thin lines
            # fig, ax = plt.subplots()
            # plt.plot(time_axis, smooth_avg_activity_left, color='blue', alpha=0.7, linewidth=3, label='Smoothed Average Left')
            # plt.plot(time_axis, smooth_avg_activity_right, color='red', alpha=0.7, linestyle='--', linewidth=3, label='Smoothed Average Right')
            
            # Plot individual trial data with thin black lines for left and dashed black lines for right
            # for trial_left, trial_right in zip(smooth_trials_left, smooth_trials_right):
            #     plt.plot(time_axis, trial_left, color='black', alpha=0.3, linewidth=1)
            #     plt.plot(time_axis, trial_right, color='black', alpha=0.3, linestyle='--', linewidth=1)
            
            # Overlay shaded rectangle for stimulus epoch
            # plt.axvspan(20, 60, color='gray', alpha=0.1, label='Stimulus Epoch')

            # plt.title(f'Average and Individual Trial Activity Dynamics for Neuron {functional_id}')
            # plt.xlabel('Time (seconds)')
            # plt.ylabel('Activity')
            
            # Set font of legend text to Arial
            # legend = plt.legend()
            # for text in legend.get_texts():
            #     text.set_fontfamily('Arial')
                
            # Set aspect ratio to 1
            # ratio = 1.0
            # x_left, x_right = ax.get_xlim()
            # y_low, y_high = ax.get_ylim()
            # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

            #Save figure 
            # path_text_file = str(ROOT_PATH) + "/" + segment_id + "/" + segment_id + "_dynamics.pdf"
            # plt.savefig(path_text_file)
            # plt.show()

# Execute the script
generate_metadata_files(df, ROOT_PATH)

# %%
# Copy non functionally imaged neurons to a separate folder for functional type prediction. 
import os
import shutil

def copy_folders_based_on_functional_id(input_path, not_functionally_imaged_dest, functionally_imaged_dest):
    # List all folders in the input path
    for folder_name in os.listdir(input_path):
        source_folder = os.path.join(input_path, folder_name)
        
        # Ensure it's a directory before proceeding
        if os.path.isdir(source_folder):
            # Define the metadata file path
            metadata_file = os.path.join(source_folder, f"{folder_name}_metadata.txt")
            
            # Check if the metadata file exists
            if os.path.isfile(metadata_file):
                # Read the metadata file and search for the functional_id
                with open(metadata_file, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if "functional_id" in line:
                            # Extract the functional_id value
                            functional_id = line.split('=')[-1].strip().replace('"', '')
                            
                            # Determine the destination folder based on the functional_id
                            if functional_id == "not functionally imaged":
                                dest_folder = os.path.join(not_functionally_imaged_dest, folder_name)
                            else:
                                dest_folder = os.path.join(functionally_imaged_dest, folder_name)
                            
                            # Check if the folder has already been copied
                            if os.path.exists(dest_folder):
                                print(f"Folder {folder_name} already exists in destination, skipping copy.")
                            else:
                                # Copy the folder to the appropriate destination
                                shutil.copytree(source_folder, dest_folder)
                                print(f"Copied {source_folder} to {dest_folder} (functional ID: {functional_id})")
                            break
                    else:
                        print(f"Functional ID not found in metadata for {source_folder}")
            else:
                print(f"Metadata file not found in {source_folder}")

# Example usage
input_path = ROOT_PATH
not_functionally_imaged_dest = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/traced_neurons/non_functionally_imaged_111224/'
functionally_imaged_dest = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/traced_neurons/functionally_imaged_111224/'

copy_folders_based_on_functional_id(input_path, not_functionally_imaged_dest, functionally_imaged_dest)


def count_folders_in_path(path):
    # List all items in the given path and filter for directories
    folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return len(folders)

# Example usage
path = dest_folder # Replace with the actual path
num_folders = count_folders_in_path(ROOT_PATH)

print(f"Number of folders in {path}: {num_folders}")
