# Connectome-constrained network modeling

### Project organization
Here is an overview to navigate the project
- `./analysis`: contains intermediate steps of analysis, which may be used to aggregate data and compute important quantities used in model simulation and figure generation
- `./figures`: contains all scripts generating raw versions of the figures in the manuscript
- `./model`: contains the implementation of the model and the training script
- `./utils`: contain useful services, constants, functions, and utility classes, used to run analysis and compute core quantities appearing in the figures

- `./env_clem_zfish1_model.yaml`: environment configuration to install dependencies
- `./noise_estimation.pkl`: precomputed estimation of the noise contribution to augment the dataset of recorded traces

### Environment setup for dependencies
Use the project-specific environment. Create it:
```bash
conda env create -f <path_to_directory_of_this_README>/env_clem_zfish1_model.yaml
```
Activate it
```bash
conda activate clem_zfish1_global
```

### Environment variables
In order for all the training and figure-generating scripts to get access to the right path and data, 
you will need to create a file named `.env` in the same directory as this README file.
Open it and copy-paste this template in:
```angular2html
PATH_DATA=<path_to_directory_containing_activity_traces>
PATH_MODELS=<path_to_directory_containing_models_to_analyze>
PATH_NOISE_ESTIMATION=<path_to_noise_estimation>  # you find a precomputed model in the the folder of this README
PATH_SAVE=<path_where_you_want_to_store_results>

PATH_DATA_NOISE=<path_to_file_with_traces_to_use_for_noise_computation>  # [OPTIONAL] only needed to compute noise estimation
PATH_MODELS_LOADMASK=<path_to_directory_containing_models_trained_with_best_mask>  # [OPTIONAL] only needed to compare loss distributions
```
Then substitute all placeholder with the actual paths to the relative directories in your system.

N.B. Changing this root .env will affect all the scripts in the project. In case you want more granular configuration,
consider creating .env 

### Data
The scripts in this project have been developed to work with the csv aggregated version of the traces, which you can 
find in the supplementary data.  