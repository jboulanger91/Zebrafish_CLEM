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
PATH_DIR="/path/to/project_root"  # Base root directory, containing data/, models/, and results/ 

# Explicit paths (OPTIONAL, used only for training)
PATH_DATA="/path/to/project_root/data"
PATH_SAVE="/path/to/project_root/models"
PATH_NOISE_ESTIMATION="/path/to/project_root/data/noise_estimation/contralateral_motion_integrator_preferred_noise_estimation.pkl"
PATH_W_CSV="/path/to/project_root/data/connectome.csv"
LABEL="connectome"
```
Then substitute all placeholder with the actual paths to the relative directories in your system.

N.B. Changing this root .env will affect all the scripts in the project. In case you want more granular configuration,
consider creating .env 

### Data
The scripts in this project have been developed to work with the csv aggregated version of the traces, which you can 
find in the supplementary data.  