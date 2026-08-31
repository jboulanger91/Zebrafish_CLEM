import sys
from pathlib import Path

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "..")

def generate_env(path_dir, path_data, path_noise_estimation, path_save, path_save_env):
    for model_filepath in Path(path_dir).glob('connectivity_mask_*.csv'):
        # initialize .env
        content = f'''
# --- benchmarking
PATH_DATA={path_data}
PATH_SAVE={path_save}
PATH_NOISE_ESTIMATION={path_noise_estimation}
'''

        model_filename = str(model_filepath.name)
        model_id = model_filename.split("_")[2].split("-")[0]
        content += f"LABEL={model_id}\n"
        content += f"PATH_W_CSV={str(model_filepath)}\n"

        f = open(f"{path_save_env}/.env.model_{model_id}", "x")
        f.write(content)

def generate_slurm(script_path, python_file_path, env_file_path_root, label_script, label_env_list):
    # init and open bash file
    sh_content = f"#!/bin/bash\n"

    for label_env in label_env_list:
        f_slurm = open(f"{script_path}/{label_script}_{label_env}.slurm", mode="a")

        # initialize slurm script
        slurm_content = '''#!/bin/bash
#SBATCH --job-name=rnn_connectome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=/home/kn/kn_kn/kn_pop542534/output/job_output_%j.out
#SBATCH --mail-type=NONE
#SBATCH --mail-user=roberto.garza@uni.kn

source $HOME/miniforge3/etc/profile.d/conda.sh

export PYTHONPATH="${PYTHONPATH}:/home/kn/kn_pop542534/code:/home/kn/kn_pop542534/code/Zebrafish_CLEM/5. Connectome_constrained_network_modeling"

    '''


        # add python command to pbs script
        slurm_content += f'conda run -n py312 python "{python_file_path}" "{env_file_path_root}.{label_env}"\n'
        f_slurm.write(slurm_content)

        # write bash file to launch pbs
        sh_content += f"sbatch {script_path}/{label_script}_{label_env}.slurm\n"

    f_sh = open(f"{script_path}/{label_script}.sh", mode="a")
    f_sh.write(sh_content)

# path_root = "C:/Users/Roberto/Academics/data"  # /home/kn/kn_kn/kn_pop542534/data
# path_experiment = "benchmarking/5_param"
# path = f"{path_root}/{path_experiment}"
# path_save = path
path_dir = "/home/kn/kn_pop542534/data/rnn_ds/connectome_enhancement"
path_data = "/home/kn/kn_pop542534/data/rnn_ds/traces/from_jon"
path_noise_estimation = "/home/kn/kn_pop542534/data/rnn_ds/traces/from_jon/noise_estimation/contralateral_motion_integrator_preferred_noise_estimation.pkl"
path_save = path_data + "/connectome_enhancement"
path_save_env = "/home/kn/kn_pop542534/code/Zebrafish_CLEM/5. Connectome_constrained_network_modeling/model/connectome_enhancement"
generate_env(path_dir, path_data, path_noise_estimation, path_save, path_save_env)

script_path = "/home/kn/kn_pop542534/script/rnn_ds/connectome_enhancement"  # where to save scripts
label_script = "rnn_connectome"  # scripts root name
python_file_path = "/home/kn/kn_pop542534/code/Zebrafish_CLEM/5. Connectome_constrained_network_modeling/model/train_rnn_connectome.py"  # path to python script to execute
env_file_path_root = path_save_env + "/.env"  # path and root label for env file
label_env_list = [str(l.name).split(".")[-1] for l in Path(path_save_env).glob('.env.*')]  # end label for env file
generate_slurm(script_path, python_file_path, env_file_path_root, label_script=label_script, label_env_list=label_env_list)

