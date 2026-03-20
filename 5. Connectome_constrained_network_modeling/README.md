
### Environment setup
Use the global environment. Optionally, create it
```bash
conda env create -f clem_zfish1_global.yaml
```
Activate it
```bash
conda activate clem_zfish1_global
```
and complement it with the missing dependencies
```bash
conda env update -n my_env --file './5. Connectome_constrained_network_modeling\env_clem_zfish1_model.yaml'   
```