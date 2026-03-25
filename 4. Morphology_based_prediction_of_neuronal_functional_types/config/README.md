# config/ -- Configuration Files

Configuration files for data path resolution and environment setup.

## File Listing

| File | Description |
|---|---|
| `path_configuration.txt` | User-to-data-path mapping for locating the `CLEM_paper_data` directory on each developer's machine |
| `environment.yml` | Conda environment specification (`hbsf` environment) |

## path_configuration.txt

Plain-text file mapping OS usernames to local filesystem paths where the shared `CLEM_paper_data` directory resides. This is the primary mechanism for cross-machine data path resolution.

### Format

```
username /absolute/path/to/CLEM_paper_data
```

Each line contains a username (matched against the current OS user at runtime) followed by a space and the absolute path. Lines starting with `#` are treated as comments. Blank lines are ignored.

### Example

```
fkampf /Users/fkampf/Documents/hindbrain_structure_function/nextcloud
arminbahl /Users/arminbahl/Nextcloud/CLEM_paper_data
ag-bahl C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data
```

### How It Works

The function `src.util.get_base_path.get_base_path()` reads this file and returns a `pathlib.Path` for the current user. If the current username is not found, a `NotSetup` exception is raised with instructions to configure the file.

The configuration file is searched for at `config/path_configuration.txt`.

## environment.yml

Conda environment definition for the `hbsf` environment. Install with:

```bash
conda env create -f config/environment.yml
conda activate hbsf
```

### Key Dependencies

| Category | Packages |
|---|---|
| Core data processing | numpy, pandas, scipy, h5py |
| Neuroanatomy | navis (>= 1.10) |
| Machine learning | scikit-learn (== 1.5.2) |
| Visualization | matplotlib, seaborn, plotly |
| Image processing | tifffile, opencv |
| Utilities | chardet, tqdm, openpyxl |

Python version requirement: >= 3.12.
