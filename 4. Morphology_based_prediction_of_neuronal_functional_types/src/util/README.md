# src.util -- Utility Module

## Overview

The `src.util` module provides centralized utility functions shared across the
entire zebrafish hindbrain structure-function codebase. Its primary
responsibility is resolving paths from a **unified project root** (`MORPH2FUNC_ROOT`),
where input data, output files, and configuration all live under a single
parent folder.

## File Listing

| File | Description | Key Functions / Classes |
|------|-------------|------------------------|
| `__init__.py` | Package init; re-exports `get_base_path`, `get_project_root`, `get_data_dir`, `get_output_root`, `NotSetup`. | -- |
| `project_root.py` | **Unified project root resolution.** All paths derive from a single root directory. Resolution order: (1) `MORPH2FUNC_ROOT` env var, (2) `~/Desktop/morph2func/` on macOS, (3) `~/morph2func/`, (4) fallback from `config/path_configuration.txt`. | `get_project_root() -> Path`, `get_data_dir() -> Path`, `get_output_root() -> Path` |
| `get_base_path.py` | Resolves the input data path. Delegates to `get_data_dir()` from `project_root.py`, falls back to legacy config file lookup. | `get_base_path() -> Path`, `NotSetup` (exception) |
| `output_paths.py` | Centralized output path configuration. Delegates to `get_output_root()` from `project_root.py`, falls back to `MORPH2FUNC_OUTPUT_ROOT` env var or platform defaults. | `get_output_dir(module, *subdirs) -> Path`, `OUTPUT_ROOT` (module-level Path) |

## Key Exports

```python
get_project_root     # Unified project root (MORPH2FUNC_ROOT)
get_data_dir         # Input data directory ({MORPH2FUNC_ROOT}/data/)
get_output_root      # Output root directory ({MORPH2FUNC_ROOT}/output/)
get_base_path        # Input data path (delegates to get_data_dir)
NotSetup             # Exception raised when base path is not configured
get_output_dir       # Centralized output directory (import from src.util.output_paths)
```

## `MORPH2FUNC_ROOT` Environment Variable

All paths derive from a single project root. Set `MORPH2FUNC_ROOT` to override
automatic detection:

```bash
export MORPH2FUNC_ROOT=~/Desktop/morph2func
```

Expected layout:
```
{MORPH2FUNC_ROOT}/
  data/          # (or morph2func_input/) read-only scientific data
  output/        # (or morph2func_output/) all generated output
```

When `MORPH2FUNC_ROOT` is not set, the root is auto-detected:
- **macOS**: `~/Desktop/morph2func/` (if it exists)
- **Linux / Windows**: `~/morph2func/` (if it exists)
- **Fallback**: derived from `config/path_configuration.txt`

The legacy `MORPH2FUNC_OUTPUT_ROOT` env var is still supported as a fallback for
output path resolution.

Usage example:

```python
from src.util.output_paths import get_output_dir

# Returns {MORPH2FUNC_ROOT}/output/classifier_pipeline/confusion_matrices/
out = get_output_dir("classifier_pipeline", "confusion_matrices")
```

## Cross-Platform Support

- **`get_base_path`**: Works on any platform; reads `config/path_configuration.txt`
  and uses `Path.home().name` to identify the current OS user.
- **`get_output_dir`**: Detects platform via `sys.platform` to choose the
  appropriate default output root.

## Dependencies on Other `src/` Modules

None. This module is a leaf dependency used by `src.myio`, `src.pipelines`,
`src.viz`, and analysis scripts throughout the repository.
