# tests/ -- Unit and Integration Test Suite

Pytest-based test suite verifying the public API, data structures, naming conventions, and color systems across the `src/` core library.

## Overview

The test suite is designed to run without any data files. Tests exercise importability, function signatures, constant values, synthetic data processing, and temporary file I/O. Where optional dependencies are not installed (notably `caveclient`), lightweight mocks are injected to keep the full test suite runnable.

## File Listing

| File | Tests | Description |
|---|---|---|
| `test_myio.py` | 13 | I/O loader module (`src.myio`): import smoke tests, API shape, `_parsing.py` utilities, table loader signatures and error handling, mesh loader signatures, integration tests for `load_cells_predictor_pipeline` |
| `test_cell_paths.py` | 14 | Cell path utilities (`src.util.cell_paths`): modality root directory mappings, cell file prefix resolution, path construction for PA/CLEM/EM modalities |
| `test_color_unification.py` | 51 | Color dictionary completeness and consistency: `FUNCTIONAL_COLORS` (hex strings), `CONNECTOME_COLORS_RGBA` (RGBA tuples), `MODALITY_COLORS` (case-insensitive), `NEURON_COLORS_RGB`, `POPULATION_COLORS` (MI/MON/SMI); utility functions (`hex_to_rgba`, `hex_to_rgb_tuple`, `get_functional_color`, `get_modality_color`); `__all__` exports; GUI consistency checks ensuring no standalone `FUNCTIONAL_COLORS` redefinition in GUI files |

## Test Organization

Tests are organized by test class within each module:

### test_myio.py

- `TestImportAndAPIShape` -- Verifies all public symbols in `src.myio.__all__` are importable
- `TestParsing` -- Tests `read_to_pandas_row` key=value parsing including edge cases
- `TestTableLoaders` -- Signature and error handling for `load_pa_table`, `load_clem_table`, `load_em_table`
- `TestMeshLoaders` -- Signature tests for `load_mesh`
- `TestIntegration` -- `load_cells_predictor_pipeline` parameter verification, return type annotations, module docstring

### test_color_unification.py

- `TestFunctionalColorsCanonical` -- Required keys, hex string format
- `TestConnectomeColorsCanonical` -- Required keys, RGBA tuple format with [0,1] range
- `TestModalityColors` -- Lowercase and uppercase modality keys
- `TestNeuronColorsRGB`, `TestPopulationColors` -- Required key presence
- `TestUtilityFunctions` -- Correctness of color conversion and lookup functions
- `TestModuleExports` -- `__all__` completeness
- `TestGUIColorConsistency` -- GUI files import from canonical source, no local redefinitions

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_myio.py

# Run a specific test class
pytest tests/test_color_unification.py::TestFunctionalColorsCanonical

# Run a specific test
pytest tests/test_myio.py::TestParsing::test_parses_key_value_format
```

## Dependencies

- `pytest` -- Test framework
- `numpy`, `pandas` -- Data structures for synthetic test data
- `unittest.mock.MagicMock` -- Mock objects for optional dependencies (`caveclient`)
- No data files required; all tests use synthetic data, temporary files, or import verification only

## Mock Strategy

The `caveclient` package is an optional dependency for cloud-based EM data access. If not installed, tests inject a `MagicMock` into `sys.modules["caveclient"]` before any `src.*` imports. This allows the full module import chain (`src.myio` -> `cave_pipeline` -> `caveclient`) to proceed, enabling API shape tests on all modules regardless of the installation environment.
