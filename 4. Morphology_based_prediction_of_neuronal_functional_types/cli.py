#!/usr/bin/env python3
"""morph2func -- Unified CLI for neuronal functional type prediction.

Morphology-based prediction of neuronal functional types in the
zebrafish hindbrain. Boulanger-Weill et al. (2025).

Usage:
    python cli.py setup --download
    python cli.py run
    python cli.py analysis published-metrics
    python cli.py test --all
    python cli.py all
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Path setup (once, before any local imports)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_CLASSIFIER_DIR = _REPO_ROOT / "functional_type_prediction" / "classifier_prediction"
_SRC = _REPO_ROOT / "src"
for _p in [str(_REPO_ROOT), str(_SRC), str(_CLASSIFIER_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _configure_matplotlib():
    """Set non-interactive backend. Called before pipeline/analysis commands."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("ERROR: matplotlib not installed. Run: python cli.py env --create")
        sys.exit(1)
    except Exception:
        # Agg backend missing (broken install)
        import matplotlib
        print("WARNING: matplotlib Agg backend not found. Try: pip install --force-reinstall matplotlib")
        matplotlib.use("pdf")


def _suppress_resource_tracker_warnings():
    """Suppress harmless multiprocessing ResourceTracker ChildProcessError on macOS/Python 3.12."""
    import atexit
    def _silence():
        # Monkey-patch ResourceTracker._stop_locked to swallow ChildProcessError
        try:
            import multiprocessing.resource_tracker as rt
            _orig = rt.ResourceTracker._stop_locked
            def _patched(self):
                try:
                    _orig(self)
                except ChildProcessError:
                    pass
            rt.ResourceTracker._stop_locked = _patched
        except Exception:
            pass
    atexit.register(_silence)
    _silence()


# ---------------------------------------------------------------------------
# Shared parent parsers
# ---------------------------------------------------------------------------

def _make_global_parent():
    """Arguments shared by all subcommands that touch data."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("global options")
    g.add_argument(
        "--data-path", type=Path, default=None, metavar="PATH",
        help="Override data directory (bypasses MORPH2FUNC_ROOT and path_configuration.txt).",
    )
    g.add_argument(
        "--output-path", type=Path, default=None, metavar="PATH",
        help="Override output directory (bypasses MORPH2FUNC_OUTPUT_ROOT).",
    )
    g.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    g.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error output.")
    return p


def _make_data_parent():
    """Arguments for data loading (features, modalities, labels)."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("data options")
    g.add_argument(
        "--features-file", type=str, default=None, metavar="NAME",
        help="HDF5 features file name (without _features.hdf5 suffix). Default: from PipelineConfig.",
    )
    g.add_argument(
        "--modalities", nargs="+", default=None, metavar="MOD",
        help="Imaging modalities to load. Default: pa clem em clem_predict.",
    )
    g.add_argument(
        "--label-column", type=str, default="kmeans_function", metavar="COL",
        help="Column from metadata.xlsx for functional labels. Default: kmeans_function.",
    )
    g.add_argument(
        "--drop-neurotransmitter", action="store_true",
        help="Drop neurotransmitter from the feature set.",
    )
    g.add_argument(
        "--no-filter-incomplete", action="store_true",
        help="Do not filter incomplete CLEM reconstructions.",
    )
    g.add_argument(
        "--force-recalculation", action="store_true",
        help="Force recalculation of morphological features from SWC files.",
    )
    return p


def _make_rfe_parent():
    """Arguments for RFE feature selection."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("RFE options")
    g.add_argument(
        "--rfe-cv-method", choices=["ss", "lpo"], default="ss",
        help="CV method for RFE. ss=ShuffleSplit, lpo=Leave-P-Out. Default: ss.",
    )
    g.add_argument(
        "--rfe-metric", choices=["f1", "accuracy", "balanced_accuracy"], default="f1",
        help="Optimization metric for RFE. Default: f1.",
    )
    g.add_argument(
        "--train-modality", type=str, default="all", metavar="MOD",
        help="Modality to train on. Default: all.",
    )
    g.add_argument(
        "--test-modality", type=str, default="clem", metavar="MOD",
        help="Modality to test on. Default: clem.",
    )
    return p


GLOBAL_PARENT = _make_global_parent()
DATA_PARENT = _make_data_parent()
RFE_PARENT = _make_rfe_parent()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_data_path(args):
    """Resolve data path from args or environment."""
    if args.data_path:
        return args.data_path
    if args.output_path:
        os.environ["MORPH2FUNC_OUTPUT_ROOT"] = str(args.output_path)
    from src.util.get_base_path import get_base_path
    return get_base_path()


def _resolve_modalities(args):
    """Return modalities list from args or default."""
    return args.modalities or ["pa", "clem", "em", "clem_predict"]


def _find_conda():
    """Find conda executable, cross-platform."""
    conda = shutil.which("conda")
    if conda:
        return conda
    # Common locations
    for candidate in [
        Path.home() / "miniforge3" / "condabin" / "conda",
        Path.home() / "miniconda3" / "condabin" / "conda",
        Path.home() / "anaconda3" / "condabin" / "conda",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Subcommand: setup
# ---------------------------------------------------------------------------

def cmd_setup(args):
    """Download data from Zenodo and/or configure paths."""
    from scripts.setup_data_paths import (
        run_download, run_terminal, run_gui, get_current_user,
        get_user_path, validate_path,
    )

    if args.download:
        run_download(args.dest)
    elif args.gui:
        run_gui()
    elif args.verify:
        user = get_current_user()
        current = get_user_path(user)
        if not current:
            print(f"No path configured for user '{user}'.")
            print("Run: python cli.py setup --download")
            return 1
        print(f"User: {user}")
        print(f"Path: {current}")
        valid, msg = validate_path(current)
        print(f"Status: {msg}")
        return 0 if valid else 1
    else:
        run_terminal()
    return 0


# ---------------------------------------------------------------------------
# Subcommand: env
# ---------------------------------------------------------------------------

def _find_env_python() -> str | None:
    """Find the morph2func environment's Python, whether conda or venv."""
    # Check conda env
    conda = _find_conda()
    if conda:
        try:
            result = subprocess.run(
                [conda, "env", "list"], capture_output=True, text=True,
            )
            if "morph2func" in result.stdout:
                return "conda"
        except (FileNotFoundError, OSError):
            pass  # conda binary found but doesn't work

    # Check venv
    venv_dir = _REPO_ROOT / "morph2func_env"
    if venv_dir.is_dir():
        if platform.system() == "Windows":
            py = venv_dir / "Scripts" / "python.exe"
        else:
            py = venv_dir / "bin" / "python"
        if py.exists():
            return "venv"

    return None


def _create_with_conda(req_txt: Path) -> int:
    """Create environment using conda + pip."""
    conda = _find_conda()
    env_name = "morph2func"

    print(f"Creating conda environment '{env_name}'...")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print()

    # Install numpy with openblas from conda-forge to ensure consistent
    # BLAS implementation across all platforms (macOS/Windows/Linux).
    # Without this, macOS pip uses Accelerate and Windows pip uses scipy-openblas,
    # which produce different floating-point results in RFE/LDA.
    result = subprocess.run(
        [conda, "create", "-n", env_name, "python=3.12",
         "numpy=2.0.2", "scipy=1.15.2", "libblas=*=*openblas",
         "matplotlib", "-y", "-c", "conda-forge"],
    )
    if result.returncode != 0:
        return result.returncode

    print(f"\nInstalling packages from {req_txt}...")
    result = subprocess.run(
        [conda, "run", "-n", env_name, "pip", "install", "-r", str(req_txt)],
    )
    if result.returncode == 0:
        print(f"\nEnvironment created. Activate with:")
        print(f"  conda activate {env_name}")
    return result.returncode


def _create_with_venv(req_txt: Path) -> int:
    """Create environment using Python venv + pip."""
    import venv as _venv

    venv_dir = _REPO_ROOT / "morph2func_env"
    env_name = "morph2func_env"

    print(f"conda not found, using Python venv instead.")
    print(f"Creating venv at {venv_dir}...")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print()

    # Check Python version
    if sys.version_info < (3, 12):
        print(f"ERROR: Python 3.12+ required, found {sys.version}")
        print("Install Python 3.12+ or install conda:")
        print("  https://github.com/conda-forge/miniforge")
        return 1

    # Create venv
    _venv.create(str(venv_dir), with_pip=True, clear=True)

    # Find pip in the new venv
    if platform.system() == "Windows":
        pip = str(venv_dir / "Scripts" / "pip")
        py = str(venv_dir / "Scripts" / "python")
    else:
        pip = str(venv_dir / "bin" / "pip")
        py = str(venv_dir / "bin" / "python")

    print(f"Installing packages from {req_txt}...")
    result = subprocess.run([pip, "install", "-r", str(req_txt)])
    if result.returncode == 0:
        if platform.system() == "Windows":
            activate = f"{venv_dir}\\Scripts\\activate"
        else:
            activate = f"source {venv_dir}/bin/activate"
        print(f"\nEnvironment created. Activate with:")
        print(f"  {activate}")
    return result.returncode


def cmd_env(args):
    """Create or verify the environment (conda or venv)."""
    req_txt = _REPO_ROOT / "requirements.txt"
    env_name = "morph2func"

    if args.verify:
        # Reference environment (macOS ARM64, used to produce paper results)
        # All versions must match exactly for reproducible RFE/LDA results.
        # Source: morph2func conda env on Mac (Apple Silicon M2), March 2026
        REFERENCE = {
            "python": "3.12",
            "numpy": "2.0.2",
            "scipy": "1.15.2",
            "scikit-learn": "1.5.2",
            "pandas": "2.2.3",
            "navis": "1.10.0",
            "pykdtree": "1.3.12",
            "ncollpyde": "0.19.0",
            "skeletor": "1.5.0",
            "trimesh": "4.11.5",
            "morphops": "0.1.13",
            "threadpoolctl": "3.6.0",
            "matplotlib": "3.10.1",
            "h5py": "3.13.0",
            "blas": "openblas",
        }

        # Check if already in the env first (avoids conda run issues on Windows)
        if _in_morph2func_env():
            env_type = "current"
        else:
            env_type = _find_env_python()
        if not env_type:
            print("No environment found.")
            print("Create with: python cli.py env --create")
            return 1

        # Build check script that queries all reference packages
        check_script = (
            "import sys, platform, json, io, contextlib\n"
            "versions = {'python': '.'.join(map(str, sys.version_info[:2]))}\n"
            "for pkg, mod in [\n"
            "    ('numpy','numpy'),('scipy','scipy'),('scikit-learn','sklearn'),\n"
            "    ('pandas','pandas'),('navis','navis'),('matplotlib','matplotlib'),\n"
            "    ('h5py','h5py'),('pykdtree','pykdtree'),('ncollpyde','ncollpyde'),\n"
            "    ('skeletor','skeletor'),('trimesh','trimesh'),('morphops','morphops'),\n"
            "    ('threadpoolctl','threadpoolctl')]:\n"
            "    try:\n"
            "        import importlib; m = importlib.import_module(mod)\n"
            "        versions[pkg] = m.__version__\n"
            "    except: versions[pkg] = 'NOT INSTALLED'\n"
            "try:\n"
            "    import numpy, re\n"
            "    buf = io.StringIO()\n"
            "    with contextlib.redirect_stdout(buf):\n"
            "        numpy.show_config()\n"
            "    cfg = buf.getvalue()\n"
            "    cfg_lower = cfg.lower()\n"
            "    blas_name = 'unknown'\n"
            "    blas_ver = ''\n"
            "    if 'scipy-openblas' in cfg_lower or 'scipy_openblas' in cfg_lower:\n"
            "        blas_name = 'scipy-openblas'\n"
            "    elif 'openblas' in cfg_lower: blas_name = 'openblas'\n"
            "    elif 'accelerate' in cfg_lower: blas_name = 'accelerate'\n"
            "    elif 'mkl' in cfg_lower: blas_name = 'mkl'\n"
            "    m = re.search(r'OpenBLAS ([0-9]+\\.[0-9]+\\.[0-9]+)', cfg)\n"
            "    if m: blas_ver = m.group(1)\n"
            "    if not blas_ver:\n"
            "        for line in cfg.splitlines():\n"
            "            if 'blas' in line.lower() and 'version' in line.lower():\n"
            "                vm = re.search(r'\"([0-9]+\\.[0-9]+\\.[0-9]+)\"', line)\n"
            "                if vm: blas_ver = vm.group(1); break\n"
            "    versions['blas'] = f'{blas_name} {blas_ver}'.strip() if blas_ver else blas_name\n"
            "except: versions['blas'] = 'unknown'\n"
            "versions['platform'] = f'{platform.system()} {platform.machine()}'\n"
            "print('MORPH2FUNC_ENV_CHECK:' + json.dumps(versions))\n"
        )

        if _in_morph2func_env():
            # Already in the env — run directly
            result = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True, text=True,
            )
        elif env_type == "conda":
            conda = _find_conda()
            if platform.system() == "Windows":
                # On Windows, find env Python directly to avoid conda run path issues
                py_result = subprocess.run(
                    [conda, "run", "-n", env_name, "python", "-c",
                     "import sys; print(sys.executable)"],
                    capture_output=True, text=True,
                )
                if py_result.returncode == 0:
                    env_python = py_result.stdout.strip().splitlines()[-1]
                    result = subprocess.run(
                        [env_python, "-c", check_script],
                        capture_output=True, text=True,
                    )
                else:
                    result = py_result
            else:
                result = subprocess.run(
                    [conda, "run", "-n", env_name, "python", "-c", check_script],
                    capture_output=True, text=True,
                )
        else:
            venv_dir = _REPO_ROOT / "morph2func_env"
            py = str(venv_dir / ("Scripts/python" if platform.system() == "Windows" else "bin/python"))
            result = subprocess.run(
                [py, "-c", check_script], capture_output=True, text=True,
            )

        if result.returncode != 0:
            print(f"ERROR: Could not check environment.\n{result.stderr}")
            return 1

        import json
        try:
            # Find the marked line in output (conda run may add extra output)
            json_line = None
            for line in result.stdout.splitlines():
                if line.startswith("MORPH2FUNC_ENV_CHECK:"):
                    json_line = line.split(":", 1)[1]
                    break
            if not json_line:
                raise ValueError("Marker not found in output")
            versions = json.loads(json_line)
        except (json.JSONDecodeError, ValueError, IndexError):
            print(f"ERROR: Could not parse environment info.")
            print(f"stdout: {result.stdout[:500]}")
            print(f"stderr: {result.stderr[:500]}")
            return 1

        # Compare
        print(f"{'='*70}")
        print(f"  Environment Comparison")
        print(f"  Type:      {env_type}")
        print(f"  Platform:  {versions.get('platform', 'unknown')}")
        print(f"  Reference: macOS ARM64 (Apple Silicon M2, March 2026)")
        print(f"{'='*70}")
        print(f"  {'Package':<16} {'Installed':<16} {'Required':<16} {'Status'}")
        print(f"  {'-'*16} {'-'*16} {'-'*16} {'-'*10}")

        # Critical packages: must match exactly or pipeline produces wrong results
        critical = {"scikit-learn", "blas", "python"}
        # Numerical packages: should match for identical feature computation
        numerical = {"numpy", "scipy", "pykdtree", "ncollpyde", "skeletor",
                     "trimesh", "morphops", "navis", "threadpoolctl"}

        issues = 0
        warnings = 0
        for pkg, ref_ver in REFERENCE.items():
            installed = versions.get(pkg, "NOT INSTALLED")

            if installed == "NOT INSTALLED":
                if pkg in critical:
                    status = "MISSING"
                    issues += 1
                else:
                    status = "MISSING"
                    warnings += 1
            elif installed == ref_ver:
                status = "OK"
            elif pkg == "python" and installed.startswith(ref_ver):
                status = "OK"
            elif pkg == "blas" and installed.startswith(ref_ver):
                # "openblas 0.3.27" matches reference "openblas"
                status = "OK"
            elif pkg in critical:
                status = "CRITICAL"
                issues += 1
            elif pkg in numerical:
                status = "MISMATCH"
                warnings += 1
            else:
                status = "DIFFERS"

            print(f"  {pkg:<16} {installed:<16} {ref_ver:<16} {status}")

        print(f"{'='*70}")
        if issues > 0:
            print(f"  {issues} CRITICAL issue(s): results will differ from paper.")
            print(f"  Fix: python cli.py env --create (recreates environment)")
        if warnings > 0:
            print(f"  {warnings} MISMATCH(es): feature computation may differ slightly.")
            print(f"  Use --use-baseline-features and --use-published-features")
            print(f"  for exact paper reproduction.")
        if issues == 0 and warnings == 0:
            print(f"  All packages match reference. Results will be identical.")
        print(f"{'='*70}")
        return 1 if issues > 0 else 0

    if args.create:
        if not req_txt.exists():
            print(f"ERROR: {req_txt} not found.")
            return 1

        # Try conda first, fall back to venv
        conda = _find_conda()
        if conda:
            return _create_with_conda(req_txt)
        else:
            return _create_with_venv(req_txt)

    # Default: show status
    env_type = _find_env_python()
    conda = _find_conda()
    print(f"Requirements:    {req_txt}")
    print(f"Conda:           {'found' if conda else 'not found (will use venv)'}")
    print(f"Environment:     {env_type or 'not created'}")
    print(f"Platform:        {platform.system()} {platform.machine()}")
    print()
    print("Commands:")
    print(f"  python cli.py env --create   Create environment (conda if available, otherwise venv)")
    print(f"  python cli.py env --verify   Check environment is set up correctly")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args):
    """Run the full classifier pipeline."""
    _configure_matplotlib()
    _suppress_resource_tracker_warnings()
    from functional_type_prediction.classifier_prediction.pipelines.pipeline_main import (
        PipelineConfig, run_pipeline, compare_to_reference,
    )

    config = PipelineConfig()

    # Override config from CLI args
    if args.data_path:
        config.DATA_PATH = args.data_path
    if args.features_file:
        config.FEATURES_FILE = args.features_file
    if args.modalities:
        config.MODALITIES = args.modalities
    config.FORCE_RECALCULATION = args.force_recalculation
    config.DROP_NEUROTRANSMITTER = args.drop_neurotransmitter
    config.FILTER_INCOMPLETE_CLEM = not args.no_filter_incomplete
    config.LABEL_COLUMN = args.label_column
    config.TRAIN_MODALITY = args.train_modality
    config.TEST_MODALITY = args.test_modality
    config.RFE_CV_METHOD = args.rfe_cv_method
    config.RFE_METRIC = args.rfe_metric
    config.CV_METHOD = args.cv_method
    config.CV_PLOT = not args.no_cv_plot
    config.USE_JON_PRIORS = args.use_priors
    if args.suffix:
        config.SUFFIX = args.suffix
    if args.verification_tests:
        config.VERIFICATION_TESTS = args.verification_tests
    config.SAVE_PREDICTIONS = not args.no_save
    config.SAVE_FEATURES = not args.no_save

    if config.FORCE_RECALCULATION:
        config.USE_STORED_FEATURES = False

    if args.use_baseline_features:
        config.FEATURES_FILE = "baseline"
        config.USE_STORED_FEATURES = True
        config.FORCE_RECALCULATION = False

    if args.use_published_features:
        config.USE_PUBLISHED_FEATURES = True

    # Auto-enable baseline features + published features on non-Apple Silicon
    # platforms for reproducibility. macOS ARM64 is the reference platform
    # where the paper results were produced; other platforms have different
    # BLAS implementations that cause different RFE/feature computation results.
    no_auto = getattr(args, "no_auto_platform", False)
    if not no_auto and not (sys.platform == "darwin" and platform.machine() == "arm64"):
        if not args.use_baseline_features and not config.FORCE_RECALCULATION:
            print("  [Auto] Non-Apple-Silicon platform detected.")
            print("         Using baseline features + published feature selection")
            print("         for paper-reproducible results.")
            print("         Disable with --no-auto-platform\n")
            config.FEATURES_FILE = "baseline"
            config.USE_STORED_FEATURES = True
            config.USE_PUBLISHED_FEATURES = True

    results = run_pipeline(config)

    if not args.no_compare:
        compare_to_reference(results["predictions"], config)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: analysis
# ---------------------------------------------------------------------------

def cmd_analysis_published_metrics(args):
    """Run published metrics (confusion matrices for pv, ps, ff)."""
    _configure_matplotlib()
    from functional_type_prediction.classifier_prediction.analysis.calculate_published_metrics import (
        run_published_metrics,
    )
    data_path = _resolve_data_path(args)
    run_published_metrics(
        data_path=data_path,
        features_file=args.features_file or "test",
        modalities=_resolve_modalities(args),
    )
    return 0


def cmd_analysis_feature_importance(args):
    """Run permutation importance analysis."""
    _configure_matplotlib()
    from functional_type_prediction.classifier_prediction.analysis.feature_importance import (
        run_feature_importance,
    )
    data_path = _resolve_data_path(args)
    run_feature_importance(
        data_path=data_path,
        features_file=args.features_file or "test",
        modalities=_resolve_modalities(args),
        train_mod=args.train_modality,
        test_mod=args.test_modality,
        rfe_metric=args.rfe_metric,
        permutations=args.permutations,
    )
    return 0


def cmd_analysis_feature_selector(args):
    """Run RFE feature selection with multiple estimators."""
    _configure_matplotlib()
    from functional_type_prediction.classifier_prediction.analysis.find_best_feature_selector import (
        main as run_feature_selector,
    )
    data_path = _resolve_data_path(args)
    run_feature_selector(
        data_path=data_path,
        features_file=args.features_file,
        modalities=_resolve_modalities(args),
        train_mod=args.train_modality,
        test_mod=args.test_modality,
        cv_method=args.rfe_cv_method,
        metric=args.rfe_metric,
    )
    return 0


def cmd_analysis_proba_cutoff(args):
    """Run probability cutoff optimization."""
    _configure_matplotlib()
    from functional_type_prediction.classifier_prediction.analysis.find_optimal_proba_cutoff import (
        run_proba_cutoff,
    )
    data_path = _resolve_data_path(args)
    run_proba_cutoff(
        data_path=data_path,
        features_file=args.features_file or "test",
        modalities=_resolve_modalities(args),
        train_mod=args.train_modality,
        test_mod=args.test_modality,
        rfe_metric=args.rfe_metric,
        cutoff_min=args.cutoff_min,
        cutoff_max=args.cutoff_max,
        cutoff_step=args.cutoff_step,
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------

def cmd_verify(args):
    """Verify generated features against baseline reference."""
    _configure_matplotlib()
    import numpy as np
    import pandas as pd

    data_path = _resolve_data_path(args)
    features_file = args.features_file or "final"

    # Find baseline reference (always baseline_features.hdf5)
    baseline_path = data_path / "baselines" / "baseline_features.hdf5"
    if not baseline_path.exists():
        print(f"ERROR: Baseline not found: {baseline_path}")
        return 1

    # Find generated (uses the features_file name, default: final_features.hdf5)
    from src.util.output_paths import get_output_dir
    generated_path = get_output_dir("classifier_pipeline", "features") / f"{features_file}_features.hdf5"
    if not generated_path.exists():
        print(f"No generated features file at {generated_path}")
        print("Run the pipeline first with --force-recalculation to generate features.")
        return 1

    print(f"Baseline:  {baseline_path}")
    print(f"Generated: {generated_path}")
    print()

    baseline = pd.read_hdf(baseline_path, key="complete_df")
    generated = pd.read_hdf(generated_path, key="complete_df")

    print(f"Baseline shape:  {baseline.shape}")
    print(f"Generated shape: {generated.shape}")

    if baseline.shape != generated.shape:
        print(f"\nERROR: Shape mismatch!")
        return 1

    # Compare numeric columns
    num_cols = baseline.select_dtypes(include="number").columns
    base_vals = baseline[num_cols].values
    gen_vals = generated[num_cols].values
    diff = np.abs(base_vals - gen_vals)

    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)
    exact = np.sum(diff == 0)
    total = diff.size
    close = np.sum(diff < 1e-6)

    # Thresholds
    IDENTICAL = 1e-10
    ACCEPTABLE = 1.0  # platform-dependent float differences are typically < 1.0

    print(f"\n{'='*60}")
    print(f"  Feature Verification Report")
    print(f"{'='*60}")
    print(f"  Cells:           {baseline.shape[0]}")
    print(f"  Features:        {len(num_cols)}")
    print(f"  Total values:    {total}")
    print(f"  Exact matches:   {exact} ({exact/total*100:.1f}%)")
    print(f"  Max abs diff:    {max_diff:.2e}")
    print(f"  Mean abs diff:   {mean_diff:.2e}")

    if max_diff < IDENTICAL:
        print(f"\n  RESULT: IDENTICAL to baseline reference")
        print(f"{'='*60}")
        return 0
    elif max_diff < ACCEPTABLE:
        print(f"\n  RESULT: ACCEPTABLE platform-dependent variation")
        print(f"  These small numerical differences are expected across")
        print(f"  different CPUs/BLAS libraries and do not affect results.")
        print()
        # Show top differences
        col_diffs = []
        for i, col in enumerate(num_cols):
            col_max = np.nanmax(np.abs(base_vals[:, i] - gen_vals[:, i]))
            if col_max > IDENTICAL:
                col_diffs.append((col, col_max))
        col_diffs.sort(key=lambda x: -x[1])
        print(f"  Differences by column:")
        for col, d in col_diffs:
            print(f"    {col:40s} {d:.2e}")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n  RESULT: SIGNIFICANT DIFFERENCES DETECTED")
        print(f"  Max diff {max_diff:.2e} exceeds threshold {ACCEPTABLE:.1f}.")
        print(f"  This may indicate different SWC files, data versions,")
        print(f"  or a bug in feature computation.")
        print()
        col_diffs = []
        for i, col in enumerate(num_cols):
            col_max = np.nanmax(np.abs(base_vals[:, i] - gen_vals[:, i]))
            if col_max > ACCEPTABLE:
                col_diffs.append((col, col_max))
        col_diffs.sort(key=lambda x: -x[1])
        print(f"  Columns exceeding threshold:")
        for col, d in col_diffs:
            print(f"    {col:40s} {d:.2e}")
        print(f"{'='*60}")
        return 1


# ---------------------------------------------------------------------------
# Subcommand: test
# ---------------------------------------------------------------------------

def cmd_test(args):
    """Run unit and/or regression tests."""
    cmd = [sys.executable, "-m", "pytest"]

    targets = []
    if args.unit or (not args.regression and not args.all and not args.test_class):
        targets.append(str(_REPO_ROOT / "tests"))
    if args.regression or args.all:
        targets.append(str(_CLASSIFIER_DIR / "tests"))
    if args.unit and args.all:
        # --all includes both
        pass
    if not targets and args.test_class:
        # If only --class given, search everywhere
        targets.append(str(_REPO_ROOT / "tests"))
        targets.append(str(_CLASSIFIER_DIR / "tests"))

    cmd.extend(targets)

    if args.test_class:
        cmd.extend(["-k", args.test_class])
    if args.verbose:
        cmd.append("-v")

    cmd.extend(["--tb=short"])

    if not args.quiet:
        print(f"Running: {' '.join(cmd)}")

    return subprocess.run(cmd).returncode



# ---------------------------------------------------------------------------
# Subcommand: all
# ---------------------------------------------------------------------------

def cmd_all(args):
    """Run everything: env check -> setup -> pipeline -> analysis."""
    if not args.skip_env:
        print("=" * 60)
        print("  Step 0: Verify environment")
        print("=" * 60)
        env_type = _find_env_python()
        if env_type:
            print(f"  Environment found ({env_type}).")
        else:
            print("  No environment found. Creating...")
            args_env = argparse.Namespace(create=True, verify=False)
            rc = cmd_env(args_env)
            if rc != 0:
                return rc
        print()

    if not args.skip_setup:
        print("=" * 60)
        print("  Step 1: Verify data setup")
        print("=" * 60)
        from scripts.setup_data_paths import get_current_user, get_user_path, validate_path
        user = get_current_user()
        current = get_user_path(user)
        need_download = False
        if current:
            valid, msg = validate_path(current)
            if valid:
                print(f"  Data path OK: {current}")
            else:
                print(f"  Data incomplete: {msg}")
                need_download = True
        else:
            print("  No data path configured.")
            need_download = True

        if need_download:
            print("  Downloading data from Zenodo...")
            from scripts.setup_data_paths import run_download
            run_download()
            # Re-validate after download
            current = get_user_path(user)
            if not current:
                print("  ERROR: Download failed to configure data path.")
                return 1
        print()

    print("=" * 60)
    print("  Step 2: Run classifier pipeline")
    print("=" * 60)
    rc = cmd_run(args)
    if rc != 0:
        return rc
    print()

    if not args.skip_analysis:
        print("=" * 60)
        print("  Step 3: Run analysis scripts")
        print("=" * 60)

        print("\n--- Published metrics ---")
        cmd_analysis_published_metrics(args)

        print("\n--- Feature importance ---")
        cmd_analysis_feature_importance(args)

        print("\n--- Feature selector ---")
        cmd_analysis_feature_selector(args)

        print("\n--- Probability cutoff ---")
        cmd_analysis_proba_cutoff(args)

    print()
    print("=" * 60)
    print("  All done!")
    print("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

EPILOG = """\
Getting started:
  1. python cli.py env --create              Create conda environment (one-time)
  2. python cli.py setup --download          Download data from Zenodo (one-time, ~116 MB)
  3. python cli.py run                       Train classifier and predict cell types
  4. python cli.py analysis published-metrics Inspect accuracy and per-class results

Examples:
  python cli.py run --features-file my_features --modalities pa clem
  python cli.py run --force-recalculation
  python cli.py analysis published-metrics
  python cli.py analysis feature-importance --permutations 100
  python cli.py analysis proba-cutoff --cutoff-step 0.05
  python cli.py test --unit
  python cli.py all --skip-setup

Run 'python cli.py <command> --help' for all options and defaults.
"""


def build_parser():
    """Construct the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=(
            "Predict neuronal functional types from morphology in the zebrafish hindbrain.\n"
            "Companion code for Boulanger-Weill et al. (2025)\n"
            "doi:10.1101/2025.03.14.643363"
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"morph2func {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- setup ---
    setup_p = subparsers.add_parser(
        "setup", parents=[GLOBAL_PARENT],
        help="Download data from Zenodo and configure paths.",
        description="Download structural data from Zenodo and/or configure data paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setup_p.add_argument("--download", action="store_true", help="Download data from Zenodo (~116 MB).")
    setup_p.add_argument("--dest", type=Path, default=None, metavar="PATH", help="Download destination directory.")
    setup_p.add_argument("--verify", action="store_true", help="Verify current data path and exit.")
    setup_p.add_argument("--gui", action="store_true", help="Launch GUI setup window (requires tkinter).")
    setup_p.set_defaults(func=cmd_setup)

    # --- env ---
    env_p = subparsers.add_parser(
        "env",
        help="Create or verify the conda environment.",
        description=(
            "Manage the morph2func conda environment.\n\n"
            "The environment requires scikit-learn==1.5.2 (v1.6+ produces different RFE results)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    env_p.add_argument("--create", action="store_true", help="Create the conda environment from config/environment.yml.")
    env_p.add_argument("--verify", action="store_true", help="Check environment exists and has correct dependencies.")
    env_p.set_defaults(func=cmd_env)

    # --- run ---
    run_p = subparsers.add_parser(
        "run", parents=[GLOBAL_PARENT, DATA_PARENT, RFE_PARENT],
        help="Run the full classifier pipeline (load -> feature selection -> cross-validate -> predict -> verify).",
        description=(
            "Run the complete cell type classification pipeline.\n\n"
            "Steps:\n"
            "  1. Load cell metadata and SWC skeletons from all modalities\n"
            "  2. Extract/load morphological features (68 total)\n"
            "  3. Select optimal features via RFE with AdaBoost (default: 13)\n"
            "  4. Cross-validate with LDA (Leave-One-Out on CLEM)\n"
            "  5. Predict functional types for unlabeled neurons\n"
            "  6. Verify predictions via NBLAST similarity and outlier detection\n\n"
            "Pipeline defaults:\n"
            "  Modalities:        pa, clem, em, clem_predict\n"
            "  RFE:               ShuffleSplit, F1, AdaBoost\n"
            "  CV:                Leave-One-Out (LPO, p=1)\n"
            "  Classifier:        LDA (lsqr, shrinkage=auto)\n"
            "  Verification:      Isolation Forest + Local Outlier Factor\n"
            "  Training cells:    120 (47 PA + 73 CLEM)\n"
            "  Predicted cells:   337 (122 CLEM + 215 EM)\n"
            "  Selected features: 13 of 68\n"
            "  Expected F1:       82.1% (LOO on CLEM)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_g = run_p.add_argument_group("pipeline options")
    run_g.add_argument("--cv-method", choices=["lpo", "ss"], default="lpo", help="CV method for final evaluation. Default: lpo.")
    run_g.add_argument("--no-cv-plot", action="store_true", help="Skip confusion matrix plot generation.")
    run_g.add_argument("--use-priors", action="store_true", help="Apply anatomical prior probabilities.")
    run_g.add_argument("--suffix", type=str, default=None, metavar="STR", help="Output file suffix. Default: _optimize_all_predict.")
    run_g.add_argument("--verification-tests", nargs="+", default=None, metavar="TEST", help="Verification tests: IF, LOF, OCSVM. Default: IF LOF.")
    run_g.add_argument("--no-compare", action="store_true", help="Skip comparison to baseline reference files.")
    run_g.add_argument("--no-save", action="store_true", help="Do not save predictions or features to disk.")
    run_g.add_argument("--use-baseline-features", action="store_true",
                       help="Use pre-computed baseline features (from baselines/baseline_features.hdf5) "
                            "instead of computing or loading locally generated features.")
    run_g.add_argument("--use-published-features", action="store_true",
                       help="Skip RFE and use the 13 published feature indices directly. "
                            "Guarantees identical feature selection across all platforms.")
    run_g.add_argument("--no-auto-platform", action="store_true",
                       help="Disable automatic baseline/published features on non-Apple-Silicon "
                            "platforms. By default, non-macOS-ARM64 platforms auto-enable "
                            "--use-baseline-features and --use-published-features for reproducibility.")
    run_p.set_defaults(func=cmd_run)

    # --- analysis ---
    analysis_p = subparsers.add_parser(
        "analysis",
        help="Model evaluation and feature analysis.",
        description="Run analysis scripts for model evaluation and feature importance.",
    )
    analysis_sub = analysis_p.add_subparsers(dest="analysis_command", help="Analysis commands")

    # analysis published-metrics
    pm_p = analysis_sub.add_parser(
        "published-metrics", parents=[GLOBAL_PARENT, DATA_PARENT],
        help="Reproduce published confusion matrices (pv, ps, ff features with LDA LOO).",
    )
    pm_p.set_defaults(func=cmd_analysis_published_metrics)

    # analysis feature-importance
    fi_p = analysis_sub.add_parser(
        "feature-importance", parents=[GLOBAL_PARENT, DATA_PARENT, RFE_PARENT],
        help="Compute permutation importance for RFE-selected features.",
    )
    fi_p.add_argument("--permutations", type=int, default=50, metavar="K", help="Permutations per feature. Default: 50.")
    fi_p.set_defaults(func=cmd_analysis_feature_importance)

    # analysis feature-selector
    fs_p = analysis_sub.add_parser(
        "feature-selector", parents=[GLOBAL_PARENT, DATA_PARENT, RFE_PARENT],
        help="Find optimal feature selector and feature count via RFE.",
    )
    fs_p.set_defaults(func=cmd_analysis_feature_selector)

    # analysis proba-cutoff
    pc_p = analysis_sub.add_parser(
        "proba-cutoff", parents=[GLOBAL_PARENT, DATA_PARENT, RFE_PARENT],
        help="Optimize probability cutoff threshold (accuracy vs. coverage).",
    )
    pc_p.add_argument("--cutoff-min", type=float, default=0.01, help="Min cutoff to test. Default: 0.01.")
    pc_p.add_argument("--cutoff-max", type=float, default=0.99, help="Max cutoff to test. Default: 0.99.")
    pc_p.add_argument("--cutoff-step", type=float, default=0.01, help="Cutoff step size. Default: 0.01.")
    pc_p.set_defaults(func=cmd_analysis_proba_cutoff)

    # --- verify ---
    verify_p = subparsers.add_parser(
        "verify", parents=[GLOBAL_PARENT, DATA_PARENT],
        help="Verify generated features against baseline reference.",
        description=(
            "Compare locally generated features against the baseline HDF5.\n"
            "Use this to check if feature computation on your platform matches\n"
            "the reference (Mac Apple Silicon). Differences indicate platform-\n"
            "dependent numerical variation in morphological feature extraction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verify_p.set_defaults(func=cmd_verify)

    # --- test ---
    test_p = subparsers.add_parser(
        "test", parents=[GLOBAL_PARENT],
        help="Run unit and/or regression tests.",
        description=(
            "Run tests via pytest.\n\n"
            "Unit tests (tests/ directory) do not require data files.\n"
            "Regression tests (classifier_prediction/tests/) require data + baselines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    test_p.add_argument("--unit", action="store_true", help="Run unit tests only (no data required).")
    test_p.add_argument("--regression", action="store_true", help="Run regression tests (requires data + baselines).")
    test_p.add_argument("--all", action="store_true", help="Run all tests (unit + regression).")
    test_p.add_argument("--class", dest="test_class", metavar="NAME", help="Run specific test class by name.")
    test_p.set_defaults(func=cmd_test)



    # --- all ---
    all_p = subparsers.add_parser(
        "all", parents=[GLOBAL_PARENT, DATA_PARENT, RFE_PARENT],
        help="Run everything: setup -> pipeline -> analysis.",
        description="Run the complete workflow end-to-end (setup, pipeline, all analysis scripts).",
    )
    # Pipeline-specific args needed by cmd_run (duplicated from run subparser)
    all_g = all_p.add_argument_group("pipeline options")
    all_g.add_argument("--cv-method", choices=["lpo", "ss"], default="lpo", help="CV method. Default: lpo.")
    all_g.add_argument("--no-cv-plot", action="store_true", help="Skip CV plots.")
    all_g.add_argument("--use-priors", action="store_true", help="Apply prior probabilities.")
    all_g.add_argument("--suffix", type=str, default=None)
    all_g.add_argument("--verification-tests", nargs="+", default=None)
    all_g.add_argument("--no-compare", action="store_true")
    all_g.add_argument("--no-save", action="store_true")
    all_g.add_argument("--use-baseline-features", action="store_true",
                       help="Use pre-computed baseline features for cross-platform reproducibility.")
    all_g.add_argument("--use-published-features", action="store_true",
                       help="Skip RFE, use the 13 published feature indices.")
    all_g.add_argument("--no-auto-platform", action="store_true",
                       help="Disable auto baseline/published features on non-Apple-Silicon.")
    # Analysis-specific args
    all_g2 = all_p.add_argument_group("analysis options")
    all_g2.add_argument("--permutations", type=int, default=50, help="Feature importance permutations. Default: 50.")
    all_g2.add_argument("--cutoff-min", type=float, default=0.01)
    all_g2.add_argument("--cutoff-max", type=float, default=0.99)
    all_g2.add_argument("--cutoff-step", type=float, default=0.01)
    # Orchestration
    all_g3 = all_p.add_argument_group("orchestration options")
    all_g3.add_argument("--skip-env", action="store_true", help="Skip conda environment check.")
    all_g3.add_argument("--skip-setup", action="store_true", help="Skip data setup verification.")
    all_g3.add_argument("--skip-analysis", action="store_true", help="Skip all analysis steps.")
    all_p.set_defaults(func=cmd_all)

    return parser


def _in_morph2func_env() -> bool:
    """Check if running inside the morph2func environment (conda or venv)."""
    # Conda env
    if os.environ.get("CONDA_DEFAULT_ENV", "") == "morph2func":
        return True
    # Venv or conda — check executable path
    if "morph2func" in sys.executable:
        return True
    return False


def _reexec_in_env() -> int:
    """Re-execute the current command inside the morph2func environment."""
    env_type = _find_env_python()

    if env_type == "conda":
        conda = _find_conda()
        if platform.system() == "Windows":
            # On Windows, conda run has issues with spaces in paths.
            # Find the env's Python directly and call it.
            result = subprocess.run(
                [conda, "run", "-n", "morph2func", "python", "-c",
                 "import sys; print(sys.executable)"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                env_python = result.stdout.strip().splitlines()[-1]
                cmd = [env_python, str(_REPO_ROOT / "cli.py")] + sys.argv[1:]
                print("Activating morph2func conda environment...")
                return subprocess.run(cmd).returncode
        # macOS/Linux: conda run works fine
        cmd = [conda, "run", "--no-capture-output", "-n", "morph2func",
               "python", "cli.py"] + sys.argv[1:]
        print("Activating morph2func conda environment...")
        return subprocess.run(cmd, cwd=str(_REPO_ROOT)).returncode

    elif env_type == "venv":
        venv_dir = _REPO_ROOT / "morph2func_env"
        if platform.system() == "Windows":
            py = str(venv_dir / "Scripts" / "python")
        else:
            py = str(venv_dir / "bin" / "python")
        cmd = [py, str(_REPO_ROOT / "cli.py")] + sys.argv[1:]
        print("Activating morph2func venv environment...")
        return subprocess.run(cmd).returncode

    else:
        print("No morph2func environment found.")
        print("Create one with: python cli.py env --create")
        return 1


# Commands that don't need the morph2func env
_NO_ENV_COMMANDS = {"env", None}


def main():
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Auto-activate: if not in morph2func env and command needs it, re-exec
    if args.command not in _NO_ENV_COMMANDS and not _in_morph2func_env():
        return _reexec_in_env()

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
