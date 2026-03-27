"""Cell Type Classification Pipeline.

Pipeline methods return typed containers for explicit data flow:

    data = predictor.load_data(...)           -> LoadedData
    rfe  = predictor.select_features_rfe(...) -> RFEResult
    cv   = predictor.cross_validate(...)      -> CVResult
    pred = predictor.predict(...)             -> PredictionResults
    pred = predictor.verify(pred, ...)        -> PredictionResults (updated)

Author: Florian Kämpf
"""

import sys
from pathlib import Path

# =============================================================================
# Path Setup
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_CLASSIFIER_DIR = _SCRIPT_DIR.parent
_FUNCTIONAL_DIR = _CLASSIFIER_DIR.parent
_REPO_ROOT = _FUNCTIONAL_DIR.parent
_SRC = _REPO_ROOT / "src"

# _CLASSIFIER_DIR: needed for bare core.* imports
# _SRC: Required for src.* imports (remove after pip install -e .)
for path in [str(_CLASSIFIER_DIR), str(_SRC)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# =============================================================================
# Imports
# =============================================================================
try:
    from src.util.get_base_path import get_base_path  # noqa: E402
    from src.util.output_paths import get_output_dir  # noqa: E402
except ModuleNotFoundError:
    from util.get_base_path import get_base_path  # noqa: E402
    from util.output_paths import get_output_dir  # noqa: E402

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import AdaBoostClassifier  # noqa: E402

_REQUIRED_SKLEARN = "1.5.2"
if sklearn.__version__ != _REQUIRED_SKLEARN:
    import warnings

    warnings.warn(
        f"sklearn {sklearn.__version__} detected, but {_REQUIRED_SKLEARN} is required "
        f"for reproducible results. RFE feature selection and classifier tie-breaking "
        f"differ across versions. Use environment py312_fish_clem_paper_skl152.",
        stacklevel=1,
    )

from src.myio.load_cells2df import normalize_label  # noqa: E402

from core.class_predictor import class_predictor  # noqa: E402

matplotlib.use("Agg")  # Non-interactive backend


# =============================================================================
# Configuration
# =============================================================================
#
# DATA FLOW SUMMARY (with current settings):
# -----------------------------------------------
# 1. load_cells_predictor_pipeline  -> 563 cells loaded from metadata
#    - 10 incomplete CLEM cells with function labels FILTERED OUT
#      (all have incomplete axon or dendrites)
#    -> 553 cells after filtering
# 2. prepare_data_for_metrics       -> 553 (0 duplicates removed)
# 3. Remove cells with 'axon' in name -> 513 cells enter feature calculation
#    Breakdown: 251 CLEM, 215 EM, 47 PA
#    - 120 training (with function): 84 MI, 21 MON, 15 SMI
#    - 380 to_predict, 13 neg_control
#    - All 73 CLEM training cells have reconstruction_complete = True
#    - No incomplete CLEM neurons with functional identity in final dataset
#
class PipelineConfig:
    """Pipeline configuration settings.

    These defaults reproduce the published results: 13 RFE features,
    82.1% weighted F1 (leave-one-out on CLEM),
    and 100% match against reference predictions.
    """

    # Data paths (resolved via config/path_configuration.txt or MORPH2FUNC_ROOT env)
    DATA_PATH = get_base_path()
    FEATURES_FILE = "final"

    # --- Data loading ---
    MODALITIES = ["pa", "clem", "em", "clem_predict"]
    USE_STORED_FEATURES = True      # Use existing HDF5 features if available
    FORCE_RECALCULATION = False
    DROP_NEUROTRANSMITTER = False  # Keep neurotransmitter as a feature
    FILTER_INCOMPLETE_CLEM = True  # Remove 11 CLEM cells without complete reconstructions
    LABEL_COLUMN = "kmeans_function"  # xlsx column for functional labels (e.g. "function")

    # --- Feature selection (Recursive Feature Elimination) ---
    TRAIN_MODALITY = "all"                           # Train on all modalities
    TEST_MODALITY = "clem"                           # Test on CLEM subset
    RFE_CV_METHOD = "ss"                             # ShuffleSplit for RFE CV
    RFE_ESTIMATOR = AdaBoostClassifier(random_state=0)  # Wrapper estimator for RFE
    RFE_METRIC = "f1"                                # Optimization metric

    # --- Cross-validation and classification ---
    CV_METHOD = "lpo"  # Leave-one-out cross-validation (LeavePOut with p=1)
    CV_PLOT = True
    CLASSIFIER = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    # --- Prediction ---
    USE_JON_PRIORS = False             # Do not apply prior probabilities
    SUFFIX = "_optimize_all_predict"   # Output file suffix

    # --- Verification ---
    VERIFICATION_TESTS = ["IF", "LOF"]  # Isolation Forest + Local Outlier Factor

    # --- Output control ---
    SAVE_FEATURES = True
    SAVE_PREDICTIONS = True
    FORCE_NEW = True    # Force save even if predictions match existing files


def run_pipeline(config: PipelineConfig = None):
    """Run the full cell type classification pipeline.

    Args:
        config: Pipeline configuration. Uses defaults if None.

    Returns
    -------
        dict: Pipeline results containing:
            - data: LoadedData container
            - rfe: RFEResult container
            - cv: CVResult container
            - predictions: PredictionResults container
    """
    if config is None:
        config = PipelineConfig()

    print("\n" + "=" * 80)
    print("CELL TYPE CLASSIFICATION PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Initialize predictor
    # -------------------------------------------------------------------------
    print("\n[1/5] Initializing predictor...")
    predictor = class_predictor(config.DATA_PATH)

    # -------------------------------------------------------------------------
    # Step 2: Load data -> LoadedData container
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading data...")
    data = predictor.load_data(
        features_file=config.FEATURES_FILE,
        modalities=config.MODALITIES,
        use_stored_features=config.USE_STORED_FEATURES,
        force_recalculation=config.FORCE_RECALCULATION,
        label_column=config.LABEL_COLUMN,
    )
    print(f"   Loaded: {data.n_training} training cells, {data.n_to_predict} to predict")
    print(f"   Features: {data.n_features}")

    # -------------------------------------------------------------------------
    # Step 3: Feature selection -> RFEResult container
    # -------------------------------------------------------------------------
    print("\n[3/5] Selecting features (RFE)...")
    rfe = predictor.select_features_rfe(
        data=data,
        train_mod=config.TRAIN_MODALITY,
        test_mod=config.TEST_MODALITY,
        cv_method_rfe=config.RFE_CV_METHOD,
        estimator=config.RFE_ESTIMATOR,
        metric=config.RFE_METRIC,
    )
    print(f"   Selected {rfe.best_n_features} features")
    print(f"   Best F1 score: {rfe.best_score:.4f}")

    # -------------------------------------------------------------------------
    # Step 4: Cross-validation -> CVResult container
    # -------------------------------------------------------------------------
    print("\n[4/5] Running cross-validation...")
    cv = predictor.cross_validate(
        data=data,
        selected_features=rfe.selected_features_idx,
        classifier=config.CLASSIFIER,
        method=config.CV_METHOD,
        plot=config.CV_PLOT,
    )
    print(f"   CV Score: {cv.score:.2f}%")
    print(f"   Confusion matrix shape: {cv.confusion_matrix.shape}")

    # -------------------------------------------------------------------------
    # Step 5: Predict cells -> PredictionResults container
    # -------------------------------------------------------------------------
    print("\n[5/5] Making predictions...")
    predictions = predictor.predict(
        data=data,
        selected_features=rfe.selected_features_idx,
        use_jon_priors=config.USE_JON_PRIORS,
        suffix=config.SUFFIX,
        save_predictions=config.SAVE_PREDICTIONS,
    )
    print(f"   Predictions: {predictions.n_cells} cells")
    print(f"   Distribution: {predictions.prediction_counts}")

    # -------------------------------------------------------------------------
    # Step 6: Verify predictions -> Updated PredictionResults
    # -------------------------------------------------------------------------
    print("\n[6/6] Verifying predictions...")
    predictions = predictor.verify(
        predictions=predictions,
        data=data,
        required_tests=config.VERIFICATION_TESTS,
        force_new=config.FORCE_NEW,
    )
    print(f"   Verified: {predictions.n_verified}/{predictions.n_cells} cells")
    print(f"   Verification rate: {predictions.verification_rate:.1%}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Training cells:     {data.n_training}")
    print(f"  Predicted cells:    {predictions.n_cells}")
    print(f"  Verified cells:     {predictions.n_verified}")
    print(f"  Selected features:  {rfe.best_n_features}")
    print(f"  CV Score:           {cv.score:.2f}%")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # Save predictions to a single xlsx with CLEM and EM sheets
    # -------------------------------------------------------------------------
    pred_df = predictions.cells
    pred_cols = ["cell_name", "imaging_modality", "function", "morphology",
                 "neurotransmitter", "prediction", "prediction_scaled",
                 "MON_proba", "cMI_proba", "iMI_proba", "SMI_proba",
                 "MON_proba_scaled", "cMI_proba_scaled", "iMI_proba_scaled", "SMI_proba_scaled"]
    # Only keep columns that exist
    pred_cols = [c for c in pred_cols if c in pred_df.columns]
    predictions_path = get_output_dir("classifier_pipeline", "predictions") / "predictions.xlsx"
    with pd.ExcelWriter(predictions_path, engine="openpyxl") as writer:
        clem_mask = pred_df["imaging_modality"] == "clem"
        em_mask = pred_df["imaging_modality"] == "EM"
        if clem_mask.any():
            pred_df.loc[clem_mask, pred_cols].to_excel(writer, sheet_name="CLEM", index=False)
        if em_mask.any():
            pred_df.loc[em_mask, pred_cols].to_excel(writer, sheet_name="EM", index=False)
    print(f"\n Predictions saved to: {predictions_path}")

    return {
        "predictor": predictor,
        "data": data,
        "rfe": rfe,
        "cv": cv,
        "predictions": predictions,
    }


def print_data_summary(data):
    """Print detailed summary of loaded data."""
    print("\n" + "-" * 60)
    print("DATA SUMMARY")
    print("-" * 60)
    print(f"Training cells:     {data.n_training}")
    print(f"To predict cells:   {data.n_to_predict}")
    print(f"Total features:     {data.n_features}")
    print("\nModality breakdown:")
    modality_counts = data.training_data.training.modality_mask.summary()
    for mod, count in modality_counts.items():
        print(f"  {mod}: {count}")
    print("\nClass distribution:")
    class_counts = data.training_data.training.class_counts
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    print("-" * 60 + "\n")


def print_prediction_summary(predictions):
    """Print detailed summary of predictions."""
    print("\n" + "-" * 60)
    print("PREDICTION SUMMARY")
    print("-" * 60)
    print(f"Total predictions:  {predictions.n_cells}")
    print(f"Verified:           {predictions.n_verified}")
    print(f"Verification rate:  {predictions.verification_rate:.1%}")
    print("\nPrediction distribution:")
    for cls, count in sorted(predictions.prediction_counts.items()):
        print(f"  {cls}: {count}")
    if predictions.scaled_predictions is not None:
        print("\nScaled prediction distribution:")
        for cls, count in sorted(predictions.scaled_prediction_counts.items()):
            print(f"  {cls}: {count}")
    print("-" * 60 + "\n")


def compare_to_reference(predictions, config: PipelineConfig = None, save_comparison: bool = True):
    """Compare predictions to baseline reference files.

    Args:
        predictions: PredictionResults container from pipeline
        config: Pipeline configuration for data path
        save_comparison: If True, save comparison Excel to output folder

    Returns
    -------
        dict: Comparison results with match rates
    """
    if config is None:
        config = PipelineConfig()

    print("\n" + "=" * 80)
    print("COMPARISON TO REFERENCE FILES")
    print("=" * 80)

    # Reference file paths (baseline predictions with modern nomenclature)
    clem_ref_path = config.DATA_PATH / "baselines" / "clem_baseline_predictions.xlsx"
    em_ref_path = config.DATA_PATH / "baselines" / "em_baseline_predictions.xlsx"

    results = {}
    pred_df = predictions.cells

    # Compare CLEM predictions
    if clem_ref_path.exists():
        print(f"\n CLEM Reference: {clem_ref_path.name}")
        clem_ref = pd.read_excel(clem_ref_path)
        # Drop any rows with NaN cell_name (summary rows)
        clem_ref = clem_ref.dropna(subset=["cell_name"])
        clem_ref["cell_name"] = clem_ref["cell_name"].astype(str).str.strip()
        # Normalize old reference labels to modern nomenclature
        for col in ["prediction", "prediction_scaled"]:
            if col in clem_ref.columns:
                clem_ref[col] = clem_ref[col].apply(normalize_label)

        clem_pred = pred_df[pred_df["imaging_modality"] == "clem"].copy()
        clem_pred["cell_name"] = clem_pred["cell_name"].astype(str).str.strip()

        print(f"   CLEM predictions: {len(clem_pred)}, CLEM reference: {len(clem_ref)}")

        # Merge on cell_name
        merged = clem_pred.merge(
            clem_ref[["cell_name", "prediction", "prediction_scaled"]],
            on="cell_name",
            suffixes=("_new", "_ref"),
            how="inner",
        )

        if len(merged) > 0:
            # Compare unscaled predictions
            matches = (merged["prediction_new"] == merged["prediction_ref"]).sum()
            total = len(merged)
            match_rate = matches / total * 100

            # Compare scaled predictions
            scaled_matches = (
                merged["prediction_scaled_new"]
                == merged["prediction_scaled_ref"]
            ).sum()
            scaled_rate = scaled_matches / total * 100

            print(f"   Matched cells: {total}")
            print(
                f"   Prediction match: "
                f"{matches}/{total} ({match_rate:.1f}%)"
            )
            print(
                f"   Scaled match:     "
                f"{scaled_matches}/{total} ({scaled_rate:.1f}%)"
            )

            # Show mismatches
            mismatches = merged[
                merged["prediction_new"] != merged["prediction_ref"]
            ]
            if len(mismatches) > 0 and len(mismatches) <= 10:
                print(f"\n   Mismatches ({len(mismatches)}):")
                for _, row in mismatches.iterrows():
                    print(
                        f"     {row['cell_name']}: "
                        f"{row['prediction_new']} vs "
                        f"{row['prediction_ref']}"
                    )
            elif len(mismatches) > 10:
                print(
                    f"\n   Mismatches: {len(mismatches)} "
                    f"cells (showing first 10)"
                )
                for _, row in mismatches.head(10).iterrows():
                    print(
                        f"     {row['cell_name']}: "
                        f"{row['prediction_new']} vs "
                        f"{row['prediction_ref']}"
                    )

            results["clem"] = {
                "total": total,
                "matches": matches,
                "match_rate": match_rate,
                "scaled_matches": scaled_matches,
                "scaled_rate": scaled_rate,
                "merged_df": merged,
            }
        else:
            print("     No matching cells found")
    else:
        print(f"\n  CLEM reference not found: {clem_ref_path}")

    # Compare EM predictions
    if em_ref_path.exists():
        print(f"\n EM Reference: {em_ref_path.name}")
        em_ref = pd.read_excel(em_ref_path)
        # Drop any rows with NaN cell_name (summary rows)
        em_ref = em_ref.dropna(subset=["cell_name"])
        # Normalize cell_name: convert to int then string to remove .0
        em_ref["cell_name"] = em_ref["cell_name"].apply(
            lambda x: str(int(float(x))) if pd.notna(x) else x
        )
        # Normalize old reference labels to modern nomenclature
        for col in ["prediction", "prediction_scaled"]:
            if col in em_ref.columns:
                em_ref[col] = em_ref[col].apply(normalize_label)

        em_pred = pred_df[
            pred_df["imaging_modality"] == "EM"
        ].copy()
        # Normalize cell_name the same way
        em_pred["cell_name"] = em_pred["cell_name"].apply(
            lambda x: str(int(float(x)))
            if pd.notna(x)
            and str(x).replace('.', '').replace('-', '').isdigit()
            else str(x)
        )

        print(f"   EM predictions: {len(em_pred)}, EM reference: {len(em_ref)}")
        print(f"   EM pred cell_names sample: {em_pred['cell_name'].head(3).tolist()}")
        print(f"   EM ref cell_names sample: {em_ref['cell_name'].head(3).tolist()}")

        # Merge on cell_name
        merged = em_pred.merge(
            em_ref[["cell_name", "prediction", "prediction_scaled"]],
            on="cell_name",
            suffixes=("_new", "_ref"),
            how="inner",
        )
        print(f"   Merged rows: {len(merged)}")

        if len(merged) > 0:
            # Compare unscaled predictions
            matches = (merged["prediction_new"] == merged["prediction_ref"]).sum()
            total = len(merged)
            match_rate = matches / total * 100

            # Compare scaled predictions
            scaled_matches = (
                merged["prediction_scaled_new"]
                == merged["prediction_scaled_ref"]
            ).sum()
            scaled_rate = scaled_matches / total * 100

            print(f"   Matched cells: {total}")
            print(
                f"   Prediction match: "
                f"{matches}/{total} ({match_rate:.1f}%)"
            )
            print(
                f"   Scaled match:     "
                f"{scaled_matches}/{total} ({scaled_rate:.1f}%)"
            )

            # Show mismatches
            mismatches = merged[
                merged["prediction_new"]
                != merged["prediction_ref"]
            ]
            if len(mismatches) > 0 and len(mismatches) <= 10:
                print(f"\n   Mismatches ({len(mismatches)}):")
                for _, row in mismatches.iterrows():
                    print(
                        f"     {row['cell_name']}: "
                        f"{row['prediction_new']} vs "
                        f"{row['prediction_ref']}"
                    )
            elif len(mismatches) > 10:
                print(
                    f"\n   Mismatches: {len(mismatches)} "
                    f"cells (showing first 10)"
                )
                for _, row in mismatches.head(10).iterrows():
                    print(
                        f"     {row['cell_name']}: "
                        f"{row['prediction_new']} vs "
                        f"{row['prediction_ref']}"
                    )

            results["em"] = {
                "total": total,
                "matches": matches,
                "match_rate": match_rate,
                "scaled_matches": scaled_matches,
                "scaled_rate": scaled_rate,
                "merged_df": merged,
            }
        else:
            print("     No matching cells found")
    else:
        print(f"\n  EM reference not found: {em_ref_path}")

    # Overall summary
    if "clem" in results and "em" in results:
        total_matches = results["clem"]["matches"] + results["em"]["matches"]
        total_cells = results["clem"]["total"] + results["em"]["total"]
        overall_rate = total_matches / total_cells * 100
        print(
            f"\n OVERALL: {total_matches}/{total_cells}"
            f" ({overall_rate:.1f}%) predictions match reference"
        )

    # Save comparison to Excel
    if save_comparison and results:
        output_dir = get_output_dir("classifier_pipeline", "predictions", "comparison")

        output_file = output_dir / "prediction_comparison.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = []
            for modality in ["clem", "em"]:
                if modality in results:
                    r = results[modality]
                    summary_data.append({
                        "modality": modality.upper(),
                        "total_cells": r["total"],
                        "prediction_matches": r["matches"],
                        "prediction_match_rate": f"{r['match_rate']:.1f}%",
                        "scaled_matches": r["scaled_matches"],
                        "scaled_match_rate": f"{r['scaled_rate']:.1f}%",
                    })
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            # Detailed comparison sheets with only the 5 key columns
            rename_cols = {
                "prediction_new": "prediction",
                "prediction_ref": "prediction_reference",
                "prediction_scaled_new": "prediction_scaled",
                "prediction_scaled_ref": "prediction_scaled_reference",
            }
            output_cols = ["cell_name", "prediction", "prediction_reference",
                          "prediction_scaled", "prediction_scaled_reference"]

            if "clem" in results:
                clem_df = results["clem"]["merged_df"].rename(columns=rename_cols)
                clem_df = clem_df[output_cols]
                clem_df.to_excel(writer, sheet_name="CLEM_Comparison", index=False)
            if "em" in results:
                em_df = results["em"]["merged_df"].rename(columns=rename_cols)
                em_df = em_df[output_cols]
                em_df.to_excel(writer, sheet_name="EM_Comparison", index=False)

        print(f"\n Comparison saved to: {output_file}")

    print("=" * 80 + "\n")
    return results


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the cell type classification pipeline.",
    )
    parser.add_argument(
        "--force-recalculation", action="store_true",
        help="Force recalculation of features from scratch",
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="Skip comparison to baseline reference files",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print data and prediction summaries only (skip comparison)",
    )

    args = parser.parse_args()

    config = PipelineConfig()

    if args.force_recalculation:
        config.FORCE_RECALCULATION = True

    # Run pipeline
    results = run_pipeline(config)

    # Print detailed summaries
    print_data_summary(results["data"])
    print_prediction_summary(results["predictions"])

    # Compare to baseline reference files
    if not args.no_compare and not args.summary_only:
        compare_to_reference(results["predictions"], config)
