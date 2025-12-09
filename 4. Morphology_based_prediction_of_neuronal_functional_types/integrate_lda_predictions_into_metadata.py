import pandas as pd
from pathlib import Path
from typing import Union


def integrate_lda_predictions_into_metadata(
    metadata_csv: Union[str, Path],
    lda_excel: Union[str, Path],
    output_csv: Union[str, Path],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Integrate LDA-based functional predictions into the master metadata table.

    This function:
      1. Loads the original metadata CSV (one row per reconstructed neuron / axon).
      2. Loads LDA prediction results from an Excel file.
      3. Normalizes functional classifier names in the metadata to the canonical set:
            - integrator_*      → motion_integrator
            - dynamic_threshold → motion_onset
            - motor_command     → slow_motion_integrator
      4. For cells that are currently "not functionally imaged" (and not myelinated),
         it looks up LDA predictions and, if they pass all tests, updates:
             - 'functional classifier' (using the canonical names above)
             - 'functional_id'  (assigned a new integer ID)
             - 'lda'            ('native' → 'predicted')
      5. Saves the updated metadata to a new CSV and returns it.

    Matching logic
    --------------
    - Metadata: candidate rows satisfy
          type == 'cell'
          functional_id == 'not functionally imaged'
          functional classifier != 'myelinated'
    - LDA table: matched by
          metadata['nucleus_id']  <->  lda_predictions['cell_name'] (with 'cell_' stripped)
    - LDA prediction is accepted only if `passed_tests` is TRUE.

    Functional classifier updates from LDA
    --------------------------------------
    - prediction in {'integrator_ipsilateral', 'integrator_contralateral', 'integrator'}
            → 'motion_integrator'
    - prediction == 'dynamic_threshold'
            → 'motion_onset'
    - prediction == 'motor_command'
            → 'slow_motion_integrator'

    Parameters
    ----------
    metadata_csv : str or Path
        Path to the original metadata CSV
        (e.g. .../all_reconstructed_neurons.csv).
    lda_excel : str or Path
        Path to the LDA prediction Excel file
        (e.g. clem_cell_prediction_optimize_all_predict_YYYY-MM-DD_hh-mm-ss.xlsx).
    output_csv : str or Path
        Path where the updated metadata CSV will be written.
    verbose : bool, optional
        If True, print progress and summary information.

    Returns
    -------
    pandas.DataFrame
        The updated metadata DataFrame (also written to `output_csv`).
    """
    metadata_csv = Path(metadata_csv)
    lda_excel = Path(lda_excel)
    output_csv = Path(output_csv)

    # ------------------------------------------------------------------
    # 1. Load metadata and LDA predictions
    # ------------------------------------------------------------------
    if verbose:
        print(f"Loading metadata from: {metadata_csv}")
    metadata_df = pd.read_csv(metadata_csv)

    if verbose:
        print(f"Loading LDA predictions from: {lda_excel}")
    lda_predictions = pd.read_excel(lda_excel)

    # ------------------------------------------------------------------
    # 2. Normalize functional classifiers & create LDA status column
    # ------------------------------------------------------------------
    # Normalize metadata functional classifiers to canonical labels
    metadata_df["functional classifier"] = (
        metadata_df["functional classifier"]
        .astype(str)
        .str.strip()
        .replace(
            {
                # Everything “integrator_*” becomes motion_integrator
                "integrator_contralateral": "motion_integrator",
                "integrator_ipsilateral": "motion_integrator",
                "integrator": "motion_integrator",

                # dynamic threshold → motion onset
                "dynamic_threshold": "motion_onset",
                "dynamic threshold": "motion_onset",

                # motor command → slow motion integrator
                "motor_command": "slow_motion_integrator",
                "motor command": "slow_motion_integrator",
            }
        )
    )

    # Add an LDA status column: default is 'native', changed to 'predicted' when updated
    metadata_df["lda"] = "native"

    # Standardize ID columns as strings for matching
    metadata_df["nucleus_id"] = metadata_df["nucleus_id"].astype(str).str.strip()
    lda_predictions["cell_name"] = (
        lda_predictions["cell_name"]
        .str.replace("cell_", "", regex=False)
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------------------
    # 3. Identify cells that are candidates for LDA-based annotation
    # ------------------------------------------------------------------
    mask_candidates = (
        (metadata_df["type"] == "cell")
        & (metadata_df["functional_id"] == "not functionally imaged")
        & (metadata_df["functional classifier"] != "myelinated")
    )

    count_cells_to_predict = metadata_df.loc[mask_candidates, "axon_id"].nunique()
    if verbose:
        print(
            "Number of unique 'axon_id' values with type='cell', "
            "functional_id='not functionally imaged', and non-myelinated: "
            f"{count_cells_to_predict}"
        )

    # ------------------------------------------------------------------
    # 4. Iterate over metadata and apply LDA predictions
    # ------------------------------------------------------------------
    functional_id_counter = 1  # new integer IDs for predicted cells

    for idx, row in metadata_df[mask_candidates].iterrows():
        nucleus = row["nucleus_id"]

        # Find matching LDA prediction by nucleus_id ↔ cell_name
        matching_row = lda_predictions[lda_predictions["cell_name"] == nucleus]

        if matching_row.empty:
            continue  # no prediction for this cell

        # Extract prediction and test status from the first match
        predicted_functional_id = str(matching_row.iloc[0]["prediction"]).strip()
        pooled_tests = matching_row.iloc[0]["passed_tests"]

        # Only accept predictions where all tests pass
        if str(pooled_tests).strip().upper() != "TRUE":
            continue

        # --------------------------------------------------------------
        # Map LDA prediction to canonical functional classifier
        # --------------------------------------------------------------
        if predicted_functional_id in {
            "integrator_ipsilateral",
            "integrator_contralateral",
            "integrator",
        }:
            new_fc = "motion_integrator"
        elif predicted_functional_id == "dynamic_threshold":
            new_fc = "motion_onset"
        elif predicted_functional_id == "motor_command":
            new_fc = "slow_motion_integrator"
        else:
            # Unknown / unsupported prediction → skip
            if verbose:
                print(
                    f"[LDA] nucleus_id={nucleus}: unsupported prediction "
                    f"'{predicted_functional_id}', skipping."
                )
            continue

        # Apply updates to the metadata table
        metadata_df.at[idx, "functional classifier"] = new_fc
        metadata_df.at[idx, "lda"] = "predicted"
        metadata_df.at[idx, "functional_id"] = functional_id_counter

        if verbose:
            print(
                f"[LDA] nucleus_id={nucleus} → "
                f"functional_id={functional_id_counter}, "
                f"prediction={predicted_functional_id} → {new_fc}, tests_passed=TRUE"
            )

        functional_id_counter += 1

    # ------------------------------------------------------------------
    # 5. Save updated metadata
    # ------------------------------------------------------------------
    metadata_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"Updated metadata with LDA predictions saved to: {output_csv}")

    return metadata_df


# Example call
updated_df = integrate_lda_predictions_into_metadata(
    metadata_csv=(
        "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
        "Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/"
        "1. Downloading_neuronal_morphologies_and_metadata/all_reconstructed_neurons.csv"
    ),
    lda_excel=(
        "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
        "Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/"
        "4. Morphology_based_prediction_of_neuronal_functional_types/LDA_predictions.xlsx"
    ),
    output_csv=(
        "/Users/jonathanboulanger-weill/Harvard University Dropbox/"
        "Jonathan Boulanger-Weill/Projects/Zebrafish_CLEM/"
        "4. Morphology_based_prediction_of_neuronal_functional_types/"
        "all_reconstructed_neurons_with_LDA_predictions.csv"
    ),
    verbose=True,
)