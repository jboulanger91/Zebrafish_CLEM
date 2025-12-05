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
      3. For cells that are currently "not functionally imaged" (and not myelinated),
         it looks up LDA predictions and, if they pass all tests, updates:
             - 'functional classifier'
             - 'functional_id'  (assigned a new integer ID)
             - 'lda'            ('native' → 'predicted')
      4. Saves the updated metadata to a new CSV and returns it.

    Matching logic
    --------------
    - Metadata: rows with
        type == 'cell'
        functional_id == 'not functionally imaged'
        functional classifier != 'myelinated'
    - LDA table: matched by
        metadata['nucleus_id'] <-> lda_predictions['cell_name'] (with 'cell_' stripped)
    - LDA prediction is accepted only if `passed_tests` is TRUE.

    Functional classifier updates
    -----------------------------
    - prediction in {'integrator_ipsilateral', 'integrator_contralateral'} → 'integrator'
    - prediction in {'dynamic_threshold', 'motor_command'}                 → that string
    - 'lda' column is set to 'predicted' for updated rows.

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
    # 2. Standardize naming / create LDA status column
    # ------------------------------------------------------------------
    # Normalize functional classifier names to snake_case where needed
    metadata_df["functional classifier"] = metadata_df["functional classifier"].replace(
        {
            "dynamic threshold": "dynamic_threshold",
            "motor command": "motor_command",
        }
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
    # 3. How many cells are candidates for prediction?
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
        predicted_functional_id = matching_row.iloc[0]["prediction"]
        pooled_tests = matching_row.iloc[0]["passed_tests"]

        # Only accept predictions where all tests pass
        if str(pooled_tests).strip().upper() != "TRUE":
            continue

        # --------------------------------------------------------------
        # Update functional classifier based on the prediction
        # --------------------------------------------------------------
        if predicted_functional_id in {"integrator_ipsilateral", "integrator_contralateral"}:
            metadata_df.at[idx, "functional classifier"] = "integrator"
        elif predicted_functional_id in {"dynamic_threshold", "motor_command"}:
            metadata_df.at[idx, "functional classifier"] = predicted_functional_id

        # Mark as LDA-predicted
        metadata_df.at[idx, "lda"] = "predicted"

        # Assign a new integer functional_id
        metadata_df.at[idx, "functional_id"] = functional_id_counter
        functional_id_counter += 1

        if verbose:
            print(
                f"[LDA] nucleus_id={nucleus} → "
                f"functional_id={functional_id_counter - 1}, "
                f"prediction={predicted_functional_id}, tests_passed=TRUE"
            )

    # ------------------------------------------------------------------
    # 5. Save updated metadata
    # ------------------------------------------------------------------
    metadata_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"Updated metadata with LDA predictions saved to: {output_csv}")

    return metadata_df