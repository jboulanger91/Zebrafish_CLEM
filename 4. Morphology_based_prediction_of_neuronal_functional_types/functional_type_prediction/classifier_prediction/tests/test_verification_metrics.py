"""Test verification metrics for cell type prediction pipeline."""

import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

# Path setup for output_paths import
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))  # Required for standalone execution

try:
    from src.util.get_base_path import get_base_path  # noqa: E402
    from src.util.output_paths import get_output_dir  # noqa: E402
except ModuleNotFoundError:
    from util.get_base_path import get_base_path  # noqa: E402
    from util.output_paths import get_output_dir  # noqa: E402

from core.class_predictor import class_predictor  # noqa: E402


def calc_validation_metric_matrix(df, variables, scaled=False):
    """Compute accuracy and F1 matrices across all variable combinations."""
    suffix = ""
    if scaled:
        suffix = "_scaled"
    combinations1 = list(
        product([True, False], repeat=len(variables[: np.floor(len(variables) / 2).astype(int)]))
    )
    combinations2 = list(
        product([True, False], repeat=len(variables[np.floor(len(variables) / 2).astype(int) :]))
    )
    clem_func_recorded = df.query(
        'imaging_modality == "clem" and function != "to_predict" and function != "neg_control"'
    )

    combinations1_names = []
    for combination in combinations1:
        temp = []
        for i, var in enumerate(variables[: np.floor(len(variables) / 2).astype(int)]):
            temp.append(var + "_" + str(combination[i]))
        combinations1_names.append(".".join(temp))

    combinations2_names = []
    for combination in combinations2:
        temp = []
        for i, var in enumerate(variables[np.floor(len(variables) / 2).astype(int) :]):
            temp.append(var + "_" + str(combination[i]))
        combinations2_names.append(".".join(temp))

    verification_accuracy_matrix = pd.DataFrame(
        index=combinations1_names, columns=combinations2_names
    )
    verification_n_cells_matrix = pd.DataFrame(
        index=combinations1_names, columns=combinations2_names
    )
    verification_accuracy_matrix_f1 = pd.DataFrame(
        index=combinations1_names, columns=combinations2_names
    )

    for row in verification_accuracy_matrix.index:
        for column in verification_accuracy_matrix.columns:
            temp_selector = []
            for var in column.split("."):
                if var.split("_")[-1] == "True":
                    temp_selector.append("_".join(var.split("_")[:-1]))
            for var in row.split("."):
                if var.split("_")[-1] == "True":
                    temp_selector.append("_".join(var.split("_")[:-1]))

            temp = clem_func_recorded[clem_func_recorded[temp_selector].all(axis=1)]

            verification_accuracy_matrix.loc[row, column] = accuracy_score(
                temp["function"], temp[f"prediction{suffix}"]
            )
            verification_accuracy_matrix_f1.loc[row, column] = f1_score(
                temp["function"], temp[f"prediction{suffix}"], average="weighted"
            )

            verification_n_cells_matrix.loc[row, column] = temp.shape[0]
    return (
        verification_accuracy_matrix,
        verification_n_cells_matrix,
        verification_accuracy_matrix_f1,
    )


def plot_validation_metric_matrix(df, title="no title", fontsize="small"):
    """Plot a validation metric matrix as an annotated heatmap."""
    plt.figure(dpi=1200)
    plt.imshow(np.array(df).astype(float))
    plt.colorbar()
    plt.xticks(
        np.arange(len(df.columns)),
        [x.replace(".", "\n") for x in df.columns],
        fontsize=fontsize * 0.8,
        rotation=90,
    )
    plt.yticks(
        np.arange(len(df.index)), [x.replace(".", "\n") for x in df.index], fontsize=fontsize * 0.8
    )

    max_flat_index = np.argmax(np.array(df).astype(float), axis=None)

    # Convert flat index to row and column index
    max_row, max_col = np.unravel_index(max_flat_index, df.shape)
    max_row, max_col = max_row - 0.5, max_col - 0.5
    plt.plot([max_col, max_col], [max_row, max_row + 1], "r-")
    plt.plot([max_col, max_col + 1], [max_row + 1, max_row + 1], "r-")
    plt.plot([max_col, max_col + 1], [max_row, max_row], "r-")
    plt.plot([max_col + 1, max_col + 1], [max_row, max_row + 1], "r-")
    plt.title(title, fontsize=fontsize)
    for irow in range(len(df.index)):
        for icol in range(len(df.columns)):
            plt.text(
                icol,
                irow,
                f"{df.iloc[irow, icol]:.5f}",
                ha="center",
                va="center",
                color="w",
                fontsize=fontsize,
            )

    savepath = get_output_dir("classifier_pipeline", "test_verification_metrics")
    plt.savefig(savepath / f"{title.replace(' ', '_').replace('.', '_')}.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # load metrics and cells
    # Was: hindbrain_structure_function/nextcloud
    _data_path = get_base_path()
    with_neurotransmitter = class_predictor(_data_path)
    with_neurotransmitter.load_cells_df(
        modalities=["pa", "clem", "em", "clem_predict"],
    )
    with_neurotransmitter.calculate_metrics(
        "test"
    )

    # with_neurotransmitter.calculate_published_metrics()
    with_neurotransmitter.load_cells_features(
        "test",
        drop_neurotransmitter=False,
    )
    # select features
    # test.select_features_RFE('all', 'clem', cv=False,cv_method_rfe='lpo') #runs through all
    # estimator
    with_neurotransmitter.select_features_RFE(
        "all",
        "clem",
        save_features=True,
        estimator=AdaBoostClassifier(random_state=0),
        cv_method_rfe="ss",
        metric="f1",
    )
    # predict cells
    # optimize_all_predict means to go for the 82.05%,
    # alternative is balance_all_pa: 79.49% ALL, 69.75% PA
    with_neurotransmitter.predict_cells(
        use_jon_priors=False,
        suffix="_optimize_all_predict",
        predict_recorded=True,
    )
    with_neurotransmitter.calculate_verification_metrics(
        calculate_smat=False, with_kunst=False, calculate4recorded=True
    )

    with_neurotransmitter.prediction_predict_df.loc[
        with_neurotransmitter.prediction_predict_df["imaging_modality"] == "clem"
    ]

    # calculate the matrices
    variables = ["OCSVM_intra_class", "IF_intra_class", "LOF_intra_class", "OCSVM", "IF", "LOF"]
    variables_scaled = [
        "NBLAST_g",
        "NBLAST_z_scaled",
        "NBLAST_ak_scaled",
        "NBLAST_ks_scaled",
        "OCSVM",
        "IF",
        "LOF",
    ]
    verification_accuracy_matrix, verification_n_cells_matrix, verification_accuracy_matrix_f1 = (
        calc_validation_metric_matrix(with_neurotransmitter.prediction_predict_df, variables)
    )
    (
        verification_accuracy_matrix_scaled,
        verification_n_cells_matrix_scaled,
        verification_accuracy_matrix_f1_scaled,
    ) = calc_validation_metric_matrix(
        with_neurotransmitter.prediction_predict_df, variables_scaled, scaled=True
    )

    # NOT SCALED

    # plot the accuracy while varying the validation metrics
    plot_validation_metric_matrix(
        verification_accuracy_matrix_f1, "F1 after applying validation metrics.", fontsize=6
    )

    # plot the number of cells while varying the validation metrics
    plot_validation_metric_matrix(
        verification_n_cells_matrix, "N cells after applying validation metrics", fontsize=6
    )

    # plot cells lost
    cells_lost = 1 - (verification_n_cells_matrix / np.max(verification_n_cells_matrix))
    plot_validation_metric_matrix(
        cells_lost, "Percent of cells lost after applying validation metrics", fontsize=6
    )

    # Visualization of Validation Metrics delta accuracy from optimal
    clem_func_recorded = with_neurotransmitter.prediction_predict_df.query(
        'imaging_modality == "clem" and function != "to_predict" and function != "neg_control"'
    )
    verification_accuracy_matrix_delta_F1 = verification_accuracy_matrix_f1 - f1_score(
        clem_func_recorded["function"], clem_func_recorded["prediction"], average="weighted"
    )
    plot_validation_metric_matrix(
        verification_accuracy_matrix_delta_F1, "∂F1 after applying validation metrics.", fontsize=6
    )
