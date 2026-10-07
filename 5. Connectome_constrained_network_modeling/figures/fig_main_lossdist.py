"""
Overview
--------
This script generates the following figure panels in Boulanger-Weiss et al. 2026:
- Fig. 5...

This script evaluates and statistically compares the performance (training MSE loss)
of recurrent neural network (RNN) dynamical models trained under different architectural,
biological, or data perturbation conditions (e.g., intact connectome vs. LDA ablation variants).

Core Pipeline & Workflow:
1. Environment and Configuration:
   - Loads filesystem paths via .env for trained model checkpoints and output directories.
   - Defines a test suite registry mapping experimental ablation conditions (such as
     intact baseline, target ablations, or matrix ablations) to their checkpoint folders.
   - Configures figure canvas dimensions, paddings, and layout origins using `RNNDSStyle`.

2. Metric Extraction & Aggregation:
   - Iterates across each experimental condition directory and loads all matching model checkpoints.
   - Extracts the mean squared error (`loss_mse`) attribute from each model instance.
   - Clips loss values to a defined ceiling (`MAX_LOSS = 10`), discards NaN entries, and
     structures the empirical loss distributions per condition.

3. Statistical Hypothesis Testing & Visualization:
   - Performs two-sided Mann-Whitney U rank-sum non-parametric tests comparing each perturbation
     condition against the baseline ("Original") condition.
   - Logs resulting p-values to stdout and marks statistically significant differences
     (p < 0.005) with asterisk annotations directly on the figure.
   - Draws violin plots illustrating the loss distributions across experimental cohorts.

4. Figure Export:
   - Saves the final vector comparison plot as a PDF with an appropriate condition label tag.
"""

import numpy as np
from pathlib import Path
from dotenv import dotenv_values
from scipy.stats import mannwhitneyu

# Manually add root path for imports to improve interoperability across parent modules
import sys; sys.path.insert(0, "..")

from style import RNNDSStyle
from utils.services.rnn_service import RNNService
from utils.figure_helper import Figure
from utils.load_model import load_model
from utils.load_connectome import get_W


# ------------------------------------------------
# Configuration
# ------------------------------------------------
# Toggle visualization of the loss comparison violin plot
show_loss = True

# Load directory paths from environment configuration (.env)
env = dotenv_values()
path_dir = Path(env["PATH_DIR"])
path_models = path_dir / "models"
path_save = path_dir / "results"

# Suffix tag applied to the exported figure filename
label_figure = "_lda"
# Registry of model directories and display labels representing distinct experimental/ablation cohorts
test_list = [
    {"path": path_models / "connectome",
     "label": "Original"},
    # {"path": path_models / "connectome_noLDA_target",
    #  "label": "No LDA in target"},
    # {"path": path_models / "connectome_noLDA_matrix",
    #  "label": "No LDA in matrix"},
    {"path": path_models / "connectome_shuffle",
     "label": "Shuffle mask"},
    {"path": path_models / "connectome_iMI_cut_50",
     "label": "Remove 50% iMI $\leftrightarrow$ iMI"},
    {"path": path_models / "connectome_iMI_cut_100",
     "label": "Remove 50% iMI $\leftrightarrow$ iMI"},
    {"path": path_models / "connectome_cut_05-0",
     "label": "Remove random 5%"},
    {"path": path_models / "connectome_cut_10-0",
     "label": "Remove random 10%"},
    {"path": path_models / "connectome_cut_25-0",
     "label": "Remove random 25%"},
    {"path": path_models / "connectome_cut_50-0",
     "label": "Remove random 50%"},
]


# ----------------------------------------------------------------
# Plot configuration (layout, sizes, padding, etc.)
# ----------------------------------------------------------------
# Initialize custom canvas styling, plot dimensions, and layout coordinates
style = RNNDSStyle()

plot_height = style.plot_size_big * 3
plot_height_small = plot_height / 2.5

plot_width_small = style.plot_width_small

plot_width = style.plot_size_big * 3

padding = style.padding / 2
padding_big = style.padding * 2
padding_vertical = style.padding

xpos_start = style.xpos_start
ypos_start = style.ypos_start - plot_height/2
xpos = xpos_start
ypos = ypos_start - padding


# ----------------------------------------------------------------
# Initialize figure container
# ----------------------------------------------------------------
# Instantiate figure canvas container
fig = Figure()


# ----------------------------------------------------------------
# Loop over all tests
# ----------------------------------------------------------------
# Preallocate list of loss containers and initialize tracking counters
loss_list = [None for _ in test_list]
N_MAX_MODELS = 0
MAX_LOSS = 10
for i_test, test in enumerate(test_list):

    # ------------------------------------------------
    # Loop over all trained models
    # ------------------------------------------------
    loss_list[i_test] = []
    model_path_list = []
    i_model = 0
    # Discover and inspect all trained model checkpoints within the condition directory
    for path_model in Path(test["path"]).glob(f"model_*"):
        print(f"Evaluating model {i_model}")
        i_model += 1

        model_path_list.append(path_model)
        # Load model instance
        model = load_model(path_model)

        # Retrieve MSE loss; apply upper bound clipping and skip invalid/missing values
        try:
            loss = np.min((model.loss_mse, MAX_LOSS))
            if np.isnan(loss): continue
        except AttributeError:
            continue

        loss_list[i_test].append(loss)

        # Track the maximum number of checkpoints observed across cohorts
        if i_model > N_MAX_MODELS:
            N_MAX_MODELS = i_model + 1
N_TESTS = len(loss_list)

if show_loss:
    # Set y-axis upper boundary for loss display
    max_value = 0.05
    # Create categorical plot panel for loss distributions
    plot_loss = fig.create_plot(plot_title="Loss across conditions",
                                     xpos=xpos, ypos=ypos, plot_height=plot_height,
                                     plot_width=plot_width,
                                     xmin=0.5, xmax=N_TESTS+0.5, xticks=np.arange(N_TESTS)+1,
                                     xticklabels=[test["label"] for test in test_list],
                                     xticklabels_rotation=45,
                                     ymin=0, ymax=max_value, yticks=[0, max_value])

    # Run non-parametric Mann-Whitney U test comparing each condition against baseline cohort (index 0)
    for i_test, loss in enumerate(loss_list[1:]):
        res = mannwhitneyu(np.array(loss_list[0]), np.array(loss))
        print(f"p-val Original vs {test_list[i_test+1]['label']}: {res.pvalue}")
        # Annotate significant differences with an asterisk marker near top of panel
        if res.pvalue < 0.005:
            ytext = max_value*9/10
            plot_loss.draw_text((i_test+1), ytext, "*")

    # Render violin distribution plots for each cohort
    plot_loss.draw_violin(loss_list, facecolor="k")

    # Advance horizontal coordinate for subsequent panels
    xpos += plot_width + padding


# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
# Ensure target output directory exists and save vector figure as PDF
path_save.mkdir(parents=True, exist_ok=True)
filename = f"figure_compare{label_figure}.pdf" if label_figure is not None else "figure_compare.pdf"
fig.save(path_save / filename, open_file=False, tight=style.page_tight)