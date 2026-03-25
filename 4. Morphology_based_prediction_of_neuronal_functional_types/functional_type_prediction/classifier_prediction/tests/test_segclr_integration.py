"""Tests for SegCLR embedding integration into ClassPredictor."""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

# Path setup
_TEST_DIR = Path(__file__).resolve().parent
_CLASSIFIER_DIR = _TEST_DIR.parent
_FUNCTIONAL_DIR = _CLASSIFIER_DIR.parent
_REPO_ROOT = _FUNCTIONAL_DIR.parent
_SRC = _REPO_ROOT / "src"

for path in [str(_CLASSIFIER_DIR), str(_SRC), str(_REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from core.class_predictor import ClassPredictor  # noqa: E402
from core.containers import CellDataset, LoadedData, ModalityMask, TrainingData  # noqa: E402


# =============================================================================
# Fixtures
# =============================================================================

N_CELLS = 20
N_EMBED_DIM = 512
N_MORPH_FEATURES = 30
LABEL_NAMES = ["motion_onset", "motion_integrator_contralateral",
               "motion_integrator_ipsilateral", "slow_motion_integrator"]


@pytest.fixture
def fake_h5(tmp_path):
    """Create a fake all_embeddings.hdf5 for testing."""
    h5_path = tmp_path / "all_embeddings.hdf5"
    rng = np.random.default_rng(42)

    cell_names = [f"cell_{i:03d}" for i in range(N_CELLS)]
    labels = rng.choice(LABEL_NAMES, size=N_CELLS)
    embeddings = rng.standard_normal((N_CELLS, N_EMBED_DIM)).astype(np.float32)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("cell_names", data=np.array(cell_names, dtype="S"))
        f.create_dataset("labels", data=np.array(labels, dtype="S"))
        f.create_dataset("label_names", data=np.array(LABEL_NAMES, dtype="S"))

    return h5_path


@pytest.fixture
def predictor(tmp_path):
    """Create a ClassPredictor with a temporary base path."""
    return ClassPredictor(tmp_path)


@pytest.fixture
def fake_morph_data():
    """Create a fake LoadedData with morphology features."""
    rng = np.random.default_rng(42)
    n_cells = N_CELLS
    n_features = N_MORPH_FEATURES

    cell_names = [f"cell_{i:03d}" for i in range(n_cells)]
    labels = rng.choice(LABEL_NAMES, size=n_cells)
    modalities = np.array(["clem"] * n_cells)

    cells_df = pd.DataFrame({
        "cell_name": cell_names,
        "function": labels,
        "imaging_modality": modalities,
    })

    feature_names = [f"morph_{i:03d}" for i in range(n_features)]
    features = rng.standard_normal((n_cells, n_features)).astype(np.float32)
    modality_mask = ModalityMask.from_series(cells_df["imaging_modality"])

    training_dataset = CellDataset(
        cells=cells_df,
        features=features,
        labels=labels,
        modality=modalities,
        modality_mask=modality_mask,
        feature_names=feature_names,
    )

    # Empty to_predict dataset
    empty_df = pd.DataFrame(columns=cells_df.columns)
    empty_modality = np.array([], dtype=str)
    empty_mask = ModalityMask(
        clem=np.array([], dtype=bool),
        photoactivation=np.array([], dtype=bool),
        em=np.array([], dtype=bool),
    )
    to_predict_dataset = CellDataset(
        cells=empty_df,
        features=np.empty((0, n_features)),
        labels=np.array([], dtype=str),
        modality=empty_modality,
        modality_mask=empty_mask,
        feature_names=feature_names,
    )

    training_data = TrainingData(
        training=training_dataset,
        to_predict=to_predict_dataset,
    )

    return LoadedData(
        training_data=training_data,
        feature_names=feature_names,
        cells_df=cells_df,
        cells_with_to_predict_df=cells_df.copy(),
    )


class TestLoadSegclrEmbeddings:
    """Tests for ClassPredictor.load_segclr_embeddings()."""

    def test_returns_loaded_data(self, predictor, fake_h5):
        """Verify load_segclr_embeddings returns LoadedData with correct shape."""
        result = predictor.load_segclr_embeddings(fake_h5)

        assert isinstance(result, LoadedData)
        assert result.training_data.training.features.shape == (N_CELLS, N_EMBED_DIM)

    def test_feature_names_are_segclr_prefixed(self, predictor, fake_h5):
        """Verify all feature names start with 'segclr_'."""
        result = predictor.load_segclr_embeddings(fake_h5)

        assert len(result.feature_names) == N_EMBED_DIM
        assert all(name.startswith("segclr_") for name in result.feature_names)

    def test_feature_names_format(self, predictor, fake_h5):
        """Verify feature names are zero-padded: segclr_000 through segclr_511."""
        result = predictor.load_segclr_embeddings(fake_h5)

        assert result.feature_names[0] == "segclr_000"
        assert result.feature_names[-1] == "segclr_511"

    def test_labels_loaded_correctly(self, predictor, fake_h5):
        """Verify labels are loaded from HDF5."""
        result = predictor.load_segclr_embeddings(fake_h5)

        labels = result.training_data.training.labels
        assert len(labels) == N_CELLS
        assert all(label in LABEL_NAMES for label in labels)

    def test_cell_names_in_dataframe(self, predictor, fake_h5):
        """Verify cell names appear in the cells DataFrame."""
        result = predictor.load_segclr_embeddings(fake_h5)

        assert "cell_name" in result.cells_df.columns
        assert len(result.cells_df) == N_CELLS


class TestCombineFeatures:
    """Tests for ClassPredictor.combine_features()."""

    def test_concatenates_features(self, predictor, fake_morph_data, fake_h5):
        """Verify combined shape is (n_cells, n_morph_selected + 512)."""
        segclr_data = predictor.load_segclr_embeddings(fake_h5)

        n_selected = 13
        morph_feature_mask = np.zeros(N_MORPH_FEATURES, dtype=bool)
        morph_feature_mask[:n_selected] = True

        combined = predictor.combine_features(
            morph_data=fake_morph_data,
            segclr_data=segclr_data,
            morph_feature_mask=morph_feature_mask,
        )

        assert isinstance(combined, LoadedData)
        expected_features = n_selected + N_EMBED_DIM
        assert combined.training_data.training.features.shape == (N_CELLS, expected_features)

    def test_combined_feature_names(self, predictor, fake_morph_data, fake_h5):
        """Verify combined feature names include both morph and segclr."""
        segclr_data = predictor.load_segclr_embeddings(fake_h5)

        n_selected = 13
        morph_feature_mask = np.zeros(N_MORPH_FEATURES, dtype=bool)
        morph_feature_mask[:n_selected] = True

        combined = predictor.combine_features(
            morph_data=fake_morph_data,
            segclr_data=segclr_data,
            morph_feature_mask=morph_feature_mask,
        )

        assert len(combined.feature_names) == n_selected + N_EMBED_DIM
        # First 13 should be morph features
        assert all(name.startswith("morph_") for name in combined.feature_names[:n_selected])
        # Last 512 should be segclr features
        assert all(name.startswith("segclr_") for name in combined.feature_names[n_selected:])
