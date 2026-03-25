"""Configuration dataclasses for the classification pipeline.

This module centralizes all configuration, constants, and default values
used throughout the pipeline. Using dataclasses ensures type safety,
documentation, and easy serialization.

Benefits of centralized configuration:
    - Single source of truth for magic numbers
    - Self-documenting default values with context
    - Easy to override for experiments
    - Type checking and validation

Example:
    >>> config = PipelineConfig(data_path=Path("/data"))
    >>> config.cv.n_repeats
    100
    >>> config.priors.as_dict()
    {'motion_onset': 0.0408, 'motion_integrator_contralateral': 0.2885, ...}

Author: Florian Kämpf
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from src.util.output_paths import get_output_dir
except ModuleNotFoundError:
    from util.output_paths import get_output_dir

# =============================================================================
# Constants - Cell Type Definitions
# =============================================================================

VALID_CELL_TYPES: list[str] = [
    "motion_onset",
    "motion_integrator_contralateral",
    "motion_integrator_ipsilateral",
    "slow_motion_integrator",
]
"""Valid cell type labels in standard order (alphabetical by abbreviation: cMI, iMI, MON, SMI)."""

CELL_TYPE_ABBREVIATIONS: dict[str, str] = {
    "motion_onset": "MON",
    "motion_integrator_contralateral": "cMI",
    "motion_integrator_ipsilateral": "iMI",
    "slow_motion_integrator": "SMI",
    "neg_control": "NC",
    "to_predict": "TP",
}
"""Maps full cell type names to standard abbreviations."""

ACRONYM_DICT: dict[str, str] = {
    "motion_onset": "mon",
    "motion_integrator_contralateral": "cmi",
    "motion_integrator_ipsilateral": "imi",
    "slow_motion_integrator": "smi",
    "neg_control": "nc",
}
"""Maps full cell type names to lowercase abbreviations (for NBLAST analysis)."""


# =============================================================================
# Constants - Visualization
# =============================================================================

CELL_TYPE_COLORS: dict[str, str] = {
    "motion_integrator_ipsilateral": "#feb326b3",  # Orange with alpha
    "motion_integrator_contralateral": "#e84d8ab3",  # Pink with alpha
    "motion_onset": "#64c5ebb3",  # Cyan with alpha
    "slow_motion_integrator": "#7f58afb3",  # Purple with alpha
}
"""Color mapping for cell type visualization (RGBA hex format).

Colors chosen for:
    - High contrast between types
    - Color-blind accessibility (distinct hues)
    - Alpha channel for overlapping neuron visualization
"""

CELL_TYPE_COLORS_SOLID: dict[str, str] = {
    "motion_integrator_ipsilateral": "#feb326",  # Orange
    "motion_integrator_contralateral": "#e84d8a",  # Pink
    "motion_onset": "#64c5eb",  # Cyan
    "slow_motion_integrator": "#7f58af",  # Purple
}
"""Solid color mapping without alpha (for confusion matrices, bar charts)."""


# =============================================================================
# Constants - Imaging Modalities
# =============================================================================

VALID_MODALITIES: list[str] = ["clem", "photoactivation", "EM"]
"""Valid imaging modality values."""

MODALITY_ALIASES: dict[str, str] = {
    "pa": "photoactivation",
    "CLEM": "clem",
    "em": "EM",
    "all": "all",
}
"""Maps common aliases to canonical modality names."""


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class CellTypePriors:
    """Prior probabilities for cell type classification.

    Based on cell counts from 2p photon imaging analysis.
    Source: Jonathans counts from CLEM functional imaging of hindbrain integrator circuit.
    Total cell count: 539 neurons in the anterior hindbrain of correct functional identiy.

    Attributes
    ----------
        motion_onset: Prior for MON cells (22/539 ≈ 4.1%).
        motion_integrator_contralateral: Prior for cMI cells (155.5/539 ≈ 28.8%).
            Contralaterally-projecting motion integrator neurons.
        motion_integrator_ipsilateral: Prior for iMI cells (155.5/539 ≈ 28.8%).
            Ipsilaterally-projecting motion integrator neurons.
        slow_motion_integrator: Prior for SMI cells (206/539 ≈ 38.2%).
            Slow motion integrator neurons.

    Note:
        iMI and cMI counts are split evenly (311/2 = 155.5) because the
        original count did not distinguish projection direction.

    Example:
        >>> priors = CellTypePriors()
        >>> priors.as_dict()
        {'motion_onset': 0.0408, 'motion_integrator_contralateral': 0.2885, ...}
        >>> sum(priors.as_dict().values())
        1.0
    """

    motion_onset: float = 22 / 539
    motion_integrator_contralateral: float = 155.5 / 539
    motion_integrator_ipsilateral: float = 155.5 / 539
    slow_motion_integrator: float = 206 / 539

    def as_dict(self) -> dict[str, float]:
        """Return priors as dictionary for sklearn classifiers."""
        return {
            "motion_onset": self.motion_onset,
            "motion_integrator_contralateral": self.motion_integrator_contralateral,
            "motion_integrator_ipsilateral": self.motion_integrator_ipsilateral,
            "slow_motion_integrator": self.slow_motion_integrator,
        }

    def as_array(self, order: list[str] | None = None) -> np.ndarray:
        """Return priors as numpy array in specified order.

        Parameters
        ----------
        order : List[str], optional
            Order of cell types. Default: VALID_CELL_TYPES.

        Returns
        -------
        np.ndarray
            Prior probabilities in specified order.
        """
        order = order or VALID_CELL_TYPES
        d = self.as_dict()
        return np.array([d[ct] for ct in order])

    def validate(self) -> bool:
        """Check that priors sum to 1.0 (within floating point tolerance)."""
        total = sum(self.as_dict().values())
        return abs(total - 1.0) < 1e-6


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation evaluation.

    Supports two CV methods:
        - 'lpo' (Leave-P-Out): Deterministic, thorough but slow for large p.
          Default p=1 gives Leave-One-Out CV.
        - 'ss' (ShuffleSplit): Stochastic, faster with multiple repeats.
          Randomly splits data into train/test for n_repeats iterations.

    Attributes
    ----------
        method: CV method ('lpo' for Leave-P-Out, 'ss' for ShuffleSplit).
        n_repeats: Number of CV iterations for ShuffleSplit. Default: 100.
            Higher values give more stable estimates but take longer.
        test_size: Fraction of data for testing in ShuffleSplit. Default: 0.3.
            Common values: 0.2-0.3 for reasonable train/test split.
        p: Number of samples to leave out in LeavePOut. Default: 1.
            p=1 is Leave-One-Out, p=2 is Leave-Two-Out, etc.
        metric: Scoring metric ('accuracy', 'f1', 'balanced_accuracy').
            'f1' is preferred for imbalanced classes. Default: 'f1'.

    Example:
        >>> cv_config = CrossValidationConfig(method="ss", n_repeats=50)
        >>> cv_config.method
        'ss'
    """

    method: str = "lpo"
    n_repeats: int = 100
    test_size: float = 0.3
    p: int = 1
    metric: str = "f1"

    def __post_init__(self):
        """Validate configuration values."""
        if self.method not in ("lpo", "ss"):
            raise ValueError(f"method must be 'lpo' or 'ss', got '{self.method}'")
        if self.metric not in ("accuracy", "f1", "balanced_accuracy"):
            raise ValueError(
                f"metric must be 'accuracy', 'f1', or 'balanced_accuracy', got '{self.metric}'"
            )
        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {self.n_repeats}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if self.p < 1:
            raise ValueError(f"p must be >= 1, got {self.p}")


@dataclass
class VerificationConfig:
    """Configuration for prediction verification.

    Verification tests whether predictions are morphologically consistent
    with their assigned cell type using NBLAST similarity and outlier detection.

    Attributes
    ----------
        required_tests: Tests that must pass for a prediction to be verified.
            Default: ['IF', 'LOF'] (Isolation Forest and Local Outlier Factor).
            Other options: 'OCSVM', 'NBLAST_g', 'NBLAST_z', 'probability_test'.
        nblast_threshold: Minimum NBLAST score for valid prediction.
            Default: 0.3. Range: 0-1 (higher = more similar morphology).
        probability_threshold: Minimum classifier confidence for 'probability_test'.
            Default: 0.7. Predictions below this threshold are less reliable.
        save_visualizations: Whether to save per-cell validation plots.
            Default: False (can generate many files).

    Example:
        >>> ver_config = VerificationConfig(required_tests=["IF", "LOF", "NBLAST_g"])
        >>> "NBLAST_g" in ver_config.required_tests
        True
    """

    required_tests: list[str] = field(default_factory=lambda: ["IF", "LOF"])
    nblast_threshold: float = 0.3
    probability_threshold: float = 0.7
    save_visualizations: bool = False


@dataclass
class FeatureConfig:
    """Configuration for feature computation and selection.

    Attributes
    ----------
        form_factor_samples: Number of samples for form factor computation.
            Default: 300. Higher values give smoother estimates.
        form_factor_cores: Number of CPU cores for parallel computation.
            Default: 15. Set to -1 for all available cores.
        persistence_samples: Number of samples for persistence vectors.
            Default: 300.

    Example:
        >>> feat_config = FeatureConfig(form_factor_cores=8)
    """

    form_factor_samples: int = 300
    form_factor_cores: int = 15
    persistence_samples: int = 300


@dataclass
class PipelineConfig:
    """Master configuration for the classification pipeline.

    Groups all configuration into a single, documented object that can be
    passed to ClassPredictor and its component modules.

    Attributes
    ----------
        data_path: Base path to data directory containing HDF5 features,
            reference predictions, and brain meshes.
        output_path: Path for saving results. Default: data_path / 'output'.
        train_modalities: Modalities to use for training classifier.
            Default: ['clem', 'photoactivation'].
        use_priors: Whether to use anatomical priors in classification.
            Default: True (improves accuracy for imbalanced classes).
        priors: Cell type prior configuration.
        cv: Cross-validation configuration.
        verification: Verification configuration.
        features: Feature computation configuration.

    Example:
        >>> config = PipelineConfig(data_path=Path("/data/hindbrain"))
        >>> config.cv.n_repeats
        100
        >>> config.priors.as_dict()["slow_motion_integrator"]
        0.382...

    Note:
        ClassPredictor also accepts a plain Path as input and will
        create default PipelineConfig internally.
    """

    data_path: Path
    output_path: Path | None = None
    train_modalities: list[str] = field(default_factory=lambda: ["clem", "photoactivation"])
    use_priors: bool = True
    priors: CellTypePriors = field(default_factory=CellTypePriors)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self):
        """Convert string paths and set defaults."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if self.output_path is None:
            self.output_path = get_output_dir("classifier_pipeline")
        elif isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

