"""Morphological Metrics Calculator Module.

Handles calculation of neuronal morphology features for classification.

Zen of Python principles applied:
    - Simple is better than complex
    - Sparse is better than dense
    - Errors should never pass silently

Classes:
    MorphologyMetrics: Calculate morphological features from neuron skeletons
    BranchMetrics: Specialized branch analysis metrics
    HemisphericMetrics: Ipsilateral/contralateral hemisphere metrics

Author: Florian Kämpf
"""


import navis
import numpy as np
import pandas as pd
from tqdm import tqdm


class MorphologyMetrics:
    """Calculate basic morphological metrics for neurons.

    Provides a clean interface for computing standard neuron morphology features
    including cable length, branching, spatial extent, and tortuosity.

    Attributes
    ----------
    BRAIN_WIDTH : float
        Width of zebrafish hindbrain in microns (default: 495.56)

    Methods
    -------
    calculate_basic_metrics(neuron)
        Calculate basic morphology metrics for a single neuron
    calculate_batch_metrics(neurons_df)
        Calculate metrics for all neurons in a DataFrame
    """

    BRAIN_WIDTH = 495.56  # Zebrafish hindbrain width in microns

    @staticmethod
    def calculate_basic_metrics(neuron: navis.TreeNeuron) -> dict[str, float]:
        """Calculate basic morphological metrics for a single neuron.

        Parameters
        ----------
        neuron : navis.TreeNeuron
            Neuron skeleton to analyze

        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values

        Notes
        -----
        Calculated metrics include:
        - cable_length: Total cable length
        - bbox_volume: Bounding box volume
        - x_extent, y_extent, z_extent: Spatial extents
        - x_avg, y_avg, z_avg: Average coordinates
        - tortuosity: Path tortuosity measure
        - n_leafs, n_branches, n_ends, n_edges: Graph topology

        Examples
        --------
        >>> metrics = MorphologyMetrics.calculate_basic_metrics(neuron)
        >>> print(f"Cable length: {metrics['cable_length']:.2f} microns")
        """
        metrics = {}

        # Cable length
        metrics["cable_length"] = neuron.cable_length

        # Bounding box metrics
        extents = neuron.extents
        metrics["x_extent"] = extents[0]
        metrics["y_extent"] = extents[1]
        metrics["z_extent"] = extents[2]
        metrics["bbox_volume"] = extents[0] * extents[1] * extents[2]

        # Average coordinates
        metrics["x_avg"] = np.mean(neuron.nodes.x)
        metrics["y_avg"] = np.mean(neuron.nodes.y)
        metrics["z_avg"] = np.mean(neuron.nodes.z)

        # Soma coordinates
        metrics["soma_x"] = neuron.nodes.loc[0, "x"]
        metrics["soma_y"] = neuron.nodes.loc[0, "y"]
        metrics["soma_z"] = neuron.nodes.loc[0, "z"]

        # Tortuosity
        metrics["tortuosity"] = navis.tortuosity(neuron)

        # Graph topology
        metrics["n_leafs"] = neuron.n_leafs
        metrics["n_branches"] = neuron.n_branches
        metrics["n_ends"] = neuron.n_ends
        metrics["n_edges"] = neuron.n_edges

        return metrics

    @staticmethod
    def calculate_strahler_metrics(neuron: navis.TreeNeuron) -> dict[str, float]:
        """Calculate Strahler index metrics.

        Parameters
        ----------
        neuron : navis.TreeNeuron
            Neuron to analyze

        Returns
        -------
        Dict[str, float]
            Strahler index metrics

        Notes
        -----
        Strahler index measures branch hierarchy complexity.
        """
        metrics = {}

        # Calculate Strahler index
        navis.strahler_index(neuron)
        metrics["max_strahler_index"] = neuron.nodes.strahler_index.max()

        return metrics

    @staticmethod
    def calculate_persistence_metrics(neuron: navis.TreeNeuron) -> dict[str, float]:
        """Calculate topological persistence metrics.

        Parameters
        ----------
        neuron : navis.TreeNeuron
            Neuron to analyze

        Returns
        -------
        Dict[str, float]
            Persistence-based metrics

        Notes
        -----
        Persistence measures the "significance" of branches based on
        how long they persist in a filtration.
        """
        metrics = {}

        # Persistence points
        persistence = navis.persistence_points(neuron)
        metrics["n_persistence_points"] = len(persistence)

        if len(persistence) > 0:
            death_birth_delta = [row["death"] - row["birth"] for _, row in persistence.iterrows()]
            metrics["avg_delta_death_birth_persistence"] = np.mean(death_birth_delta)
            metrics["median_delta_death_birth_persistence"] = np.median(death_birth_delta)
            metrics["std_delta_death_birth_persistence"] = np.std(death_birth_delta)
        else:
            metrics["avg_delta_death_birth_persistence"] = 0
            metrics["median_delta_death_birth_persistence"] = 0
            metrics["std_delta_death_birth_persistence"] = 0

        return metrics

    @staticmethod
    def calculate_sholl_metrics(
        neuron: navis.TreeNeuron, radii: np.ndarray | None = None
    ) -> dict[str, float]:
        """Calculate Sholl analysis metrics.

        Parameters
        ----------
        neuron : navis.TreeNeuron
            Neuron to analyze
        radii : np.ndarray, optional
            Radii at which to sample. Default: np.arange(10, 200, 10)

        Returns
        -------
        Dict[str, float]
            Sholl analysis metrics

        Notes
        -----
        Sholl analysis measures branching at increasing distances from soma.
        """
        if radii is None:
            radii = np.arange(10, 200, 10)

        metrics = {}

        # Perform Sholl analysis
        sholl = navis.sholl_analysis(neuron, radii=radii, center="root")

        # Distance with maximum branches
        max_branch_dist = sholl.branch_points.idxmax()
        metrics["sholl_distance_max_branches"] = max_branch_dist
        metrics["sholl_distance_max_branches_cable_length"] = sholl.cable_length[max_branch_dist]

        return metrics

    @staticmethod
    def calculate_branchpoint_metrics(neuron: navis.TreeNeuron) -> dict[str, float]:
        """Calculate branch point related metrics.

        Parameters
        ----------
        neuron : navis.TreeNeuron
            Neuron to analyze

        Returns
        -------
        Dict[str, float]
            Branch point metrics
        """
        metrics = {}

        # Main branch point
        metrics["main_branchpoint"] = navis.find_main_branchpoint(neuron)

        return metrics


class HemisphericMetrics:
    """Calculate hemisphere-specific metrics for bilateral neurons.

    For zebrafish hindbrain neurons that cross the midline,
    computes separate metrics for ipsilateral and contralateral projections.

    Methods
    -------
    calculate_hemispheric_metrics(neuron, brain_width)
        Calculate ipsi/contra metrics
    calculate_ic_index(x_coords, brain_width)
        Calculate ipsilateral-contralateral index
    """

    @staticmethod
    def calculate_ic_index(x_coords: np.ndarray, brain_width: float = 495.56) -> float:
        """Calculate ipsilateral-contralateral index.

        Measures the weighted average position relative to midline.

        Parameters
        ----------
        x_coords : np.ndarray
            X-coordinates of neuron nodes
        brain_width : float, optional
            Brain width in microns. Default: 495.56

        Returns
        -------
        float
            IC index (-1 = fully contra, 0 = midline, 1 = fully ipsi)

        Notes
        -----
        Formula: ic_index = mean((midline - x) / (midline))
        """
        midline = brain_width / 2
        distances = [(midline - x) / midline for x in x_coords]
        return np.mean(distances)

    @staticmethod
    def calculate_hemispheric_metrics(
        neuron: navis.TreeNeuron, brain_width: float = 495.56
    ) -> dict[str, float]:
        """Calculate hemisphere-specific morphology metrics.

        Parameters
        ----------
        neuron : navis.TreeNeuron
            Neuron to analyze
        brain_width : float, optional
            Brain width in microns. Default: 495.56

        Returns
        -------
        Dict[str, float]
            Hemispheric metrics

        Notes
        -----
        Calculates separate metrics for nodes in ipsilateral
        (x < midline) vs contralateral (x > midline) hemispheres.

        Examples
        --------
        >>> metrics = HemisphericMetrics.calculate_hemispheric_metrics(neuron)
        >>> print(f"Fraction contra: {metrics['fraction_contra']:.2%}")
        """
        metrics = {}
        midline = brain_width / 2

        # Node counts
        ipsi_mask = neuron.nodes.x < midline
        contra_mask = neuron.nodes.x > midline

        metrics["n_nodes_ipsi_hemisphere"] = ipsi_mask.sum()
        metrics["n_nodes_contra_hemisphere"] = contra_mask.sum()

        total_nodes = len(neuron.nodes)
        metrics["n_nodes_ipsi_hemisphere_fraction"] = ipsi_mask.sum() / total_nodes
        metrics["n_nodes_contra_hemisphere_fraction"] = contra_mask.sum() / total_nodes
        metrics["fraction_contra"] = contra_mask.sum() / total_nodes

        # IC index
        metrics["x_location_index"] = HemisphericMetrics.calculate_ic_index(
            neuron.nodes.x.values, brain_width
        )

        # Branch counts
        branch_mask = neuron.nodes.type == "branch"
        metrics["ipsilateral_branches"] = (ipsi_mask & branch_mask).sum()
        metrics["contralateral_branches"] = (contra_mask & branch_mask).sum()

        # Ipsilateral extents
        if ipsi_mask.any():
            ipsi_nodes = neuron.nodes[ipsi_mask]
            metrics["y_extent_ipsi"] = ipsi_nodes.y.max() - ipsi_nodes.y.min()
            metrics["z_extent_ipsi"] = ipsi_nodes.z.max() - ipsi_nodes.z.min()
            metrics["max_x_ipsi"] = ipsi_nodes.x.max()
            metrics["max_y_ipsi"] = ipsi_nodes.y.max()
            metrics["max_z_ipsi"] = ipsi_nodes.z.max()
            metrics["min_x_ipsi"] = ipsi_nodes.x.min()
            metrics["min_y_ipsi"] = ipsi_nodes.y.min()
            metrics["min_z_ipsi"] = ipsi_nodes.z.min()
        else:
            for key in [
                "y_extent_ipsi",
                "z_extent_ipsi",
                "max_x_ipsi",
                "max_y_ipsi",
                "max_z_ipsi",
                "min_x_ipsi",
                "min_y_ipsi",
                "min_z_ipsi",
            ]:
                metrics[key] = 0

        # Contralateral extents
        if contra_mask.any():
            contra_nodes = neuron.nodes[contra_mask]
            metrics["y_extent_contra"] = contra_nodes.y.max() - contra_nodes.y.min()
            metrics["z_extent_contra"] = contra_nodes.z.max() - contra_nodes.z.min()
            metrics["max_x_contra"] = contra_nodes.x.max()
            metrics["max_y_contra"] = contra_nodes.y.max()
            metrics["max_z_contra"] = contra_nodes.z.max()
            metrics["min_x_contra"] = contra_nodes.x.min()
            metrics["min_y_contra"] = contra_nodes.y.min()
            metrics["min_z_contra"] = contra_nodes.z.min()
        else:
            for key in [
                "y_extent_contra",
                "z_extent_contra",
                "max_x_contra",
                "max_y_contra",
                "max_z_contra",
                "min_x_contra",
                "min_y_contra",
                "min_z_contra",
            ]:
                metrics[key] = 0

        return metrics


class BranchMetrics:
    """Calculate detailed branch-level metrics.

    Analyzes individual branches and their properties including
    main path, first branch, and largest branches.

    Methods
    -------
    calculate_first_branch_metrics(neuron)
        Metrics for the first major branch
    calculate_main_path_metrics(neuron, branches_df)
        Metrics for the main axonal path
    """

    @staticmethod
    def calculate_first_branch_metrics(neuron: navis.TreeNeuron) -> dict[str, float]:
        """Calculate metrics for the first branch.

        Parameters
        ----------
        neuron : navis.TreeNeuron
            Neuron to analyze

        Returns
        -------
        Dict[str, float]
            First branch metrics

        Notes
        -----
        Analyzes the initial branch after the soma, measuring
        cable length and distance to first branch point.
        """
        metrics = {}

        try:
            # Split into fragments at all leaf nodes
            fragments = navis.split_into_fragments(neuron, neuron.n_leafs)

            if len(fragments) > 1:
                # First branch is fragment 1 (0 is typically soma)
                first_branch = fragments[1]
                metrics["first_branch_longest_neurite"] = navis.longest_neurite(
                    first_branch
                ).cable_length
                metrics["first_branch_total_branch_length"] = first_branch.cable_length
            else:
                metrics["first_branch_longest_neurite"] = 0
                metrics["first_branch_total_branch_length"] = 0

            # Cable length to first branch
            pruned = navis.prune_twigs(neuron, 5, recursive=True)
            if pruned.n_branches > 0:
                first_branch_node = pruned.nodes.loc[pruned.nodes.type == "branch", "node_id"].iloc[
                    0
                ]
                cut_result = navis.cut_skeleton(pruned, first_branch_node)
                metrics["cable_length_2_first_branch"] = cut_result[1].cable_length

                # Z distance to first branch
                first_z = cut_result[1].nodes.iloc[0].z
                last_z = cut_result[1].nodes.iloc[-1].z
                metrics["z_distance_first_2_first_branch"] = first_z - last_z
            else:
                metrics["cable_length_2_first_branch"] = 0
                metrics["z_distance_first_2_first_branch"] = 0

        except Exception:
            # Handle edge cases gracefully
            for key in [
                "first_branch_longest_neurite",
                "first_branch_total_branch_length",
                "cable_length_2_first_branch",
                "z_distance_first_2_first_branch",
            ]:
                metrics[key] = 0

        return metrics


class MetricsBatchCalculator:
    """Batch calculation of all metrics for multiple neurons.

    Provides high-level interface for calculating all morphological
    metrics for a DataFrame of neurons with progress tracking.

    Methods
    -------
    calculate_all_metrics(neurons_df, resample_resolution)
        Calculate all metrics for all neurons
    """

    @staticmethod
    def calculate_all_metrics(
        neurons_df: pd.DataFrame,
        resample_resolution: str = "0.5 micron",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Calculate all morphological metrics for a DataFrame of neurons.

        Parameters
        ----------
        neurons_df : pd.DataFrame
            DataFrame with 'swc' column containing navis.TreeNeuron objects
        resample_resolution : str, optional
            Resampling resolution. Default: "0.5 micron"
        show_progress : bool, optional
            Show progress bar. Default: True

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added metric columns

        Notes
        -----
        This is a comprehensive wrapper that calculates:
        - Basic morphology metrics
        - Strahler index
        - Persistence metrics
        - Sholl analysis
        - Branch metrics
        - Hemispheric metrics

        Examples
        --------
        >>> enriched_df = MetricsBatchCalculator.calculate_all_metrics(
        ...     cells_df, resample_resolution="0.5 micron"
        ... )
        >>> print(enriched_df.columns)
        """
        print("\n Calculating morphological metrics...")
        print(f"   Number of neurons: {len(neurons_df)}")
        print(f"   Resample resolution: {resample_resolution}")

        # Store original neurons
        neurons_df["not_resampled_swc"] = neurons_df["swc"]

        # Resample neurons
        print("   Resampling neurons...")
        neurons_df["swc"] = neurons_df.swc.apply(lambda x: x.resample(resample_resolution))

        # Initialize metric calculators
        morph = MorphologyMetrics()
        hemi = HemisphericMetrics()
        branch = BranchMetrics()

        # Calculate metrics for each neuron
        iterator = tqdm(neurons_df.iterrows(), total=len(neurons_df), disable=not show_progress)

        for i, cell in iterator:
            neuron = cell["swc"]

            # Basic metrics
            basic = morph.calculate_basic_metrics(neuron)
            for key, value in basic.items():
                neurons_df.loc[i, key] = value

            # Strahler metrics
            strahler = morph.calculate_strahler_metrics(neuron)
            for key, value in strahler.items():
                neurons_df.loc[i, key] = value

            # Persistence metrics
            persistence = morph.calculate_persistence_metrics(neuron)
            for key, value in persistence.items():
                neurons_df.loc[i, key] = value

            # Sholl metrics
            sholl = morph.calculate_sholl_metrics(neuron)
            for key, value in sholl.items():
                neurons_df.loc[i, key] = value

            # Branch metrics
            branch_metrics = branch.calculate_first_branch_metrics(neuron)
            for key, value in branch_metrics.items():
                neurons_df.loc[i, key] = value

            # Hemispheric metrics
            hemi_metrics = hemi.calculate_hemispheric_metrics(neuron)
            for key, value in hemi_metrics.items():
                neurons_df.loc[i, key] = value

        print(" Morphological metrics calculated\n")
        return neurons_df
