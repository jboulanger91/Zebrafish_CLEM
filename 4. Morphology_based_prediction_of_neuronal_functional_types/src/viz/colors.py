"""Unified color schemes for hindbrain structure-function visualizations.

This module centralizes all color definitions used across the visualization
codebase. Use these constants instead of defining colors locally to ensure
consistency across all figures.

Color Schemes:
    FUNCTIONAL_COLORS: Colors for functional cell types (MON, MI, SMI, etc.)
    POPULATION_COLORS: Colors for population imaging classifications
    MODALITY_COLORS: Colors for imaging modalities (CLEM, PA, EM)

Usage:
    >>> from src.viz.colors import FUNCTIONAL_COLORS, get_functional_color
    >>> color = FUNCTIONAL_COLORS["motion_integrator"]
    >>> rgba = hex_to_rgba("#e84d8a", alpha=0.7)

Author: Florian Kampf
"""

from __future__ import annotations

# =============================================================================
# Functional Cell Type Colors
# =============================================================================
# Standard colors for functional cell types from single-cell analysis
# Format: hex without alpha (add alpha with utility functions as needed)

FUNCTIONAL_COLORS: dict[str, str] = {
    # Primary functional types
    "motion_integrator": "#e84d8a",            # Pink
    "motion_onset": "#64c5eb",                 # Cyan
    "slow_motion_integrator": "#7f58af",       # Purple
    "neg_control": "#a8c256",                  # Green (negative control)
    # Motion integrator subtypes (by morphology)
    "motion_integrator_ipsilateral": "#feb326",     # Orange
    "motion_integrator_contralateral": "#e84d8a",   # Pink (same as motion_integrator)
}

# Abbreviations used in some contexts (e.g., confusion matrices)
FUNCTIONAL_ABBREV: dict[str, str] = {
    "MON": "motion_onset",
    "cMI": "motion_integrator_contralateral",
    "iMI": "motion_integrator_ipsilateral",
    "SMI": "slow_motion_integrator",
    "MI": "motion_integrator",
    "NC": "neg_control",
}

# Integer label to class name mapping (from k-means clustering)
INT_TO_CLASS: dict[int, str] = {
    0: "slow_motion_integrator",
    1: "motion_integrator",
    2: "motion_onset",
    3: "motion_integrator",  # Second motion_integrator cluster
}


# =============================================================================
# Population Imaging Colors
# =============================================================================
# Colors for whole-brain GCaMP imaging classifications

POPULATION_COLORS: dict[str, str] = {
    "MON": "#68C7EC",   # Light blue - motoneuron-like
    "MI": "#ED7658",    # Coral - motor integrator
    "SMI": "#7F58AF",   # Purple - sensory-motor integrator
    "none": "#808080",  # Gray - unclassified
}


# =============================================================================
# Imaging Modality Colors
# =============================================================================
# Colors for different imaging modalities

MODALITY_COLORS: dict[str, str] = {
    "CLEM": "#2ecc71",          # Green
    "clem": "#2ecc71",          # Alias lowercase
    "PA": "#3498db",            # Blue
    "pa": "#3498db",            # Alias lowercase
    "photoactivation": "#3498db",  # Alias full name
    "EM": "#e74c3c",            # Red
    "em": "#e74c3c",            # Alias lowercase
    "all": "#9b59b6",           # Purple (for combined)
}


# =============================================================================
# Navis/Neuron Visualization Colors
# =============================================================================
# Colors for 3D neuron rendering (RGB tuples for navis compatibility)

NEURON_COLORS_RGB: dict[str, tuple[int, int, int, float]] = {
    "motion_integrator": (255, 99, 71, 1.0),       # Tomato
    "motion_onset": (100, 197, 235, 1.0),           # Light blue
    "slow_motion_integrator": (127, 88, 175, 1.0),  # Purple
    "nan": (60, 60, 60, 1.0),                       # Dark gray
    "unknown": (128, 128, 128, 0.5),                # Gray semi-transparent
}


# =============================================================================
# Connectome Visualization Colors
# =============================================================================
# Colors for fish15 connectome analysis (RGBA tuples, 0-1 range)
# Used for connectivity matrices and connectome visualization

CONNECTOME_COLORS_RGBA: dict[str, tuple[float, float, float, float]] = {
    # Functional cell types
    "motion_integrator_ipsilateral": (254/255, 179/255, 38/255, 0.7),      # Yellow-orange
    "motion_integrator_contralateral": (232/255, 77/255, 138/255, 0.7),    # Magenta-pink
    "motion_onset": (100/255, 197/255, 235/255, 0.7),                     # Light blue
    "slow_motion_integrator": (127/255, 88/255, 175/255, 0.7),            # Purple
    # Structural/morphological types
    "myelinated": (80/255, 220/255, 100/255, 0.7),                  # Bright teal
    "raphe": (34/255, 139/255, 34/255, 0.7),                        # Forest green
    "axon": (0.2, 0.2, 0.2, 0.7),                                   # Dark gray
    "axon_rostral": (105/255, 105/255, 105/255, 0.7),               # Dim gray
    "axon_caudal": (192/255, 192/255, 192/255, 0.7),                # Silver/light gray
    "not functionally imaged": (0.5, 0.5, 0.5, 0.7),                # Gray
}

# Extended colors with hemisphere suffix (for LR-specific plots)
CONNECTOME_COLORS_LR_RGBA: dict[str, tuple[float, float, float, float]] = {
    **{f"{k}_left": v for k, v in CONNECTOME_COLORS_RGBA.items()},
    **{f"{k}_right": v for k, v in CONNECTOME_COLORS_RGBA.items()},
}


# =============================================================================
# Utility Functions
# =============================================================================

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert hex color to RGBA tuple (0-1 range).

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., "#e84d8a" or "e84d8a")
    alpha : float
        Alpha value (0-1). Default: 1.0

    Returns
    -------
    Tuple[float, float, float, float]
        RGBA tuple with values in 0-1 range

    Examples
    --------
    >>> hex_to_rgba("#000000")
    (0.0, 0.0, 0.0, 1.0)
    >>> hex_to_rgba("#ffffff")
    (1.0, 1.0, 1.0, 1.0)
    >>> hex_to_rgba("#ffffff", alpha=0.5)
    (1.0, 1.0, 1.0, 0.5)
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b, alpha)


def hex_to_rgb_tuple(hex_color: str, alpha: float = 1.0) -> tuple[int, int, int, float]:
    """Convert hex color to RGB tuple (0-255 range) for navis compatibility.

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., "#e84d8a")
    alpha : float
        Alpha value (0-1). Default: 1.0

    Returns
    -------
    Tuple[int, int, int, float]
        RGB tuple with values in 0-255 range plus alpha

    Examples
    --------
    >>> hex_to_rgb_tuple("#e84d8a")
    (232, 77, 138, 1.0)
    >>> hex_to_rgb_tuple("#000000")
    (0, 0, 0, 1.0)
    >>> hex_to_rgb_tuple("#ffffff", alpha=0.5)
    (255, 255, 255, 0.5)
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def get_functional_color(
    cell_type: str,
    alpha: float | None = None,
    format: str = "hex",
) -> str | tuple:
    """Get color for a functional cell type.

    Parameters
    ----------
    cell_type : str
        Cell type name (e.g., "motion_integrator", "MON", "motion_onset")
    alpha : float, optional
        Alpha value to apply. If None, returns opaque color.
    format : str
        Output format: "hex", "rgba", or "rgb_tuple"

    Returns
    -------
    str or Tuple
        Color in requested format

    Examples
    --------
    >>> get_functional_color("motion_integrator")
    '#e84d8a'
    >>> get_functional_color("MON")
    '#64c5eb'
    >>> get_functional_color("unknown_type")
    '#808080'
    """
    # Handle abbreviations
    if cell_type.upper() in FUNCTIONAL_ABBREV:
        cell_type = FUNCTIONAL_ABBREV[cell_type.upper()]

    # Get base color
    hex_color = FUNCTIONAL_COLORS.get(cell_type.lower(), "#808080")

    if format == "hex":
        if alpha is not None:
            # Append alpha as hex
            alpha_hex = format(int(alpha * 255), "02x")
            return f"{hex_color}{alpha_hex}"
        return hex_color
    elif format == "rgba":
        return hex_to_rgba(hex_color, alpha if alpha is not None else 1.0)
    elif format == "rgb_tuple":
        return hex_to_rgb_tuple(hex_color, alpha if alpha is not None else 1.0)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'hex', 'rgba', or 'rgb_tuple'")


def get_modality_color(modality: str, format: str = "hex") -> str | tuple:
    """Get color for an imaging modality.

    Parameters
    ----------
    modality : str
        Modality name (e.g., "CLEM", "PA", "EM", "photoactivation")
    format : str
        Output format: "hex" or "rgba"

    Returns
    -------
    str or Tuple
        Color in requested format

    Examples
    --------
    >>> get_modality_color("CLEM")
    '#2ecc71'
    >>> get_modality_color("PA")
    '#3498db'
    >>> get_modality_color("unknown_modality")
    '#808080'
    """
    hex_color = MODALITY_COLORS.get(modality, MODALITY_COLORS.get(modality.lower(), "#808080"))

    if format == "hex":
        return hex_color
    elif format == "rgba":
        return hex_to_rgba(hex_color)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'hex' or 'rgba'")


# =============================================================================
# GUI Default Colors
# =============================================================================
# Nested color dict used by the 3 GUI modules (projection_gui, interactive_gui,
# video_gui). Built from canonical flat dicts above to keep a single source.

DEFAULT_GUI_COLORS: dict[str, dict[str, str]] = {
    "functional_type": {
        "motion_integrator_ipsilateral": FUNCTIONAL_COLORS["motion_integrator_ipsilateral"],
        "motion_integrator_contralateral": FUNCTIONAL_COLORS["motion_integrator_contralateral"],
        "motion_onset": FUNCTIONAL_COLORS["motion_onset"],
        "slow_motion_integrator": FUNCTIONAL_COLORS["slow_motion_integrator"],
        "other": FUNCTIONAL_COLORS["neg_control"],
    },
    "validation_status": {
        "validated": "#2ecc71",
        "predicted": "#f39c12",
        "unclassified": "#95a5a6",
    },
    "modality": {
        "pa": MODALITY_COLORS["pa"],
        "clem": MODALITY_COLORS["clem"],
        "em": MODALITY_COLORS["em"],
    },
    "object_type": {
        "cell": "#3498db",
        "axon": "#e74c3c",
    },
    "neurotransmitter": {
        "excitatory": "#e74c3c",
        "inhibitory": "#3498db",
        "unknown": "#95a5a6",
    },
    "morphology": {
        "ipsilateral": "#f39c12",
        "contralateral": "#9b59b6",
        "unknown": "#95a5a6",
    },
    "other": {
        "none": "#95a5a6",
    },
}


# =============================================================================
# Aliases
# =============================================================================

DEFAULT_COLORS = {
    key: f"{color}b3"  # Add alpha=0.7 as hex suffix (b3 = 179/255 = 0.7)
    for key, color in FUNCTIONAL_COLORS.items()
}

DEFAULT_INT2CLASS = INT_TO_CLASS


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Color dictionaries
    "FUNCTIONAL_COLORS",
    "FUNCTIONAL_ABBREV",
    "INT_TO_CLASS",
    "POPULATION_COLORS",
    "MODALITY_COLORS",
    "NEURON_COLORS_RGB",
    "CONNECTOME_COLORS_RGBA",
    "CONNECTOME_COLORS_LR_RGBA",
    # Utility functions
    "hex_to_rgba",
    "hex_to_rgb_tuple",
    "get_functional_color",
    "get_modality_color",
    # GUI colors
    "DEFAULT_GUI_COLORS",
    # Legacy compatibility
    "DEFAULT_COLORS",
    "DEFAULT_INT2CLASS",
]
