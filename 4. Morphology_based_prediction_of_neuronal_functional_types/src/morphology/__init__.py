"""Morphological analysis functions for cell neurons.

This module provides functions for:
- Analyzing and repairing neuron skeleton (SWC) data
- Finding and extracting branch structures
- Fragmenting neurons into segments for analysis
- Computing NBLAST similarity scores between neurons

The morphology analysis supports cells from all imaging modalities
(photoactivation, CLEM, EM) after their skeletons have been loaded.
"""

from .find_branches import find_branches
from .fragment_neurite import (
    angle_between_vectors,
    direct_angle_and_crossing_extraction,
    find_crossing_neurite,
    find_end_neurites,
    fragment_neuron_into_segments,
)
from .nblast import (
    compute_nblast_within_and_between,
    nblast_one_group,
    nblast_two_groups,
    nblast_two_groups_custom_matrix,
)
from .repair_swc import repair_neuron

__all__ = [
    # Branch analysis
    "find_branches",
    # Neurite fragmentation
    "find_end_neurites",
    "fragment_neuron_into_segments",
    "find_crossing_neurite",
    "angle_between_vectors",
    "direct_angle_and_crossing_extraction",
    # Skeleton repair
    "repair_neuron",
    # NBLAST similarity
    "nblast_one_group",
    "nblast_two_groups",
    "nblast_two_groups_custom_matrix",
    "compute_nblast_within_and_between",
]
