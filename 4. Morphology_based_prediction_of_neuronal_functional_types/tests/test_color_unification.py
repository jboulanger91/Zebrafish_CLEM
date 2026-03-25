"""Verify color dictionaries have a single canonical source.

Tests that src.viz.colors provides all required color dictionaries,
that they are complete, and that GUI modules either import from the
canonical source or at minimum stay consistent with it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Canonical dictionary completeness
# =============================================================================

class TestFunctionalColorsCanonical:
    """FUNCTIONAL_COLORS covers all needed functional cell types."""

    def test_import(self):
        """Verify FUNCTIONAL_COLORS is importable and is a dict."""
        from src.viz.colors import FUNCTIONAL_COLORS
        assert isinstance(FUNCTIONAL_COLORS, dict)

    @pytest.mark.parametrize("key", [
        "motion_integrator",
        "motion_integrator_ipsilateral",
        "motion_integrator_contralateral",
        "motion_onset",
        "slow_motion_integrator",
        "neg_control",
    ])
    def test_required_keys(self, key):
        """Verify each required functional type key exists."""
        from src.viz.colors import FUNCTIONAL_COLORS
        assert key in FUNCTIONAL_COLORS, f"Missing key: {key}"

    def test_values_are_hex_strings(self):
        """Verify all values are 7-char hex color strings."""
        from src.viz.colors import FUNCTIONAL_COLORS
        for key, val in FUNCTIONAL_COLORS.items():
            assert isinstance(val, str), f"{key}: expected str, got {type(val)}"
            assert val.startswith("#"), f"{key}: hex should start with #"
            assert len(val) == 7, f"{key}: expected #RRGGBB (7 chars), got {val!r}"


class TestConnectomeColorsCanonical:
    """CONNECTOME_COLORS_RGBA covers all connectome cell types."""

    def test_import(self):
        """Verify CONNECTOME_COLORS_RGBA is importable and is a dict."""
        from src.viz.colors import CONNECTOME_COLORS_RGBA
        assert isinstance(CONNECTOME_COLORS_RGBA, dict)

    @pytest.mark.parametrize("key", [
        "motion_integrator_ipsilateral",
        "motion_integrator_contralateral",
        "motion_onset",
        "slow_motion_integrator",
    ])
    def test_required_functional_keys(self, key):
        """Verify each required connectome key exists."""
        from src.viz.colors import CONNECTOME_COLORS_RGBA
        assert key in CONNECTOME_COLORS_RGBA, f"Missing key: {key}"

    def test_values_are_rgba_tuples(self):
        """Verify all values are 4-element RGBA tuples in [0, 1]."""
        from src.viz.colors import CONNECTOME_COLORS_RGBA
        for key, val in CONNECTOME_COLORS_RGBA.items():
            assert isinstance(val, tuple), f"{key}: expected tuple"
            assert len(val) == 4, f"{key}: expected 4-element RGBA tuple"
            for component in val:
                assert 0.0 <= component <= 1.0, (
                    f"{key}: RGBA values must be in [0, 1], got {val}"
                )


class TestModalityColors:
    """MODALITY_COLORS covers pa, clem, em in both cases."""

    def test_import(self):
        """Verify MODALITY_COLORS is importable and is a dict."""
        from src.viz.colors import MODALITY_COLORS
        assert isinstance(MODALITY_COLORS, dict)

    @pytest.mark.parametrize("modality", ["pa", "clem", "em"])
    def test_lowercase_keys(self, modality):
        """Verify lowercase modality keys exist."""
        from src.viz.colors import MODALITY_COLORS
        assert modality in MODALITY_COLORS, f"Missing lowercase key: {modality}"

    @pytest.mark.parametrize("modality", ["PA", "CLEM", "EM"])
    def test_uppercase_keys(self, modality):
        """Verify uppercase modality keys exist."""
        from src.viz.colors import MODALITY_COLORS
        assert modality in MODALITY_COLORS, f"Missing uppercase key: {modality}"


class TestNeuronColorsRGB:
    """NEURON_COLORS_RGB exists for navis rendering."""

    def test_import(self):
        """Verify NEURON_COLORS_RGB is importable and is a dict."""
        from src.viz.colors import NEURON_COLORS_RGB
        assert isinstance(NEURON_COLORS_RGB, dict)

    @pytest.mark.parametrize("key", ["motion_integrator", "motion_onset", "slow_motion_integrator"])
    def test_required_keys(self, key):
        """Verify each required neuron color key exists."""
        from src.viz.colors import NEURON_COLORS_RGB
        assert key in NEURON_COLORS_RGB, f"Missing key: {key}"


class TestPopulationColors:
    """POPULATION_COLORS exists for population imaging."""

    def test_import(self):
        """Verify POPULATION_COLORS is importable and is a dict."""
        from src.viz.colors import POPULATION_COLORS
        assert isinstance(POPULATION_COLORS, dict)

    @pytest.mark.parametrize("key", ["MON", "MI", "SMI"])
    def test_required_keys(self, key):
        """Verify each required population key exists."""
        from src.viz.colors import POPULATION_COLORS
        assert key in POPULATION_COLORS, f"Missing key: {key}"


class TestUtilityFunctions:
    """Utility functions are importable and work correctly."""

    def test_hex_to_rgba(self):
        """Verify hex_to_rgba returns a 4-element tuple with correct alpha."""
        from src.viz.colors import hex_to_rgba
        result = hex_to_rgba("#e84d8a", alpha=0.7)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert abs(result[3] - 0.7) < 1e-6

    def test_hex_to_rgb_tuple(self):
        """Verify hex_to_rgb_tuple returns correct 0-255 RGB values."""
        from src.viz.colors import hex_to_rgb_tuple
        result = hex_to_rgb_tuple("#e84d8a")
        assert isinstance(result, tuple)
        assert len(result) == 4
        # RGB values in 0-255 range
        assert result[0] == 232  # 0xe8

    def test_get_functional_color(self):
        """Verify get_functional_color returns correct hex for full name."""
        from src.viz.colors import get_functional_color
        assert get_functional_color("motion_integrator") == "#e84d8a"

    def test_get_functional_color_abbreviation(self):
        """Verify get_functional_color resolves abbreviations like MON."""
        from src.viz.colors import get_functional_color
        # "MON" should resolve to motion_onset
        assert get_functional_color("MON") == "#64c5eb"

    def test_get_modality_color(self):
        """Verify get_modality_color returns correct hex for CLEM."""
        from src.viz.colors import get_modality_color
        assert get_modality_color("CLEM") == "#2ecc71"


class TestModuleExports:
    """__all__ exports all public names."""

    def test_all_defined(self):
        """Verify __all__ is defined on the colors module."""
        from src.viz import colors
        assert hasattr(colors, "__all__")

    @pytest.mark.parametrize("name", [
        "FUNCTIONAL_COLORS",
        "FUNCTIONAL_ABBREV",
        "INT_TO_CLASS",
        "POPULATION_COLORS",
        "MODALITY_COLORS",
        "NEURON_COLORS_RGB",
        "CONNECTOME_COLORS_RGBA",
        "CONNECTOME_COLORS_LR_RGBA",
        "hex_to_rgba",
        "hex_to_rgb_tuple",
        "get_functional_color",
        "get_modality_color",
    ])
    def test_exported_name(self, name):
        """Verify each expected name is in __all__ and accessible."""
        from src.viz import colors
        assert name in colors.__all__, f"{name} not in __all__"
        assert hasattr(colors, name), f"{name} not defined on module"


# =============================================================================
# GUI consistency checks
# =============================================================================

class TestGUIColorConsistency:
    """GUI DEFAULT_COLORS inline dicts should be consistent with canonical source.

    The GUI files define a nested DEFAULT_COLORS dict (keyed by category like
    'functional_type', 'modality', etc.). These are GUI-specific structures,
    but the underlying hex values should match the canonical source.
    """

    GUI_FILES = [
        ROOT / "visualization" / "projection_2d" / "projection_gui.py",
        ROOT / "visualization" / "plot_3d_interactive" / "interactive_gui.py",
        ROOT / "visualization" / "video" / "video_gui.py",
    ]

    def test_interactive_gui_imports_functional_colors(self):
        """interactive_gui.py should import FUNCTIONAL_COLORS from canonical source."""
        path = ROOT / "visualization" / "plot_3d_interactive" / "interactive_gui.py"
        if not path.exists():
            pytest.skip(f"{path} not found")
        source = path.read_text()
        assert "from src.viz.colors import FUNCTIONAL_COLORS" in source, (
            "interactive_gui.py should import FUNCTIONAL_COLORS from src.viz.colors"
        )

    @pytest.mark.parametrize("gui_file", GUI_FILES, ids=lambda p: p.name)
    def test_no_standalone_functional_colors_dict(self, gui_file):
        """GUI files should not define their own FUNCTIONAL_COLORS dict.

        The GUI files have DEFAULT_COLORS (nested by category) which is a
        different structure from the canonical flat FUNCTIONAL_COLORS.
        But they should NOT redefine FUNCTIONAL_COLORS itself.
        """
        if not gui_file.exists():
            pytest.skip(f"{gui_file} not found")

        import ast

        source = gui_file.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "FUNCTIONAL_COLORS":
                        pytest.fail(
                            f"{gui_file.name} defines FUNCTIONAL_COLORS locally "
                            f"instead of importing from src.viz.colors"
                        )
