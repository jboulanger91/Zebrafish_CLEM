"""Utilities for retrieving the configured dataset base path."""

from __future__ import annotations

import tempfile
from pathlib import Path

PLACEHOLDER_PATH = "/YOUR/PATH/TO/CLEM_paper_data/HERE"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_LOCATIONS = (
    _REPO_ROOT / "config" / "path_configuration.txt",
    _REPO_ROOT / "functional_type_prediction" / "FK_tools" / "path_configuration.txt",
)
_NOT_SETUP_MESSAGE = (
    "Path isn't configured for user {user}. Please modify path_configuration.txt "
    "with your path to the CLEM_paper_data which you should download from the "
    "nextcloud. A user profile has been created for you."
)


class NotSetup(Exception):
    """Raised when a base path hasn't been configured for the active user."""


def _resolve_config_path() -> Path:
    """Return the configuration file path, creating a default file if necessary."""
    for candidate in _CONFIG_LOCATIONS:
        if candidate.exists():
            return candidate

    default_path = _CONFIG_LOCATIONS[0]
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.touch(exist_ok=True)
    return default_path


def _load_user_paths(config_path: Path) -> dict[str, str]:
    """Parse the configuration file into a mapping of usernames to base paths."""
    user_paths: dict[str, str] = {}

    with config_path.open("r", encoding="utf-8") as config_file:
        for line in config_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                user, path = parts
                user_paths[user] = path

    return user_paths


def _append_placeholder_profile(username: str, config_path: Path) -> None:
    """Append a placeholder entry for the provided user to the configuration file."""
    prefix = "\n" if config_path.stat().st_size > 0 else ""
    with config_path.open("a", encoding="utf-8") as config_file:
        config_file.write(f"{prefix}{username} {PLACEHOLDER_PATH}\n")


def get_base_path() -> Path:
    """Return the configured dataset base path for the current OS user.

    Tries the unified project root first (``{HBSF_ROOT}/data/``), then
    falls back to the legacy ``config/path_configuration.txt`` lookup.

    Raises
    ------
        NotSetup: If the user is missing from the configuration file or still uses
        the placeholder path.
    """
    # Try unified project root first
    try:
        from src.util.project_root import get_data_dir  # noqa: E402

        data_dir = get_data_dir()
        if data_dir.is_dir():
            return data_dir
    except (FileNotFoundError, ImportError):
        pass

    # Legacy fallback: config-file-based resolution
    config_path = _resolve_config_path()
    user_paths = _load_user_paths(config_path)
    current_user = Path.home().name
    configured_path = user_paths.get(current_user)

    if not configured_path:
        _append_placeholder_profile(current_user, config_path)
        raise NotSetup(_NOT_SETUP_MESSAGE.format(user=current_user))

    if configured_path == PLACEHOLDER_PATH:
        raise NotSetup(_NOT_SETUP_MESSAGE.format(user=current_user))

    return Path(configured_path).expanduser()


def _self_test() -> None:
    """Run lightweight checks for helper functions.

    Raises
    ------
        AssertionError: If any helper behaves unexpectedly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "path_configuration.txt"
        config_path.write_text("foo /path/to/data\n", encoding="utf-8")

        user_paths = _load_user_paths(config_path)
        assert user_paths["foo"] == "/path/to/data"

        _append_placeholder_profile("bar", config_path)
        appended = config_path.read_text(encoding="utf-8").splitlines()
        assert appended[-1].split(maxsplit=1) == ["bar", PLACEHOLDER_PATH]


if __name__ == "__main__":
    print("Running helper self-test ... ", end="")
    _self_test()
    print("ok")

    try:
        resolved_path = get_base_path()
    except NotSetup as error:
        print(f"get_base_path() failed: {error}")
    else:
        print(f"Current base path: {resolved_path}")
