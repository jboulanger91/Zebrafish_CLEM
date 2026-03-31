import sys
from pathlib import Path
from dotenv import dotenv_values

env = dotenv_values(".env")
ROOT_PATH = env["ROOT_PATH"]

def add_import_path(path_str):
    EXTRA_ROOT = path_str.resolve()
    # Add it to sys.path if not already there
    if str(EXTRA_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTRA_ROOT))

def fix_imports_root(root_path=None):
    if root_path is None:
        root_path = ROOT_PATH
    root_path = Path(root_path).resolve()
    print(f"DEBUG | {root_path}")
    for path in root_path.glob("*"):
        if path.is_dir() and ".idea" not in str(path) and ".git" not in str(path):
            add_import_path(path)
