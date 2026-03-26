#!/usr/bin/env python3
"""Interactive setup for data path configuration.

Configures ``config/path_configuration.txt`` so that all modules can find
the CLEM_paper_data directory via ``get_base_path()``.

Usage (terminal):
    python scripts/setup_data_paths.py

Usage (GUI):
    python scripts/setup_data_paths.py --gui

Usage (verify):
    python scripts/setup_data_paths.py --verify

Usage (download):
    python scripts/setup_data_paths.py --download
    python scripts/setup_data_paths.py --download --dest ~/Desktop/morph2func/morph2func_input

Reference:
    Reads/writes: config/path_configuration.txt
    Used by: src.util.get_base_path.get_base_path()
    Zenodo: https://zenodo.org/records/19235597
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "config" / "path_configuration.txt"

# Directories we expect inside a valid CLEM_paper_data root
EXPECTED_SUBDIRS = ["clem_zfish1", "em_zfish1", "paGFP"]

# Zenodo record for the structural data
ZENODO_RECORD_ID = "19235597"
ZENODO_FILES = [
    ("metadata.xlsx", 93_647),
    ("custom_nblast_matrix.csv", 1_304),
    ("baselines.zip", 148_741),
    ("paGFP.zip", 4_752_958),
    ("em_zfish1.zip", 50_413_892),
    ("clem_zfish1.zip", 60_150_911),
]
DEFAULT_DEST = Path.home() / "Desktop" / "morph2func" / "morph2func_input"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_current_user() -> str:
    """Return the current OS username."""
    return Path.home().name


def load_config() -> dict[str, str]:
    """Load path_configuration.txt into a username -> path dict."""
    entries: dict[str, str] = {}
    if not CONFIG_FILE.exists():
        return entries
    for line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) == 2:
            entries[parts[0]] = parts[1]
    return entries


def save_config(entries: dict[str, str]) -> None:
    """Write username -> path entries back to path_configuration.txt."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{user} {path}" for user, path in entries.items()]
    CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_path(path_str: str) -> tuple[bool, str]:
    """Check whether a path looks like a valid CLEM_paper_data directory.

    Returns (is_valid, message).
    """
    p = Path(path_str).expanduser()
    if not p.exists():
        return False, f"Path does not exist: {p}"
    if not p.is_dir():
        return False, f"Path is not a directory: {p}"

    found = []
    missing = []
    for subdir in EXPECTED_SUBDIRS:
        if (p / subdir).is_dir():
            found.append(subdir)
        else:
            missing.append(subdir)

    if not found:
        return False, (
            f"Directory exists but none of the expected subdirectories "
            f"were found: {', '.join(EXPECTED_SUBDIRS)}"
        )

    msg = f"Found: {', '.join(found)}"
    if missing:
        msg += f" (missing: {', '.join(missing)} -- may be OK for partial setups)"

    return True, msg


def get_user_path(user: str) -> str | None:
    """Return the configured path for a user, or None."""
    entries = load_config()
    return entries.get(user)


# ---------------------------------------------------------------------------
# Download from Zenodo
# ---------------------------------------------------------------------------

def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """Return a progress bar string like [========>           ] 42%."""
    if total <= 0:
        return ""
    frac = min(current / total, 1.0)
    filled = int(width * frac)
    bar = "=" * filled
    if filled < width:
        bar += ">"
        bar += " " * (width - filled - 1)
    else:
        bar = "=" * width
    pct = frac * 100
    mb_done = current / 1_048_576
    mb_total = total / 1_048_576
    return f"[{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB"


def _download_file(url: str, dest: Path, expected_size: int) -> None:
    """Download a single file with progress bar."""
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  {dest.name}: already exists, skipping")
        return

    label = dest.name
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "morph2func-setup/1.0"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            downloaded = 0
            while True:
                chunk = resp.read(262_144)  # 256 KB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                bar = _progress_bar(downloaded, expected_size)
                print(f"\r  {label:30s} {bar}", end="", flush=True)
        print()
        tmp.rename(dest)
    except Exception as exc:
        print(f"\r  {label:30s} FAILED: {exc}")
        if tmp.exists():
            tmp.unlink()
        raise


def _unzip_and_remove(zip_path: Path, dest_dir: Path) -> None:
    """Unzip a file into dest_dir and remove the zip."""
    print(f"  Extracting {zip_path.name} ... ", end="", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    print("done")


def run_download(dest: Path | None = None) -> None:
    """Download structural data from Zenodo and set up directory structure."""
    dest = dest or DEFAULT_DEST

    print("=" * 60)
    print("  Download Structural Data from Zenodo")
    print(f"  Record: https://zenodo.org/records/{ZENODO_RECORD_ID}")
    print(f"  Destination: {dest}")
    print("=" * 60)
    print()

    total_bytes = sum(size for _, size in ZENODO_FILES)
    print(f"Total download: {total_bytes / 1_048_576:.1f} MB ({len(ZENODO_FILES)} files)")
    print()

    if dest.exists():
        valid, msg = validate_path(str(dest))
        if valid:
            print(f"Data directory already exists and looks valid:")
            print(f"  {msg}")
            confirm = input("\nRe-download and overwrite? [y/N]: ").strip().lower()
            if confirm != "y":
                print("Skipping download.")
                _configure_after_download(dest)
                return

    dest.mkdir(parents=True, exist_ok=True)

    print("\nDownloading files:")
    download_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"
    for filename, expected_size in ZENODO_FILES:
        url = f"{download_url}/{filename}/content"
        _download_file(url, dest / filename, expected_size)

    print("\nExtracting archives:")
    for filename, _ in ZENODO_FILES:
        if filename.endswith(".zip"):
            _unzip_and_remove(dest / filename, dest)

    print()
    valid, msg = validate_path(str(dest))
    if valid:
        print(f"Validation: {msg}")
    else:
        print(f"WARNING: {msg}")

    _configure_after_download(dest)


def _configure_after_download(dest: Path) -> None:
    """Configure path_configuration.txt after download."""
    user = get_current_user()
    entries = load_config()
    entries[user] = str(dest)
    save_config(entries)
    print(f"\nConfigured: {user} -> {dest}")
    print(f"Config file: {CONFIG_FILE}")
    print("\nSetup complete! You can now run the pipeline:")
    print("  python functional_type_prediction/classifier_prediction/pipelines/pipeline_main.py")


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------

def run_terminal() -> None:
    """Interactive terminal setup for data path configuration."""
    print("=" * 60)
    print("  Data Path Setup -- morph2func")
    print(f"  Config file: {CONFIG_FILE}")
    print("=" * 60)
    print()

    user = get_current_user()
    entries = load_config()
    current_path = entries.get(user)

    # Show all configured users
    if entries:
        print("Currently configured users:")
        for u, p in entries.items():
            marker = " <-- you" if u == user else ""
            print(f"  {u}: {p}{marker}")
    else:
        print("No users configured yet.")
    print()

    # Show current user status
    if current_path:
        valid, msg = validate_path(current_path)
        status = "VALID" if valid else "INVALID"
        print(f"Your path ({user}): {current_path}")
        print(f"  Status: {status} -- {msg}")
    else:
        print(f"No path configured for user '{user}'.")
    print()

    # Menu
    print("Choose an option:")
    print("  [1] Set/update your data path")
    print("  [2] Verify your current path")
    print("  [3] Add another user")
    print("  [4] Show expected directory structure")
    print("  [q] Quit")
    print()

    choice = input("Enter choice [1/2/3/4/q]: ").strip().lower()

    if choice == "1":
        print()
        print("Enter the path to your CLEM_paper_data directory.")
        print("This should contain subdirectories like clem_zfish1/, em_zfish1/, etc.")
        print()
        new_path = input("Path: ").strip()
        if not new_path:
            print("No path entered. Aborting.")
            return

        # Expand ~ and env vars
        new_path = os.path.expanduser(os.path.expandvars(new_path))

        valid, msg = validate_path(new_path)
        print(f"\n  Validation: {msg}")

        if not valid:
            confirm = input("  Path looks invalid. Save anyway? [y/N]: ").strip().lower()
            if confirm != "y":
                print("  Aborted.")
                return

        entries[user] = new_path
        save_config(entries)
        print(f"\n  Saved: {user} -> {new_path}")
        print(f"  Config file: {CONFIG_FILE}")

    elif choice == "2":
        if not current_path:
            print("\nNo path configured. Run option 1 first.")
            return
        print(f"\nValidating: {current_path}")
        valid, msg = validate_path(current_path)
        print(f"  {msg}")
        if valid:
            # Count contents
            p = Path(current_path).expanduser()
            for subdir in EXPECTED_SUBDIRS:
                sd = p / subdir
                if sd.is_dir():
                    n_items = sum(1 for _ in sd.iterdir())
                    print(f"  {subdir}/: {n_items} items")
            print("\nPath is valid and accessible.")
        else:
            print("\nPath has issues. Run option 1 to reconfigure.")

    elif choice == "3":
        print()
        new_user = input("Username to add: ").strip()
        if not new_user:
            print("No username entered. Aborting.")
            return
        new_path = input(f"Path for {new_user}: ").strip()
        if not new_path:
            print("No path entered. Aborting.")
            return
        new_path = os.path.expanduser(os.path.expandvars(new_path))
        entries[new_user] = new_path
        save_config(entries)
        print(f"\n  Saved: {new_user} -> {new_path}")

    elif choice == "4":
        print()
        print("Expected directory structure under CLEM_paper_data:")
        print()
        print("  CLEM_paper_data/")
        print("    metadata.xlsx              (cell inventory: PA, CLEM, EM sheets)")
        print("    clem_zfish1/")
        print("      clem_zfish1_cell_*/      (per-neuron CLEM data directories)")

        print("    em_zfish1/")
        print("      em_fish1_*/              (per-neuron EM data directories)")
        print("    paGFP/")
        print("      20YYMMDD.N/             (per-session PA cell directories)")

    elif choice == "q":
        print("Bye.")

    else:
        print(f"Unknown option: {choice}")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_gui() -> None:
    """GUI setup for data path configuration."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print("ERROR: tkinter is not available. Use the terminal version.")
        sys.exit(1)

    root = tk.Tk()
    root.title("Data Path Setup -- morph2func")
    root.geometry("580x480")
    root.resizable(False, False)

    user = get_current_user()
    entries = load_config()

    # Header
    ttk.Label(
        root, text="Data Path Configuration",
        font=("Helvetica", 16, "bold"),
    ).pack(pady=(15, 5))

    ttk.Label(
        root,
        text=f"User: {user}\nConfig: {CONFIG_FILE}",
        justify="center",
    ).pack(pady=(0, 10))

    # Current status
    status_frame = ttk.LabelFrame(root, text="Current Configuration", padding=10)
    status_frame.pack(fill="x", padx=20, pady=5)

    current_path = entries.get(user, "")
    status_text = f"Path: {current_path}" if current_path else "Not configured"
    status_var = tk.StringVar(value=status_text)
    ttk.Label(status_frame, textvariable=status_var, wraplength=500).pack()

    # Path entry
    path_frame = ttk.LabelFrame(root, text="Set Data Path", padding=10)
    path_frame.pack(fill="x", padx=20, pady=5)

    path_var = tk.StringVar(value=current_path)
    entry_row = ttk.Frame(path_frame)
    entry_row.pack(fill="x")
    path_entry = ttk.Entry(entry_row, textvariable=path_var, width=50)
    path_entry.pack(side="left", fill="x", expand=True, pady=5)

    def on_browse():
        folder = filedialog.askdirectory(
            title="Select CLEM_paper_data directory",
            initialdir=path_var.get() or str(Path.home()),
        )
        if folder:
            path_var.set(folder)

    ttk.Button(entry_row, text="Browse", command=on_browse).pack(
        side="right", padx=(5, 0), pady=5,
    )

    # Log
    log_frame = ttk.LabelFrame(root, text="Log", padding=5)
    log_frame.pack(fill="both", expand=True, padx=20, pady=5)

    log_text = tk.Text(
        log_frame, height=8, width=60, state="disabled",
        font=("Menlo", 10),
    )
    log_text.pack(fill="both", expand=True)

    def log(msg: str) -> None:
        log_text.config(state="normal")
        log_text.insert("end", msg + "\n")
        log_text.see("end")
        log_text.config(state="disabled")
        root.update_idletasks()

    def refresh_status() -> None:
        reloaded = load_config()
        p = reloaded.get(user, "")
        if p:
            status_var.set(f"Path: {p}")
        else:
            status_var.set("Not configured")

    # Buttons
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=20, pady=5)

    def on_save():
        p = path_var.get().strip()
        if not p:
            messagebox.showwarning("No Path", "Enter a path first.")
            return
        p = os.path.expanduser(os.path.expandvars(p))
        valid, msg = validate_path(p)
        log(f"Validation: {msg}")
        if not valid and not messagebox.askyesno(
            "Invalid Path",
            f"Path looks invalid:\n{msg}\n\nSave anyway?",
        ):
            return
        reloaded = load_config()
        reloaded[user] = p
        save_config(reloaded)
        log(f"Saved: {user} -> {p}")
        refresh_status()
        messagebox.showinfo("Saved", f"Path saved for user '{user}'.")

    def on_validate():
        p = path_var.get().strip()
        if not p:
            log("No path entered.")
            return
        p = os.path.expanduser(os.path.expandvars(p))
        valid, msg = validate_path(p)
        log(f"Validation: {msg}")
        if valid:
            pp = Path(p)
            for subdir in EXPECTED_SUBDIRS:
                sd = pp / subdir
                if sd.is_dir():
                    n = sum(1 for _ in sd.iterdir())
                    log(f"  {subdir}/: {n} items")
            messagebox.showinfo("Valid", "Path is valid and accessible.")
        else:
            messagebox.showwarning("Invalid", f"Path issue:\n{msg}")

    def on_show_users():
        reloaded = load_config()
        if not reloaded:
            log("No users configured.")
            return
        log("Configured users:")
        for u, p in reloaded.items():
            marker = " <-- you" if u == user else ""
            log(f"  {u}: {p}{marker}")

    ttk.Button(btn_frame, text="Save", command=on_save).grid(
        row=0, column=0, padx=3, pady=3,
    )
    ttk.Button(btn_frame, text="Validate", command=on_validate).grid(
        row=0, column=1, padx=3, pady=3,
    )
    ttk.Button(btn_frame, text="Show Users", command=on_show_users).grid(
        row=0, column=2, padx=3, pady=3,
    )
    ttk.Button(btn_frame, text="Browse", command=on_browse).grid(
        row=0, column=3, padx=3, pady=3,
    )

    root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args and dispatch to terminal or GUI."""
    parser = argparse.ArgumentParser(
        description="Configure data paths for morph2func.",
        epilog=(
            "Examples:\n"
            "  python scripts/setup_data_paths.py          # interactive terminal\n"
            "  python scripts/setup_data_paths.py --gui    # GUI mode\n"
            "  python scripts/setup_data_paths.py --verify     # check current path\n"
            "  python scripts/setup_data_paths.py --download   # download data from Zenodo\n"
            "  python scripts/setup_data_paths.py --download --dest /path/to/data\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch the graphical setup window.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify the current user's configured path and exit.",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download structural data from Zenodo and configure paths.",
    )
    parser.add_argument(
        "--dest", type=Path, default=None,
        help=f"Destination directory for download (default: {DEFAULT_DEST}).",
    )
    args = parser.parse_args()

    if args.download:
        run_download(args.dest)
    elif args.gui:
        run_gui()
    elif args.verify:
        user = get_current_user()
        current = get_user_path(user)
        if not current:
            print(f"No path configured for user '{user}'.")
            print("Run: python scripts/setup_data_paths.py")
            sys.exit(1)
        print(f"User: {user}")
        print(f"Path: {current}")
        valid, msg = validate_path(current)
        print(f"Status: {msg}")
        sys.exit(0 if valid else 1)
    else:
        run_terminal()


if __name__ == "__main__":
    main()
