#!/usr/bin/env python
"""Test runner script for pipeline regression tests.

This script provides a convenient way to run all regression tests and capture baselines.

Usage:
    # Capture all baselines (run once before refactoring)
    python run_tests.py --capture-baselines

    # Run all tests
    python run_tests.py --all

    # Run basic tests only
    python run_tests.py --basic

    # Run detailed tests only
    python run_tests.py --detailed

    # Run specific test classes
    python run_tests.py --class TestLoadCellsDF
    python run_tests.py --class TestConfusionMatrixValues

    # Run with verbose output
    python run_tests.py --all -v
"""

import argparse
import subprocess
import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=_TEST_DIR)
    return result.returncode == 0


def capture_baselines():
    """Capture all baseline values."""
    print("\n" + "=" * 60)
    print("CAPTURING BASELINE VALUES")
    print("=" * 60)

    success = True

    # Capture basic baseline
    print("\n[1/2] Capturing basic baseline values...")
    if not run_command(
        [sys.executable, str(_TEST_DIR / "capture_baseline.py")], "Capturing basic baseline values"
    ):
        print("WARNING: Failed to capture basic baseline")
        success = False

    # Capture detailed baseline
    print("\n[2/2] Capturing detailed baseline values...")
    if not run_command(
        [sys.executable, str(_TEST_DIR / "capture_detailed_baseline.py")],
        "Capturing detailed baseline values",
    ):
        print("WARNING: Failed to capture detailed baseline")
        success = False

    if success:
        print("\n" + "=" * 60)
        print("ALL BASELINES CAPTURED SUCCESSFULLY")
        print("=" * 60)
        print("\nBaseline files created:")
        print(f"  - {_TEST_DIR / 'baseline_values.json'}")
        print(f"  - {_TEST_DIR / 'detailed_baseline_values.json'}")
    else:
        print("\n" + "=" * 60)
        print("BASELINE CAPTURE COMPLETED WITH WARNINGS")
        print("=" * 60)

    return success


def run_tests(test_type="all", test_class=None, verbose=False):
    """Run pytest with specified options."""
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if test_type == "basic":
        cmd.append(str(_TEST_DIR / "test_pipeline_regression.py"))
    elif test_type == "detailed":
        cmd.append(str(_TEST_DIR / "test_detailed_regression.py"))
    elif test_type == "all":
        cmd.extend(
            [
                str(_TEST_DIR / "test_pipeline_regression.py"),
                str(_TEST_DIR / "test_detailed_regression.py"),
            ]
        )

    if test_class:
        cmd.extend(["-k", test_class])

    # Add useful pytest options
    cmd.extend(
        [
            "--tb=short",  # Shorter tracebacks
            "-x",  # Stop on first failure
        ]
    )

    return run_command(cmd, f"Running {test_type} tests")


def main():
    """Parse arguments and run pipeline regression tests."""
    parser = argparse.ArgumentParser(
        description="Test runner for pipeline regression tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--capture-baselines",
        action="store_true",
        help="Capture baseline values (run once before refactoring)",
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--basic", action="store_true", help="Run basic regression tests only")
    parser.add_argument(
        "--detailed", action="store_true", help="Run detailed regression tests only"
    )
    parser.add_argument(
        "--class",
        dest="test_class",
        help="Run specific test class (e.g., TestConfusionMatrixValues)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # If no args, show help
    if not any([args.capture_baselines, args.all, args.basic, args.detailed, args.test_class]):
        parser.print_help()
        return 1

    success = True

    if args.capture_baselines:
        success = capture_baselines() and success

    if args.all:
        success = run_tests("all", verbose=args.verbose) and success
    elif args.basic:
        success = run_tests("basic", verbose=args.verbose) and success
    elif args.detailed:
        success = run_tests("detailed", verbose=args.verbose) and success
    elif args.test_class:
        success = run_tests("all", test_class=args.test_class, verbose=args.verbose) and success

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
