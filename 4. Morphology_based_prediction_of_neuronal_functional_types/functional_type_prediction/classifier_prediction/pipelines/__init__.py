"""Production Pipeline Scripts.

This package contains production-ready pipeline scripts for running
cell type prediction workflows.

Scripts
-------
pipeline_main : Main production pipeline
  - Integrates all modular components
  - Validates against reference files
  - Includes Gregor's Feb 2025 EM data
  - Uses AdaBoost classifier
  - Automated execution (non-interactive matplotlib backend)

Usage
-----
Run the main pipeline from the command line:
    python pipelines/pipeline_main.py

Or import and use programmatically:
    from classifier_prediction.pipelines import pipeline_main
"""

__all__ = [
    "pipeline_main",
]
