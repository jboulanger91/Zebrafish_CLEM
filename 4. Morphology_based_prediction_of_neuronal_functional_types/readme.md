# LDA-Based Functional Prediction

This module integrates morphology-based LDA predictions into the master metadata table.
It updates neurons originally labeled *not functionally imaged* with predicted functional
classes (motion_integrator, motion_onset, slow_motion_integrator), assigns new
functional IDs, and adds an `lda` column (`native` or `predicted`).

Use the script:

```
python3 integrate_lda_predictions_into_metadata.py \
    --metadata-csv all_reconstructed_neurons.csv \
    --lda-excel LDA_predictions.xlsx \
    --output-csv all_reconstructed_neurons_with_LDA_predictions.csv
```

This updated metadata can then be used to generate LDA-aware connectivity matrices and diagrams.
