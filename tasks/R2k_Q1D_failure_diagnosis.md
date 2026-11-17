# R2k-Q1D — Read-only failure diagnosis

Verify and merge accepted Q1 PR #21. Use CPU only, the accepted Q1 artefacts and the original train-side Calibration and Holdout images. The latter is exclusively post-hoc evidence. Verify original fingerprints, reconstruct scores from the frozen quality head, and record aligned healthy-versus-transformed descriptor changes, logit contributions, Platt and quantization effects, locked-threshold recall, and ROI pixel differences. Keep heads, raw image arrays and predictions local. Preserve all accepted outputs and keep the scientific Q1 FAIL unchanged.

Propose at most two Q1-v2 options, including matched controls, separate Fit/Calibration design constraints and entirely new independent data for a future confirmatory experiment. No new fitting, choice based on Holdout, score reversal, official test access, segmentation or GPU work. Open the Q1D PR and pause.
