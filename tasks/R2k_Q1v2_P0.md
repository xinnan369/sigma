# R2k-Q1v2-P0 — CPU-only multiscale candidate

Freeze one candidate before reading image pixels. Reuse the byte-pinned Q1 Fit, Calibration, sample stream and optimizer settings. The only image reader is dispatched with one of those two partitions. Old Holdout, blur, semantic labels, Sigma, GPU and official test are outside the run scope.

Use box windows 3, 9 and 21 pixels, a 60×80 grid aligned to original pixel centers at 4+8k, and identical grayscale input shapes. The candidate's six descriptors are two log frequency/gradient ratios, a peak-to-local gradient width proxy, coarse gradient, saturation and exact zero fill. The coarse-gradient indicator is reported separately without excluding points. Output bad probability as uint8/255, with quality equal to one minus that probability.

The six-feature Q1 refit uses the original logistic fitting budget. Thermal and RGB share the identical multiscale operation sequence. The ablation scans the same features, then replaces both ratio channels by local mean and variance. This preserves nonconstant inputs and the same trainable dimensionality. Each learned arm uses seven fitted parameters and up to two Platt parameters, the identical selected grid points and the same optimizer cap; realized iterations may differ. The shuffle arm permutes targets only, and a constant arm uses the calibration prevalence. Compare fixed seen-family diagnostics, provenance and CPU costs. Independent external confirmation needs a separate approval under the frozen evaluation protocol.

Run in Linux with CUDA hidden and numerical libraries set to one thread:

```bash
python -m experiments.r2k_q1v2.run --root "$MFNET_ROOT" --local "$P0_LOCAL" --evidence reports/evidence/r2k-q1v2-p0
python -m experiments.r2k_q1v2.audit --repo . --local "$P0_LOCAL" --evidence reports/evidence/r2k-q1v2-p0 --output reports/evidence/r2k-q1v2-p0/cpu-audit.json
```

Keep parameters, raw image data and prediction arrays local. Publish hashes, model engineering diagnostics and non-image aggregates. Open the PR and stop.
