# R2h — Corrupt-Context Counterfactual State Repair

Use the frozen R2g assets and R2f CUDA recurrence/FP32 FMA readout, with the same limits. RGB is the unmodified RGB tensor in the natural paired Corrupt condition. Thermal has only the frozen central 64×64 occlusion. Both inputs to every repair replay are exactly those of the natural Corrupt forward. Unselected scans execute their native code on the current downstream activations, which may change as a result of upstream intervention.

Bits 1, 2, 4 represent h, C, D*x, respectively. A set bit selects the original natural Corrupt component. Mask 7 must reproduce native Corrupt logits bitwise. Mask 6 restores only h, 5 only C, and 3 only D*x. Masks 4, 2, 1 and 0 restore two or three components. Components at selected scans are frozen from the two natural traces. The mismatch control uses Clean encoder h from the next ID in the sorted 157-image cohort, cyclic, with the target's Corrupt C and D*x. Pilot and full use the same donor map.

The primary endpoint is ROI-outside net-error change for encoder mask 6 in each arm. Negative indicates improvement. Clean establishes which original errors were introduced by occlusion. Recovery fraction is the count of these errors corrected by repair divided by their original count, pooled within each seed and bootstrap draw. Harm counts all newly incorrect pixels relative to natural Corrupt, including pixels already wrong in Clean. Net error equals harm minus all errors corrected relative to Corrupt. Report h vs C and D*x, encoder vs decoder and same-image vs mismatch with paired inference. Report ROI-outside counts and rates, distance bands, class-specific metrics, full and foreground mIoU. A drop in disagreement alone does not satisfy the accuracy criterion. Require improved net errors with an interval below zero, and report seed directions and whether induced errors recovered exceed repair-induced errors. The design tests fixed-model boundary interventions. Secondary and factorial estimates have unadjusted intervals and multiplicity risk.

Run the 12-image three-arm pilot before launching full. Both phases use SQLite journal transactions with immutable per-image hashes and preserved failures. Keep dataset, checkpoints, state tensors and journals local. Publish summaries, hashes, logs and certificates. Use 10,000 shared crossed resamples of five seeds and 54 frozen visual proxy groups, with seed-level t intervals and individual seed points.

```bash
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1
opts=(--upstream "$SIGMA_UPSTREAM" --init "$VMAMBA_INIT" --root "$MFNET_ROOT"
      --checkpoint-dir "$R2D_CHECKPOINTS" --evidence "$R2H_RUN")
python -u -m experiments.r2h.run pilot "${opts[@]}"
python -u -m experiments.r2h.run full "${opts[@]}"
python -u -m experiments.r2h.analysis --evidence "$R2H_RUN"
```
