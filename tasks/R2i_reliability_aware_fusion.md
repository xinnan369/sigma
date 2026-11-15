# R2i — Reliability-aware RGB-T feature fusion

## Frozen hypotheses

H1 predicts a reduction in real ROI-outside semantic errors under thermal degradation. H2 predicts an advantage over a parameter/operation-matched RGB-only scoring control and persistence under matched augmentation. H3 requires Clean overall and foreground mIoU noninferiority with a 0.5 percentage-point margin. The formal primary endpoint is candidate minus same-track natural SSM baseline ROI-outside error rate for center thermal occlusion, with a paired interval below zero. Verify this for the gated arm versus both controls, and disclose all seeds. H2 requires a gated advantage within matched training tracks. Report Clean noninferiority with paired seed/group inference. This pilot only assesses engineering feasibility.

## Design and frozen inputs

Use the pinned Sigma-Tiny architecture and official VMamba-T classification initialization. At all four CrossMamba input boundaries, learn a per-location bounded residual scale. A three-feature linear score observes RGB absolute magnitude, thermal energy and adjacent-channel cross-modal product, then tanh bounds the two complementary scales. The control substitutes RGB for Thermal only in the scoring features, with identical operations and trainable shape. Both original feature tensors still enter CrossMamba. A zero score is bitwise Identity on finite inputs. No masks, labels or IDs enter the gate. All other graph paths remain native.

Use seed 3407, R1g frozen data and native preprocessing, FP32, AdamW, learning rate 6e-5, microbatch 1 and eight-image accumulation with valid-pixel normalization. Run three optimizer updates per arm per track. The clean-training track uses native augmentation only. The augmented track additionally applies a 64×64 raw-zero equivalent Thermal patch after native crop/pad/normalization with probability 0.5, position uniformly drawn across the final training tensor. SHA256 of seed/epoch/record ID seeds an independent NumPy PCG64 generator, with a Bernoulli draw then y and x draws. Reuse the same record order and random streams within a track. Save locally and reload after window 1.

Evaluate all 157 internal holdout images and 54 visual groups, all arms and both tracks. Report clean, R2a centered occlusion, deterministic random 64×64 occlusion and centered Gaussian sigma 20 seed 3407. For random occlusion use the independent hash RNG with epoch 0, seeds and target ID, without the Bernoulli draw. Metric denominator is valid GT pixels outside the corresponding ROI. Compare each candidate with the baseline on the same corrupted inputs and keep paired confusion, net, new, corrected, class and distance summaries. Use 10,000 group bootstrap draws for the pilot, with no estimate of training-seed variation. This reused holdout is historical and is not a blind test. Keep the official test pixels unopened for future one-time independent confirmation.

## Follow-on budget

Only after independent acceptance run five seeds 3407–3411 with all three arms and two training tracks, five epochs per cell, 785 updates per cell. Maintain exact tail normalization. Use the final epoch as primary and use common paired seed/group resamples. Archive per-seed, per-group results, Clean intervals, design limits and secondary multiplicity risk. The pilot launcher has a fixed three-window cap.

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 python -u -m experiments.r2i.run \
  --upstream "$SIGMA_UPSTREAM" --init "$VMAMBA_INIT" --root "$MFNET_ROOT" --output "$R2I_RUN"
```
