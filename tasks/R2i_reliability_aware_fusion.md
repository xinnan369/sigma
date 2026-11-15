# R2i — Reliability-aware RGB-T feature fusion

## Frozen hypotheses

H1 predicts a reduction in real ROI-outside semantic errors under thermal degradation. H2 predicts an advantage over a parameter/operation-matched RGB-only scoring control and persistence under matched augmentation. H3 requires Clean overall and foreground mIoU noninferiority with a 0.5 percentage-point margin. The formal primary endpoint is candidate minus same-track natural SSM baseline ROI-outside error rate for center thermal occlusion, with a paired interval below zero. Verify this for the gated arm versus both controls, and disclose all seeds. H2 requires a gated advantage within matched training tracks. Report Clean noninferiority with paired seed/group inference. This pilot only assesses engineering feasibility.

## Design and frozen inputs

Use the pinned Sigma-Tiny architecture and official VMamba-T classification initialization. At all four CrossMamba input boundaries, learn a per-location bounded residual scale. A three-feature linear score observes RGB absolute magnitude, thermal energy and adjacent-channel cross-modal product, then tanh bounds the two complementary scales. The control substitutes RGB for Thermal only in the scoring features, with identical operations and trainable shape. Both original feature tensors still enter CrossMamba. A zero score is bitwise Identity on finite inputs. No masks, labels or IDs enter the gate. All other graph paths remain native.

Use seed 3407, R1g frozen data and native preprocessing, FP32, AdamW, learning rate 6e-5, microbatch 1 and eight-image accumulation with valid-pixel normalization. Run three optimizer updates per arm per track. The clean-training track uses native augmentation only. The augmented track additionally applies a 64×64 raw-zero equivalent Thermal patch after native crop/pad/normalization with probability 0.5, position uniformly drawn across the final training tensor. SHA256 of seed/epoch/record ID seeds an independent NumPy PCG64 generator, with a Bernoulli draw then y and x draws. Reuse the same record order and random streams within a track. Save locally and reload after window 1.

Evaluate all 157 internal holdout images and 54 visual groups, all arms and both tracks. Report clean, R2a centered occlusion, deterministic random 64×64 occlusion and centered Gaussian sigma 20 seed 3407. For random occlusion use the independent hash RNG with epoch 0, seeds and target ID, without the Bernoulli draw. Metric denominator is valid GT pixels outside the corresponding ROI. Compare each candidate with the baseline on the same corrupted inputs and keep paired confusion, net, new, corrected, class and distance summaries. Use 10,000 group bootstrap draws for the pilot, with no estimate of training-seed variation. This reused holdout is historical and is not a blind test. Keep the official test pixels unopened for future one-time independent confirmation.

## Staged formal experiment

The three-window pilot has a fixed engineering budget. After independent acceptance of the pilot, run only seed 3407 for the full formal budget. Every seed uses three arms and two training tracks, five epochs and 785 updates per cell with valid-pixel tail normalization. Preserve the registered architecture, split, evaluation protocol, augmentations and training hyperparameters.

1. Run seed 3407. Archive all six cells, Clean mIoU, perturbation metrics, parameters and compute, submit results in a PR, then pause for independent ChatGPT acceptance.
2. After acceptance and a decision to continue the study, add only seeds 3408 and 3409. Archive the cumulative results and pause for a second independent acceptance.
3. After the second acceptance, add seeds 3410 and 3411. Report all five seeds and paired seed/group intervals using the final epoch as the primary endpoint.

Early-stage results are observations within the evaluated seeds. A poor first-stage result can end the experiment with a documented negative outcome. Any design change requires a versioned experiment and explicit separation of exploratory and confirmatory results. A runner must never advance automatically to the next seed batch.


```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 python -u -m experiments.r2i.run \
  --upstream "$SIGMA_UPSTREAM" --init "$VMAMBA_INIT" --root "$MFNET_ROOT" --output "$R2I_RUN"
```
