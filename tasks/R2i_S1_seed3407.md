# R2i-S1 — Seed 3407 formal training

R2i preregistration governs H1–H3, the architectures, initialization and perturbation definitions. The sole seed is 3407. Run clean and augmented tracks, with baseline, gated and capacity control in each. Each cell runs five epochs and 785 updates. For each epoch, sample all 1,254 training records once, with 156 eight-record windows and one six-record window. Use native FP32 preprocessing and CE, valid-pixel gradient normalization, microbatch 1 and constant AdamW lr 6e-5. The augmentation stream uses the existing hash RNG with the actual training epoch.

Resume is at a persisted optimizer boundary. Restart verifies all saved state and audited permutation. Failed windows preserve logs and stop execution. Complete the two-process GPU recovery test before launching any formal group.

At Epoch 5 evaluate all four frozen conditions and compare gated with baseline and capacity control, and capacity control with baseline. Report ROI-outside new, corrected and net counts, with negative net change favoring the candidate. Use common 10,000 visual-group draws for paired endpoints, including both Clean mIoU differences and the 0.5 percentage-point noninferiority margin. Group intervals describe one training seed, and secondary comparisons are unadjusted. The historically used holdout is not blind. Keep official test pixels unopened.

Use `experiments.r2i_s1.driver` with `--arm`, `--track`, and external `--output`. For the isolated recovery gate use `--smoke` in two separate processes, adding `--resume` in process two. Run `experiments.r2i_s1.analyze` after all six complete. Keep checkpoints, prediction arrays and raw data local. Submit the S1 PR and stop for acceptance. No further seeds are in scope.
