# R2g — Full-Scale Causal Attribution of Cross-Modal Selective Scan Paths

## Frozen cohort and inputs

Run each of the 157 R1g internal validation images on the 15 R2d Epoch 5 models
(SSM, CNN, Attention × seeds 3407–3411). Use the fixed 54 visual proxy groups and
the R2a thermal-only center occlusion at y=[208,272), x=[288,352), fill=0 before
normalization. Verify the R1g isolation certificate, 471 selected validation
files, upstream and repository sources, extension binary, and all checkpoint
ledgers. The official test ID list is used solely to verify split isolation.
No test images, training, or checkpoint writes are part of this task.

## Experiment

Use the accepted R2f CUDA basis-vector state recovery and explicit FP32 FMA,
with the R2f numerical limits unchanged. Trace native Clean and Corrupt first
and verify their prediction hashes against R2d. Run the eight binary masks with
bits h=1, C=2 and D*x=4 in Clean input context. Mask 0 is the bitwise Identity
control. Mask 7 is joint replacement using the naturally Corrupt readout at
each selected scan. Treat mask 7 and the natural Corrupt model pass as distinct
conditions. Under joint replay, a selected downstream scan receives its frozen
readout, so its natural response to an upstream intervention is bypassed.
These are fixed-model boundary interventions, and do not identify independent
natural causal contributions or a decomposition of all network computation.

The scopes are encoder, decoder, shared encoder+decoder, each CrossMamba and
ConcatMamba scale in SSM, their joint fusion scope, and all paths. CNN and
Attention retain SSM paths in their shared encoder and decoder. All eight
readouts, native endpoints and CUDA chunk ends are audited per scan per image.
A failed sample blocks the cohort and is recorded permanently in its journal.

## Inference and outputs

Compute ratios using pooled pixel counts within each sampled visual group.
Average over training seeds with equal seed weight. Resample the five paired
seeds and 54 paired visual groups with common bootstrap multiplicities
(10,000 draws; RNG seed 3407). Keep all images in each sampled visual group.
Report percentile intervals, per-seed directions and seed-level t intervals.
The 10,000 draws are repeated resamples of five trainings. Use finite unions for
mIoU and exclude undefined denominators. Report two- and three-way interactions
by inclusion–exclusion on metric contrasts relative to Clean.

Prespecified comparisons are h−C and h−D*x within matched scopes, and
encoder−decoder and fusion−shared at the same mask. Report the four
ROI-outside rates and distance bands. Individual scales and interactions are
exploratory, with unadjusted intervals and multiple-comparison risk. Pooled
metrics apply to visual proxy groups without verified temporal independence.

## Reproduction and recovery

From the research repository in the validated WSL2 CUDA environment:

```bash
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
opts=(--upstream "$SIGMA_UPSTREAM" --init "$VMAMBA_INIT" --root "$MFNET_ROOT"
      --checkpoint-dir "$R2D_CHECKPOINTS" --evidence "$R2G_RUN")
python -m experiments.r2g.run pilot "${opts[@]}"
python -m experiments.r2g.run full "${opts[@]}"
python -m experiments.r2g.analysis --evidence "$R2G_RUN" --output "$R2G_RUN/paired-statistics.json"
```

The full phase requires a separate R2g pilot certificate over 12 frozen images
on the three seed-3407 models. Preserve the local run directory on restart.
The per-cohort SQLite journals use transactions and per-record SHA256.
The full launcher rechecks the frozen fingerprint and all completed rows on
reopen, rejects duplicates and stops on saved failures. Completed cohorts
export compressed JSONL summaries and layouts; keep SQLite files and model
assets local.

