# R2k-Q1v2-P1 data readiness

**Admission status: BLOCKED_NO_ELIGIBLE_INDEPENDENT_DATA.** No source has verified rights, original-byte and visual independence, synchronized and compatible inputs, and the complete group roster required by P0. Source-by-source evidence and open requirements are in the [data matrix](../docs/R2k_Q1v2_DATA_SOURCE_MATRIX.md). New image acquisition, quality evaluation, GPU execution and scientific gate judgement were **NOT RUN**.

## Accepted baseline and freeze audit

P0 PR #23 was merged after verifying its accepted head `9978711f28fcf7b333602cb283a169cfb48a1b8e`, passing Push and PR CI, and a clean merge state. The main merge commit is `154d164658385295727cba23e1fbf227beec27fd`.

Local [freeze audit](evidence/r2k-q1v2-p1/freeze-audit.json) **PASS**. The P0 parameter digest is `fbc78757c2711558623d4c11f7efdc3693c70bb1711e7b5e105de63b161b6e08`. The audit checked the accepted configuration and decision protocol, ten Q1/P0 source hashes, historical MFNet manifest bytes and six frozen threshold bins. Local parameters were read for verification only. No model code was invoked.

## Admission handoff

For a future authorized source, provide a sealed `rgbt-p1-v1` manifest. Record the source URL, rights for research/local derivatives/local audit, the applicable terms and a reviewer receipt. Include a duplicate-review receipt and method covering MFNet and every available historical RGB-T source. Record reviewed near-duplicate, neighbor, derivative and repeat edges and any known historical matches. A separate receipt documents controlled physical defocus with instrument-recorded focus changes.

Each intact pair needs a stable ID, globally canonical group/site/sequence IDs, a timezone-aware timestamp, sensor provenance, day/night from the capture log, an alignment receipt and frozen-blur eligibility. RGB and Thermal entries include relative paths and original-byte SHA256. Same-site or same-sequence frames stay together. Reviewed links must not cross groups. Keep the exclusion ledger and rerun the checks against the eligible subset after any exclusions. The tool checks exact digests against archived RGB/Modal MFNet entries without opening MFNet pixels. Visual duplicates and transformed copies require an authorized independent review and a canonical link roster. The byte check cannot detect them by itself.

Run the metadata and freeze checks after authorized acquisition:

```sh
python -m experiments.r2k_q1v2_p1.admission --frozen-parameters /local/frozen-parameters.json --metadata /local/manifest.json --data-root /local/authorized-originals
```

The tool's highest status is `READY_FOR_INDEPENDENT_REVIEW`. Receipts are inputs to independent review. Their presence does not grant rights or establish truthful provenance. Once admitted, freeze the manifest, registration and intensity representation, the P0 grid and the synthetic blur recipe before any scoring. The grid has stride 8 and offset 4. The referenced Q1 recipe uses a 96 × 96 ROI, 12-pixel margin, and Gaussian kernel/sigma pairs (7, 1.6) and (17, 4.0). Use the frozen per-image center generation and pair both tiers with the same healthy frame and unchanged RGB. Maintain untouched originals and derived images within the same group. Keep any instrument-labelled physical defocus in a separate secondary analysis.

## Synthetic statistical preflight

The synthetic suite passed 5 tests on WSL CPU. The full CPU suite passed 124 tests with 3 skips (37.69 seconds). Tests check legal attestations, SHA mismatch and traversal, historical matches, sequence and reviewed-link closure, support, tied-bin AP and paired group bootstrap, balanced Brier, operating-point recall, healthy FPR including day/night/low texture, zero observability with full retained coverage, and decision branches. The synthetic bootstrap used the prescribed 10,000 shared group draws and PCG64 seed 3407. No candidate datasets were loaded, scored, or used to tune thresholds. A result from this fixture is an engineering assertion only.

## Acquisition decision

Request the MS² site/sequence metadata and calibration release information first. Advance that source only if an authorized inventory supports at least 40 groups, 15 day and 15 night groups, 15 positive-blur groups and a reviewable historical exclusion ledger. A companion controlled collection can supply the secondary physical-defocus protocol, with its own permissions, paired sensor settings and provenance. If a public source lacks the roster or cannot support the counts, arrange a new authorized synchronized capture campaign at independent sites, sampling above the required floors to allow for exclusions. Document permissions, calibration, time, actual lighting, eligibility and optical-focus settings before sealing the roster. Preserve the two historical validation sets in their original roles.
