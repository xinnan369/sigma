# R2k-Q1v2-P3R — EXPLORATORY_LLVIP_PILOT

**CPU Pilot completed: 64/64 paired test images, 128/128 blur views.** P0's frozen multiscale Thermal head did not respond in the expected direction on this sample and input adapter. This pilot does not resolve LLVIP's original admission constraints or pass the Q1-v2 scientific gate. Do not expand the same run or change the frozen parameters based on these results.

## Audit reproduction and review

The new CPU scanner reproduced all fields of **46,465** file records, keyed by relative path, including raw-file SHA256, decoded pixel SHA256, image format and sample precision, ZIP-entry CRC32, and XML checks. Its pairing, duplicate, perceptual candidate, and annotation summaries match the P3 archive. The new scanner writes JSONL in case-sensitive POSIX lexical path order instead of the old Windows path order, and explicitly outputs ambiguous pairing IDs (zero). Counts: 12,025 train and 3,463 test pairs, no missing sides or decode errors, three exact same-split duplicate groups, no exact cross-split duplicates, five malformed box geometries, seven RGB cross-split perceptual candidates. Historical comparison remains the original P3 evidence; the rerun did not open any MFNet or archived Holdout images.

Local review material is under `%LOCALAPPDATA%/sigma/llvip-p3r-review/`. Open `review.html` there to see RGB and Thermal counterparts side by side. The page links to the original local files; it does not copy or upload images. All seven decisions remain **UNKNOWN**. Screenshot, semantic judgment, and adjudication were not performed.

## Frozen pilot design

Before any scoring, `freeze-roster` selected the first 64 complete official LLVIP test pairs ranked by lexical SHA256 of `3407/llvip-test/<id>`. The [roster](evidence/llvip-pilot/roster.json) stores both source paths and hashes; each source hash was rechecked on read. Both modalities used the same OpenCV 4.9.0 pipeline: native 1280×1024 uint8 BGR decode without EXIF rotation, crop `[x0,y0,x1,y1)=[0,32,1280,992)`, INTER_AREA resize to 640×480, then BGR-to-gray. The [protocol](../configs/R2k_Q1v2_LLVIP_PILOT.json) binds the adapter, feature and scoring sources, P0 and Q1 protocols, and locally retained frozen model objects. The [per-image ledger](evidence/llvip-pilot/per-image.jsonl) links IDs to the two adapted pixel hashes, ROI coordinates and per-view counts. No transformed frames or model parameters are published.

Each adapted Thermal had one untouched reference and two 96×96 Gaussian-blur ROIs sampled by the unchanged Q1 seed/ID/tier logic, with kernels 7/17 and σ 1.6/4.0. Shape and support were checked; pixels outside each ROI were unchanged. “Healthy” here means no synthetic perturbation applied; sensor health was not independently certified. All 4,800 grid points per view were retained. The archived Q1 Thermal, P0 Thermal and P0 RGB-only heads retained their own frozen weights, calibration and threshold bins (7, 6, 7). RGB-only was evaluated again in each condition and its output bins were identical to healthy for every image, view and pixel.

## Results

| Head | Healthy FPR | Blur ROI recall | AUPRC | Brier | Balanced Brier | Mean ROI Δ bad probability |
|---|---:|---:|---:|---:|---:|---:|
| Q1 six-descriptor Thermal | 27.01% | 15.13% | 0.02505 | 0.10828 | 0.50908 | −6.37 pp |
| Q1v2 multiscale Thermal | 13.73% | 5.78% | 0.02818 | 0.09691 | 0.51548 | −3.52 pp |
| P0 RGB-only | 0.34% | 0.47% | 0.03092 | 0.02916 | 0.47837 | 0.00 pp |

AUPRC is grouped-by-score average precision on both blurred tiers; prevalence and chance reference are 0.03000. Brier uses all blurred-view grid points; balanced Brier gives equal weight to ROI and non-ROI errors. ROI changes compare the score at the same position in the untouched reference, using the corresponding tier's ROI. Both Thermal heads had median change zero. The multiscale mean decreased at both tiers (−4.18 and −2.86 pp); only 5.91% of ROI scores increased and 77.94% tied. Its recall decreased from 6.80% to 4.75% between tiers. Both Thermal heads exceeded the original 7% healthy FPR reference; the multiscale head reduced false alarms relative to Q1 but still flagged nearly twice the reference rate. RGB-only's AP near prevalence is consistent with its unchanged inputs, not sensing Thermal blur. Aggregate AP, Brier, recall, healthy FPR, count totals and RGB invariance were independently recomputed from the archived per-image histograms or output checks.

## Execution and limits

CPU runs: native Windows scanner with Python 3.13.2, Pillow 12.2.0, NumPy 2.3.5, SciPy 1.17.1, four workers; pilot on WSL2 Python 3.11.11, NumPy 1.26.4, SciPy 1.13.0, OpenCV 4.9.0, one OpenCV/BLAS/OMP thread, CUDA devices disabled. Evaluation took **5.45 s** and **153.1 MiB peak RSS**. Windows synthetic tests: **3 passed**. WSL existing CPU suite: **129 passed, 3 skipped** (upstream assets/CUDA); the three new tests were run on Windows rather than the WSL environment's older NumPy. The pilot uses the pinned historical CPU feature environment. No training, refitting, recalibration, threshold adjustment, historical asset evaluation or GPU use.

Commands (run from the repository; choose new local output paths where needed):

```powershell
python -m experiments.r2k_q1v2_llvip.scan --root D:/BaiduNetdiskDownload/LLVIP/LLVIP --zip D:/BaiduNetdiskDownload/LLVIP.zip --output "$env:TEMP/sigma-llvip-p3r-rescan" --review-local "$env:LOCALAPPDATA/sigma/llvip-p3r-review"
python -m pytest -q tests/test_r2k_q1v2_llvip.py
```

```sh
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m experiments.r2k_q1v2_llvip.pilot freeze-roster --manifest reports/evidence/llvip-precheck/file_manifest.json --output "$PILOT_OUTPUT"
python -m experiments.r2k_q1v2_llvip.pilot evaluate --manifest reports/evidence/llvip-precheck/file_manifest.json --output "$PILOT_OUTPUT" --root /mnt/d/BaiduNetdiskDownload/LLVIP/LLVIP --q1 "$HOME/sigma-r2k-q1/frozen-fit.json" --p0 "$HOME/sigma-r2k-q1v2-p0/frozen-parameters.json"
```

There are no independent-group confidence intervals, pixel-level significance claims or confirmatory PASS. The sample could include related views. The LLVIP adapter changes the image domain and may interact with the quality features; this run cannot attribute the response to the adapter, sensor, scene content or the head independently. The original P3 rights and scene-metadata limitations remain open; no image derivatives were retained or distributed. A larger same-protocol run is not recommended on this evidence. Any proposed further work needs separate review of rights, scene grouping and its scientific purpose.

Evidence: [reproduced audit](evidence/llvip-pilot/audit-reproduction.json), [full metrics and resources](evidence/llvip-pilot/results.json).
