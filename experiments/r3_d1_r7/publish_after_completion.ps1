param(
    [Parameter(Mandatory=$true)][string]$Checkout,
    [Parameter(Mandatory=$true)][string]$Commit,
    [string]$StateCommit,
    [ValidateRange(3408, 3411)][int]$LastSeed = 3409,
    [switch]$ResumeBatch,
    [switch]$PublishOnly
)
$ErrorActionPreference = 'Stop'
if (-not $StateCommit) { $StateCommit = $Commit }
$branch = 'codex/R3-D1B-R7-remaining-four-seeds'
$base = 'codex/R3-D1B-R3-independent-authorization-audit'
$repo = 'xinnan369/sigma'
$out = 'D:/work/sigma-r3-d1-r7-formal'
$proof = 'reports/evidence/r3-d1b-r7/completed'
$report = 'reports/R3_D1B_R7_FIVE_SEED_EXPERIMENTS.md'
$head = git -C $Checkout rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve checkout HEAD' }
$alreadyCommitted = $head -ne $Commit
if ($alreadyCommitted) {
    $parent = git -C $Checkout rev-parse 'HEAD^'
    if (-not $PublishOnly -or $LASTEXITCODE -ne 0 -or $parent -ne $Commit) {
        throw 'Runtime checkout is not the verified training commit'
    }
}

if (-not $PublishOnly) {
    $extra = @()
    if ($ResumeBatch) { $extra += '--resume-batch' }
    wsl -d Ubuntu-22.04 --cd /mnt/d/work/sigma-r7-runtime -- env `
        CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 `
        PYTHONDONTWRITEBYTECODE=1 /home/xinnan/.venvs/sigma-r1-cu121/bin/python -u `
        -m experiments.r3_d1_r7.run_remaining --output /mnt/d/work/sigma-r3-d1-r7-formal `
        --data /mnt/d/BaiduNetdiskDownload/LLVIP/LLVIP --upstream /home/xinnan/sigma-r1-upstream `
        --commit $StateCommit --last-seed $LastSeed @extra
    if ($LASTEXITCODE -ne 0) { throw "Training or audit exited $LASTEXITCODE. Evidence preserved at $out" }
    if ($LastSeed -lt 3411) {
        $stage = Get-Content -LiteralPath "$out/stage-through-$LastSeed-summary.json" -Raw | ConvertFrom-Json
        if ($stage.status -ne 'STAGE_COMPLETE' -or $stage.last_scheduled_seed -ne $LastSeed) {
            throw 'The requested stage did not complete its audit'
        }
        Write-Output "Completed the audited stage through seed $LastSeed"
        return
    }
}

$batch = Get-Content -LiteralPath "$out/batch-summary.json" -Raw | ConvertFrom-Json
$matrix = Get-Content -LiteralPath "$Checkout/$proof/five-seed-audit.json" -Raw | ConvertFrom-Json
$manifest = Get-Content -LiteralPath "$Checkout/$proof/verification.json" -Raw | ConvertFrom-Json
if ($batch.status -ne 'COMPLETE' -or $batch.commit -ne $StateCommit -or $batch.new_updates -ne 397120 `
    -or $matrix.status -ne 'PASS' -or $matrix.training_commit -ne $StateCommit -or $matrix.runs.Count -ne 20 `
    -or $matrix.all_updates -ne 496400 -or $matrix.official_test_pixel_reads -ne 0 -or $manifest.status -ne 'PASS') {
    throw 'The complete five-seed proof is required to publish'
}
if (@($manifest.files_sha256.PSObject.Properties).Count -ne 5) { throw 'Wrong evidence scope' }
foreach ($entry in $manifest.files_sha256.PSObject.Properties) {
    $path = Join-Path "$Checkout/$proof" $entry.Name
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
        throw "Evidence digest differs for $($entry.Name)"
    }
}
if ($alreadyCommitted) {
    $staged = @(git -C $Checkout diff-tree --no-commit-id --name-only -r HEAD)
} else {
    git -C $Checkout add -- $report $proof
    if ($LASTEXITCODE -ne 0) { throw 'Could not stage the report' }
    git -C $Checkout diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Final evidence contains whitespace errors' }
    $staged = @(git -C $Checkout diff --cached --name-only)
}
if ($LASTEXITCODE -ne 0 -or $staged.Count -ne 7 -or @($staged | Where-Object {
    $_ -ne $report -and -not $_.StartsWith("$proof/")
}).Count -gt 0) { throw 'Unexpected tree changes' }
if (-not $alreadyCommitted) {
    git -C $Checkout commit -m 'Archive the complete R7 five-seed experiment'
    if ($LASTEXITCODE -ne 0) { throw 'Could not commit the final report' }
}

$published = git -C $Checkout rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the final commit' }
$pushSucceeded = $false
foreach ($attempt in 1..3) {
    git -C $Checkout push origin "HEAD:refs/heads/$branch"
    if ($LASTEXITCODE -eq 0) { $pushSucceeded = $true; break }
    if ($attempt -lt 3) { Start-Sleep -Seconds 30 }
}
if (-not $pushSucceeded) { throw "Could not push final commit $published" }

$body = @"
# R7 five-seed comparison complete

The original seed 3407 results were retained. The 16 approved runs for seeds 3408–3411 completed in order, with 397,120 new AdamW updates and 496,400 total across the 20-run matrix. All evaluation used the frozen internal Validation split. Official Test pixel reads were zero.

The report includes per-run AP50/AP50:95, five-seed means and sample standard deviations, paired Mamba-minus-CNN differences, final checkpoint hashes, provenance, and failure/recovery records.

[Full report](https://github.com/$repo/blob/$published/$report) · [Five-seed audit](https://github.com/$repo/blob/$published/$proof/five-seed-audit.json) · [Per-seed evidence and SHA256](https://github.com/$repo/tree/$published/$proof)

The licensed data, weights and full event logs remain in the local run directory. This PR is ready for independent review.
"@
$bodyPath = Join-Path ([System.IO.Path]::GetTempPath()) ('sigma-r7-pr-' + [guid]::NewGuid().ToString() + '.md')
Set-Content -LiteralPath $bodyPath -Value $body -Encoding utf8NoBOM
$pr = $null
foreach ($attempt in 1..3) {
    $pr = gh pr create -R $repo --base $base --head $branch --title '[R3-D1B-R7] Complete the frozen five-seed RGB-T detection experiments' --body-file $bodyPath
    if ($LASTEXITCODE -eq 0 -and $pr) { break }
    $existing = gh pr view $branch -R $repo --json url,headRefOid,state
    if ($LASTEXITCODE -eq 0) {
        $snapshot = $existing | ConvertFrom-Json
        if ($snapshot.state -eq 'OPEN' -and $snapshot.headRefOid -eq $published) { $pr = $snapshot.url; break }
    }
    if ($attempt -lt 3) { Start-Sleep -Seconds 30 }
}
if (-not $pr) { throw "Final evidence pushed at $published. Creating the PR failed" }
[pscustomobject]@{status='OPEN'; pr=$pr; head=$published; trainingCommit=$StateCommit} |
    ConvertTo-Json -Compress | Set-Content -LiteralPath "$out/publication.json" -Encoding utf8NoBOM
Write-Output $pr
