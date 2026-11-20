"""Execute and audit the approved R7 batch; never schedule seed 3407."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

REPO = Path(__file__).resolve().parents[2]
ARMS = 'ABCD'
SEEDS = range(3408, 3412)
RUNS = [(arm, seed) for seed in SEEDS for arm in ARMS]
UPDATES_PER_EPOCH = 2482
EPOCHS = 10
RECEIPT = REPO / 'reports/evidence/r3-d1b-r2r/authorization-candidate.json'
SPLIT = REPO / 'reports/evidence/r3-d0/split.json'
EVIDENCE = REPO / 'reports/evidence/r3-d1b-r7'
R6_EVIDENCE = REPO / 'reports/evidence/r3-d1b-r6'
CST = timezone(timedelta(hours=8))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_bytes())


def bytes_hash(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return bytes_hash((json.dumps(value, sort_keys=True, separators=(',', ':'),
                                  allow_nan=False) + '\n').encode())


def write(path, data):
    """Publish atomically; recoverable saves never replace raw training artifacts."""
    temp = path.with_name(path.name + '.pending')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def local_time(timestamp):
    return datetime.fromtimestamp(timestamp, CST).isoformat(timespec='seconds')


def checkpoint(run, signed, env, complete=False):
    ledger = read(run / 'checkpoint-ledger.json')
    require(read(run / 'initialization.json')['signature_sha256'] ==
            ledger['signature_sha256'] == signed, 'Checkpoint signature differs')
    require(read(run / 'runtime.json') == env, 'Checkpoint runtime differs')
    for kind in ('last', 'best'):
        wanted = ledger[kind]
        if wanted is None:
            require(kind == 'best' and not complete, 'Missing terminal checkpoint')
            continue
        require(len(wanted) == 64 and all(c in '0123456789abcdef' for c in wanted),
                'Invalid checkpoint object name')
        obj = run / 'checkpoint-objects' / (wanted + '.pt')
        require(hash_file(obj) == wanted, 'Checkpoint object SHA256 differs')
        alias = run / (kind + '.pt')
        if alias.exists():
            allowed = {wanted} if complete else {wanted, ledger['previous'][kind]}
            require(hash_file(alias) in allowed, 'Checkpoint alias differs')
        else:
            require(not complete, 'Missing terminal alias')
    data = (run / 'events.jsonl').read_bytes()
    offset = ledger['event_log']['offset']
    require(0 < offset <= len(data) and bytes_hash(data[:offset]) == ledger['event_log']['sha256'],
            'Committed event prefix differs')
    require(not complete or offset == len(data), 'Trailing terminal events')
    return ledger, data


def audit_run(run, arm, seed, approved, receipt_record, fit, attempts):
    signed = approved['runtime_signatures'][f'{arm}/{seed}']['runtime_signature_sha256']
    ledger, raw = checkpoint(run, signed, approved['environment'], complete=True)
    history = read(run / 'epochs.json')
    require(ledger['update'] == EPOCHS * UPDATES_PER_EPOCH and len(history) == EPOCHS,
            'Incomplete terminal update/epoch count')
    best = max(history, key=lambda row: (row['ap50_95'], row['ap50'], -row['epoch']))
    events = iter(raw.splitlines())
    fit_reads, val_reads, peak = 0, 0, 0
    timing = {'update_pipeline': 0., 'validation': 0.}
    orders = []
    for epoch, row in enumerate(history):
        batches = []
        loss = dict.fromkeys(('cls', 'box', 'center'), 0.)
        for index in range(UPDATES_PER_EPOCH):
            event = json.loads(next(events))
            update = epoch * UPDATES_PER_EPOCH + index + 1
            ids = event['ids']
            count = 2 if index == UPDATES_PER_EPOCH - 1 else 4
            require((event['kind'], event['epoch'], event['batch_cursor'], event['update'],
                     len(ids), event['lr']) == ('update', epoch, index + 1, update, count, .0003),
                    'Update or batch order differs')
            flips = [hashlib.sha256(f'{seed}/{epoch}/{i}/flip'.encode()).digest()[0] < 128 for i in ids]
            require(event['flips'] == flips and math.isfinite(event['gradient_l2']) and
                    event['gradient_l2'] >= 0 and all(math.isfinite(v) for v in event['loss'].values()),
                    'Augmentation or update diagnostics differ')
            for name in loss:
                loss[name] += event['loss'][name] * count
            fit_reads += count
            batches.append(ids)
            peak = max(peak, event['peak_gpu_allocated_bytes'])
            timing['update_pipeline'] += event['seconds']
        flat = [i for batch in batches for i in batch]
        order = digest(batches)
        orders.append(order)
        require(len(flat) == len(set(flat)) == len(fit) and set(flat) == fit and
                order == receipt_record['fit_orders'][epoch], 'Signed Fit coverage differs')
        require(json.loads(next(events)) == {'kind': 'epoch', **row} and row['epoch'] == epoch and
                row['update'] == (epoch + 1) * UPDATES_PER_EPOCH and row['images'] == 2087 and
                row['gt'] == 5536 and all(0 <= row[k] <= 1 for k in ('ap50', 'ap50_95')) and
                all(math.isclose(row['training_loss'][name], losses / len(flat), rel_tol=1e-12)
                    for name, losses in loss.items()), 'Epoch metrics, evaluated population, or loss differs')
        val_reads += row['images']
        timing['validation'] += row['validation_seconds']
    require(next(events, None) is None, 'Unexpected event log tail')
    require(fit_reads == 99260 and val_reads == 20870, 'Dataset read counts differ')
    files = {p.relative_to(run).as_posix(): {'bytes': p.stat().st_size, 'sha256': hash_file(p)}
             for p in sorted(run.rglob('*')) if p.is_file()}
    return dict(status='COMPLETE', arm=arm, seed=seed, updates=ledger['update'],
                best_validation=best, completed_epochs=history, best_sha256=ledger['best'],
                last_sha256=ledger['last'], artifact_manifest=files, signed_run_sha256=signed,
                epoch_order_sha256=orders, train_pair_reads=fit_reads, validation_pair_reads=val_reads,
                peak_gpu_allocated_bytes=peak,
                wall_seconds=sum(x['wall_seconds'] for x in attempts if x['wall_seconds'] is not None),
                unknown_attempt_durations=sum(x['wall_seconds'] is None for x in attempts),
                timing_seconds=timing, attempts=attempts, resume_audit=read(run/'resume-audit.json'),
                official_test_pixel_reads=0)


def validate_approval():
    approved = read(EVIDENCE / 'scope-extension-approval.json')
    gate = read(EVIDENCE / 'preflight-verification.json')
    receipt = read(RECEIPT)
    ledger = read(RECEIPT.parent / 'run-signatures.json')['runs']
    keys = [f'{arm}/{seed}' for arm, seed in RUNS]
    require(approved['status']=='APPROVED' and gate['status']=='PASS' and gate['approved_cases']==20 and
            gate['approval_file_sha256']==hash_file(EVIDENCE/'scope-extension-approval.json') and
            gate['receipt_sha256']==approved['candidate_sha256']==hash_file(RECEIPT) and
            approved['approved_new_run_keys']==keys and
            approved['split_sha256']==receipt['split_sha256']==hash_file(SPLIT), 'Approval or gate differs')
    require(approved['registry_sha256']==hash_file(REPO/'experiments/r3_d1/review_registry.py') and
            approved['authorization_source_sha256']==hash_file(REPO/'experiments/r3_d1/authorization.py'),
            'Production authorization sources changed')
    for key in keys:
        row = approved['runtime_signatures'][key]
        original = ledger[key]['signature']
        require(digest(original)==row['receipt_signature_sha256']==receipt['run_signatures'][key],
                'Original receipt signature differs')
        paths = dict(original['source_sha256'])
        paths['experiments/r3_d1/authorization.py'] = approved['authorization_source_sha256']
        require(all(hash_file(REPO/path)==h for path,h in paths.items()) and
                digest(dict(original, source_sha256=paths)) == row['runtime_signature_sha256'],
                'Frozen runtime fingerprint differs')
    return approved, ledger, set(read(SPLIT)['fit'])


def launch(args, state, item, approved, ledger, fit):
    arm, seed = item
    key = f'{arm}/{seed}'
    run = args.output/arm/str(seed)
    audit = args.output/f'{arm}-{seed}-audit.json'
    attempts = state['attempts'].setdefault(key, [])
    if audit.exists():
        fixed = read(audit)
        require(fixed==audit_run(run,arm,seed,approved,ledger[key],fit,attempts),
                f'{key} completed audit changed')
        return fixed
    resume = run.exists()
    if resume:
        require(args.resume_batch and attempts, f'{key} output exists without explicit recovery')
        checkpoint(run,approved['runtime_signatures'][key]['runtime_signature_sha256'],approved['environment'])
        require(all(a['exit_code'] is not None for a in attempts), 'Interrupted attempt needs inspection')
    logs = args.output/'runner-logs'
    log = logs/f'{arm}-{seed}.attempt-{len(attempts)+1}.log'
    argv = [sys.executable,'-u','-m','experiments.r3_d1.train','--root',str(args.data),
            '--split',str(SPLIT),'--upstream',str(args.upstream),'--output',str(args.output),
            '--seed',str(seed),'--arm',arm,'--authorize-formal-training','--authorization',str(RECEIPT)]
    if resume:
        argv.append('--resume')
    started = time.time()
    clock = time.monotonic()
    with log.open('xb') as stream:
        attempt = dict(started_at=local_time(started), resume=resume, log=log.relative_to(args.output).as_posix(),
                       argv=argv, exit_code=None)
        attempts.append(attempt)
        with subprocess.Popen(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT) as worker:
            state.update(active=key, active_pid=worker.pid, status='RUNNING')
            write(args.output/'batch-status.json',state)
            attempt['exit_code'] = worker.wait()
    attempt.update(wall_seconds=time.monotonic()-clock, log_sha256=hash_file(log),
                   ended_at=local_time(time.time()))
    write(args.output/'batch-status.json',state)
    require(attempt['exit_code']==0, f'{key} exited {attempt["exit_code"]}; preserved {log}')
    result = audit_run(run,arm,seed,approved,ledger[key],fit,attempts)
    write(audit,result)
    return result


def seed_audit(output, seed, runs):
    ordered = [runs[f'{a}/{seed}'] for a in ARMS]
    require(all(row['epoch_order_sha256']==ordered[0]['epoch_order_sha256'] for row in ordered),
            f'{seed} paired sampling order differs')
    doc = dict(status='PASS',seed=seed,arms=ordered,total_updates=sum(row['updates'] for row in ordered))
    path = output/f'seed-{seed}-audit.json'
    if path.exists():
        require(read(path)==doc,'Existing seed audit changed')
    else:
        write(path,doc)


def recover_parent(output, state, approved):
    if state['status'] in ('STOPPED', 'STAGE_COMPLETE'):
        return
    require(state['status']=='RUNNING', 'A completed batch cannot be relaunched')
    key = state.get('active')
    if key is None:
        return
    pid = state.get('active_pid')
    if pid:
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.exists():
            running = cmdline.read_bytes()
            require(b'experiments.r3_d1.train' not in running or
                    os.fsencode(output) not in running, 'The previous training worker is still active')
    attempts = state['attempts'][key]
    last = attempts[-1]
    if last['exit_code'] is None:
        checkpoint(output / key,approved['runtime_signatures'][key]['runtime_signature_sha256'],
                   approved['environment'])
        last.update(exit_code='UNKNOWN_INTERRUPTION',wall_seconds=None,
                    log_sha256=hash_file(output/last['log']),
                    interruption_verified_at=local_time(time.time()))


def scheduled_runs(last_seed):
    return [(arm, seed) for arm, seed in RUNS if seed <= last_seed]


def run(args):
    approved, ledger, fit = validate_approval()
    scheduled = scheduled_runs(args.last_seed)
    keys = {f'{arm}/{seed}' for arm, seed in scheduled}
    args.output = args.output.resolve()
    require(not args.output.is_relative_to(REPO) and args.data.is_dir() and args.upstream.is_dir(),
            'External output or frozen assets are invalid')
    args.output.mkdir(exist_ok=args.resume_batch)
    (args.output/'runner-logs').mkdir(exist_ok=args.resume_batch)
    status = args.output/'batch-status.json'
    with (args.output/'batch.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume_batch:
            state = read(status)
            require(state['commit']==args.commit,'Recovery commit differs')
            require(set(state['attempts']) <= keys and set(state['runs']) <= keys,
                    'Existing work extends beyond the requested seed stage')
            recover_parent(args.output,state,approved)
        else:
            state = dict(status='RUNNING',commit=args.commit,started_at=local_time(time.time()),
                         approved_scope=approved['approved_new_run_keys'],attempts={},runs={})
        state.update(status='RUNNING', last_scheduled_seed=args.last_seed,
                     scheduled_run_keys=[f'{arm}/{seed}' for arm, seed in scheduled])
        state.pop('error',None)
        write(status,state)
        for arm, seed in scheduled:
            key = f'{arm}/{seed}'
            try:
                state['runs'][key] = launch(args,state,(arm,seed),approved,ledger,fit)
                if arm=='D':
                    seed_audit(args.output,seed,state['runs'])
                write(status,state)
            except Exception as exc:
                state.update(status='STOPPED',active=key,active_pid=None,error=str(exc))
                write(status,state)
                raise
        full = args.last_seed == max(SEEDS)
        state.update(status='COMPLETE' if full else 'STAGE_COMPLETE',
                     active=None,active_pid=None,completed_at=local_time(time.time()),
                     new_updates=sum(r['updates'] for r in state['runs'].values()))
        require(state['new_updates'] == len(scheduled)*EPOCHS*UPDATES_PER_EPOCH,
                'Wrong scheduled update count')
        write(status,state)
        if full:
            write(args.output/'batch-summary.json',state)
            finalize(args.output,state,approved)
        else:
            write(args.output/f'stage-through-{args.last_seed}-summary.json',state)
            print(f'PASS stage through {args.last_seed}: {len(scheduled)} audited runs')


def final_rows(state):
    r6 = R6_EVIDENCE/'completed/final-training-audit.json'
    manifest = read(R6_EVIDENCE/'completed/final-verification.json')
    require(hash_file(r6)==manifest['files_sha256'][r6.name],'R6 historical proof changed')
    reference = read(r6)
    old = reference['arm_results']
    require(reference['total_optimizer_updates']==99280 and len(old)==4,'R6 historical matrix incomplete')
    return [old[a] for a in ARMS] + [state['runs'][f'{a}/{s}'] for a,s in RUNS]


def selection(row, rule):
    return row['best_validation'] if rule=='best' else row['completed_epochs'][-1]


def describe(numbers):
    return {'mean':statistics.mean(numbers),'sample_std':statistics.stdev(numbers)}


def statistics_for(rows):
    by_key = {(row['arm'],row['seed']):row for row in rows}
    require(len(by_key)==20,'Matrix is incomplete')
    result = {}
    for rule in ('best','epoch_10'):
        arm_summary = {}
        for arm in ARMS:
            arm_summary[arm] = {metric:describe([selection(by_key[arm,s],rule)[metric]
                                for s in range(3407,3412)]) for metric in ('ap50','ap50_95')}
        pairs = {str(s):{metric:selection(by_key['D',s],rule)[metric]-
                 selection(by_key['C',s],rule)[metric] for metric in ('ap50','ap50_95')}
                 for s in range(3407,3412)}
        result[rule] = dict(arms=arm_summary,paired_D_minus_C=pairs,
                            paired_summary={metric:describe([row[metric] for row in pairs.values()])
                                            for metric in ('ap50','ap50_95')})
    return result


def finalize(output, state, approved):
    rows = final_rows(state)
    stats = statistics_for(rows)
    results = EVIDENCE/'completed'
    results.mkdir()
    for seed in SEEDS:
        name = f'seed-{seed}-audit.json'
        (results/name).write_bytes((output/name).read_bytes())
    combined = dict(status='PASS',training_commit=state['commit'],started_at=state['started_at'],
                    completed_at=state['completed_at'],seed_count=5,new_updates=state['new_updates'],
                    all_updates=sum(row['updates'] for row in rows),runs=rows,statistics=stats,
                    environment=approved['environment'],upstream=approved['upstream'],
                    candidate_sha256=approved['candidate_sha256'],split_sha256=approved['split_sha256'],
                    protocol_sha256=approved['protocol_sha256'],
                    authorization_source_sha256=approved['authorization_source_sha256'],
                    official_test_pixel_reads=0,local_artifact_root=str(output))
    require(combined['all_updates']==496400,'Invalid total training count')
    write(results/'five-seed-audit.json',combined)
    write(results/'verification.json',dict(status='PASS',files_sha256=
          {p.name:hash_file(p) for p in sorted(results.iterdir())}))
    report = REPO/'reports/R3_D1B_R7_FIVE_SEED_EXPERIMENTS.md'
    before = report.read_text(encoding='utf-8')
    with report.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(render_report(before,combined))+'\n')
    print(f'PASS complete 20-run matrix with {combined["all_updates"]} updates')


def render_report(before, report):
    stats = report['statistics']
    lines = [before.replace('**状态为训练准备。**','**状态为五种子训练和审计已完成。**').rstrip(),
             '', '## 五随机种子完整结果', '',
             f'R7 结束于 {report["completed_at"]}。16 个新增运行共 {report["new_updates"]:,} 次更新，'
             f'含历史 seed 3407 的 20 个运行共 {report["all_updates"]:,} 次。官方 Test 像素读取为 0。',
             '', '| Seed | 臂 | 最佳轮 | 最佳 AP50 | 最佳 AP50:95 | 第 10 轮 AP50 | 第 10 轮 AP50:95 | 墙钟小时 | 峰值 MiB |',
             '|---:|:---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['runs']:
        best, end = selection(row,'best'), selection(row,'epoch_10')
        lines.append(f'| {row["seed"]} | {row["arm"]} | {best["epoch"]+1} | {best["ap50"]:.2%} | '
                     f'{best["ap50_95"]:.2%} | {end["ap50"]:.2%} | {end["ap50_95"]:.2%} | '
                     f'{row["wall_seconds"]/3600:.2f} | {row["peak_gpu_allocated_bytes"]/2**20:.1f} |')
    lines += ['', '## 均值、样本标准差与配对差', '',
              '四臂均值与样本标准差以五个种子为单位。配对差取同一 seed 的 Mamba（D）减 CNN（C），单位为百分点。',
              '', '| 端点 | 实验臂 | AP50 均值 ± 样本 SD（%） | AP50:95 均值 ± 样本 SD（%） |',
              '|---|---|---:|---:|']
    for key, label in (('best','最高 AP50:95 轮'),('epoch_10','第 10 轮')):
        for a, row in stats[key]['arms'].items():
            v, w = row['ap50'], row['ap50_95']
            lines.append(f'| {label} | {a} | {v["mean"]*100:.2f} ± {v["sample_std"]*100:.2f} | '
                         f'{w["mean"]*100:.2f} ± {w["sample_std"]*100:.2f} |')
    lines += ['', '| Seed | 最佳 AP50 差 | 最佳 AP50:95 差 | 第 10 轮 AP50 差 | 第 10 轮 AP50:95 差 |',
              '|---:|---:|---:|---:|---:|']
    for s in range(3407,3412):
        diff = [stats[k]['paired_D_minus_C'][str(s)][m]*100 for k in ('best','epoch_10')
                for m in ('ap50','ap50_95')]
        lines.append(f'| {s} | ' + ' | '.join(f'{x:+.2f}' for x in diff) + ' |')
    pairs = [stats[k]['paired_summary'][m] for k in ('best','epoch_10') for m in ('ap50','ap50_95')]
    lines.append('| 均值 ± 样本 SD | ' +
                 ' | '.join(f'{r["mean"]*100:+.2f} ± {r["sample_std"]*100:.2f}' for r in pairs) + ' |')
    attempts = [a for row in report['runs'] for a in row.get('attempts',[])]
    failures = sum(a['exit_code'] != 0 for a in attempts)
    resumes = sum(a['resume'] for a in attempts)
    unknown = sum(row.get('unknown_attempt_durations',0) for row in report['runs'])
    lines += ['', f'R7 失败或中断记录 {failures} 次，签名与哈希核验后恢复 {resumes} 次。'
              f'墙钟用时不含 {unknown} 次未记录结束时间的运行尝试。'
              '每个种子的独立审计保存逐轮指标、退出状态、运行命令、完整环境指纹与 best/last checkpoint 哈希。'
              '指标限于冻结的内部 Validation。',
              '', '[五种子机器可读结果](evidence/r3-d1b-r7/completed/five-seed-audit.json) · '
              '[复核文件 SHA256](evidence/r3-d1b-r7/completed/verification.json)']
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('output','data','upstream'):
        parser.add_argument(f'--{option}',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--resume-batch',action='store_true')
    parser.add_argument('--last-seed',type=int,choices=SEEDS,default=3409)
    run(parser.parse_args())
