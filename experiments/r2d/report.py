"""Render the measured results in the R2d report after the paired analysis."""
import argparse
from pathlib import Path

from experiments.r1g.partition import read, require
from experiments.r2c.protocol import ARMS

ARM_NAMES = {'ssm': 'SSM', 'cnn': 'CNN', 'attention': 'Attention'}
MARKER = '<!-- Completed results generated below. -->'


def percent(value):
    return 'NA' if value is None else f'{100 * value:.5f}%'


def contrast(value):
    lo, hi = value['ci95']
    return f'{100 * value["point"]:+.5f} [{100 * lo:+.5f}, {100 * hi:+.5f}]'


def render(evidence, report):
    stats = read(evidence / 'multiseed-summary.json')
    seeds = stats['seeds']
    require(stats['status'] == 'PASS' and len(seeds) >= 3, 'Insufficient audited results')
    p = stats['cases']['primary']
    root = report.read_text(encoding='utf-8')
    require(MARKER in root, 'R2d report marker missing')
    lines = ['## 完成的训练与 Epoch 5 指标', '',
             f'完成 {len(seeds)} 个种子和 {3 * len(seeds)} 次独立五轮训练。种子为 '
             + '、'.join(map(str, seeds)) + '。']
    if len(seeds) < 5:
        lines += ['种子层面不确定性较大，以下结果为探索性比较。']
    lines += ['', '| Seed | 融合 | Clean mIoU | 前景 mIoU | Epoch 5 train / val CE | '
              '新增率 | 净错误率 | 分歧率 | 最佳 epoch / mIoU | 训练 (min) | 峰值 (GiB) |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    minutes = {arm: 0. for arm in ARMS}
    for idx, seed in enumerate(seeds):
        for arm in ARMS:
            h = read(evidence / f'seed{seed}/training/{arm}/epochs.json')
            require([r['epoch'] for r in h] == [1, 2, 3, 4, 5] and h[-1]['updates'] == 785,
                    'Training history differs')
            best = max(h, key=lambda r: r['validation']['all']['miou_including_0'])
            train_minutes = sum(r['train_seconds'] for r in h) / 60
            minutes[arm] += train_minutes
            final = h[-1]
            v = p['arms'][arm]
            cells = [str(seed), ARM_NAMES[arm], percent(v['clean_miou']['per_seed'][idx]),
                     percent(v['clean_foreground_miou']['per_seed'][idx]),
                     f'{final["train_mean_pixel_loss"]:.5f} / {final["validation"]["all"]["mean_pixel_loss"]:.5f}',
                     *(percent(v[k]['per_seed'][idx]) for k in ('new', 'net', 'disagree')),
                     f'{best["epoch"]} / {percent(best["validation"]["all"]["miou_including_0"])}',
                     f'{train_minutes:.2f}', f'{max(r["peak_allocated_bytes"] for r in h) / 2**30:.3f}']
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f'GPU 训练窗口合计 {sum(minutes.values()) / 60:.2f} 小时。逐 epoch CE、总体与前景 mIoU、'
              '九类 IoU、独立 checkpoint 指纹、扰动和 CI 见各种子证据。']
    for seed in seeds:
        base = f'evidence/r2d/seed{seed}'
        lines.append(f'- Seed {seed} [训练记录]({base}/training) · [最终审计]({base}/audit-epoch5.json) · '
                     f'[扰动逐图]({base}/perturbation) · [组级结果]({base}/analysis/summary.json)')
    lines += ['', '## 种子均值与配对区间', '',
              '每格为等权种子均值 ± 种子样本标准差。配对区间同时对种子和 54 个视觉组重采样。', '',
              '| 融合 | Clean mIoU | 前景 mIoU | ROI 外新增 | ROI 外纠正 | ROI 外净增 | ROI 外分歧 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a = p['arms'][arm]
        cells = [f'{percent(a[k]["point"])} ± {percent(a[k]["seed_sd"])}' for k in
                 ('clean_miou', 'clean_foreground_miou', 'new', 'corrected', 'net', 'disagree')]
        lines.append(f'| {ARM_NAMES[arm]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| SSM 减对照 | 指标 | 差值与 95% CI（百分点） | SSM 较低 / 较高 / 相同的种子数 |',
              '|---|---|---:|---:|']
    for other in ('cnn', 'attention'):
        for metric, label in (('new', '新增'), ('net', '净错误'), ('disagree', '分歧'),
                              ('clean_miou', 'Clean mIoU'), ('delta_miou', '全图 mIoU 变化'),
                              ('delta_foreground_miou', '前景 mIoU 变化')):
            v = p['contrasts'][f'ssm_minus_{other}'][metric]
            d = v['seed_directions']
            lines.append(f'| {ARM_NAMES[other]} | {label} | {contrast(v)} | '
                         f'{d["negative"]} / {d["positive"]} / {d["zero"]} |')
    lines += ['', '## 科研判断', '']
    for question, other in (('A', 'cnn'), ('B', 'attention')):
        results = p['contrasts'][f'ssm_minus_{other}']
        common_advantage = all(results[k]['ci95'][1] < 0 and results[k]['seed_directions']['negative'] == len(seeds)
                               for k in ('new', 'net'))
        assessment = '两个主要指标均呈稳定优势' if common_advantage else '两个主要指标未共同显示稳定优势'
        lines.append(f'{question}．SSM 相对 {ARM_NAMES[other]}，{assessment}。新增及净错误率的配对差分别为 '
                     f'{contrast(results["new"])} 和 {contrast(results["net"])} 个百分点。')
    old = read(evidence.parent / 'r2c/analysis/summary.json')['cases']['primary']['contrasts']
    lines += ['', 'C．以下比较 R2c 的未匹配单种子点估计与 R2d 匹配后的多种子估计。'
              '差异同时可能来自宽度、随机种子和训练后的权重。', '',
              '| 比较 | R2c 新增 / 净差（百分点） | R2d 新增 / 净差（百分点） |', '|---|---:|---:|']
    for other in ('cnn', 'attention'):
        key = f'ssm_minus_{other}'
        earlier = ' / '.join(f'{old[key]["outside"][k]["point"] * 100:+.5f}' for k in ('new', 'net'))
        later = ' / '.join(f'{p["contrasts"][key][k]["point"] * 100:+.5f}' for k in ('new', 'net'))
        lines.append(f'| SSM−{ARM_NAMES[other]} | {earlier} | {later} |')
    lines += ['']
    for other in ('cnn', 'attention'):
        key = f'ssm_minus_{other}'
        for metric, label in (('new', '新增'), ('net', '净错误')):
            value = p['contrasts'][key][metric]
            same = value['point'] * old[key]['outside'][metric]['point'] > 0
            directions = '保留 R2c 的点估计方向' if same else '未保留 R2c 的点估计方向'
            lines.append(f'SSM−{ARM_NAMES[other]} 的{label}率差{directions}。')
    lines += ['', 'D．将每个种子的两项架构配对各计一次。较高的 clean mIoU 同时伴随较高扰动错误率'
              '时计为权衡方向，仅作描述。']
    for metric, label in (('new', '新增'), ('net', '净错误')):
        products = [x * y for c in p['contrasts'].values()
                    for x, y in zip(c['clean_miou']['per_seed'], c[metric]['per_seed'], strict=True)]
        n = sum(x > 0 for x in products)
        lines.append(f'{label}率有 {n}/{len(products)} 项配对符合该方向。架构与训练随机性均影响两类指标，'
                     '此设计无法确立因果权衡。')
    lines += ['', 'E．值得继续以 R2b 的内部状态干预检验局部因果路径。当前三臂共享含 SSM 的编码器与解码器，'
              '整体扰动差异的归因需要单独控制这些状态路径。', '',
              '[多层 Bootstrap 汇总](evidence/r2d/multiseed-summary.json)']
    report.write_text(root.split(MARKER, 1)[0] + MARKER + '\n\n' + '\n'.join(lines) + '\n',
                      encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence, args.report)


if __name__ == '__main__':
    main()
