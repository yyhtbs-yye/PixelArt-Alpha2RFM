"""Compare uninterrupted/resumed RFM updates and publish a local HTML report."""
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'output/training_dependencies')]
import numpy as np
from PIL import Image
import torch

from diffusion.model.builder import build_model
from diffusion.rectified_flow import flow_matching_loss, sample_rfm
from datasets.rfm_data import read_cache


def compare(left, right, path, report):
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'{path}: shape/dtype mismatch')
        report['tensors_compared'] += 1
        if not torch.equal(left, right):
            difference = (left.float() - right.float()).abs().max().item()
            report['differences'].append(dict(path=path, max_abs=difference))
            tolerance = 2e-5 if path.startswith('optimizer.') else 1e-6
            if not left.is_floating_point() or not torch.allclose(left, right, atol=tolerance, rtol=1e-5):
                report['outside_numerical_tolerance'].append(path)
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            report['differences'].append(dict(path=path, kind='numpy'))
            report['outside_numerical_tolerance'].append(path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{path}: keys mismatch')
        for key in left:
            compare(left[key], right[key], f'{path}.{key}', report)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f'{path}: length mismatch')
        for index, (first, second) in enumerate(zip(left, right)):
            compare(first, second, f'{path}.{index}', report)
    elif left != right:
        report['differences'].append(dict(path=path, kind='scalar'))
        report['outside_numerical_tolerance'].append(path)


def main():
    torch.set_num_threads(4)
    work = ROOT / 'output/rfm_mix50_smoke'
    uninterrupted = torch.load(work/'uninterrupted/checkpoints/update-000020.pth',
                                map_location='cpu', weights_only=False, mmap=True)
    resumed = torch.load(work/'resumed/checkpoints/update-000020.pth',
                         map_location='cpu', weights_only=False, mmap=True)
    report = dict(tensors_compared=0, differences=[], outside_numerical_tolerance=[],
                  resume_tolerance=dict(model_atol=1e-6, optimizer_atol=2e-5, rtol=1e-5))
    for key in ('state_dict', 'state_dict_ema', 'optimizer', 'scheduler', 'progress', 'rank_rng'):
        compare(uninterrupted[key], resumed[key], key, report)
    report['resume_bitwise_equal'] = not report['differences']
    report['resume_numerically_close'] = not report['outside_numerical_tolerance']
    report['resume_rng_cursor_scheduler_exact'] = not any(
        item['path'].split('.')[0] in ('rank_rng', 'progress', 'scheduler') for item in report['differences'])
    report['resume_max_abs'] = {key: max((item['max_abs'] for item in report['differences']
        if item['path'].startswith(key + '.') and 'max_abs' in item), default=0.)
        for key in ('state_dict', 'state_dict_ema', 'optimizer')}
    records = [json.loads(line) for line in (work/'uninterrupted/train.jsonl').read_text().splitlines()]
    replay = [json.loads(line) for line in (work/'resumed/train.jsonl').read_text().splitlines()]
    report['replayed_metrics_equal'] = all(
        a[key] == b[key] for a, b in zip(records[10:], replay)
        for key in ('step', 'loss', 'grad_norm', 'learning_rate', 'samples_seen', 'time_bins_sum_count'))
    report['replayed_steps'] = [row['step'] for row in replay]
    report['replayed_loss_max_abs'] = max(abs(a['loss']-b['loss']) for a,b in zip(records[10:], replay))
    assert report['replayed_steps'] == list(range(11, 21))
    report['all_steps_balanced'] = all(row['source_counts_per_update'] == [16, 16] for row in records + replay)
    report['all_metrics_finite'] = all(np.isfinite([r['loss'], r['grad_norm']]).all() for r in records + replay)
    report['peak_training_gpu_gib'] = max(max(r['peak_gpu_gib_per_rank']) for r in records + replay)
    report['steady_training_seconds_per_step'] = float(np.median([r['seconds_per_step'] for r in records[2:]]))
    report['validation_images'] = []
    for folder in ('uninterrupted', 'resumed'):
        for path in sorted((work/folder/'validation/update-000020').glob('*.png')):
            with Image.open(path) as image:
                image.load()
                assert image.size == (512, 512) and image.mode == 'RGB'
            report['validation_images'].append(str(path.relative_to(work)))
    assert len(report['validation_images']) == 16
    config = uninterrupted['metadata']['training_config']
    source = Path(config['source_model'])/'transformer/diffusion_pytorch_model.safetensors'
    with source.open('rb') as handle:
        report['source_checkpoint_unchanged'] = hashlib.file_digest(handle, 'sha256').hexdigest() == config['source_checkpoint_sha256']
    initial = torch.load(config['init_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    head = ['final_layer.linear.weight', 'final_layer.linear.bias']
    report['changed_backbone_tensors'] = sum(not torch.equal(value, initial['state_dict'][key])
        for key, value in uninterrupted['state_dict'].items() if key not in head)
    report['head_max_abs'] = uninterrupted['state_dict'][head[0]].abs().max().item()
    assert report['changed_backbone_tensors'] > 0 and report['head_max_abs'] > 0
    del resumed, initial

    # Fixed data/noise/time probe: confirm a real forward after optimization.
    device = torch.device('cuda', 0)
    model = build_model('PixArt_XL_2', input_size=64, pred_sigma=False, model_max_length=120,
                        class_dropout_prob=0., qk_norm='rms_norm', use_grad_checkpoint=False, lewei_scale=1.).float().to(device).eval()
    model.load_state_dict(uninterrupted['state_dict'], strict=True)
    fixed = []
    for name in config['data_roots']:
        root = Path(name)
        rows = [json.loads(line) for line in (root/'training_metadata.jsonl').read_text().splitlines()]
        fixed.extend(read_cache((root, row)) for row in rows[:4])
    mean, std, caption, mask = [torch.from_numpy(np.stack([row[column] for row in fixed])).to(device)
                               for column in range(4)]
    caption, mask = caption.bfloat16(), mask.bool()
    generator = torch.Generator(device=device).manual_seed(914)
    clean = (mean + std * torch.randn(mean.shape, generator=generator, device=device)) * .18215
    noise = torch.randn(clean.shape, generator=generator, device=device)
    times = torch.linspace(0, 1, 8, device=device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        losses = flow_matching_loss(model, clean, noise, times, caption, mask)
    report['fixed_probe'] = dict(time=times.tolist(), zero_head_loss=(clean-noise).square().flatten(1).mean(1).tolist(),
                                 trained_loss=losses.tolist())
    assert torch.isfinite(losses).all()
    text = torch.load(config['conditioning_cache'], map_location='cpu', weights_only=True)
    sample_noise = torch.cat([torch.randn(1, 4, 64, 64, device=device,
                              generator=torch.Generator(device=device).manual_seed(seed))
                              for seed in config['validation_seeds']])
    heun, heun_report = sample_rfm(model, sample_noise, text['condition'].to(device).bfloat16(),
        text['mask'].to(device).bool(), text['empty'].to(device).bfloat16(),
        text['empty_mask'].to(device).bool(), method='heun', steps=50)
    reference = torch.load(work/'uninterrupted/validation/update-000020/raw.pt', weights_only=True)['latents'].to(device)
    report['heun_check'] = dict(**heun_report, relative_latent_l2=float((heun-reference).norm()/reference.norm()))
    torch.save(dict(latents=heun.cpu(), **heun_report), work/'heun_raw.pt')
    report['passed'] = all(report[key] for key in ('resume_numerically_close', 'resume_rng_cursor_scheduler_exact',
                         'all_steps_balanced', 'all_metrics_finite', 'source_checkpoint_unchanged'))
    report['status'] = ('PASS' if report['resume_bitwise_equal'] else 'PASS_WITH_NUMERICAL_DRIFT') if report['passed'] else 'FAIL'
    (work/'verification.json').write_text(json.dumps(report, indent=2))

    images = ''.join(f'<figure><img loading="lazy" src="{path}"><figcaption>{path}</figcaption></figure>'
                     for path in report['validation_images'] if path.startswith('uninterrupted/'))
    summary = html.escape(json.dumps({key: value for key, value in report.items()
                          if key not in ('validation_images', 'fixed_probe', 'differences')}, indent=2, ensure_ascii=False))
    source_rows = json.loads((work/'source_audit.json').read_text())
    dataset_rows = ''.join(f'<tr><td>{html.escape(row["source"])}</td><td>{row["images"]:,}</td><td>50%</td><td>256</td></tr>' for row in source_rows)
    (work/'report.html').write_text(f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PixArt RFM 50/50 短程验证</title>
<style>body{{font:16px/1.7 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#172333;background:#f6f8fb}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;border:1px solid #d5dce5;text-align:left}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}
figcaption{{font-size:12px;overflow-wrap:anywhere}}pre{{background:#e9eef5;padding:20px;overflow:auto}}.notice{{background:#fff1c9;padding:18px;border-radius:8px}}</style>
<h1>PixArt RFM · 50/50 数据混合短程验证</h1>
<p>状态：<strong>{report['status']}</strong>。4 GPUs × batch 8，无梯度累积，BF16，关闭 gradient checkpointing。</p>
<p class="notice">本次只验证工程流程：20 iter 连续训练，与第 10 iter 恢复后再训练到 20 iter。
输出头从零开始，且尚处于 500 iter warmup，以下图像主要是噪声，不能作为生成质量或收敛结论。
未启动 100K 正式训练，未完成全量 latent/T5 缓存。</p>
<h2>数据</h2><table><tr><th>来源</th><th>全量清单</th><th>每 batch 比例</th><th>本次缓存</th></tr>{dataset_rows}</table>
<p>全量检查为路径存在、文件名/索引唯一、元数据尺寸和提示词检查；实际解码与编码的是本次 512 张样本。
使用 model_prompt 原始生成提示词。两个来源独立打乱、循环，不截断较大的数据集。</p>
<h2>训练定义</h2><p>x(t)=(1−t)ε+t·z；target=z−ε；t~Uniform(0,1)；普通 MSE，无 VB。
时间 embedding 使用连续 999(1−t)，不使用旧 VP β schedule。保留 QK RMSNorm 骨干，4 通道输出投影零初始化；EMA 从新 RAW 复制。</p>
<h2>验证结果</h2><p>恢复检查不满足逐位一致：RAW/EMA 和 optimizer 存在下列微小数值差异。
随机状态、数据游标、scheduler 完全一致；单独按报告列出的绝对/相对容差检查模型和 optimizer。
误差来源尚未单独定位，不声称严格确定性恢复。</p><pre>{summary}</pre>
<p><a href="verification.json">完整验证 JSON</a> · <a href="source_audit.json">全量清单检查</a> ·
<a href="uninterrupted/train.jsonl">连续训练日志</a> · <a href="resumed/train.jsonl">恢复日志</a> ·
<a href="uninterrupted/initialization.json">初始化记录</a> · <a href="uninterrupted/validation/update-000020/sampler.json">采样器记录</a></p>
<h2>RAW / EMA 流程输出</h2><div class="gallery">{images}</div>
<h2>下一阶段</h2><p>完成两个来源的全量缓存后，正式配置会检查 50,000 / 76,708 条记录。
正式训练默认 100K iter，每 500 iter 生成 RAW/EMA 各 4 张，每 5000 iter 保存。
先在早期 checkpoint 观察目标适配、时间分桶 loss 与生成质量，再判断 RFM 是否改善质感。</p>
</html>''')
    print(json.dumps({key: value for key, value in report.items() if key not in ('validation_images', 'fixed_probe', 'differences')}, indent=2))
    if not report['passed']:
        raise AssertionError('RFM smoke verification failed; see verification.json')


if __name__ == '__main__':
    main()
