"""PixArt RFM training: use torchrun directly, or --launch for background cache preparation and training."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'output/training_dependencies')]

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from diffusion.model.builder import build_model
from diffusion.rectified_flow import flow_matching_loss, sample_rfm
from diffusion.utils.rfm_checkpoint import (drop_to_empty_text, rng_state, restore_rng,
                                         update_ema, save_training_checkpoint)
from datasets.rfm_data import CachedDataset, BalancedFlowSampler


@torch.no_grad()
def validate(model, ema, vae, text, config, step, device):
    folder = Path(config['work_dir']) / 'validation' / f'update-{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'prompts.json').write_text(json.dumps(text['prompts'], indent=2))
    condition = text['condition'].to(device, dtype=torch.bfloat16)
    mask = text['mask'].to(device, dtype=torch.bool)
    empty = text['empty'].to(device, dtype=torch.bfloat16)
    empty_mask = text['empty_mask'].to(device, dtype=torch.bool)
    noise = torch.cat([torch.randn(1, 4, 64, 64, device=device,
                       generator=torch.Generator(device=device).manual_seed(seed))
                       for seed in config['validation_seeds']])
    reports = {}
    for label, network in [('raw', model), ('ema', ema)]:
        network.eval()
        latents, report = sample_rfm(network, noise, condition, mask, empty, empty_mask,
                                    config['cfg_scale'], config['sampling_method'], config['sampling_steps'],
                                    config['sampling_atol'], config['sampling_rtol'])
        report['latent_rms'] = latents.square().mean().sqrt().item()
        for index, latent in enumerate(latents):
            decoded = vae.decode(latent[None] / .18215).sample[0]
            if not torch.isfinite(decoded).all():
                raise FloatingPointError('Nonfinite VAE decode')
            pixels = ((decoded / 2 + .5).clamp(0, 1) * 255).round().byte().permute(1, 2, 0)
            Image.fromarray(pixels.cpu().numpy()).save(folder / f'{label}-{index}.png')
        torch.save(dict(latents=latents.cpu(), seeds=config['validation_seeds'], **report), folder / f'{label}.pt')
        reports[label] = report
    (folder / 'sampler.json').write_text(json.dumps(reports, indent=2))
    model.train()
    return reports


def publish_status(folder, stage, **details):
    status = dict(stage=stage, controller_pid=os.getpid(),
                  updated_at=datetime.now(timezone.utc).isoformat(), **details)
    temporary = folder / 'status.tmp'
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(folder / 'status.json')


def launch_training(args, config):
    """Prepare caches and run a detached controller, or execute that controller."""
    overrides = []
    for key in ('work_dir', 'max_train_steps', 'validate_every_updates', 'save_every_updates'):
        value = getattr(args, key)
        if value is not None:
            overrides.extend(['--' + key.replace('_', '-'), str(value)])
    work = Path(config['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    preparation = work / 'preparation'
    preparation.mkdir(exist_ok=True)
    if not args.controller:
        lock = (preparation / 'launch.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (preparation / 'controller.json').exists():
            previous = json.loads((preparation / 'controller.json').read_text())
            proc = Path(f'/proc/{previous["pid"]}/cmdline')
            if proc.exists() and b'--controller' in proc.read_bytes() and b'train_rfm' in proc.read_bytes():
                raise RuntimeError(f'RFM preparation/training already active: PID {previous["pid"]}')
        if (work / 'train.jsonl').exists():
            raise FileExistsError('An existing training run requires an explicit resume, not a fresh launch')
        verification = json.loads((ROOT / 'output/rfm_mix50_smoke/verification.json').read_text())
        if not verification['passed']:
            raise RuntimeError('The RFM smoke verification has not passed')
        files = [args.config, 'scripts/train_rfm.py', 'datasets/rfm_data.py',
                 'diffusion/rectified_flow.py',
                 'diffusion/model/nets/PixArt.py', 'diffusion/model/nets/PixArt_blocks.py',
                 'diffusion/model/builder.py', 'diffusion/utils/rfm_checkpoint.py',
                 'scripts/preparation/cache_synthetic_dataset.py', 'scripts/preparation/encode_image_latents.py',
                 'diffusion/model/t5.py', 'requirements.txt']
        hashes = {}
        for name in files:
            path = Path(name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else Path('config.py')
            destination = preparation / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        (preparation / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
        shutil.copy2('output/rfm_mix50_smoke/verification.json', preparation / 'smoke_verification.json')
        (work / 'config.json').write_text(json.dumps(config, indent=2))
        environment = dict(os.environ, OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
                           PYTHONPATH='output/training_dependencies:output/t5_dependencies:.')
        command = [sys.executable, '-u', 'scripts/train_rfm.py', args.config, '--controller', *overrides]
        with (preparation / 'controller.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(pid=process.pid, command=command, config=args.config,
                       started_at=datetime.now(timezone.utc).isoformat())
        (preparation / 'controller.json').write_text(json.dumps(launch, indent=2))
        print(json.dumps(launch, indent=2))
        return

    lock = (preparation / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    children = []
    completed_sources = []
    gpu_ids = list(range(config['world_size']))
    if gpu_ids != [0, 1, 2, 3] or config['batch_size'] != 8:
        raise ValueError('This launcher is configured for four GPUs, eight samples per GPU')
    try:
        publish_status(preparation, 'preflight', target_steps=config['max_train_steps'], gpu_ids=gpu_ids)
        required = shutil.disk_usage(work).free
        if required < 350 * 2**30:
            raise RuntimeError('Need at least 350 GiB free for full caches and retained checkpoints')
        for source, cache, expected in zip(config['source_datasets'], config['data_roots'], config['source_counts']):
            source, cache = Path(source), Path(cache)
            cache.mkdir(parents=True, exist_ok=True)
            # These immutable NPZ records were encoded with the identical VAE/T5 settings.
            pilot_name = 'localized_narratives' if 'localized_narratives' in source.name else 'flux1_dev'
            pilot = Path('output/rfm_mix50_smoke/cache') / pilot_name
            settings = json.loads((pilot / 'cache_settings.json').read_text())
            if settings['source'] != str(source.resolve()):
                raise ValueError('Pilot cache source mismatch')
            if hashlib.sha256((source / 'metadata.jsonl').read_bytes()).hexdigest() != settings['source_metadata_sha256']:
                raise ValueError('Source metadata changed since smoke test')
            for subfolder in ('latents', 't5_embeddings'):
                (cache / subfolder).mkdir(exist_ok=True)
                for existing in (pilot / subfolder).glob('*.npz'):
                    destination = cache / subfolder / existing.name
                    if not destination.exists():
                        os.link(existing, destination)
            children = []
            for gpu in gpu_ids:
                command = [sys.executable, '-u', 'scripts/preparation/cache_synthetic_dataset.py',
                           '--source', str(source), '--output', str(cache), '--shards', '4',
                           '--shard', str(gpu), '--batch-size', '32']
                with (preparation / f'cache-{source.name}-{gpu}.log').open('a') as log:
                    child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                children.append(child)
            while True:
                progress = []
                for gpu, child in zip(gpu_ids, children):
                    path = cache / f'progress-{gpu}.json'
                    try:
                        row = json.loads(path.read_text()) if path.exists() else {}
                    except json.JSONDecodeError:
                        row = {}  # Worker progress files are informational and may be mid-write.
                    progress.append(dict(gpu=gpu, pid=child.pid, returncode=child.poll(), **row))
                publish_status(preparation, 'caching', source=str(source), cache=str(cache),
                               source_images=expected, completed_sources=completed_sources,
                               workers=progress, training_started=False)
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError(f'Cache worker failed for {source}; inspect preparation/cache-*.log')
                if all(child.poll() == 0 for child in children):
                    break
                time.sleep(10)
            subprocess.run([sys.executable, 'scripts/preparation/cache_synthetic_dataset.py', '--source', str(source),
                            '--output', str(cache), '--shards', '4', '--finalize'], check=True)
            summary = json.loads((cache / 'summary.json').read_text())
            if not summary['complete'] or summary['images'] != expected:
                raise RuntimeError('Finalized cache count mismatch')
            completed_sources.append(dict(source=str(source), cache=str(cache), images=expected))
        hashes = json.loads((preparation / 'source_sha256.json').read_text())
        for name, expected_hash in hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected_hash:
                raise RuntimeError(f'{name} changed during cache preparation; training was not started')
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   'scripts/train_rfm.py', args.config, *overrides]
        with (work / 'console.log').open('a') as log:
            trainer = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children = [trainer]
        launch = dict(pid=trainer.pid, command=command, started_at=datetime.now(timezone.utc).isoformat(),
                       source_sha256=hashes, completed_sources=completed_sources)
        (work / 'launch.json').write_text(json.dumps(launch, indent=2))
        (work / 'train.pid').write_text(str(trainer.pid) + '\n')
        while True:
            metrics = work / 'train.jsonl'
            latest = None
            if metrics.exists():
                lines = metrics.read_text().splitlines()
                if lines:
                    try:
                        latest = json.loads(lines[-1])
                    except json.JSONDecodeError:
                        latest = json.loads(lines[-2]) if len(lines) > 1 else None
            validations = sorted((work / 'validation').glob('update-*/sampler.json'))
            publish_status(preparation, 'training_running' if latest else 'training_starting',
                           trainer_pid=trainer.pid, latest_metrics=latest,
                           latest_validation=str(validations[-1]) if validations else None,
                           completed_sources=completed_sources, training_started=bool(latest))
            result = trainer.poll()
            if result is not None:
                if result or not (work / 'completed.json').exists():
                    raise RuntimeError(f'RFM trainer exited with status {result}; inspect console.log')
                publish_status(preparation, 'completed', trainer_pid=trainer.pid,
                               progress=json.loads((work / 'completed.json').read_text()))
                break
            time.sleep(10)
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
        publish_status(preparation, 'failed', error=str(error), completed_sources=completed_sources)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='configs/rfm.py')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--launch', action='store_true', help='Prepare caches and launch fresh training in the background')
    mode.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--resume')
    parser.add_argument('--work-dir')
    parser.add_argument('--max-train-steps', type=int)
    parser.add_argument('--validate-every-updates', type=int)
    parser.add_argument('--save-every-updates', type=int)
    args = parser.parse_args()
    if args.launch or args.controller:
        if any(key in os.environ for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')):
            parser.error('Launch mode must run outside torchrun')
        if args.resume:
            parser.error('Launch mode starts fresh training; use torchrun with --resume to continue a checkpoint')
        os.chdir(ROOT)
    elif not all(key in os.environ for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')):
        parser.error('Use torchrun for direct training, or --launch for background preparation and training')
    config = dict(runpy.run_path(args.config)['config'])
    for key in ('work_dir', 'max_train_steps', 'validate_every_updates', 'save_every_updates'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    required = dict(prediction_type='flow_velocity', path='linear', time_sampling='uniform_continuous',
                    time_embedding='999*(1-t)', loss='unweighted_mse', out_channels=4,
                    head_initialization='zero', qk_norm='rms_norm', mixed_precision='bf16',
                    grad_checkpointing=False, accumulation_steps=1, source_weights='raw',
                    source_weights_per_batch=[.5, .5])
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('Unsupported RFM numerical/architecture configuration')
    for key in ('max_train_steps', 'save_every_updates', 'validate_every_updates', 'log_every_updates',
                'warmup_updates', 'dataloader_workers', 'prefetch_factor'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if args.launch or args.controller:
        if config['world_size'] != 4 or config['batch_size'] != 8:
            parser.error('Launch mode requires four GPUs and eight samples per GPU')
        launch_training(args, config)
        return
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != config['world_size']:
        raise ValueError('World size differs from configuration')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    torch.set_num_threads(4)
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    torch.manual_seed(config['seed'] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    work = Path(config['work_dir'])
    if rank == 0:
        work.mkdir(parents=True, exist_ok=True)
        if (work / 'train.jsonl').exists():
            raise FileExistsError('Use a fresh work directory for each launch, including resume verification')
        (work / 'config.json').write_text(json.dumps(config, indent=2))
    dist.barrier()

    records, lengths, hashes = [], [], {}
    if any(len(config[key]) != 2 for key in ('data_roots', 'source_datasets', 'source_counts')):
        raise ValueError('Exactly two cache roots, original sources and expected counts are required')
    for name, source, expected in zip(config['data_roots'], config['source_datasets'], config['source_counts']):
        root = Path(name)
        manifest = root / 'training_metadata.jsonl'
        content = manifest.read_bytes()
        hashes[str(manifest)] = hashlib.sha256(content).hexdigest()
        rows = [json.loads(line) for line in content.splitlines()]
        settings = json.loads((root / 'cache_settings.json').read_text())
        source_hash = hashlib.sha256((Path(source) / 'metadata.jsonl').read_bytes()).hexdigest()
        if settings['source'] != str(Path(source).resolve()) or settings['source_metadata_sha256'] != source_hash:
            raise ValueError('Cached source provenance mismatch')
        if settings['count'] != len(rows) or len(rows) != expected:
            raise ValueError(f'Unexpected cache count for {root}: {len(rows)} != {expected}')
        if settings['latent_scaled'] or settings['scaling_factor'] != .18215 or settings['text_preprocessing']:
            raise ValueError('Unexpected cache preprocessing')
        if len({row['file_name'] for row in rows}) != len(rows):
            raise ValueError('Duplicate cache records')
        if any(not (root / row[key]).is_file() for row in rows for key in ('latent_file', 't5_file')):
            raise FileNotFoundError('Incomplete cache')
        records.extend((root, row) for row in rows)
        lengths.append(len(rows))
    if len(lengths) != 2:
        raise ValueError('Exactly two datasets are required')
    text_path = Path(config['conditioning_cache'])
    hashes[str(text_path)] = hashlib.sha256(text_path.read_bytes()).hexdigest()
    text = torch.load(text_path, map_location='cpu', weights_only=True)
    if text['seeds'] != config['validation_seeds'] or len(text['prompts']) != 4:
        raise ValueError('Validation text/seeds mismatch')
    empty = text['empty'].to(device, dtype=torch.bfloat16)
    empty_mask = text['empty_mask'].to(device, dtype=torch.bool)

    model = build_model(config['model'], input_size=64, pred_sigma=False, model_max_length=120,
                        class_dropout_prob=0., qk_norm='rms_norm', use_grad_checkpoint=False,
                        lewei_scale=1.).float()
    # The approved converted epsilon checkpoint already strictly maps the original RAW.
    initial = torch.load(config['init_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    provenance = initial['import_provenance']
    if (provenance['source_weights'] != 'raw' or provenance['source_sha256'] != config['source_checkpoint_sha256']
            or provenance['conditioning_sha256'] != hashes[str(text_path)]
            or initial['metadata']['prediction_type'] != 'epsilon'):
        raise ValueError('Initialization provenance mismatch')
    original = Path(config['source_model']) / 'transformer' / 'diffusion_pytorch_model.safetensors'
    with original.open('rb') as handle:
        source_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if source_digest != config['source_checkpoint_sha256']:
        raise ValueError('Original RAW checkpoint changed')
    state = dict(initial['state_dict'])
    reset = ['final_layer.linear.weight', 'final_layer.linear.bias']
    for key in reset:
        state[key] = torch.zeros_like(model.state_dict()[key])
    model.load_state_dict(state, strict=True)
    if not torch.equal(model.y_embedder.y_embedding, empty[0, 0].cpu().float()):
        raise ValueError('Null conditioning differs from initialization')
    if rank == 0:
        matched = [key for key in model.state_dict() if key not in reset]
        if not all(torch.equal(model.state_dict()[key], initial['state_dict'][key]) for key in matched):
            raise AssertionError('Backbone transfer was not exact')
        (work / 'initialization.json').write_text(json.dumps(dict(source=str(original),
            source_sha256=config['source_checkpoint_sha256'], source_weights='raw',
            preserved_tensors=len(matched), zero_initialized=reset, out_channels=4,
            optimizer_reset=True, ema_initialization='copy_migrated_raw'), indent=2))
    del state, initial
    model.to(device).train()
    if any(getattr(module, 'grad_checkpointing', False) for module in model.modules()):
        raise AssertionError('Gradient checkpointing must be disabled')
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=config['betas'],
                                  eps=config['adam_eps'], weight_decay=config['weight_decay'], fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / config['warmup_updates'], 1.))
    metadata = dict(format_version='rfm_v1', prediction_type='flow_velocity', path='linear',
                    time_embedding=config['time_embedding'], vae_scaling_factor=.18215,
                    training_config=config, manifest_hashes=hashes)
    progress = dict(micro_step=0, optimizer_step=0, consumed_batches=0, samples_seen=0, accumulation_steps=1)
    saved_rng = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
        old = checkpoint['metadata']
        if old['format_version'] != 'rfm_v1' or old['manifest_hashes'] != hashes:
            raise ValueError('Resume format or dataset mismatch')
        mutable = {'work_dir', 'max_train_steps', 'save_every_updates', 'validate_every_updates', 'log_every_updates'}
        if any(old['training_config'].get(key) != config.get(key)
               for key in set(old['training_config']) | set(config) if key not in mutable):
            raise ValueError('Resume changed training semantics')
        progress = dict(checkpoint['progress'])
        if (len(checkpoint['rank_rng']) != world or progress['optimizer_step'] != progress['consumed_batches']
                or progress['micro_step'] != progress['optimizer_step']
                or progress['samples_seen'] != progress['optimizer_step'] * world * config['batch_size']):
            raise ValueError('Invalid resume cursor')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        ema.load_state_dict(checkpoint['state_dict_ema'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        saved_rng = checkpoint['rank_rng'][rank]
        if rank == 0:
            (work / 'resume.json').write_text(json.dumps(dict(checkpoint=args.resume, **progress), indent=2))
        del checkpoint
    if config['max_train_steps'] <= progress['optimizer_step']:
        raise ValueError('Training budget must extend beyond the resume step')
    model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False,
                                    gradient_as_bucket_view=True, static_graph=True)
    sampler = BalancedFlowSampler(lengths, config['batch_size'], world, rank, config['seed'],
                                  progress['consumed_batches'], config['max_train_steps'])
    loader = DataLoader(CachedDataset(records), batch_sampler=sampler, num_workers=config['dataloader_workers'],
                        prefetch_factor=config['prefetch_factor'], persistent_workers=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(config['seed'] + rank), multiprocessing_context='spawn')
    vae = None
    if rank == 0:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(config['vae_path'], local_files_only=True).float().to(device).eval().requires_grad_(False)
        if not np.isclose(vae.config.scaling_factor, .18215):
            raise ValueError('VAE scale mismatch')
        (work / 'dataset.json').write_text(json.dumps(dict(counts=lengths, source_fraction=[.5, .5],
            batch_per_gpu=config['batch_size'], source_per_gpu=[config['batch_size']//2]*2,
            global_batch=world*config['batch_size'], sampler='independent_shuffled_source_cycles',
            cache_loading='dataloader_prefetch'), indent=2))
    dist.barrier()
    # Setup/validation must not consume the training RNG stream, including on resume.
    if saved_rng is not None:
        restore_rng(saved_rng)
    else:
        random.seed(config['seed'] + rank)
        np.random.seed(config['seed'] + rank)
        torch.manual_seed(config['seed'] + rank)
    torch.cuda.reset_peak_memory_stats()
    for mean, std, caption, mask in loader:
        started = time.monotonic()
        mean, std, caption, mask = [value.to(device, non_blocking=True) for value in (mean, std, caption, mask)]
        caption, mask = drop_to_empty_text(caption, mask, empty, empty_mask, config['class_dropout_prob'])
        clean = (mean + std * torch.randn_like(mean)) * .18215
        noise = torch.randn_like(clean)
        flow_time = torch.rand(len(clean), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            per_sample = flow_matching_loss(model, clean, noise, flow_time, caption, mask)
            loss = per_sample.mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], foreach=True)
        finite = (torch.isfinite(loss) & torch.isfinite(norm)).int()
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Nonfinite RFM loss or gradients')
        used_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        update_ema(ema, model.module, config['ema_decay'])
        for key in ('optimizer_step', 'micro_step', 'consumed_batches'):
            progress[key] += 1
        progress['samples_seen'] += world * config['batch_size']
        step = progress['optimizer_step']
        if step % config['log_every_updates'] == 0 or step == config['max_train_steps']:
            values = torch.stack([loss.detach(), norm.detach()])
            dist.all_reduce(values)
            bins = torch.zeros(5, 2, device=device)
            for index in range(5):
                selected = (flow_time >= index/5) & (flow_time < (index+1)/5)
                bins[index, 0] = per_sample.detach()[selected].sum()
                bins[index, 1] = selected.sum()
            dist.all_reduce(bins)
            memory = [None] * world
            dist.all_gather_object(memory, torch.cuda.max_memory_allocated() / 2**30)
            if rank == 0:
                report = dict(step=step, loss=float(values[0]/world), grad_norm=float(values[1]/world),
                    learning_rate=used_lr, samples_seen=progress['samples_seen'], peak_gpu_gib_per_rank=memory,
                    seconds_per_step=time.monotonic()-started, source_counts_per_update=[world*config['batch_size']//2]*2,
                    time_bins_sum_count=bins.tolist())
                with (work / 'train.jsonl').open('a') as handle:
                    handle.write(json.dumps(report) + '\n')
                print(json.dumps(report), flush=True)
        if step % config['validate_every_updates'] == 0 or step == config['max_train_steps']:
            state = rng_state()
            dist.barrier()
            if rank == 0:
                print(f'Validation {step}: {validate(model.module, ema, vae, text, config, step, device)}', flush=True)
            dist.barrier()
            restore_rng(state)
        if step % config['save_every_updates'] == 0 or step == config['max_train_steps']:
            states = [None] * world
            dist.all_gather_object(states, rng_state())
            if rank == 0:
                path = work / 'checkpoints' / f'update-{step:06d}.pth'
                save_training_checkpoint(path, model.module, ema, optimizer, scheduler, metadata, progress, states)
                (work / 'latest_checkpoint.txt').write_text(str(path) + '\n')
                print(f'Saved {path}', flush=True)
            dist.barrier()
    if rank == 0:
        (work / 'completed.json').write_text(json.dumps(progress, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
