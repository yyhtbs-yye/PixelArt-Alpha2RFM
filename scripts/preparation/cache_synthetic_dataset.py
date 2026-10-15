"""Cache synthetic image posteriors and original generation-prompt T5 features.

Each worker owns a disjoint shard. Features retain only unmasked tokens; masked
padding is reconstructed during training without changing attention outputs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'output/t5_dependencies'))
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL
from diffusion.model.t5 import T5Embedder
from scripts.preparation.encode_image_latents import load_image, save_latent, verify_latent


def save_caption(job):
    path, feature = job
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, caption_feature=feature,
                            attention_mask=np.ones((1, feature.shape[1]), dtype=np.int64))
    temporary.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/synth_images/pixelart_synth512')
    parser.add_argument('--output', default='output/datasets_512/PixelArtSynth')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.batch_size < 1 or args.limit < 0:
        raise ValueError('Invalid shard, batch size or limit')
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    snapshot = Path.home()/'.cache/huggingface/hub/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots/50f702106901db6d0f8b67eb88e814c56ded2692'
    content = (source/'metadata.jsonl').read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    if len({row['file_name'] for row in rows}) != len(rows):
        raise ValueError('Duplicate source filenames')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('latents', 't5_embeddings'):
        (output/name).mkdir(exist_ok=True)
    settings = dict(source=str(source), source_metadata_sha256=hashlib.sha256(content).hexdigest(),
                    snapshot=str(snapshot), shards=args.shards, count=len(rows),
                    latent_dtype='float32', latent_scaled=False, scaling_factor=.18215,
                    t5_inference_dtype='bfloat16', t5_storage_dtype='float32', max_length=120,
                    prompt_field='model_prompt', text_preprocessing=False,
                    caption_storage='unmasked_tokens_only', vae_tf32=False, cache_format_version=1)
    settings_path = output/'cache_settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError('Existing cache settings differ; use a new output directory')
    else:
        # Same immutable settings in every shard; atomic publication.
        temporary = output/f'settings-{args.shard}.tmp'
        temporary.write_text(json.dumps(settings, indent=2)+'\n')
        temporary.replace(settings_path)
    if args.finalize:
        completed = []
        for shard in range(args.shards):
            summary = json.loads((output/f'completed-{shard}.json').read_text())
            if summary['settings'] != settings:
                raise ValueError('Shard settings mismatch')
            completed.extend(json.loads(line) for line in (output/f'metadata-{shard}.jsonl').read_text().splitlines())
        completed.sort(key=lambda row: row['index'])
        if [row['index'] for row in completed] != [row['index'] for row in rows]:
            raise ValueError('Missing, duplicate or unexpected records')
        if any(not (output/row[key]).is_file() for row in completed for key in ('latent_file','t5_file')):
            raise ValueError('Missing cache files')
        temporary = output/'training_metadata.tmp'
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in completed))
        temporary.replace(output/'training_metadata.jsonl')
        summary = dict(settings, complete=True, images=len(completed),
                       total_caption_tokens=sum(row['t5_length'] for row in completed))
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)
        sys.exit(0)
    assigned = rows[args.shard::args.shards]
    for row in assigned:
        if row['width'] != 512 or row['height'] != 512 or not isinstance(row['model_prompt'], str):
            raise ValueError(f'Invalid source row: {row["index"]}')
        row['file_name'] = str(source/row['file_name'])
        row['prompt'] = row['model_prompt']
        row['latent_file'] = f'latents/{row["index"]:06d}.npz'
        row['t5_file'] = f't5_embeddings/{row["index"]:06d}.npz'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['HF_HUB_OFFLINE'] = '1'
    vae = AutoencoderKL.from_pretrained(snapshot/'vae', local_files_only=True,
        torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    text = T5Embedder('cuda', dir_or_name=snapshot.name, cache_dir=str(snapshot.parent),
                     local_cache=True, torch_dtype=torch.bfloat16, use_text_preprocessing=False)
    text.model.eval().requires_grad_(False)
    started = time.monotonic()
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=8) as pool:
        with (output/f'metadata-{args.shard}.jsonl').open('w') as journal:
            for start in range(0,len(assigned),args.batch_size):
                batch = assigned[start:start+args.batch_size]
                pending = [row for row in batch if not (output/row['latent_file']).exists()]
                if pending:
                    pixels = torch.from_numpy(np.stack(list(pool.map(load_image, [row['file_name'] for row in pending])))).cuda()
                    posterior = vae.encode(pixels).latent_dist
                    means, stds = posterior.mean.cpu().numpy(), posterior.std.cpu().numpy()
                    if not np.isfinite(means).all() or not np.isfinite(stds).all() or not (stds > 0).all():
                        raise ValueError('Invalid VAE posterior')
                    list(pool.map(save_latent, [(output/row['latent_file'], mean, std, .18215)
                        for row,mean,std in zip(pending,means,stds)]))
                    del pixels, posterior
                pending = [row for row in batch if not (output/row['t5_file']).exists()]
                if pending:
                    feature, mask = text.get_text_embeddings([row['prompt'] for row in pending])
                    feature, mask = feature.float().cpu().numpy(), mask.cpu().numpy()
                    if not np.isfinite(feature).all():
                        raise ValueError('Nonfinite caption features')
                    jobs = [(output/row['t5_file'], feature[i:i+1,:int(mask[i].sum())]) for i,row in enumerate(pending)]
                    list(pool.map(save_caption, jobs))
                # Verify saved or resumed files before publishing their manifest records.
                for row in batch:
                    verify_latent(output/row['latent_file'])
                    with np.load(output/row['t5_file'], allow_pickle=False) as cached:
                        value, attention = cached['caption_feature'], cached['attention_mask']
                        length = value.shape[1]
                        if value.shape != (1,length,4096) or not 1 <= length <= 120 or value.dtype != np.float32:
                            raise ValueError('Invalid caption shape or dtype')
                        if not np.isfinite(value).all() or attention.shape != (1,length) or not (attention == 1).all():
                            raise ValueError('Invalid cached caption values/mask')
                    row['t5_length'] = length
                    journal.write(json.dumps(row, ensure_ascii=False)+'\n')
                journal.flush()
                saved = start+len(batch)
                if saved % (args.batch_size*10) == 0 or saved == len(assigned):
                    elapsed = time.monotonic()-started
                    progress = dict(saved=saved,total=len(assigned),seconds=elapsed,images_per_second=saved/elapsed)
                    (output/f'progress-{args.shard}.json').write_text(json.dumps(progress)+'\n')
                    print(f'Shard {args.shard}: {progress}',flush=True)
    (output/f'completed-{args.shard}.json').write_text(json.dumps(dict(settings=settings, count=len(assigned)))+'\n')
