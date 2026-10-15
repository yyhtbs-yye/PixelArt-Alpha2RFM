"""Run the original RAW/EMA validation from an existing RFM checkpoint."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy

import torch
from diffusers import AutoencoderKL

from diffusion.model.builder import build_model
from scripts.train_rfm import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Python config with resource paths and validation sampling settings')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conditioning-cache')
    parser.add_argument('--vae-path')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {output}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    metadata = checkpoint['metadata']
    for key, expected in dict(format_version='rfm_v1', prediction_type='flow_velocity',
                              path='linear', time_embedding='999*(1-t)',
                              vae_scaling_factor=.18215).items():
        if metadata.get(key) != expected:
            raise ValueError(f'Incompatible RFM checkpoint: {key}')
    config = dict(metadata['training_config'])
    conditioning_hash = metadata['manifest_hashes'][config['conditioning_cache']]
    if args.config:
        runtime = runpy.run_path(args.config)['config']
        for key in ('model', 'image_size', 'prediction_type', 'path', 'time_embedding', 'out_channels', 'qk_norm'):
            if key in runtime and runtime[key] != config.get(key):
                raise ValueError(f'Config differs from checkpoint architecture/convention: {key}')
        for key in ('conditioning_cache', 'vae_path', 'cfg_scale', 'sampling_method',
                    'sampling_steps', 'sampling_atol', 'sampling_rtol'):
            if key in runtime:
                config[key] = runtime[key]
    config['work_dir'] = str(output)
    if args.conditioning_cache:
        config['conditioning_cache'] = args.conditioning_cache
    if args.vae_path:
        config['vae_path'] = args.vae_path
    if hashlib.sha256(Path(config['conditioning_cache']).read_bytes()).hexdigest() != conditioning_hash:
        raise ValueError('Conditioning cache differs from the original training cache')
    text = torch.load(config['conditioning_cache'], map_location='cpu', weights_only=True)
    if text['seeds'] != config['validation_seeds'] or len(text['prompts']) != 4:
        raise ValueError('Conditioning prompts/seeds do not match the training configuration')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(config['model'], input_size=64, pred_sigma=False, model_max_length=120,
                        class_dropout_prob=0., qk_norm=config['qk_norm'],
                        use_grad_checkpoint=False, lewei_scale=1.).float()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    ema = deepcopy(model)
    ema.load_state_dict(checkpoint['state_dict_ema'], strict=True)
    for network in (model, ema):
        if not torch.equal(network.y_embedder.y_embedding, text['empty'][0, 0].float()):
            raise ValueError('Unconditional embedding differs from the training cache')
        network.cuda().eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(config['vae_path'], local_files_only=True)
    vae = vae.cuda().float().eval().requires_grad_(False)
    if abs(vae.config.scaling_factor - .18215) > 1e-6:
        raise ValueError('Unexpected VAE scaling factor')
    step = checkpoint['progress']['optimizer_step']
    del checkpoint
    report = validate(model, ema, vae, text, config, step, torch.device('cuda'))
    (output / 'validation_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
