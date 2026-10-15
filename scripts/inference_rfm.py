"""Text-only inference from a trained four-channel PixArt rectified-flow checkpoint."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import runpy

from PIL import Image
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusion.model.builder import build_model
from diffusion.rectified_flow import sample_rfm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Python config with resource paths and sampling settings')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompts', required=True, help='JSON list of case_id, prompt and seed objects')
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', choices=('raw', 'ema'), default='raw')
    parser.add_argument('--conditioning-cache')
    parser.add_argument('--vae-path')
    parser.add_argument('--cfg', type=float)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--method', choices=('heun', 'dopri5'))
    parser.add_argument('--atol', type=float)
    parser.add_argument('--rtol', type=float)
    args = parser.parse_args()
    runtime = runpy.run_path(args.config)['config'] if args.config else {}
    for name, key, default in [('cfg', 'cfg_scale', 2.), ('steps', 'sampling_steps', 40),
                               ('method', 'sampling_method', 'heun'),
                               ('atol', 'sampling_atol', 1e-6), ('rtol', 'sampling_rtol', 1e-3)]:
        if getattr(args, name) is None:
            setattr(args, name, runtime.get(key, default))
    if args.steps < 1 or args.atol <= 0 or args.rtol <= 0:
        raise ValueError('Sampling steps and tolerances must be positive')
    if args.method not in ('heun', 'dopri5'):
        raise ValueError('Supported sampling methods: heun, dopri5')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to overwrite existing results: {output}')
    entries = json.loads(Path(args.prompts).read_text())
    ids = [entry['case_id'] for entry in entries]
    if not entries or len(set(ids)) != len(ids) or any(Path(name).name != name or name in ('.', '..') for name in ids):
        raise ValueError('Each prompt requires a unique filename-safe case_id')
    if any(not entry['prompt'].strip() or not isinstance(entry['seed'], int) or entry['seed'] < 0 for entry in entries):
        raise ValueError('Each case requires a nonempty prompt and nonnegative integer seed')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    metadata = checkpoint['metadata']
    for key, expected in dict(prediction_type='flow_velocity', path='linear',
                              time_embedding='999*(1-t)', vae_scaling_factor=.18215).items():
        if metadata.get(key) != expected:
            raise ValueError(f'Incompatible RFM checkpoint convention: {key}')
    config = dict(metadata['training_config'])
    for key in ('model', 'image_size', 'prediction_type', 'path', 'time_embedding', 'out_channels', 'qk_norm'):
        if key in runtime and runtime[key] != config.get(key):
            raise ValueError(f'Config differs from checkpoint architecture/convention: {key}')
    for key in ('conditioning_cache', 'vae_path'):
        override = getattr(args, key)
        if override is not None:
            config[key] = override
        elif key in runtime:
            config[key] = runtime[key]
    if config['model'] != 'PixArt_XL_2' or config['qk_norm'] != 'rms_norm':
        raise ValueError('Expected the pretrained PixArt XL/2 RFM with QK RMSNorm')
    state_key = 'state_dict' if args.weights == 'raw' else 'state_dict_ema'
    state = checkpoint[state_key]
    if 'y_embedder.y_embedding' not in state or 'blocks.0.cross_attn.q_linear.weight' not in state:
        raise ValueError('Expected the original T2I model with text cross-attention')
    original_conditioning = torch.load(config['conditioning_cache'], map_location='cpu', weights_only=True)
    if not torch.equal(state['y_embedder.y_embedding'], original_conditioning['empty'][0, 0].float()):
        raise ValueError('Unconditional text embedding differs from the original training cache')
    snapshot = Path(config['vae_path']).parent
    tokenizer = AutoTokenizer.from_pretrained(snapshot / 'tokenizer', local_files_only=True)
    # The synthetic training cache used T5Embedder(use_text_preprocessing=False),
    # which lowercases and strips but does not perform native caption cleaning.
    prompts = [entry['prompt'].lower().strip() for entry in entries]
    token_counts = [len(tokenizer(prompt, add_special_tokens=True)['input_ids']) for prompt in prompts]
    if max(token_counts) > 120:
        raise ValueError(f'Prompts exceed the 120-token training limit: {token_counts}')
    tokens = tokenizer(prompts, max_length=120, padding='max_length', truncation=False,
                       return_tensors='pt').to('cuda')
    encoder = T5EncoderModel.from_pretrained(snapshot / 'text_encoder', local_files_only=True,
                                             torch_dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    with torch.inference_mode():
        condition = encoder(**tokens).last_hidden_state[:, None].clone()
    mask = tokens.attention_mask.bool()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(condition=condition.cpu(), mask=mask.cpu(), prompts=prompts,
                    t5_snapshot=str(snapshot), preprocessing='lowercase and strip',
                    token_counts=token_counts), output / 'prompt_embeddings.pt')
    del encoder, tokens
    gc.collect()
    torch.cuda.empty_cache()
    model = build_model(config['model'], input_size=64, pred_sigma=False, model_max_length=120,
                         class_dropout_prob=0., qk_norm='rms_norm', use_grad_checkpoint=False,
                         lewei_scale=1.).float()
    # Load the trained output head unchanged: initialization-time head reset must
    # never be repeated when sampling an already trained RFM checkpoint.
    model.load_state_dict(state, strict=True)
    model.cuda().eval().requires_grad_(False)
    empty = original_conditioning['empty'].cuda().bfloat16()
    empty_mask = original_conditioning['empty_mask'].cuda().bool()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(config['vae_path'], local_files_only=True).cuda().float().eval().requires_grad_(False)
    if abs(vae.config.scaling_factor - .18215) > 1e-6:
        raise ValueError('Unexpected VAE scaling factor')
    report = dict(model='PixArt_XL_2_T2I_RFM', weights=args.weights, state_key=state_key,
                   optimizer_step=checkpoint['progress']['optimizer_step'],
                   checkpoint=str(Path(args.checkpoint).resolve()), prediction_type='flow_velocity',
                   conditioning='text_only', semantic_maps_used=False, cfg_scale=args.cfg,
                   sampling_method=args.method, sampling_steps=args.steps if args.method == 'heun' else None,
                   sampling_atol=args.atol, sampling_rtol=args.rtol, resolution=[512, 512],
                   t5_snapshot=str(snapshot), text_preprocessing='lowercase and strip',
                   unconditional_source=config['conditioning_cache'], cases=[])
    print(json.dumps(dict(event='ready', token_counts=token_counts, weights=args.weights)), flush=True)
    with torch.inference_mode():
        for index, entry in enumerate(entries):
            noise = torch.randn(1, 4, 64, 64, device='cuda',
                                generator=torch.Generator(device='cuda').manual_seed(entry['seed']))
            latent, sampling = sample_rfm(model, noise, condition[index:index+1], mask[index:index+1],
                                           empty, empty_mask, cfg_scale=args.cfg, method=args.method, steps=args.steps,
                                           atol=args.atol, rtol=args.rtol)
            decoded = vae.decode(latent.float() / .18215).sample[0]
            if not torch.isfinite(decoded).all():
                raise FloatingPointError(f'Nonfinite decoded image: {entry["case_id"]}')
            pixels = ((decoded / 2 + .5).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(pixels).save(output / f'{entry["case_id"]}.png')
            report['cases'].append(dict(case_id=entry['case_id'], prompt=entry['prompt'],
                                        encoded_prompt=prompts[index], token_count=token_counts[index],
                                        seed=entry['seed'], image=f'{entry["case_id"]}.png',
                                        latent_rms=latent.square().mean().sqrt().item(), sampler=sampling))
            print(json.dumps(dict(event='image_saved', case_id=entry['case_id'])), flush=True)
    with Path(args.checkpoint).open('rb') as handle:
        report['checkpoint_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
