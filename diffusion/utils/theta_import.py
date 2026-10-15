"""Strict Diffusers Theta RAW -> native PixArt parameter mapping."""
import hashlib
import importlib.metadata
import subprocess

import numpy as np
import torch


def import_theta_state(model, source, empty_embedding):
    source = dict(source)
    state = {}
    names = {
        'x_embedder.proj': 'pos_embed.proj',
        'y_embedder.y_proj.fc1': 'caption_projection.linear_1',
        'y_embedder.y_proj.fc2': 'caption_projection.linear_2',
        't_embedder.mlp.0': 'adaln_single.emb.timestep_embedder.linear_1',
        't_embedder.mlp.2': 'adaln_single.emb.timestep_embedder.linear_2',
        't_block.1': 'adaln_single.linear',
        'final_layer.linear': 'proj_out',
    }
    for target, original in names.items():
        for suffix in ('weight', 'bias'):
            state[f'{target}.{suffix}'] = source.pop(f'{original}.{suffix}')
    state['final_layer.scale_shift_table'] = source.pop('scale_shift_table')
    for index in range(len(model.blocks)):
        target = f'blocks.{index}'
        original = f'transformer_blocks.{index}'
        state[f'{target}.scale_shift_table'] = source.pop(f'{original}.scale_shift_table')
        for suffix in ('weight', 'bias'):
            state[f'{target}.attn.qkv.{suffix}'] = torch.cat([
                source.pop(f'{original}.attn1.to_{letter}.{suffix}') for letter in ('q', 'k', 'v')])
            state[f'{target}.cross_attn.kv_linear.{suffix}'] = torch.cat([
                source.pop(f'{original}.attn2.to_{letter}.{suffix}') for letter in ('k', 'v')])
        layers = {'attn.proj': 'attn1.to_out.0', 'cross_attn.q_linear': 'attn2.to_q',
                  'cross_attn.proj': 'attn2.to_out.0', 'mlp.fc1': 'ff.net.0.proj',
                  'mlp.fc2': 'ff.net.2'}
        for new_name, old_name in layers.items():
            for suffix in ('weight', 'bias'):
                state[f'{target}.{new_name}.{suffix}'] = source.pop(f'{original}.{old_name}.{suffix}')
        for new_name, old_name in [('attn', 'attn1'), ('cross_attn', 'attn2')]:
            for letter in ('q', 'k'):
                state[f'{target}.{new_name}.{letter}_norm.weight'] = source.pop(
                    f'{original}.{old_name}.norm_{letter}.weight')
    if source:
        raise ValueError(f'Unmapped source tensors: {sorted(source)}')
    state['pos_embed'] = model.pos_embed.detach().clone()
    state['y_embedder.y_embedding'] = empty_embedding.float().reshape_as(model.y_embedder.y_embedding)
    model.load_state_dict(state, strict=True)


def make_initial_metadata(config, scaling_factor, model, manifest_hashes=None):
    if config['prediction_type'] != 'epsilon':
        raise ValueError('Initialization requires the original epsilon checkpoint')
    # Preserve the source checkpoint metadata without importing a VP sampler.
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    versions = {}
    for package in ('torch', 'numpy', 'accelerate', 'diffusers', 'transformers', 'mmcv', 'timm', 'xformers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(
        format_version=1, prediction_type=config['prediction_type'],
        validation_prediction_path='epsilon_to_x0',
        schedule=dict(name='linear', steps=1000, beta_sha256=hashlib.sha256(betas.tobytes()).hexdigest()),
        vae_scaling_factor=float(scaling_factor), latent_cache_scaled=False,
        architecture={key: list(value.shape) for key, value in model.state_dict().items()},
        source_model=config['source_model'], source_prediction_type=config.get('source_prediction_type', 'epsilon'),
        source_weights=config.get('source_weights', 'raw'),
        source_optimizer_step=config.get('source_optimizer_step'),
        source_checkpoint_sha256=config.get('source_checkpoint_sha256'),
        source_checkpoint=config.get('init_checkpoint'), training_config=dict(config),
        versions=versions, manifest_hashes=manifest_hashes,
        git_commit=revision.stdout.strip() if revision.returncode == 0 else None,
    )
