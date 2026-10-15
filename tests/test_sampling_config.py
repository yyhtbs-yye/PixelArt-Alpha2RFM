import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest
import torch

from scripts import inference_rfm, validate_rfm


@pytest.mark.parametrize('module', [validate_rfm, inference_rfm])
@pytest.mark.parametrize('override', [False, True])
def test_runtime_config_paths_and_cli_override_reach_conditioning_load(tmp_path, monkeypatch, module, override):
    original = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'configs/rfm.py'))['config']
    original['conditioning_cache'] = 'original/location.pt'
    runtime_cache = tmp_path / 'runtime.pt'
    cli_cache = tmp_path / 'cli.pt'
    for path in (runtime_cache, cli_cache):
        path.write_bytes(b'original conditioning bytes')
    runtime_path = tmp_path / 'config.py'
    runtime_path.write_text(f'config = dict(conditioning_cache={str(runtime_cache)!r}, vae_path="relocated/vae")\n')
    checkpoint = dict(metadata=dict(
        format_version='rfm_v1', prediction_type='flow_velocity', path='linear',
        time_embedding='999*(1-t)', vae_scaling_factor=.18215, training_config=original,
        manifest_hashes={'original/location.pt': hashlib.sha256(runtime_cache.read_bytes()).hexdigest()}),
        state_dict={'y_embedder.y_embedding': torch.zeros(1), 'blocks.0.cross_attn.q_linear.weight': torch.zeros(1)})
    calls = []

    def load(path, **kwargs):
        calls.append(str(path))
        if len(calls) == 1:
            return checkpoint
        raise RuntimeError('conditioning load reached')

    monkeypatch.setattr(torch, 'load', load)
    arguments = ['program', '--config', str(runtime_path), '--checkpoint', 'model.pth',
                 '--output', str(tmp_path / 'output')]
    if module is inference_rfm:
        prompts = tmp_path / 'prompts.json'
        prompts.write_text(json.dumps([dict(case_id='test', prompt='A red panda', seed=42)]))
        arguments.extend(['--prompts', str(prompts)])
    if override:
        arguments.extend(['--conditioning-cache', str(cli_cache)])
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(RuntimeError, match='conditioning load reached'):
        module.main()
    assert calls == ['model.pth', str(cli_cache if override else runtime_cache)]
    assert original['conditioning_cache'] == 'original/location.pt'


@pytest.mark.parametrize('module', [validate_rfm, inference_rfm])
def test_runtime_config_cannot_change_checkpoint_architecture(tmp_path, monkeypatch, module):
    config = tmp_path / 'config.py'
    config.write_text("config = dict(model='incompatible-model')\n")
    checkpoint = dict(metadata=dict(
        format_version='rfm_v1', prediction_type='flow_velocity', path='linear',
        time_embedding='999*(1-t)', vae_scaling_factor=.18215,
        training_config=dict(model='PixArt_XL_2', conditioning_cache='cache.pt'),
        manifest_hashes={'cache.pt': 'unused'}))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: checkpoint)
    arguments = ['program', '--config', str(config), '--checkpoint', 'model.pth', '--output', str(tmp_path / 'output')]
    if module is inference_rfm:
        prompts = tmp_path / 'prompts.json'
        prompts.write_text(json.dumps([dict(case_id='test', prompt='A red panda', seed=42)]))
        arguments.extend(['--prompts', str(prompts)])
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(ValueError, match='architecture/convention: model'):
        module.main()
