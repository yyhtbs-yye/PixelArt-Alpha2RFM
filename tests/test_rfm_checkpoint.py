from copy import deepcopy
import hashlib
import random

import numpy as np
import pytest
import torch

from diffusion.utils.rfm_checkpoint import rng_state, restore_rng, update_ema, save_training_checkpoint
from diffusion.utils.theta_import import make_initial_metadata


def test_checkpoint_restores_the_next_update_and_all_random_streams(tmp_path):
    model = torch.nn.Linear(2, 2)
    model.register_buffer('updates', torch.tensor(0))
    ema = deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    loss = model(torch.randn(3, 2)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.updates.add_(1)
    update_ema(ema, model, .9)
    progress = dict(micro_step=1, optimizer_step=1, consumed_batches=1,
                    accumulation_steps=1, samples_seen=3)
    metadata = dict(format_version='rfm_v1', prediction_type='flow_velocity',
                    path='linear', time_embedding='999*(1-t)', vae_scaling_factor=.18215)
    path = tmp_path / 'checkpoints/update-000001.pth'
    save_training_checkpoint(path, model, ema, optimizer, scheduler, metadata, progress, [rng_state()])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['progress'] == progress
    assert checkpoint['metadata'] == dict(metadata, optimizer_step=1, samples_seen=3)
    assert not path.with_suffix('.tmp').exists()

    results = []
    for resumed in (False, True):
        if resumed:
            model.load_state_dict(checkpoint['state_dict'])
            ema.load_state_dict(checkpoint['state_dict_ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            restore_rng(checkpoint['rank_rng'][0])
        inputs = torch.randn(3, 2) * (random.random() + np.random.rand())
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
        scheduler.step()
        model.updates.add_(1)
        previous_ema = ema.weight.detach().clone()
        update_ema(ema, model, .9)
        torch.testing.assert_close(ema.weight, previous_ema * .9 + model.weight * .1)
        assert torch.equal(ema.updates, model.updates)
        results.append(deepcopy(dict(model=model.state_dict(), ema=ema.state_dict(),
                                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())))
    for key in ('model', 'ema'):
        for name, value in results[0][key].items():
            assert torch.equal(value, results[1][key][name])
    for index, state in results[0]['optimizer']['state'].items():
        for name, value in state.items():
            assert torch.equal(value, results[1]['optimizer']['state'][index][name])
    assert results[0]['scheduler'] == results[1]['scheduler']


def test_checkpoint_rejects_incomplete_updates(tmp_path):
    with pytest.raises(ValueError, match='optimizer update boundary'):
        save_training_checkpoint(tmp_path / 'incomplete.pth', None, None, None, None, {},
                                 dict(micro_step=1, accumulation_steps=2), [])
    assert not (tmp_path / 'incomplete.pth').exists()


def test_initial_metadata_without_git_history_preserves_source_schedule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = dict(prediction_type='epsilon', source_model='assets/source_model', source_weights='raw')
    metadata = make_initial_metadata(config, .18215, torch.nn.Linear(2, 2))
    assert metadata['prediction_type'] == 'epsilon'
    assert metadata['validation_prediction_path'] == 'epsilon_to_x0'
    assert metadata['git_commit'] is None
    assert metadata['schedule'] == dict(name='linear', steps=1000,
        beta_sha256=hashlib.sha256(np.linspace(.0001, .02, 1000, dtype=np.float64).tobytes()).hexdigest())
