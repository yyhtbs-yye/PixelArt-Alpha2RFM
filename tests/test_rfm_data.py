import numpy as np
import pytest
import torch

from datasets.rfm_data import CachedDataset, read_cache


@pytest.fixture
def cache_record(tmp_path):
    np.savez(tmp_path / 'latent.npz', latent=np.ones((4, 64, 64), dtype=np.float32),
             std=np.full((4, 64, 64), .5, dtype=np.float32), scaled=False, scaling_factor=.18215)
    np.savez(tmp_path / 'caption.npz', caption_feature=np.ones((1, 3, 4096), dtype=np.float32),
             attention_mask=np.ones((1, 3), dtype=np.int64))
    return tmp_path, dict(file_name='example.png', latent_file='latent.npz',
                          t5_file='caption.npz', t5_length=3)


def test_cached_dataset_preserves_unscaled_posterior_and_pads_text(cache_record):
    mean, std, caption, mask = CachedDataset([cache_record])[0]
    assert mean.shape == std.shape == (4, 64, 64)
    assert mean.dtype == std.dtype == torch.float32
    assert torch.all(mean == 1) and torch.all(std == .5)
    assert caption.shape == (1, 120, 4096) and caption.dtype == torch.bfloat16
    assert torch.all(caption[:, :3] == 1) and torch.all(caption[:, 3:] == 0)
    assert mask.dtype == torch.bool and mask.shape == (120,)
    assert mask[:3].all() and not mask[3:].any()


@pytest.mark.parametrize('invalid', ['scaled', 'negative_std', 'non_right_padding'])
def test_cache_rejects_incompatible_records(cache_record, invalid):
    root, row = cache_record
    path = root / row['t5_file' if invalid == 'non_right_padding' else 'latent_file']
    with np.load(path) as original:
        data = dict(original)
    if invalid == 'scaled':
        data['scaled'] = True
    elif invalid == 'negative_std':
        data['std'][0, 0, 0] = -.1
    else:
        data['attention_mask'] = np.array([[1, 0, 1]])
    np.savez(path, **data)
    with pytest.raises(ValueError):
        read_cache(cache_record)
