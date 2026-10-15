"""Cached latent/text loading and balanced, resumable RFM batches."""
import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def read_cache(record):
    root, row = record
    with np.load(root / row['latent_file'], allow_pickle=False) as data:
        mean, std = data['latent'], data['std']
        if bool(data['scaled']) or not np.isclose(float(data['scaling_factor']), 0.18215):
            raise ValueError(f'Unexpected latent scaling: {row["latent_file"]}')
    with np.load(root / row['t5_file'], allow_pickle=False) as data:
        feature, mask = data['caption_feature'], data['attention_mask']
    length = feature.shape[1] if feature.ndim == 3 else 0
    if mean.shape != (4, 64, 64) or std.shape != mean.shape or feature.shape != (1, length, 4096) or not 1 <= length <= 120 or mask.shape != (1, length):
        raise ValueError(f'Unexpected cache shape: {row["file_name"]}')
    if not all(np.isfinite(value).all() for value in (mean, std, feature)) or np.any(std < 0):
        raise ValueError(f'Invalid cached values: {row["file_name"]}')
    if not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError(f'Invalid caption mask: {row["file_name"]}')
    if np.any(np.diff(mask[0].astype(np.int64)) > 0):
        raise ValueError('Caption masks must use right padding')
    expected_length = row.get('t5_length', min(row.get('t5_token_count', 120), 120))
    if ('t5_length' in row or 't5_token_count' in row) and int(mask.sum()) != expected_length:
        raise ValueError('Caption length differs from manifest')
    if length < 120:
        feature = np.pad(feature, ((0, 0), (0, 120-length), (0, 0)))
        mask = np.pad(mask, ((0, 0), (0, 120-length)))
    return mean, std, feature, mask[0]


class CachedDataset(Dataset):
    """Read only requested NPZ records; workers prepare tensors in host memory."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        mean, std, caption, mask = read_cache(self.records[index])
        return (torch.from_numpy(mean).float(), torch.from_numpy(std).float(),
                torch.from_numpy(caption).to(torch.bfloat16), torch.from_numpy(mask).bool())


class BalancedFlowSampler(Sampler):
    def __init__(self, lengths, batch_size, world_size, rank, seed, start_step, stop_step):
        if len(lengths) != 2 or batch_size % 2 or min(lengths) < world_size * batch_size // 2:
            raise ValueError('Two sources need at least one global half-batch each')
        self.lengths = lengths
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.start_step = start_step
        self.stop_step = stop_step

    def __iter__(self):
        half = self.batch_size // 2
        global_half = half * self.world_size
        cycles = [-1, -1]
        orders = [None, None]
        for step in range(self.start_step, self.stop_step):
            batch = []
            offset = 0
            for source, length in enumerate(self.lengths):
                batches = math.ceil(length / global_half)
                cycle, position = divmod(step, batches)
                if cycles[source] != cycle:
                    generator = torch.Generator().manual_seed(self.seed + source * 1000000007 + cycle)
                    order = torch.randperm(length, generator=generator)
                    # Pad only the final global half-batch; no source is truncated.
                    padding = batches * global_half - length
                    orders[source] = torch.cat([order, order[:padding]])
                    cycles[source] = cycle
                begin = position * global_half + self.rank * half
                batch.extend((orders[source][begin:begin + half] + offset).tolist())
                offset += length
            yield batch

    def __len__(self):
        return self.stop_step - self.start_step
