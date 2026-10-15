"""RFM checkpoint serialization, random state, EMA and text dropout."""
import random
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def update_ema(ema, model, decay):
    for destination, source in zip(ema.parameters(), model.parameters()):
        destination.mul_(decay).add_(source.detach().float(), alpha=1 - decay)
    for destination, source in zip(ema.buffers(), model.buffers()):
        destination.copy_(source)


def drop_to_empty_text(caption, mask, empty, empty_mask, probability):
    dropped = torch.rand(caption.shape[0], device=caption.device) < probability
    return (torch.where(dropped[:, None, None, None], empty, caption),
            torch.where(dropped[:, None], empty_mask, mask))


def save_training_checkpoint(path, model, ema, optimizer, scheduler, metadata, progress, rank_rng):
    if progress['micro_step'] % progress['accumulation_steps']:
        raise ValueError('Checkpoint must be saved at a complete optimizer update boundary')
    metadata = dict(metadata, optimizer_step=progress['optimizer_step'], samples_seen=progress['samples_seen'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(state_dict=model.state_dict(), state_dict_ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    metadata=metadata, progress=progress, rank_rng=rank_rng), temporary)
    temporary.replace(path)
