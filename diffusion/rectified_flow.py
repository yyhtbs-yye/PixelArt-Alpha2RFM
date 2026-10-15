"""Linear noise-to-data flow matching; no VP schedule or variance objective."""
import torch


def flow_matching_loss(model, clean, noise, time, condition, mask):
    if clean.shape != noise.shape or time.shape != (clean.shape[0],):
        raise ValueError('Noise and continuous time must match the data batch')
    fraction = time.float()[:, None, None, None]
    noisy = (1 - fraction) * noise.float() + fraction * clean.float()
    target = clean.float() - noise.float()
    # Preserve the pretrained embedding's scale/direction, not its VP schedule.
    prediction = model(noisy, 999 * (1 - time.float()), condition, mask=mask)
    if prediction.shape != clean.shape:
        raise ValueError('RFM must predict exactly four latent channels')
    return (prediction.float() - target).square().flatten(1).mean(1)


@torch.no_grad()
def sample_rfm(model, noise, condition, mask, empty, empty_mask, cfg_scale=4.,
               method='dopri5', steps=50, atol=1e-6, rtol=1e-3):
    """Integrate dx/dt=u from noise at t=0 to data at t=1 in FP32."""
    evaluations = 0
    condition = torch.cat([empty.expand_as(condition), condition])
    mask = torch.cat([empty_mask.expand_as(mask), mask])

    def velocity(time, latent):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 10000:
            raise RuntimeError('ODE exceeded 10000 evaluations; inspect model stability')
        inputs = torch.cat([latent, latent])
        # As in SiT's ODE wrapper, retain continuous RK evaluation times.
        timestep = (999 * (1 - time)).expand(len(inputs)).float()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=latent.is_cuda):
            output = model(inputs, timestep, condition, mask=mask).float()
        if output.shape != inputs.shape:
            raise ValueError('RFM sampler received a non-flow output head')
        unconditional, conditional = output.chunk(2)
        result = unconditional + cfg_scale * (conditional - unconditional)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite flow velocity')
        return result

    latent = noise.float().clone()
    grid = torch.linspace(0, 1, steps + 1, device=noise.device)
    if method == 'heun':
        for start, end in zip(grid[:-1], grid[1:]):
            delta = end - start
            first = velocity(start, latent)
            second = velocity(end, latent + delta * first)
            latent = latent + delta * (first + second) / 2
    elif method == 'dopri5':
        from torchdiffeq import odeint
        latent = odeint(velocity, latent, grid[[0, -1]], method='dopri5',
                        atol=atol, rtol=rtol)[-1]
    else:
        raise ValueError('Supported RFM solvers: dopri5, heun')
    if not torch.isfinite(latent).all():
        raise FloatingPointError('Nonfinite integrated latent')
    return latent, dict(method=method, nfe=evaluations, cfg_scale=cfg_scale,
                        atol=atol, rtol=rtol, steps=steps if method == 'heun' else None,
                        direction='noise_to_data', time_embedding='999*(1-t)')
