"""PixArt XL/2 RFM: 512px, balanced 50/50 data, 0–100K updates.

Paths are relative to the repository root.
"""
config = dict(
    model='PixArt_XL_2', image_size=512, prediction_type='flow_velocity',
    path='linear', time_sampling='uniform_continuous', time_embedding='999*(1-t)',
    loss='unweighted_mse', out_channels=4, head_initialization='zero',
    qk_norm='rms_norm', grad_checkpointing=False, mixed_precision='bf16',
    source_model='assets/source_model',
    source_weights='raw',
    source_checkpoint_sha256='fecb75bebf36bcadc3e1ce97ffecf2739482d3f0fe5e61ee33b91ccf30a2de02',
    init_checkpoint='output/theta_raw_to_v_setup/initial_raw_epsilon.pth',
    conditioning_cache='output/theta_raw_to_v_setup/conditioning.pt',
    vae_path='assets/pixart/vae',
    source_datasets=['data/synth_images/flux1_dev_localized_narratives_synth512',
                     'data/synth_images/flux1_dev_synth512'],
    source_counts=[50000, 76708], source_weights_per_batch=[0.5, 0.5],
    data_roots=['output/datasets_512/FluxLocalizedNarratives', 'output/datasets_512/FluxSynth'],
    world_size=4, batch_size=8, accumulation_steps=1,
    dataloader_workers=4, prefetch_factor=2,
    learning_rate=1e-5, betas=(0.9, 0.999), adam_eps=1e-8, weight_decay=0.,
    warmup_updates=500, max_grad_norm=1., ema_decay=0.9999, class_dropout_prob=0.1,
    max_train_steps=100000, save_every_updates=5000, validate_every_updates=500,
    log_every_updates=10, seed=43,
    sampling_method='dopri5', sampling_steps=50, sampling_atol=1e-6, sampling_rtol=1e-3,
    cfg_scale=4., validation_seeds=[42, 43, 44, 45],
    work_dir='output/rfm_mix50',
)
