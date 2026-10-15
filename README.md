# pixelart-alpha2rfm

A standalone repository for text-conditioned Rectified Flow Matching (RFM) with
PixArt Alpha. The model starts from pretrained RAW epsilon-prediction weights and
trains for 100,000 optimizer updates, from step 0 to step 100K.

This repository contains the model, dataset definitions, training, preparation,
validation, and inference code. `scripts/train_rfm.py` is the single training
entry point. All project code is local to this repository; datasets, caches,
checkpoints, and pretrained VAE/T5 assets must be supplied separately.

## Experiment configuration

The experiment is defined in [configs/rfm.py](configs/rfm.py).

| Setting | Value |
| --- | --- |
| Model | PixArt XL/2, 512 × 512, QK RMSNorm |
| Initialization | RAW epsilon weights from `A_flux1_dev_mix50/checkpoint-010000` |
| Output head | Four channels; zero-initialized for new training |
| Flow path | `x(t) = (1 - t) * noise + t * data` |
| Target and loss | `data - noise`, unweighted MSE |
| Time sampling | Continuous uniform sampling; model time embedding `999 * (1 - t)` |
| Datasets | FluxLocalizedNarratives: 50,000 images; FluxSynth: 76,708 images |
| Data mixture | Exactly four samples from each dataset per GPU batch |
| Batch size | Four GPUs × eight samples; global batch size 32; no gradient accumulation |
| Precision | BF16; gradient checkpointing disabled |
| Optimizer | AdamW, learning rate `1e-5`, 500 warmup updates, zero weight decay |
| EMA decay | `0.9999` |
| Text dropout | `0.1` |
| Training duration | 100,000 optimizer updates, from step 0 to step 100K |
| Validation | RAW and EMA weights; `dopri5`, CFG 4, `atol=1e-6`, `rtol=1e-3` |
| Validation seeds | 42, 43, 44, 45 |

The trained output head is restored unchanged when resuming or sampling.
`dopri5` uses adaptive steps: `sampling_steps=50` does not impose 50 sampling
steps. Each validation run records its sampler settings and number of function
evaluations in `sampler.json`.

## Repository layout

```text
run.sh                                # Bash launcher for train, validate, and inference
configs/
  rfm.py                              # Experiment settings and resource paths
examples/
  validation_prompts.json             # Four fixed validation prompts and seeds
  inference_prompts.json              # Example custom prompts with case IDs
datasets/
  rfm_data.py                         # CachedDataset, cache loading, and balanced sampling
diffusion/
  rectified_flow.py                   # RFM loss and Heun/dopri5 sampling
  model/                              # PixArt XL/2 and T5 text encoding
  utils/rfm_checkpoint.py             # Checkpoints, EMA, RNG state, and text dropout
  utils/theta_import.py               # Pretrained weight conversion and provenance
scripts/
  train_rfm.py                        # Training, resume, validation, and optional background launch
  validate_rfm.py                     # RAW/EMA validation from a checkpoint
  inference_rfm.py                    # Custom text-to-image inference
  preparation/
    cache_synthetic_dataset.py        # VAE posterior and T5 embedding caches
    encode_image_latents.py           # Image preprocessing and latent caching
    prepare_theta_v.py                # Initialization weights and fixed text conditioning
    verify_rfm_smoke.py               # Continuous-training versus resume verification
tests/                               # Loss, sampling, data, entry point, and checkpoint checks
```

Dataset definitions and runtime data loading live in `datasets/`. Offline asset
preparation tools live in `scripts/preparation/`.

## Installation

The observed runtime used Linux, Python 3.12.13, and PyTorch `2.9.1+cu129`.
[requirements.txt](requirements.txt) records the main dependency versions observed
during extraction; it is not a complete lockfile of the original training
environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
```

Use a CUDA driver and GPU compatible with the selected PyTorch/xformers build.
The default training configuration uses four GPUs with BF16 support. An existing
compatible PixArt environment can also be used.

Run the following commands from the repository root unless stated otherwise.

## Bash launcher

```text
bash run.sh <train|validate|inference> <config.py> [arguments...]
```

The configuration path is required. After preparing the resources described
below, use:

```bash
# Train using config['world_size'] distributed workers.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train configs/rfm.py

# Validate both RAW and EMA weights.
CUDA_VISIBLE_DEVICES=0 bash run.sh validate configs/rfm.py \
  --checkpoint output/rfm_mix50/checkpoints/update-100000.pth \
  --output output/validation-100000

# Generate images from custom prompts using EMA weights.
CUDA_VISIBLE_DEVICES=0 bash run.sh inference configs/rfm.py \
  --checkpoint output/rfm_mix50/checkpoints/update-100000.pth \
  --prompts examples/inference_prompts.json \
  --weights ema --output output/inference-100000
```

The launcher changes to the repository root, so all relative paths are resolved
from there. Set `PYTHON_BIN` to select an interpreter:

```bash
PYTHON_BIN=/path/to/env/bin/python bash run.sh train configs/rfm.py
```

`CUDA_VISIBLE_DEVICES` is passed through unchanged. Validation and inference use
the first visible GPU. Training arguments such as `--resume` and `--work-dir` are
forwarded to the trainer. `bash run.sh train configs/rfm.py --help` displays help
without starting distributed workers.

Validation and inference read resource paths and sampling settings from the
specified configuration. Its model architecture and flow conventions must match
the checkpoint. Use `--conditioning-cache` and `--vae-path` to override resource
paths. Both modes require a new or empty output directory.

Validation uses the checkpoint's fixed prompts and seeds and verifies the
conditioning cache against its recorded SHA-256 hash. Inference reads prompts
and seeds from `--prompts`; `--cfg`, `--steps`, `--method`, `--atol`, and `--rtol`
override the sampling configuration. `--steps` affects Heun only.

## Prepare external resources

The default configuration expects this layout. Resources may be copied into the
repository or provided through symbolic links to existing storage.

```text
assets/source_model/                  # Pretrained checkpoint with transformer/ and scheduler/
assets/pixart/                        # Model snapshot with vae/, tokenizer/, and text_encoder/
data/synth_images/flux1_dev_localized_narratives_synth512/
data/synth_images/flux1_dev_synth512/
output/theta_raw_to_v_setup/           # initial_raw_epsilon.pth and conditioning.pt
output/datasets_512/FluxLocalizedNarratives/
output/datasets_512/FluxSynth/
```

Each source dataset needs a `metadata.jsonl` file. Each completed training cache
needs `cache_settings.json`, `training_metadata.jsonl`, `latents/*.npz`, and
`t5_embeddings/*.npz`.

The trainer checks source metadata, cache counts, source paths, and SHA-256 hashes.
Moving a source dataset can invalidate the paths recorded in an existing cache;
regenerate the cache or preserve its original source paths.

### Initialization weights and validation conditioning

```bash
python -m scripts.preparation.prepare_theta_v \
  --source assets/source_model \
  --snapshot assets/pixart \
  --prompts examples/validation_prompts.json \
  --output output/theta_raw_to_v_setup
```

The reference experiment used the PixArt snapshot
`50f702106901db6d0f8b67eb88e814c56ded2692`.

### Dataset caches

`cache_synthetic_dataset.py` currently uses that snapshot at:

```text
~/.cache/huggingface/hub/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots/50f702106901db6d0f8b67eb88e814c56ded2692
```

Provide the model there or update the script's `snapshot` variable to point to
`assets/pixart` before generating caches.

Generate and finalize the first dataset cache with one process:

```bash
python -m scripts.preparation.cache_synthetic_dataset \
  --source data/synth_images/flux1_dev_localized_narratives_synth512 \
  --output output/datasets_512/FluxLocalizedNarratives --shards 1 --shard 0

python -m scripts.preparation.cache_synthetic_dataset \
  --source data/synth_images/flux1_dev_localized_narratives_synth512 \
  --output output/datasets_512/FluxLocalizedNarratives --shards 1 --finalize
```

Repeat both commands for the second dataset, using
`data/synth_images/flux1_dev_synth512` as the source and
`output/datasets_512/FluxSynth` as the output.

For four-GPU cache preparation, run four workers with `--shards 4` and distinct
`--shard` values from 0 through 3, assigning one GPU to each worker. Once every
worker finishes, finalize with `--shards 4 --finalize`.

## Training

After preparing all resources, start a new run:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run.sh train configs/rfm.py
```

The default output directory is `output/rfm_mix50`. Use a new `--work-dir` when
starting another run or resuming an existing one. The launcher defaults
`OMP_NUM_THREADS` to 4 and starts `torch.distributed.run` with the configured
number of workers.

### Optional background preparation and training

The training entry point also supports the original background controller:

```bash
bash run.sh train configs/rfm.py --launch
```

This mode requires a passing `output/rfm_mix50_smoke/verification.json`, the
corresponding pilot caches, initialization weights, and model assets. It enforces
four GPUs and a per-GPU batch size of eight. If the full caches already exist,
use direct training as shown above; the smoke verification artifacts are not
required for that path.

Controller state and logs are written to `preparation/status.json` and
`preparation/controller.log` under the work directory. Training output is logged
to `console.log`.

Both launch modes accept `--work-dir`, `--max-train-steps`,
`--validate-every-updates`, and `--save-every-updates`. Background mode forwards
these settings to the training process. `--launch` is for new training only and
cannot be combined with `--resume` or invoked under `torchrun`.

### Resume training

Use the same configuration and resource paths as the saved run, with a new output
directory. For example, resume a run from step 5,000 toward a total of 100,000
optimizer updates:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run.sh train configs/rfm.py \
  --resume output/rfm_mix50/checkpoints/update-005000.pth \
  --work-dir output/rfm_mix50_resumed \
  --max-train-steps 100000
```

The trainer strictly compares the checkpoint's configuration and data hashes.
Only the work directory, total update count, and save/validation/logging intervals
may change. Preserve all other training settings and resource paths.

To resume a checkpoint from the original project, restore the configuration
values recorded in that checkpoint and make its paths resolve correctly. The
example paths `assets/source_model` and `assets/pixart/vae` cannot replace
historical paths during resume without triggering `Resume changed training
semantics`. No separate historical configuration is maintained in this repository.

## Validation and reproduction

`scripts/validate_rfm.py` calls the same validation function used during training.
It produces four RAW and four EMA images, along with latent tensors, prompts,
and sampler metadata under `<output>/validation/update-<step>/`. The effective
configuration is saved as `<output>/validation_config.json`.

For the original 100K checkpoint, the following command uses assets from the
parent workspace and the local Hugging Face cache:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  bash run.sh validate configs/rfm.py \
  --checkpoint ../output/rfm_from_A_flux_raw_mix50_100k/resume-030000-to-100000/checkpoints/update-100000.pth \
  --conditioning-cache ../output/theta_raw_to_v_setup/conditioning.pt \
  --vae-path "$HOME/.cache/huggingface/hub/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots/50f702106901db6d0f8b67eb88e814c56ded2692/vae" \
  --output output/revalidate-100000
```

These resource locations are specific to the original workspace. Adjust them for
another machine, and select an interpreter with the required dependencies using
`PYTHON_BIN` if needed.

A local verification through `run.sh` reproduced all eight original 100K PNGs
byte for byte and both latent tensors exactly. RAW and EMA sampling used 182 and
176 function evaluations, respectively. Identical results are not guaranteed
across GPU models or dependency versions.

When calling `python -m scripts.validate_rfm` directly without `--config`, sampler
settings and default resource paths come from the checkpoint. Resource paths can
still be overridden with `--conditioning-cache` and `--vae-path`.

## Custom inference

Supply a JSON list with a unique filename-safe `case_id`, a nonempty `prompt`,
and a nonnegative integer `seed` for each image:

```json
[
  {
    "case_id": "example-01",
    "prompt": "A small cabin beside a lake at sunrise",
    "seed": 42
  }
]
```

Use [examples/inference_prompts.json](examples/inference_prompts.json) as a
starting point. Select weights with `--weights raw` or `--weights ema`; the
default is `raw`.

Inference needs `tokenizer/` and `text_encoder/` alongside the configured `vae/`
directory. Prompts are lowercased and stripped before encoding and must fit the
120-token training limit. The output includes one PNG per case,
`prompt_embeddings.pt`, and a `validation.json` report.

Through `run.sh`, sampling defaults come from the supplied configuration:
`configs/rfm.py` uses dopri5 and CFG 4. Direct Python invocation without `--config`
uses checkpoint resource paths and defaults to Heun with 40 steps and CFG 2.

## Checks

```bash
python -m pytest -q tests
bash run.sh train configs/rfm.py --help
bash run.sh validate configs/rfm.py --help
bash run.sh inference configs/rfm.py --help
```

The tests run on CPU and cover RFM loss and sampling, dataset loading, sampling
cursors, checkpoint restoration, and launcher behavior. Training and image
generation require CUDA.

## License

The code retains the original project's Apache-2.0 [LICENSE](LICENSE) and the
author and license notices in copied files. External models and datasets remain
subject to their respective licenses.
