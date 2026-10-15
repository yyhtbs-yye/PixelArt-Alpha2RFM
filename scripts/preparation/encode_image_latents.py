"""Cache PixArt-Alpha VAE posterior statistics as one NPZ per image.

Each file contains unscaled FP32 `latent` (posterior mean) and `std`, both
shaped (4, 64, 64), plus scalar `scaling_factor` and `scaled=False`.
For diffusion training: z = (latent + std * noise) * scaling_factor.
For deterministic reconstruction: vae.decode(latent).

Run one shard per GPU with CUDA_VISIBLE_DEVICES and --shard-index.
After all shards finish, --finalize verifies every NPZ and exports
latent_metadata.jsonl without changing the image/caption metadata.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
REVISION = "50f702106901db6d0f8b67eb88e814c56ded2692"


def load_image(path):
    with Image.open(path) as image:
        if image.size != (512, 512) or image.mode != "RGB":
            raise ValueError(f"Expected a 512x512 RGB image: {path}")
        pixels = np.asarray(image, dtype=np.float32)
    return (pixels.transpose(2, 0, 1).copy() / 255.0 - 0.5) / 0.5


def save_latent(job):
    path, mean, std, scaling_factor = job
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, latent=mean, std=std,
            scaling_factor=np.float32(scaling_factor), scaled=np.bool_(False),
        )
    os.replace(temporary, path)


def verify_latent(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"latent", "std", "scaling_factor", "scaled"}:
            raise ValueError(f"Unexpected NPZ keys: {path}")
        for key in ("latent", "std"):
            array = data[key]
            if array.shape != (4, 64, 64) or array.dtype != np.float32:
                raise ValueError(f"Unexpected {key} shape or dtype: {path}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite {key}: {path}")
        if np.any(data["std"] <= 0):
            raise ValueError(f"Nonpositive posterior standard deviation: {path}")
        if data["scaled"].shape != () or bool(data["scaled"]):
            raise ValueError(f"Expected an unscaled posterior: {path}")
        if data["scaling_factor"].shape != () or not np.isclose(float(data["scaling_factor"]), 0.18215):
            raise ValueError(f"Unexpected scaling factor: {path}")
    return path.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("output/datasets_512"))
    parser.add_argument("--datasets", nargs="+", default=["LSDIR", "Unsplash"])
    parser.add_argument("--vae-path", type=Path, default=(
        Path.home() / ".cache/huggingface/hub"
        / "models--PixArt-alpha--PixArt-XL-2-512x512" / "snapshots" / REVISION / "vae"
    ))
    parser.add_argument("--latent-dir", default="latents")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards or min(args.batch_size, args.workers) < 1:
        parser.error("Invalid shard, batch size, or worker count")
    if args.limit < 0 or (args.finalize and args.limit):
        parser.error("Limit must be nonnegative; pilot runs cannot be finalized")
    if Path(args.latent_dir).name != args.latent_dir or args.latent_dir in (".", "..", "images"):
        parser.error("latent-dir must be a separate simple directory name")

    if len(set(args.datasets)) != len(args.datasets) or any(Path(name).name != name or name in (".", "..") for name in args.datasets):
        parser.error("Dataset names must be unique simple directory names")

    datasets = []
    hashes = {}
    for name in args.datasets:
        root = args.dataset_root / name
        content = (root / "metadata.jsonl").read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        rows = [json.loads(line) for line in content.splitlines()]
        if len({row["file_name"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate image references in {name}")
        for row in rows:
            row["dataset"] = name
            row["latent_file"] = str(Path(args.latent_dir) / (Path(row["file_name"]).stem + ".npz"))
        if len({row["latent_file"] for row in rows}) != len(rows):
            raise ValueError(f"Latent filename collision in {name}")
        datasets.append(rows)
    records = [row for pair in itertools.zip_longest(*datasets) for row in pair if row]
    if args.limit:
        records = records[:args.limit]
    config_path = args.vae_path / "config.json"
    weight_path = args.vae_path / "diffusion_pytorch_model.safetensors"
    settings = {
        "vae_model": MODEL_ID,
        "vae_revision": REVISION,
        "vae_subfolder": "vae",
        "vae_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "vae_weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashes,
        "image_size": [512, 512],
        "latent_shape": [4, 64, 64],
        "dtype": "float32",
        "latent": "unscaled posterior mean (deterministic mode)",
        "std": "unscaled posterior standard deviation",
        "scaling_factor": 0.18215,
        "scaled": False,
        "normalization": "(RGB / 255 - 0.5) / 0.5",
        "num_shards": args.num_shards,
        "limit": args.limit,
        "versions": {package: importlib.metadata.version(package)
                     for package in ("torch", "diffusers", "numpy")},
    }
    run_directory = args.dataset_root / "latent_runs" / args.latent_dir
    run_directory.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        for shard in range(args.num_shards):
            recorded = json.loads((run_directory / f"settings-{shard:03d}.json").read_text())
            if recorded != settings:
                raise ValueError(f"Settings mismatch for shard {shard}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for name, rows in zip(args.datasets, datasets):
                root = args.dataset_root / name
                paths = [root / row["latent_file"] for row in rows]
                if set(paths) != set((root / args.latent_dir).glob("*.npz")):
                    raise ValueError(f"Missing or extra latents in {name}")
                total_bytes = sum(pool.map(verify_latent, paths))
                destination = root / "latent_metadata.jsonl"
                content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                if destination.exists() and destination.read_text() != content:
                    raise ValueError(f"Refusing to overwrite different metadata: {destination}")
                temporary = destination.with_suffix(".jsonl.tmp")
                temporary.write_text(content)
                os.replace(temporary, destination)
                summary = dict(settings, dataset=name, count=len(rows), complete=True,
                               all_npz_verified=True, total_bytes=total_bytes)
                (root / "latent_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(f"{name}: verified and indexed {len(rows):,} NPZ files", flush=True)
        return

    settings_path = run_directory / f"settings-{args.shard_index:03d}.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Cache settings have changed; use a new --latent-dir")
    assigned = records[args.shard_index::args.num_shards]
    pending = []
    for row in assigned:
        path = args.dataset_root / row["dataset"] / row["latent_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not settings_path.exists():
                raise ValueError(f"Existing latent has no provenance: {path}")
            verify_latent(path)
        else:
            pending.append(row)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    saved = len(assigned) - len(pending)
    print(f"Shard {args.shard_index}: {saved:,} complete, {len(pending):,} pending", flush=True)
    if not pending:
        return

    import torch
    from diffusers import AutoencoderKL

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
    ).to("cuda").eval().requires_grad_(False)
    if vae.config.latent_channels != 4 or vae.config.scaling_factor != settings["scaling_factor"]:
        raise ValueError("Unexpected VAE configuration")
    started = time.monotonic()
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            paths = [args.dataset_root / row["dataset"] / row["file_name"] for row in batch]
            pixels = torch.from_numpy(np.stack(list(pool.map(load_image, paths)))).to("cuda")
            posterior = vae.encode(pixels).latent_dist
            means = posterior.mean.cpu().numpy()
            stds = posterior.std.cpu().numpy()
            if means.shape != (len(batch), 4, 64, 64) or not np.isfinite(means).all():
                raise ValueError("Invalid VAE means")
            if not np.isfinite(stds).all() or np.any(stds <= 0):
                raise ValueError("Invalid VAE standard deviations")
            jobs = [(args.dataset_root / row["dataset"] / row["latent_file"], mean, std,
                     vae.config.scaling_factor) for row, mean, std in zip(batch, means, stds)]
            list(pool.map(save_latent, jobs))
            saved += len(batch)
            if (start // args.batch_size + 1) % 10 == 0 or saved == len(assigned):
                elapsed = time.monotonic() - started
                print(f"Shard {args.shard_index}: {saved:,}/{len(assigned):,} saved; "
                      f"{(start + len(batch)) / elapsed:.2f} images/s; {elapsed:.1f}s elapsed",
                      flush=True)


if __name__ == "__main__":
    main()
