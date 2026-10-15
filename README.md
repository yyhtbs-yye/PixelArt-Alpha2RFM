# pixelart-alpha2rfm

从 PixArt Alpha 的 epsilon RAW 权重初始化、训练文本条件 Rectified Flow Matching（RFM）的独立代码仓库。
RFM 从第 0 步训练至第 100,000 步，共完成 100,000 次优化器更新；第 0 步使用上述预训练权重初始化。
本仓库对应以下训练结果，包含训练、数据缓存、权重转换、采样、验证及其本地代码依赖：

```text
output/rfm_from_A_flux_raw_mix50_100k/
└── resume-030000-to-100000/validation/update-100000
```

仓库仅保留 RFM 训练及其准备、验证和推理流程，唯一训练入口是 `scripts/train_rfm.py`。
数据读取、RFM loss、采样和 checkpoint 格式保持兼容。仓库可以单独移动、提交和配置 remote。
数据集、训练缓存、初始化权重、训练 checkpoint、VAE/T5 模型属于外部资源，未复制进仓库。
运行时需要安装 Python 依赖并准备这些资源，不需要导入原项目的 Python 源码。

## 对应的实验

| 项目 | 设置 |
| --- | --- |
| 模型 | PixArt XL/2，512×512，QK RMSNorm |
| 初始化 | `A_flux1_dev_mix50/checkpoint-010000` 的 RAW epsilon 权重 |
| 输出头 | 4 通道，初始训练时将输出头置零；续训及验证加载已训练输出头 |
| RFM 路径 | `x(t) = (1-t) × noise + t × data` |
| 训练目标 | `data - noise`，无加权 MSE |
| 时间 | 连续均匀采样，模型时间嵌入 `999 × (1-t)` |
| 数据 | FluxLocalizedNarratives 50,000 张；FluxSynth 76,708 张 |
| 混合比例 | 每张 GPU 每个 batch 严格 4+4，两组各 50% |
| 训练规模 | 4 GPU，每卡 batch 8，全局 batch 32，无梯度累积 |
| 数值设置 | BF16，关闭梯度检查点 |
| 优化器 | AdamW，LR `1e-5`，warmup 500，weight decay 0 |
| EMA / 文本 dropout | `0.9999` / `0.1` |
| 训练步数 | 0–100K，共 100,000 次优化器更新 |
| 验证 | RAW 和 EMA；`dopri5`，CFG 4，atol `1e-6`，rtol `1e-3` |
| 验证种子 | 42、43、44、45 |

`dopri5` 使用自适应步长，配置中的 `sampling_steps=50` 不表示固定执行 50 步。
每次验证的采样参数和函数调用次数保存在验证输出目录的 `sampler.json` 中。

## 目录

```text
run.sh                                # train / validate / inference 统一 Bash 启动器
configs/
  rfm.py                              # 唯一训练配置，参数和资源路径直接在此修改
examples/
  validation_prompts.json              # 初始化使用的四组验证 prompt/seed
  inference_prompts.json               # 自定义推理输入示例，含 case_id
diffusion/
  rectified_flow.py                    # RFM loss、Heun / dopri5 采样
  model/                              # PixArt XL/2 网络及 T5 文本编码
  utils/rfm_checkpoint.py             # RFM checkpoint、EMA、随机状态、文本 dropout
  utils/theta_import.py               # 初始化权重转换及来源记录
datasets/
  rfm_data.py                          # CachedDataset 定义及配套读取、采样逻辑
scripts/
  train_rfm.py                         # 直接训练或 --launch 后台准备/启动，含 RAW/EMA 验证
  validate_rfm.py                      # 从 checkpoint 调用原始验证函数
  inference_rfm.py                     # 自定义文本推理，Heun 采样
  preparation/
    cache_synthetic_dataset.py         # 生成 VAE posterior / T5 缓存
    encode_image_latents.py            # 图像预处理和 latent 缓存生成
    prepare_theta_v.py                 # RAW epsilon 权重转换和固定文本条件缓存
    verify_rfm_smoke.py                # 连续训练/续训对照验证
tests/                                # RFM、缓存校验、QKNorm、文本条件及 checkpoint 恢复
```

`datasets/` 用于数据集定义，包含 `CachedDataset` 及其配套的读取、采样逻辑；缓存生成、权重转换等离线准备工具放在 `scripts/preparation/`。
`scripts/train_rfm.py` 同时提供直接训练和后台准备/启动两种方式。初始化脚本仍读取 epsilon 预训练权重；
RFM 训练和采样仅使用 `rectified_flow.py`，不再依赖旧版 VP diffusion 求解器。

## 安装

提取时使用 Linux、Python 3.12.13、PyTorch `2.9.1+cu129`。
`requirements.txt` 记录本次观察到的主要依赖版本；它不是训练当日完整环境锁文件。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
```

CUDA 驱动和 GPU 需要与所选 Torch/xformers 环境兼容。完整训练使用 4 张支持 BF16 的 GPU。
已存在兼容 PixArt 环境时，也可以沿用它。

下面除特别注明外，均在本仓库根目录运行，使用 `python -m ...` 使本仓库代码可导入。

## 统一 Bash 启动器

接口为 `bash run.sh <train|validate|inference> <config.py> [其他参数]`，配置文件是必填参数。
准备好下面列出的数据和模型资源后，可以使用：

```bash
# 训练：自动按 config['world_size'] 启动 torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh train configs/rfm.py

# 验证：同时生成 RAW 和 EMA 结果
CUDA_VISIBLE_DEVICES=0 bash run.sh validate configs/rfm.py \
  --checkpoint output/rfm_mix50/checkpoints/update-100000.pth \
  --output output/validation-100000

# 自定义文本推理
CUDA_VISIBLE_DEVICES=0 bash run.sh inference configs/rfm.py \
  --checkpoint output/rfm_mix50/checkpoints/update-100000.pth \
  --prompts examples/inference_prompts.json \
  --weights ema --output output/inference-100000
```

启动器会切换到仓库根目录，所有相对路径均以仓库根目录为基准。设置 `PYTHON_BIN` 可选择
Python 环境，例如 `PYTHON_BIN=/path/to/env/bin/python bash run.sh train configs/rfm.py`。
`CUDA_VISIBLE_DEVICES` 原样传递；validate/inference 使用可见设备中的第一张 GPU。

训练模式将额外参数直接传给 trainer，支持 `--resume`、`--work-dir` 等；传入 `--launch`
则使用下文的后台准备流程。`bash run.sh train configs/rfm.py --help` 只显示帮助，不启动训练进程。

验证和推理读取指定配置的 `conditioning_cache`、`vae_path` 和采样参数，模型结构必须与
checkpoint 相符。`--conditioning-cache`、`--vae-path` 可覆盖资源路径。
验证保留 checkpoint 的固定 prompt/seed，并校验 conditioning 缓存哈希；推理的 prompt/seed
由 `--prompts` 指定，可用 `--cfg`、`--steps`、`--method`、`--atol`、`--rtol` 覆盖采样配置。
当前 `configs/rfm.py` 使用 dopri5、CFG 4；dopri5 的调用次数由自适应求解决定，`--steps` 只影响 Heun。

## 外部资源和从零训练

推荐为 `configs/rfm.py` 准备以下布局。资源可以复制，也可以由你建立指向数据盘的软链接：

```text
assets/source_model/                  # 原 checkpoint-010000，含 transformer/、scheduler/
assets/pixart/                        # 官方模型 snapshot，含 vae/、tokenizer/、text_encoder/
data/synth_images/flux1_dev_localized_narratives_synth512/
data/synth_images/flux1_dev_synth512/
output/theta_raw_to_v_setup/          # initial_raw_epsilon.pth、conditioning.pt
output/datasets_512/FluxLocalizedNarratives/
output/datasets_512/FluxSynth/
```

两个原始数据集各自需要 `metadata.jsonl`。训练缓存目录各自需要 `cache_settings.json`、
`training_metadata.jsonl`、`latents/*.npz` 和 `t5_embeddings/*.npz`。
训练入口会检查源 metadata、缓存数量、来源路径及 SHA-256。移动原始数据路径后，
已有缓存的来源路径可能不再匹配；应重新准备缓存，或保留原资源路径。

生成初始化权重和固定验证条件：

```bash
python -m scripts.preparation.prepare_theta_v \
  --source assets/source_model \
  --snapshot assets/pixart \
  --prompts examples/validation_prompts.json \
  --output output/theta_raw_to_v_setup
```

本次使用的官方 snapshot ID 是 `50f702106901db6d0f8b67eb88e814c56ded2692`。
原 `cache_synthetic_dataset.py` 保留了 `Path.home()/.cache/huggingface/hub/...` 下的该 snapshot 路径；
缓存生成前需在那个位置提供模型，或修改脚本中的 `snapshot` 指向 `assets/pixart`。
例如，单进程生成第一组缓存：

```bash
python -m scripts.preparation.cache_synthetic_dataset \
  --source data/synth_images/flux1_dev_localized_narratives_synth512 \
  --output output/datasets_512/FluxLocalizedNarratives --shards 1 --shard 0
python -m scripts.preparation.cache_synthetic_dataset \
  --source data/synth_images/flux1_dev_localized_narratives_synth512 \
  --output output/datasets_512/FluxLocalizedNarratives --shards 1 --finalize
```

第二组同样执行以上两条命令，将 source 换为 `data/synth_images/flux1_dev_synth512`，
output 换为 `output/datasets_512/FluxSynth`。多 GPU 缓存可以使用 `--shards 4`，
分别启动 `--shard 0` 至 `--shard 3`，完成后用相同 `--shards 4 --finalize` 合并。

资源准备好后启动新训练：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_rfm.py configs/rfm.py
```

新训练输出至 `output/rfm_mix50`。重新启动或续训必须使用新的 `--work-dir`。
原启动器已合并到同一入口。需要运行原来的“缓存准备 → 后台训练”流程时，使用：

```bash
python -m scripts.train_rfm configs/rfm.py --launch
```

`--launch` 只用于新训练，需要先提供 `output/rfm_mix50_smoke/verification.json` 中通过的验证记录、
对应的 pilot cache，以及初始化权重和模型资源；它保留原控制器的四 GPU、每卡 batch 8 检查。
缓存已经准备好时，直接使用上面的 `torchrun` 命令即可，无需这些 smoke 记录。

后台流程的状态和控制器日志保存在工作目录的 `preparation/status.json`、`preparation/controller.log`，
训练日志保存在 `console.log`。`--work-dir`、`--max-train-steps`、`--validate-every-updates` 和
`--save-every-updates` 在两种方式下均可使用，后台模式会将它们传递给实际训练进程。
断点恢复使用下面的直接训练方式；不要将 `--launch` 与 `--resume` 或 `torchrun` 一起使用。

## 断点恢复（可选）

使用同一份 `configs/rfm.py` 和训练时的资源路径，从已保存的 checkpoint 继续训练。
例如，从本仓库新训练的第 5,000 步恢复，输出到新目录：

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_rfm.py configs/rfm.py \
  --resume output/rfm_mix50/checkpoints/update-005000.pth \
  --work-dir output/rfm_mix50_resumed \
  --max-train-steps 100000
```

trainer 会严格比较 checkpoint 的训练配置和数据哈希，只允许改变工作目录、
总步数以及保存/验证/日志频率。恢复前不要修改配置中的其他参数或资源路径。

若要恢复原项目生成的 checkpoint，需将 `configs/rfm.py` 的相关字段设为该 checkpoint
记录的原值，并使用能正确解析那些路径的工作目录；不能直接使用当前示例中的
`assets/source_model`、`assets/pixart/vae` 替换历史路径，否则会触发
`Resume changed training semantics`。仓库不再维护另一份历史配置。

## 重新生成 100,000 步验证图

新增入口 `scripts/validate_rfm.py` 直接调用复制的 `train_rfm.validate`，
使用 checkpoint 中的 sampler 配置和原始固定条件，同时生成 RAW/EMA。
以下命令可在当前机器的本仓库根目录执行：

```bash
RFM_ASSET_WORKSPACE=/home/yyh/python_workspaces/pixelart-alpha-v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m scripts.validate_rfm \
  --checkpoint "$RFM_ASSET_WORKSPACE/output/rfm_from_A_flux_raw_mix50_100k/resume-030000-to-100000/checkpoints/update-100000.pth" \
  --conditioning-cache "$RFM_ASSET_WORKSPACE/output/theta_raw_to_v_setup/conditioning.pt" \
  --vae-path /home/yyh/.cache/huggingface/hub/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots/50f702106901db6d0f8b67eb88e814c56ded2692/vae \
  --output output/revalidate-100000
```

结果保存在 `output/revalidate-100000/validation/update-100000/`。
不同 GPU 或依赖版本下重新生成的结果不保证逐像素一致。

自定义文本推理可用 `python -m scripts.inference_rfm --help`。
该脚本接收包含 `case_id`、`prompt`、`seed` 的 JSON 列表。通过 Bash 启动器或显式 `--config`
调用时使用指定配置的资源路径和采样参数；直接调用 Python 且不传 `--config` 时，保留旧行为：
从 checkpoint 读取资源路径，使用 Heun 40 步、CFG 2。

## 检查代码

```bash
python -m pytest -q tests
python -m scripts.validate_rfm --help
python -m scripts.train_rfm --help
```

这些测试可以在 CPU 上运行，覆盖 RFM loss/采样、数据读取、采样游标及 checkpoint 恢复。
完整训练和图像验证需要 CUDA。

## 许可
沿用原项目的 Apache-2.0 [LICENSE](LICENSE)，并保留复制文件中的作者及许可证声明。
外部模型和数据的许可应按各自来源处理。
