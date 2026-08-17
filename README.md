# Tiny π0

Tiny π0 是一个用于学习 Vision-Language-Action 模型的 PyTorch 端到端项目。仓库只保留一套可缩放的 Tiny π0 架构，并打通了：

- PaliGemma 2 中冻结的 SigLIP、multimodal projector 与 Gemma token embedding；
- 自主实现的双专家 decoder、GQA、RoPE、block attention 与 prefix KV cache；
- LeRobot v3 双摄 SO101 数据读取、动作分块、mask 与归一化；
- Flow Matching 训练、验证、checkpoint、断点恢复和部署 artifact；
- RTX 4090 单卡训练、RTX 3050 Ti 推理，以及 SO101 双进程真机控制。

这不是 Physical Intelligence 官方仓库，也不是官方 π0 权重的等形重实现。当前 decoder 从头训练，不能加载官方 π0 decoder 参数；冻结前端来自 `google/paligemma2-3b-pt-224`。

## 两个容量，一个架构

| 配置 | 用途 | Prefix expert | Action expert |
|---|---|---:|---:|
| `TINY_PI0` | CPU 单测、快速理解数据流 | width 128 / depth 2 | width 64 / depth 2 |
| `SO101_TINY` | 当前双摄 SO101 训练与部署 | width 1024 / depth 8 | width 512 / depth 8 |

两者的 forward、mask、Flow Matching 和 checkpoint 逻辑完全一致。项目不再包含 Full 或 Large 配置。

当前 step7000 artifact 的可训练部分为约 2.58 亿参数；PaliGemma 前端约 10.08 亿参数全部冻结。历史 artifact 文件夹仍叫 `pi0_so101_recommended_step7000`，这是旧训练任务名，不代表仓库还存在 Recommended/Full 模型分支。

## 环境与验证

要求 Python 3.11。推荐在 Ubuntu 22.04 或 WSL2 Ubuntu 中使用 `uv`：

```bash
cd ~/pi0/Tiny_pi0
uv sync

uv run python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

uv run ruff check configs pi0 scripts tests
uv run pytest -q
```

查看两档容量：

```bash
uv run python -m scripts.inspect_config
```

## 准备 PaliGemma 2

先在 Hugging Face 的模型页面接受许可，再登录并下载：

```bash
uv run hf auth login
scripts/hf_download_server.sh google/paligemma2-3b-pt-224
```

默认缓存位置为：

```text
~/.cache/huggingface/hub/models--google--paligemma2-3b-pt-224/snapshots/<revision>/
```

代码只从第一 safetensors 分片选择性加载 SigLIP、multimodal projector 和 token embedding，不加载 Gemma 语言 decoder。

## 检查 LeRobot 数据

默认数据集位置是：

```text
~/.cache/huggingface/lerobot/Rich-RZ/so101_chocolates_to_bowl_v1
```

执行：

```bash
uv run python -m scripts.inspect_so101_dataset
```

当前数据约定：双摄 `front`/`wrist`、6 维 SO101 state/action、模型内部补零到 32 维、50 步 action chunk。尾部不足 50 步时重复 episode 最后一个动作，但 loss 通过 `action_valid_mask` 忽略这些补齐步；第 7–32 维通过 `action_dim_mask` 排除。

## RTX 4090 训练

先做短程验收：

```bash
uv run python -m scripts.train_so101 \
  --profile so101 \
  --max-steps 100 \
  --micro-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --validation-interval 20 \
  --validation-batches 8 \
  --checkpoint-interval 50 \
  --output-dir checkpoints/so101_tiny_smoke
```

再正式训练：

```bash
uv run python -m scripts.train_so101 \
  --profile so101 \
  --max-steps 30000 \
  --micro-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --end-learning-rate 1e-5 \
  --warmup-steps 1000 \
  --validation-interval 500 \
  --validation-batches 32 \
  --validation-action-batches 1 \
  --validation-sampling-steps 10 \
  --checkpoint-interval 1000 \
  --output-dir checkpoints/so101_tiny
```

训练参数和 AdamW 状态保持 FP32，CUDA 前向使用 BF16 autocast。指标写入 `metrics.jsonl`，checkpoint 保存可训练参数、优化器、normalizer、随机数状态和配置。恢复训练时保持参数一致并添加：

```bash
--resume
```

不要把旧 BF16-master checkpoint 当作新 FP32-master checkpoint 恢复。

## 离线推理

当前已验证的 artifact 是 step7000：

```bash
uv run python -m scripts.infer_deploy_artifact \
  --artifact-dir artifacts/pi0_so101_recommended_step7000 \
  --num-steps 10 \
  --sample-index 0 \
  --output-json artifacts/pi0_so101_recommended_step7000/offline-actions.json
```

该命令读取录制数据，不连接机器人。它会报告动作 shape、耗时、显存、首动作跳变、相对录制动作的误差和训练分位数覆盖情况。

## SO101 真机部署

模型环境与 LeRobot 环境分成两个进程：策略服务持有 GPU；机器人客户端持有串口和相机。二者只通过 `127.0.0.1` 上的 HTTP JSON/JPEG 协议通信。

终端 A：

```bash
cd ~/pi0/Tiny_pi0

uv run python -m scripts.serve_so101_policy \
  --artifact-dir artifacts/pi0_so101_recommended_step7000 \
  --host 127.0.0.1 \
  --port 8000

curl http://127.0.0.1:8000/health
```

终端 B 先运行 dry-run：

```bash
cd ~/pi0/Tiny_pi0
conda activate lerobot

python -m scripts.run_so101_real \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79017734-if00 \
  --robot-id my_follower \
  --front-camera /dev/v4l/by-id/usb-BC-231019-A_XWF-1080P-video-index0 \
  --wrist-camera /dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0 \
  --front-camera-fps 30 \
  --wrist-camera-fps 60 \
  --control-fps 20 \
  --num-steps 10 \
  --actions-per-inference 1 \
  --max-cycles 10
```

确认设备、相机角色、图像方向和输出正常后，才添加 `--execute`。默认每轮只执行动作块的第一步再重新观察，适合闭环测试：

```bash
python -m scripts.run_so101_real \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79017734-if00 \
  --robot-id my_follower \
  --front-camera /dev/v4l/by-id/usb-BC-231019-A_XWF-1080P-video-index0 \
  --wrist-camera /dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0 \
  --front-camera-fps 30 \
  --wrist-camera-fps 60 \
  --control-fps 20 \
  --num-steps 10 \
  --actions-per-inference 1 \
  --max-relative-target 2 \
  --max-cycles 1 \
  --execute
```

客户端默认检查 NaN/Inf、训练动作范围和目标跳变，之后 LeRobot 还会按 `max_relative_target` 裁剪每条指令。`--lerobot-safety-only` 会关闭策略层范围/跳变拒绝，但不会关闭协议检查和 LeRobot 裁剪。动作块可一次执行 1–50 步；步数越多越开环，真实环境变化后模型越无法及时纠正。

在 WSL2 中相机或串口突然消失时，先在 Windows 执行 `usbipd list` 并重新 attach，再检查 `/dev/serial/by-id/` 和 `/dev/v4l/by-id/`。wrist 相机当前以 MJPG 640×480、60 FPS 打开，强设 30 FPS 会被设备拒绝。

## 代码地图

- `configs/`：Tiny 架构与训练配置。
- `pi0/paligemma_prefix.py`：选择性加载并冻结 PaliGemma 前端。
- `pi0/prefix_embedding.py`：组装图像/语言 prefix。
- `pi0/action_embedding.py`：构造 state、noisy action、time suffix。
- `pi0/joint_decoder_layer.py`：双专家联合注意力与缓存路径。
- `pi0/core.py`：Flow Matching loss 与 Euler 采样。
- `pi0/lerobot_dataset.py`：LeRobot 数据和 action chunk。
- `pi0/training.py`：优化器、验证、checkpoint 与 resume。
- `pi0/deployment.py`：严格恢复 deployment artifact。
- `scripts/serve_so101_policy.py`、`scripts/run_so101_real.py`：真机服务端与客户端。

完整的实现原理与学习顺序见 [PI0_END_TO_END_GUIDE_ZH.md](PI0_END_TO_END_GUIDE_ZH.md)。

## 当前定位

该模型已经证明整条工程链路可运行，但 step7000 的真实动作效果不理想，可能出现重复动作。它适合作为端到端学习和部署 demo，不应把单次 flow loss 最低直接等同于任务成功率。下一阶段若目标转为更高成功率，应使用更多样的数据、按 episode 做严格验证，并与 LeRobot 官方提供的预训练 π0 微调方案进行对照。
