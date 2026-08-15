# Tiny π0

一个配置可缩放的 PyTorch π0 端到端复现项目，用于学习并实现从模型结构、
LeRobot 双摄数据处理、Flow Matching 训练、KV Cache 推理到 checkpoint 恢复的完整流程。

当前项目包含：

- Tiny、SO101 Recommended、SO101 Large 和官方尺寸配置；
- 冻结的 PaliGemma 2 / SigLIP 图像与文本前端；
- 独立实现的双专家 Gemma decoder、GQA、RoPE 和 block attention；
- LeRobot v3 双摄数据读取、动作分块、padding mask 和分位数归一化；
- Flow Matching loss、Euler 动作采样与 prefix KV cache；
- AdamW、warmup/cosine schedule、验证、原子 checkpoint 和断点恢复；
- RTX 3050 Ti 本地 Tiny 验证与 RTX 4090 训练配置。

## 环境

```bash
uv sync
uv run pytest -q
```

项目使用 Python 3.11，依赖由 `pyproject.toml` 和 `uv.lock` 固定。

## 检查真实 SO101 数据

```bash
uv run python -m scripts.inspect_so101_dataset
```

## 本机端到端推理

```bash
uv run python -m scripts.smoke_real_policy
```

## RTX 4090 训练验收

```bash
uv run python -m scripts.train_so101 \
  --profile recommended \
  --max-steps 100 \
  --micro-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-workers 4 \
  --validation-interval 20 \
  --validation-batches 8 \
  --checkpoint-interval 50 \
  --output-dir checkpoints/so101_4090_smoke
```

断点恢复时使用相同参数并增加 `--resume`。

## 文档

完整的中文学习路线和架构说明见
[PI0_END_TO_END_GUIDE_ZH.md](PI0_END_TO_END_GUIDE_ZH.md)。

## 说明

这是独立学习实现，不是 Physical Intelligence 官方仓库。Tiny 和 SO101 缩放配置与
官方完整模型参数形状不同，因此不能直接加载官方完整 π0 decoder 权重；项目中的
PaliGemma 2 / SigLIP 前端权重仍需按其原始许可证和访问条款获取。
