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

## 在服务器下载 Hugging Face 文件

如果 SSH 会话导出了指向本机的 `HTTP_PROXY/HTTPS_PROXY`，使用下面的脚本下载。
它会仅对下载进程移除 SSH 代理，并由服务器直连 HF 镜像：

```bash
scripts/hf_download_server.sh google/paligemma2-3b-pt-224
```

PaliGemma 是受限模型，首次下载前仍需在 Hugging Face 页面接受许可并执行
`hf auth login`。同一脚本也能下载数据集，例如：

```bash
scripts/hf_download_server.sh OWNER/DATASET --repo-type dataset
```

脚本默认使用 `https://hf-mirror.com`，缓存到 `~/.cache/huggingface`，并且
不会修改当前 shell 的代理变量。

## 检查真实 SO101 数据

```bash
uv run python -m scripts.inspect_so101_dataset
```

## 本机端到端推理

```bash
uv run python -m scripts.smoke_real_policy
```

上面的 smoke test 使用随机初始化的 decoder，只验证架构。训练结束后，应先解压
deployment artifact，再对录制数据执行真实权重离线推理：

```bash
mkdir -p artifacts/pi0_so101_recommended_step10000
tar -xzf artifacts/pi0_so101_recommended_step10000_deploy.tar.gz \
  -C artifacts/pi0_so101_recommended_step10000

uv run python -m scripts.infer_deploy_artifact \
  --artifact-dir artifacts/pi0_so101_recommended_step10000 \
  --num-steps 10 \
  --sample-index 0 \
  --output-json artifacts/pi0_so101_recommended_step10000/offline-actions.json
```

该脚本只读取一条已录制的双摄观测，严格加载模型与归一化参数并生成动作，不连接
SO101，也不会发送电机指令。输出中的动作仅前6维属于当前SO101。脚本还会将预测
首步与当前关节、录制动作进行比较；默认任一关节的首步变化超过10度就以非零状态
退出并显示 `ready_for_hardware: false`。

## RTX 4090 训练验收

训练器保持可训练参数和 AdamW 状态为 FP32，并在 CUDA 前向中使用 BF16 autocast；
SO101 的 loss 和扩散噪声只作用于前6个真实动作维。不要从旧的 BF16-master
checkpoint 恢复，应使用一个全新的输出目录：

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
  --output-dir checkpoints/so101_fp32_smoke
```

100步验收通过后，正式训练建议：

```bash
uv run python -m scripts.train_so101 \
  --profile recommended \
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
  --output-dir checkpoints/so101_fp32_v2
```

每一步指标追加到 `metrics.jsonl`；验证点同时记录 flow loss、真实6维动作块 MAE
和首动作 MAE。断点恢复时必须使用同一组参数和同一个新格式目录，并增加
`--resume`。训练器会拒绝旧 BF16-master checkpoint，避免静默继续无效训练。

## 文档

完整的中文学习路线和架构说明见
[PI0_END_TO_END_GUIDE_ZH.md](PI0_END_TO_END_GUIDE_ZH.md)。

## 说明

这是独立学习实现，不是 Physical Intelligence 官方仓库。Tiny 和 SO101 缩放配置与
官方完整模型参数形状不同，因此不能直接加载官方完整 π0 decoder 权重；项目中的
PaliGemma 2 / SigLIP 前端权重仍需按其原始许可证和访问条款获取。
