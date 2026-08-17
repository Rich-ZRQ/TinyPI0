# Tiny π0：基于配置缩放的 PyTorch 端到端复刻指南

> 适用代码版本：`openpi` commit `15a9616`  
> 主线模型：**π0（flow matching）**，不是 π0-FAST，也不是 π0.5  
> 本机硬件：RTX 3050 Ti Laptop，4 GB VRAM  
> 最终目标：在 `Tiny_pi0` 中用一套代码实现 Tiny/Full 两种规模的模型、数据、训练、参数加载、推理与部署  
> 对照实验：`openpi` 的 `pi0_aloha_sim`（作为可执行标准答案和数值基线）
> 自研框架：**PyTorch**；主要参考官方 `models_pytorch`，JAX 仅用于一次性的原始权重转换

## 0. 先说清楚“复现”的边界

这份文档的目标不是教你把 openpi 当黑盒调用，而是让你以官方 PyTorch 版为标准答案，在 `Tiny_pi0` 中自己实现一套端到端 π0。由于本机只有 4 GB 显存，本地使用 `tiny` 配置开发；迁移到服务器后切换为 `full` 配置。两者必须实例化同一组类、执行同一条 forward/loss/sampler 路径，只允许配置中的宽度、深度、头数和训练资源参数不同。

这里的“无差别复刻”专指**架构算法和代码路径无差别**：Tiny 仍包含 SigLIP/PaliGemma 视觉语言前缀、Action Expert、block attention、flow matching、Euler sampler 和 KV cache。Tiny 的参数 shape 与 Full 不同，因此数值结果不同，也不能加载官方 Full 权重；只有切换到 Full 配置后，才能加载 `pi0_base` 并与官方模型逐值对齐。不要把“同构”误写成“Tiny 与官方模型数值相同”。

但不能从零完美复现 Physical Intelligence 官方的 `pi0_base` 预训练。仓库只说明基础模型使用了 10k+ 小时机器人数据，完整数据混合、采集数据、过滤规则、采样权重、训练算力和全部超参数并未公开。因此最严谨、也最有学习价值的目标是：

1. 用 Tiny 配置在 4 GB 本机实现并验证完整架构；
2. 用 Tiny 配置跑通数据、loss、反向传播、checkpoint、采样和 mock 部署；
3. 在服务器仅切换为 Full 配置，确认无需修改模型代码；
4. 转换并加载 `pi0_base`，用相同输入逐模块对齐官方 `PI0Pytorch`；
5. 使用自己的真机采集数据在服务器微调；
6. 服务器运行 Full policy，机器人端运行客户端和安全执行层。

## 1. 你最终要掌握的全链路

```text
LeRobot 原始样本
  -> 时间窗口采样（长度 action_horizon）
  -> repack_transforms：数据集字段重排
  -> data_transforms：机器人坐标/相机/动作空间适配
  -> Normalize：state 与 actions 归一化
  -> model_transforms：图像缩放、文本分词、维度补零
  -> Observation + Actions batch
  -> Pi0.compute_loss：条件流匹配训练
  -> AdamW 更新（可选 EMA）
  -> safetensors + optimizer checkpoint
  -> Policy：输入变换
  -> Pi0.sample_actions：从高斯噪声积分到动作块
  -> 反归一化 + 机器人输出变换
  -> action chunk
```

采用“双轨、双配置开发”：`openpi/` 只用于参考，`Tiny_pi0/` 放独立实现；`tiny` 用于本机功能测试，`full` 用于服务器官方参数对齐、微调和推理。Tiny 阶段做不依赖官方参数的算法测试；Full 阶段才做逐张量数值等价测试。

## 2. π0、π0-FAST、π0.5 的边界

| 模型 | openpi 类型 | 动作生成方式 | 本文是否展开 |
|---|---|---|---|
| π0 | `ModelType.PI0` | flow matching，连续动作块 | 是，唯一主线 |
| π0-FAST | `ModelType.PI0_FAST` | FAST 动作 tokenizer + 自回归 | 否 |
| π0.5 | `ModelType.PI05` | 仓库中仍使用 flow head，但状态编码和归一化结构不同 | 仅作对照 |

最可靠的辨别方法不是配置名称，而是看模型配置：

```python
pi0_config.Pi0Config(pi05=False)  # π0
pi0_config.Pi0Config(pi05=True)  # π0.5
```

本文避免使用 `pi05_*` 和 `pi0_fast_*` 配置。

## 3. 环境准备

本节不是要求一次性执行所有命令。按当前 Tiny→Full 路线分为四组：

| 阶段 | 现在是否执行 | 内容 |
|---|---|---|
| A. 本机基础检查 | 现在 | WSL2、GPU、Git、uv、Python 3.11 |
| B. Tiny 自研环境 | 项目骨架建立时 | 独立 `.venv`、PyTorch、测试与基础依赖 |
| C. 官方 openpi 对照环境 | 需要运行官方代码时 | 子模块、openpi `uv sync`、transformers patch |
| D. Full 权重与训练环境 | 服务器阶段 | `pi0_base` 转换、Full 对齐、微调和正式推理 |

### 3.1 硬件和系统

你的 3050 Ti 只有 4 GB 显存，不能承担完整 π0 的官方权重加载、正式推理或微调。本机职责是 Tiny 模型开发、数据管线、小规模训练和 mock 部署；Full 权重转换、数值对齐、微调和推理放到远程大显存服务器。仓库主要在 Ubuntu 22.04 上测试；Windows 本机使用 WSL2 Ubuntu，并确认 WSL 能识别 GPU。

```bash
nvidia-smi
git --version
```

当前机器已经通过 GPU 检查：WSL2 能识别 RTX 3050 Ti Laptop GPU，显存 4096 MiB。以后如果 Windows 驱动发生变化，再重复执行即可。

### 3.2 安装 uv 和依赖

本机当前没有 `uv`，现在需要安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

然后让 uv 安装项目统一使用的 Python 3.11。不要使用当前 Conda base 的 Python 3.13 作为本项目解释器：

```bash
uv python install 3.11
uv python find 3.11
```

到这里是**现在必须完成**的环境准备。先不要下载官方 Full 权重。

### 3.3 Tiny 自研环境（建立项目骨架时执行）

`Tiny_pi0` 应有自己的 `pyproject.toml` 和 `.venv`，与 openpi 官方环境隔离。项目骨架尚未建立时无需提前手工安装一堆包；建立后执行：

```bash
cd ~/pi0/Tiny_pi0
uv venv --python 3.11
uv sync
uv run python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

Tiny 项目的首批依赖应由自己的 `pyproject.toml` 固定，至少包括 PyTorch、NumPy、einops、safetensors、Pillow、pytest 和配置/CLI 库。不要让 Tiny 项目隐式依赖 openpi 的 `.venv`。

### 3.4 官方 openpi 对照环境（稍后按需执行）

只有当你需要运行官方 `PI0Pytorch`、官方数据加载器或转换脚本时，才执行：

```bash
cd ~/pi0/openpi
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

快速自检：

```bash
uv run python - <<'PY'
import torch
from openpi.training import config

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
cfg = config.get_config("pi0_aloha_sim")
print(cfg)
print("model type:", cfg.model.model_type)
PY
```

预期模型类型为 `ModelType.PI0`，并且 `CUDA available` 为 `True`。这套官方环境依赖较多、下载量较大，不是开始写 Tiny 配置和纯模块代码的前置条件。

### 3.5 官方 PyTorch transformers patch（最后执行）

openpi 已提供 `PI0Pytorch`、PyTorch 训练入口和 JAX→PyTorch 参数转换器，因此直接把官方 PyTorch 实现作为逐段重写的标准答案。官方实现要求 `transformers==4.53.2`，并把定制文件复制进 transformers：

```bash
cd ~/pi0/openpi
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

这个 patch 会改动当前虚拟环境中的 transformers；uv 默认 hardlink 还可能让改动进入 uv cache。你的 `Tiny_pi0` 最好使用独立虚拟环境；自研实现也应尽量把必要改动放进自己的模块，不要永久修改第三方包源码。官方环境若需彻底撤销，应按仓库提示执行 `uv cache clean transformers` 后重建环境。

不要现在执行这个 patch。等第一次需要实例化官方 `openpi.models_pytorch.PI0Pytorch` 做 Full 对照时再执行；Tiny 自研环境不应用这个 patch。

## 4. 本机第一个里程碑：Tiny 配置跑通训练骨架

官方 `debug` 并不等于你的最终 Tiny profile，而且仍可能实例化较重的视觉组件。本机首要目标是让你自己的同构 `Pi0Model(TINY_PI0)` 使用假数据、batch size 1 和少量 step，验证 PyTorch forward/backward、优化器和 checkpoint 全链路。

```bash
cd ~/pi0/Tiny_pi0
python -m scripts.train \
  --profile tiny \
  --data-profile fake \
  --max-steps 10 \
  --batch-size 1
```

这是本指南规定的目标 CLI；需要在实现训练器时让它成立。

成功后应看到：

```text
checkpoints/tiny/debug/<step>/
```

这一步失败时先不要下载大权重。先打印每个阶段的 shape 和 `torch.cuda.max_memory_allocated()`；必要时关闭 compile、保持 batch 1、减少 dataloader worker。若仍超出 4 GB，使用 `smoke` profile 定位，再回到保持正式接口的 `tiny` profile。

## 5. 模型搭建：从配置走到 Pi0

### 5.1 代码入口

按下面顺序阅读官方 PyTorch 实现：

1. `src/openpi/models/pi0_config.py`：框架共享的模型尺寸与输入输出规格；
2. `src/openpi/models_pytorch/pi0_pytorch.py`：π0 主体、embedding、loss、采样；
3. `src/openpi/models_pytorch/gemma_pytorch.py`：PaliGemma 与 Action Expert 的联合实现；
4. `src/openpi/models_pytorch/preprocessing_pytorch.py`：PyTorch 图像预处理；
5. `scripts/train_pytorch.py`：训练、DDP、优化器与 checkpoint；
6. `examples/convert_jax_model_to_pytorch.py`：官方参数转换规则。

`src/openpi/models/pi0.py` 和 `gemma.py` 只在遇到 PyTorch 代码注释不充分或需要确认原始数学语义时查阅，不作为你的主要照抄对象。

默认 `Pi0Config`：

```python
Pi0Config(
    dtype="bfloat16",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    action_dim=32,
    action_horizon=50,
    max_token_len=48,
    pi05=False,
)
```

输入张量约定（`B` 为 batch）：

| 输入 | 形状 | 含义 |
|---|---:|---|
| 三路图像 | `[B, 224, 224, 3]` | base、left wrist、right wrist |
| image mask | `[B]`/每路 | 对应视角是否有效 |
| state | `[B, 32]` | 原维度不足时补零 |
| tokenized prompt | `[B, 48]` | PaliGemma tokenizer 输出 |
| actions | `[B, 50, 32]` | 一次预测的动作块 |

实际 ALOHA 状态/动作是 14 维；`PadStatesAndActions(32)` 把它们补到 32 维，策略输出阶段再裁回 14 维。**不要把 padding 维度当成真实机器人自由度。**

### 5.2 Tiny/Full 配置设计

不要创建 `TinyPi0Model` 和 `FullPi0Model` 两套类。只创建一个 `Pi0Model(config)`，所有尺寸都来自不可变配置。建议分成结构配置和运行配置：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class VisionConfig:
    image_size: int
    patch_size: int
    width: int
    depth: int
    mlp_dim: int
    num_heads: int


@dataclass(frozen=True)
class Pi0Config:
    vision: VisionConfig
    paligemma: TransformerConfig
    action_expert: TransformerConfig
    vocab_size: int = 257_152
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    dtype: str = "float32"


@dataclass(frozen=True)
class RuntimeConfig:
    batch_size: int
    gradient_checkpointing: bool
    compile_model: bool
    num_workers: int
```

本机 Tiny profile 建议先保持正式数据契约不变，只缩小神经网络容量：

```python
TINY_PI0 = Pi0Config(
    vision=VisionConfig(
        image_size=224,
        patch_size=14,
        width=128,
        depth=2,
        mlp_dim=256,
        num_heads=4,
    ),
    paligemma=TransformerConfig(
        width=128,
        depth=2,
        mlp_dim=256,
        num_heads=4,
        num_kv_heads=1,
        head_dim=32,
    ),
    action_expert=TransformerConfig(
        width=64,
        depth=2,
        mlp_dim=128,
        num_heads=4,
        num_kv_heads=1,
        head_dim=32,
    ),
    action_dim=32,
    action_horizon=50,
    max_token_len=48,
    dtype="float32",
)

LOCAL_RUNTIME = RuntimeConfig(
    batch_size=1,
    gradient_checkpointing=False,
    compile_model=False,
    num_workers=0,
)
```

Full profile 必须填写官方尺寸：SigLIP So400m/14、Gemma 2B、Gemma 300M、`action_dim=32`、`action_horizon=50` 和 `max_token_len=48`。其精确字段以当前 commit 的官方 PyTorch/config 源码为准，并给配置保存版本号与 hash，避免手抄后悄悄漂移。

必须在构造时断言双 expert 的兼容约束：层数、attention head 数、KV head 数和 head dimension 满足官方联合 attention 的要求。不要在 forward 中用 `if config.profile == "tiny"` 切换算法；Tiny/Full 的区别只能体现在层循环次数和张量尺寸。

建议保持 224 图像、32 维 padded action 和 50 步 horizon，以便本地就测试正式接口。如果 4 GB 下视觉反向仍超限，可增加 `smoke` profile 把图像或 horizon 临时缩小，但它只用于冒烟测试，不能替代 `tiny` 的正式接口测试。

### 5.3 三个核心组件

`Pi0.__init__` 搭建：

- PaliGemma 主干：SigLIP `So400m/14` 图像编码器 + Gemma 2B 语言专家；
- action expert：Gemma 300M 配置，与主干在每层注意力中联合计算；
- 连续动作头：`state_proj`、`action_in_proj`、时间/动作 MLP、`action_out_proj`。

关键点：它不是“先让 VLM 输出文字，再把文字转动作”。图像/语言构成 prefix，状态、带噪动作和 flow 时间构成 suffix；两个专家在同一个注意力结构中交换上下文，动作专家直接回归速度场。

### 5.4 prefix 与 suffix

`embed_prefix()`：

1. 每路图像经 SigLIP 变成视觉 token；
2. prompt token 经 Gemma embedding；
3. 图像和语言 token 组成 prefix；
4. prefix 内使用全注意力，并用 image/token mask 屏蔽 padding。

`embed_suffix()`（π0 分支）：

1. `state_proj(state)` 产生一个 state token；
2. `action_in_proj(noisy_actions)` 投影整段带噪动作；
3. 标量时间 `t` 经正弦/余弦位置编码；
4. 每个动作 token 与时间 embedding 拼接后经过 MLP；
5. suffix 中 state/action 块按 `ar_mask` 形成块级注意力关系。

`make_attn_mask()` 用 `ar_mask` 的累积和构造 prefix-LM/块因果 mask。阅读模型时务必手算一个小 mask；这是理解 π0 信息流最有效的一步。

## 6. 训练目标：条件流匹配

给定真实动作块 `a`：

```text
ε ~ N(0, I)
t ~ Beta(1.5, 1)，并限制在 [0.001, 1]
x_t = t ε + (1 - t) a
u_t = ε - a
```

网络接收观测、`x_t` 和 `t`，输出速度场 `v_θ(x_t, obs, t)`。代码中的逐时间步损失为：

```text
L = mean_action_dim[(v_θ - u_t)²]
```

注意仓库的时间约定：`t=1` 是噪声，`t=0` 是动作，和 π0 论文中的记法相反。物理过程没有变，只是变量方向相反。

对应源码是 `Pi0.compute_loss()`。阅读时逐行标出：随机增强 RNG、噪声 RNG、时间 RNG、插值、联合 forward、输出投影、MSE。

## 7. 数据处理：从 LeRobot 到模型 batch

### 7.1 ALOHA Sim 主线配置

仓库的端到端公开示例：

```python
TrainConfig(
    name="pi0_aloha_sim",
    model=Pi0Config(),
    data=LeRobotAlohaDataConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        default_prompt="Transfer cube",
        use_delta_joint_actions=False,
    ),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=20_000,
)
```

这里选择绝对动作（`use_delta_joint_actions=False`），并从官方 `pi0_base` 初始化。

### 7.2 动作 chunk 如何生成

`create_torch_dataset()` 读取 LeRobot metadata 中的 FPS，并为每个样本请求：

```python
delta_timestamps = {key: [t / fps for t in range(action_horizon)]}
```

因此单个时刻会配上未来 `action_horizon=50` 帧动作，而不是只学习下一步动作。这就是 action chunking 的数据来源。

### 7.3 完整变换顺序

训练输入严格按以下顺序执行：

```text
repack_transforms.inputs
-> data_transforms.inputs
-> Normalize
-> model_transforms.inputs
```

对 `pi0_aloha_sim` 展开：

1. `RepackTransform`：把 LeRobot 的 `observation.images.top`、`observation.state`、`action` 重排为统一键；
2. `AlohaInputs`：相机名映射到 `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`，转换 HWC 图像、状态和动作约定；
3. `Normalize`：π0 使用 mean/std z-score；
4. `InjectDefaultPrompt("Transfer cube")`；
5. `ResizeImages(224, 224)`；
6. `TokenizePrompt(PaligemmaTokenizer(max_len=48))`；
7. `PadStatesAndActions(32)`。

由于 π0 的 `use_quantile_norm=False`，其公式是：

```text
x_norm = (x - mean) / (std + 1e-6)
```

不要把 π0.5 默认使用的 q01/q99 quantile normalization 套到 π0 主线上。

### 7.4 检查真实 batch

先算归一化统计（下一节），然后用下面脚本检查模型真正收到的形状：

```bash
uv run python - <<'PY'
import torch
from openpi.training import config, data_loader

cfg = config.get_config("pi0_aloha_sim")
loader = data_loader.create_data_loader(
    cfg, shuffle=False, num_batches=1, framework="pytorch"
)
obs, actions = next(iter(loader))

print("images:", {k: (v.shape, v.dtype) for k, v in obs.images.items()})
print("masks:", {k: v.shape for k, v in obs.image_masks.items()})
print("state:", obs.state.shape, obs.state.dtype)
print("tokens:", obs.tokenized_prompt.shape)
print("actions:", actions.shape, actions.dtype)
print("device before trainer transfer:", actions.device)
print("CUDA available:", torch.cuda.is_available())
PY
```

期望最后两维分别接近 `state[..., 32]` 和 `actions[..., 50, 32]`。

## 8. 计算 normalization stats

训练前执行：

```bash
cd ~/pi0/openpi
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim
```

统计结果写入：

```text
assets/pi0_aloha_sim/<asset_id>/norm_stats.json
```

统计脚本使用相同的数据集、repack 和 robot-specific transforms，但在真正 Normalize 前收集 state/actions 分布。训练 checkpoint 会复制对应 assets；推理必须使用与训练一致的统计量。

三条硬规则：

- 改了动作表达（绝对/增量、关节/末端速度）后必须重算；
- 改了状态/动作维度或单位后必须重算；
- 推理不能拿另一个机器人或另一份数据的 stats 混用。

## 9. 正式微调 π0

### 9.1 官方 JAX 原始基线（选读，不是自研主线）

以下命令用于确认原始发布实现；采用 PyTorch 自研时可以在 PyTorch baseline 跑通后再选读。

```bash
cd ~/pi0/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_aloha_sim \
  --exp-name=pi0_foundation_run \
  --overwrite
```

首次执行会下载数据和 `gs://openpi-assets/checkpoints/pi0_base/params`。下载缓存默认在 `~/.cache/openpi`；可通过 `OPENPI_DATA_HOME` 修改。

如果不想启用 W&B：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_aloha_sim \
  --exp-name=pi0_foundation_run \
  --overwrite \
  --wandb-enabled=false
```

断点恢复时不要同时传 `--overwrite`：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_aloha_sim \
  --exp-name=pi0_foundation_run \
  --resume
```

### 9.2 PyTorch 训练入口内部发生了什么

`scripts/train_pytorch.py` 的主调用链：

```text
CLI -> get_config/tyro
-> setup_ddp + select torch.device
-> create_data_loader(framework="pytorch")
-> PI0Pytorch(model_cfg).to(device)
-> load model.safetensors from pytorch_weight_path
-> torch.optim.AdamW + learning-rate schedule
-> training loop
   -> model(observation, actions)
   -> loss.backward()
   -> optimizer.step()
-> save model/optimizer/global_step checkpoint
```

数据配置和主要超参数仍复用 `TrainConfig`，但不能假设 JAX 专属能力也存在。以 `scripts/train_pytorch.py` 实际实现为准。

### 9.3 显存不够怎么办

PyTorch 主线按优先级处理：

1. 开启/保留官方 PyTorch 模型已有的 gradient checkpointing；
2. 使用 DDP 把 global batch 分配到多张 GPU；
3. 减小 global batch size（会改变优化条件，学习率可能也要调）；
4. 只做源码学习时用 PyTorch `debug`，不要把 dummy 模型的结果当 π0 实验结果。

官方当前 PyTorch trainer 不支持 LoRA、FSDP 和 mixed-precision training，因此不能照搬 JAX 低显存配置并期待直接工作。

不要仅把真实配置的模型宽度改小后声称复现 π0；那只是结构实验。

### 9.4 官方 PyTorch 主线

官方仓库的 PyTorch 路线是：先把 JAX `pi0_base` 转成 `model.safetensors`，再交给 `scripts/train_pytorch.py`。这也正是你的自研加载器应参考的参数语义基线。

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_base \
  --config_name pi0_aloha_sim \
  --output_path ./converted_checkpoints/pi0_base_pytorch \
  --precision float32
```

随后在复制出的自定义 `TrainConfig` 中设置：

```python
pytorch_weight_path = "./converted_checkpoints/pi0_base_pytorch"
```

单卡/多卡训练入口分别是：

```bash
uv run python scripts/train_pytorch.py pi0_aloha_sim \
  --exp_name pytorch_baseline

uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi0_aloha_sim \
  --exp_name pytorch_baseline
```

命令参数使用下划线是该 PyTorch 脚本 README 示例的写法。运行前以本地 `--help` 为准。

当前官方 PyTorch 支持不是 JAX 路线的完整功能等价物：仓库说明尚不支持 π0-FAST、mixed precision training、FSDP、LoRA 和 EMA。普通 π0 的单机/DDP 微调与推理可用。你的第一版自研训练器应先对齐普通全量微调；LoRA、EMA、AMP/FSDP 应作为后续明确验证的扩展，不要默认已有官方等价保证。

## 10. checkpoint 的结构与初始化语义

微调 checkpoint 位于：

```text
checkpoints/pi0_aloha_sim/pi0_foundation_run/<step>/
```

核心内容包括参数、训练状态及 assets。`CheckpointWeightLoader(pi0_base/params)` 是训练初始化；训练后的某个 `<step>` 目录则是创建推理 policy 的输入。两者不要混淆：

- `pi0_base`：跨机器人基础模型，用来微调；
- `pi0_aloha_sim/.../20000`：适配 ALOHA Sim 后的策略，可用于相应环境推理。

对你的独立实现而言，“下载到了官方参数”不等于“能够加载”。官方 JAX checkpoint 保存的是与 Flax/NNX 模块树绑定的参数 PyTree；你的类名、层级、专家编号、矩阵轴顺序或 dtype 只要不同，直接加载就会失败或产生错误结果。你需要实现一个明确的转换层：

```text
Orbax pi0_base checkpoint
-> 展平官方参数 PyTree
-> 官方 key 到自研 key 的映射表
-> 必要的 transpose/reshape/expert slice
-> shape 与 dtype 校验
-> 加载到自研模型
-> 保存为自研 checkpoint 格式
```

本文只走 **PyTorch 自研路线**：用 `nn.Module` 自己实现，先读官方转换器生成的 safetensors，再映射到你的 state dict。需要理解原始权重转换时，参考 `examples/convert_jax_model_to_pytorch.py` 中的 transpose、Q/K/V 拆分、专家切片和 tied weights 处理；目标 key 应对应你自己的模块，而不是照搬官方 `PI0Pytorch` 类名。

无论选哪种，加载器都必须输出报告：已加载 key、缺失 key、多余 key、每个张量的源/目标 shape、发生过的轴变换。禁止用 `strict=False` 后忽略报告。

## 11. 推理原理：噪声如何变成动作块

`Pi0.sample_actions()`：

1. 图像缩放/预处理；
2. 计算 prefix token，一次 forward 填充 KV cache；
3. 初始化 `x_1 ~ N(0, I)`；
4. 默认使用 10 个 Euler step，`dt=-1/10`；
5. 每步通过 action expert 预测 `v_t`；
6. 更新 `x <- x + dt * v_t`，从 `t=1` 积分到 `t=0`；
7. 得到 `[action_horizon, action_dim]` 动作块。

伪代码：

```python
x = normal_noise()
prefix_cache = encode_images_and_prompt(observation)
for t in [1.0, 0.9, ..., 0.1]:
    v = action_expert(prefix_cache, state, x, t)
    x = x - 0.1 * v
return x
```

prefix KV cache 避免在 10 次积分中重复编码图像和语言，是推理速度的关键设计。

## 12. 从自己训练的 checkpoint 推理

### 12.1 进程内推理

新建临时学习脚本时可使用以下最小逻辑（也可以直接在 notebook 中运行）：

```python
from openpi.policies import aloha_policy, policy_config
from openpi.training import config as train_config

cfg = train_config.get_config("pi0_aloha_sim")
policy = policy_config.create_trained_policy(
    cfg,
    "checkpoints/pi0_aloha_sim/pi0_foundation_run/20000",
)

example = aloha_policy.make_aloha_example()
result = policy.infer(example)
print(result["actions"].shape)  # 通常为 (50, 14)
print(result["policy_timing"])
```

`Policy.infer()` 的完整顺序：输入 transform → 加 batch 维 → `Observation.from_dict` → `sample_actions` → 去 batch 维 → output transform。

输出侧顺序是训练输入侧的逆过程：

```text
model actions
-> Unnormalize
-> data_transforms.outputs
-> robot-native actions
```

对于 ALOHA，`AlohaOutputs` 裁掉 padding，仅保留 14 维并恢复机器人动作约定。

### 12.2 启动策略服务器

```bash
cd ~/pi0/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_aloha_sim \
  --policy.dir=checkpoints/pi0_aloha_sim/pi0_foundation_run/20000
```

服务默认监听 8000 端口。另一个终端运行 ALOHA Sim 客户端：

```bash
uv run examples/aloha_sim/main.py
```

架构为：

```text
仿真/机器人客户端 --WebSocket observation--> GPU policy server
仿真/机器人客户端 <--WebSocket action chunk--- GPU policy server
```

机器人控制循环通常不会无条件执行全部 50 步；可每执行若干动作后重新观察、重新规划。重规划间隔越短反馈越及时，但推理频率与网络负载越高。

## 13. 关键代码地图

| 学习问题 | 文件/符号 |
|---|---|
| π0 默认尺寸、输入规格 | `src/openpi/models/pi0_config.py::Pi0Config` |
| PyTorch prefix/suffix、loss、sampling | `src/openpi/models_pytorch/pi0_pytorch.py::PI0Pytorch` |
| PyTorch attention mask | `src/openpi/models_pytorch/pi0_pytorch.py::make_att_2d_masks` |
| PyTorch 双专家与视觉主干 | `src/openpi/models_pytorch/gemma_pytorch.py` |
| PyTorch 图像预处理 | `src/openpi/models_pytorch/preprocessing_pytorch.py` |
| Observation 数据契约 | `src/openpi/models/model.py::Observation` |
| 图像训练增强 | `src/openpi/models/model.py::preprocess_observation` |
| π0 模型级 transforms | `src/openpi/training/config.py::ModelTransformFactory` |
| ALOHA 数据配置 | `src/openpi/training/config.py::LeRobotAlohaDataConfig` |
| 主线训练配置 | `src/openpi/training/config.py` 中 `pi0_aloha_sim` |
| 数据集和 transform 顺序 | `src/openpi/training/data_loader.py` |
| 通用 transforms | `src/openpi/transforms.py` |
| ALOHA 输入/输出适配 | `src/openpi/policies/aloha_policy.py` |
| normalization stats | `scripts/compute_norm_stats.py` |
| PyTorch 训练循环与 DDP | `scripts/train_pytorch.py` |
| JAX→PyTorch 官方权重转换 | `examples/convert_jax_model_to_pytorch.py` |
| 权重加载 | `src/openpi/training/weight_loaders.py` |
| checkpoint 保存恢复 | `src/openpi/training/checkpoints.py` |
| policy 构造 | `src/openpi/policies/policy_config.py` |
| 单次推理封装 | `src/openpi/policies/policy.py::Policy.infer` |
| WebSocket 服务 | `scripts/serve_policy.py` |
| ALOHA Sim 客户端 | `examples/aloha_sim/main.py` |

## 14. 在 `Tiny_pi0` 中从头搭建配置化兼容实现

### 14.1 建议的独立工程边界

你的代码以 PyTorch 为唯一训练/部署框架：模型继承 `torch.nn.Module`，batch 使用 `torch.Tensor`，训练使用 autograd/`torch.optim`，checkpoint 首选 safetensors 加独立 JSON 配置。代码不应从官方 `PI0Pytorch` 继承，也不应在自研 forward 中调用官方实现。Tiny 和 Full 必须共用全部模型源文件。推荐目录：

```text
Tiny_pi0/
├── configs/
│   ├── schema.py
│   ├── tiny.py
│   ├── full.py
│   ├── data.py
│   └── train.py
├── pi0/
│   ├── types.py
│   ├── attention_mask.py
│   ├── siglip.py
│   ├── gemma.py
│   ├── action_expert.py
│   ├── model.py
│   ├── flow_matching.py
│   └── sampler.py
├── data/
│   ├── lerobot_dataset.py
│   ├── transforms.py
│   ├── normalization.py
│   ├── aloha_adapter.py
│   └── collate.py
├── checkpoints/
│   ├── read_openpi.py
│   ├── parameter_map.py
│   └── save_restore.py
├── training/
│   ├── optimizer.py
│   ├── train_state.py
│   └── trainer.py
├── inference/
│   ├── policy.py
│   ├── server.py
│   └── client.py
├── tests/
│   ├── test_masks.py
│   ├── test_transforms.py
│   ├── test_parameter_loading.py
│   ├── test_forward_equivalence.py
│   ├── test_loss_equivalence.py
│   └── test_sampling_equivalence.py
└── scripts/
    ├── compute_norm_stats.py
    ├── convert_openpi_checkpoint.py
    ├── train.py
    └── serve.py
```

这不是要求一次创建所有空文件，而是模块完成到哪里，测试和脚本就跟到哪里。

禁止出现以下设计：

```python
if config.profile == "tiny":
    return tiny_forward(...)
else:
    return full_forward(...)
```

正确设计是同一个 forward 自然读取 `config.depth/width/num_heads`。这样在服务器从 Tiny 切到 Full 时，只允许改配置、checkpoint 路径、device/batch 和训练资源参数；任何模型源码改动都意味着“同构复刻”尚未完成。

### 14.2 正确的实现顺序

按依赖关系逐层实现：

1. `Observation/Actions` 数据契约、shape assertion；
2. attention mask、sin/cos 时间编码、RMSNorm、RoPE 等纯函数；
3. Gemma attention/MLP 和双 expert 参数布局；
4. SigLIP 图像编码器及图像 token 输出；
5. prefix/suffix embedding 和完整 forward；
6. flow matching 样本构造与 loss；
7. Euler sampler 与 prefix KV cache；
8. Tiny 单 batch overfit、保存恢复和 mock serving；
9. 数据集、transforms、normalization、batch loader；
10. optimizer、EMA、保存/恢复和训练循环；
11. policy 输入/输出逆变换；
12. server/client 与机器人安全执行层；
13. 在服务器切换 Full 配置；
14. 官方 checkpoint 转换、参数加载和逐层数值对齐。

参数加载放在 Tiny 全链路稳定、Full 模型能无代码改动实例化之后。Tiny shape 与官方参数不匹配是预期行为，不要裁剪官方权重硬塞进 Tiny。未验证 Full 官方权重加载前，不要开始昂贵微调。

### 14.3 官方实现与自研实现的对齐协议

完整数值对齐只在服务器 Full profile 上执行：同一份输入分别送进官方 `PI0Pytorch` 和你的 Full PyTorch 实现。比较时必须固定：

- 完全相同的预处理后 batch；
- eval mode，关闭图像随机增强；
- 相同参数和 dtype；
- 相同的 flow `t`、高斯噪声 `ε` 和采样初始 noise；
- 相同 attention mask、position ids 和 `num_steps`。

直接和 `openpi.models_pytorch.pi0_pytorch.PI0Pytorch` 对齐。两边使用相同的 torch tensor、device 和 dtype，可以逐层 hook 中间结果，不需要把 JAX 纳入日常测试。只有转换官方原始 checkpoint 出错时，才检查 Flax Linear `[in,out]`↔PyTorch Linear `[out,in]`、QKV 切片等跨框架布局规则。

按层级验收，不要只比较最终动作：

| 层级 | 对齐对象 | 验收内容 |
|---|---|---|
| 数据 | transform 后字典 | key、shape、dtype、数值 |
| 视觉 | image tokens | shape、均值/方差、逐元素误差 |
| 文本 | token ids/embedding | tokenizer 输出和 embedding |
| mask | attention mask/positions | 完全相等 |
| prefix | prefix hidden states | 数值误差在 dtype 合理范围内 |
| suffix | state/action/time tokens | 完全相同语义与排列 |
| 模型 | predicted velocity `v_t` | 固定输入下近似相等 |
| 训练 | per-element loss/grad | loss 近似相等，关键梯度方向一致 |
| 推理 | 每个 Euler step 的 `x_t` | 误差不随 step 异常放大 |

具体容差应按 float32/bfloat16 和设备实测设定，不要武断规定一个全局阈值。先记录绝对误差、相对误差和 cosine similarity，再为每类张量制定阈值。

### 14.4 官方参数加载的完成标准

参数转换器至少完成以下检查：

```text
源 checkpoint 可读取
-> 每个官方 tensor 有确定去向或明确标为不需要
-> 每个自研 trainable tensor 都已初始化
-> shape 全部匹配
-> dtype 转换明确
-> 参数数量和总元素数可解释
-> 固定 batch 的 v_t 与官方实现对齐
-> 固定 noise 的完整 action chunk 与官方实现对齐
```

最后两项才是真正证明“官方参数已经正确加载”。单纯显示 `load_state_dict succeeded` 不足以验收。

### 14.5 自采数据接口

先定义机器人无关的 canonical sample，再写采集格式到 canonical sample 的 adapter：

```python
sample = {
    "image": {
        "base_0_rgb": uint8_hwc,
        "left_wrist_0_rgb": uint8_hwc,
        "right_wrist_0_rgb": uint8_hwc,
    },
    "image_mask": {...},
    "state": float32_state,
    "actions": float32_future_action_chunk,
    "prompt": "pick up the object",
}
```

数据采集阶段还要保存时间戳、episode 边界、控制频率、相机标定/语义、state/action 单位、关节顺序、失败/成功标签和软件版本。构造 action chunk 时不得跨 episode；末尾 padding 策略必须明确。train/validation split 应按 episode 或场景划分，不能随机打散相邻帧造成泄漏。

### 14.6 自研训练器的最低功能

你的训练器应亲自实现并测试：seed 管理、batch sharding/设备搬运、forward/loss、梯度计算、梯度裁剪（若配置启用）、AdamW、学习率调度、冻结参数、EMA、日志、定期保存、断点恢复以及异常数值检查。

第一轮不要追求分布式和极致性能：先在 dummy 模型上 overfit 一个 batch，再在少量真实数据上 overfit 一个 episode，确认 loss 能显著下降且 checkpoint 恢复后完全连续，最后才上完整数据。

本项目的 SO101 训练还遵守两条不可省略的数值约束：

1. 从零初始化的 projection 和双专家参数、梯度、AdamW moments 必须保持 FP32；
   RTX 4090 只通过 BF16 autocast 降低矩阵计算和 activation 成本。不要把可训练参数
   本身直接转换成 BF16，否则 `1e-4` 以下的更新可能在写回权重时被舍入掉。
2. 模型保留32维动作契约，但 SO101 只有前6维是真实关节。训练 loss、初始 flow
   noise 和 Euler 更新都必须使用 action-dimension mask；后26维始终为0，不能让
   padding 维主导 loss 或成为随机干扰输入。

验证不能只记录 flow velocity MSE。当前训练器还会用固定随机种子运行完整动作采样，
记录真实动作维的 action chunk MAE 和 first-action MAE，并逐步追加到
`metrics.jsonl`。只有这些物理动作指标持续改善，checkpoint 才有进入离线安全验证的
资格。

### 14.7 自研部署架构

建议把部署拆成四层：

```text
Robot Adapter
  采集相机/状态，统一时间戳和字段
        ↓
Policy Client
  编码请求、超时、重试、action chunk 缓冲
        ↓ WebSocket/其他 RPC
Inference Server
  preprocess -> model sampler -> postprocess
        ↓
Safety & Control Executor
  限位、限速、插值、碰撞检查、watchdog、急停
```

训练/推理必须复用同一份输入 transform、normalization stats 和输出逆变换。服务端启动时应校验 model checkpoint、stats、robot adapter schema 和配置版本，避免“模型能加载但动作语义不匹配”。部署验收顺序是：离线录制数据回放 → 仿真/数字孪生 → 真机低速空载 → 单任务小范围 → 正常运行。

## 15. 推荐的五阶段 Tiny→Full 路线

### 阶段 A：建立配置骨架与 Tiny 模型

- 建立 `schema.py`、`tiny.py`、`full.py`；
- 只实现一个 `Pi0Model(config)`；
- 实现 Tiny 视觉主干、语言主干、Action Expert 和投影层；
- 打印参数量、每层 shape 和显存峰值；
- 手画 prefix/suffix 与 attention mask。

验收：Tiny forward/backward 能在 4 GB 本机运行，代码中不存在 Tiny 专属 forward。

### 阶段 B：本机打通 Tiny 训练和数据

- 使用少量公开或自采 episode；
- 在每一级 transform 后打印 key、shape、dtype、min/max；
- 运行 normalization stats；
- 验证 action chunk 的时间窗口；
- overfit 单 batch 和单 episode；
- 验证 checkpoint 恢复后 loss 连续。

验收：Tiny 使用完整数据契约完成训练、保存、恢复和 Euler sampling。

### 阶段 C：本机打通 Tiny 部署闭环

- 启动 Tiny policy server；
- 用 Mock Robot Client 发送图像、状态和 prompt；
- 实现超时、重连、action chunk buffer 和 watchdog；
- 用录制数据离线回放；
- 验证输入/输出 transform 可逆。

验收：本机能从 observation 请求完整走到安全处理后的 action chunk。

### 阶段 D：服务器切换 Full 并加载官方参数

- 只把 profile 从 `tiny` 改为 `full`；
- 确认没有修改模型源文件；
- 实现 checkpoint key 映射与转换报告；
- 加载转换后的 `pi0_base` safetensors；
- 对齐固定 `t/noise` 下的 `v_t`；
- 对齐每个 Euler step 和最终 action chunk。

验收：Full 自研模型不调用官方 forward，却能加载 `pi0_base` 并在固定输入下复现官方 PyTorch 输出。

### 阶段 E：真机数据微调与远程部署

- 将数据转成 LeRobot；
- 明确相机、state、action 的单位和坐标系；
- 新建 Inputs/Outputs；
- 新建 DataConfigFactory 和 TrainConfig；
- 计算新 stats；
- 先取一个 batch，再做短训练，最后才跑长训练；
- 部署时加动作限幅、安全检查和急停。

验收：训练与推理共用同一对可逆的 robot-specific transforms。

## 16. 自定义机器人时的最小改造清单

1. 数据字段能表示当前观测和未来动作序列；
2. 明确真实 action dim 与模型 padding dim；
3. 固定每一路相机语义，不要训练/部署时互换；
4. 缺失视角补黑图，同时正确设置 `image_mask=False`；
5. state/action 单位、关节顺序和 gripper 约定完全一致；
6. 决定 absolute 还是 delta action，并提供对应逆变换；
7. prompt 对每个样本可用；
8. 重新计算 normalization stats；
9. policy 输出要反归一化并恢复机器人原生坐标；
10. 真机执行前必须限幅、限速、碰撞检查和急停。

## 17. 最容易踩的坑

### 配置名字和模型类型混淆

始终打印 `cfg.model.model_type` 和 `cfg.model.pi05`。不要因文件都叫 `pi0_config.py` 就认为 π0.5 与 π0 数据处理相同。

### 忘记子模块或绕过 uv

统一使用 `uv run ...`；安装后仍执行一次 `git submodule update --init --recursive`。

### 没算 norm stats

错误通常会直接提示运行 `compute_norm_stats.py`。不要通过 `skip_norm_stats=True` 来开始正式训练。

### transforms 训练/推理不对称

训练中做过 delta action、坐标转换、归一化，推理输出必须逆变换。否则 loss 可能下降，但机器人动作完全错误。

### 图像布局错误

外部 ALOHA 示例接收 CHW，`AlohaInputs` 转成 HWC；模型统一使用 HWC。自定义输入若本来已经 HWC，不能再次错误转置。

### 直接让 base checkpoint 控制新机器人

`pi0_base` 是微调初始化，不是任意机器人的即插即用策略。至少需要匹配的适配层、stats 和目标机器人数据。

### `--overwrite` 与 `--resume` 同时使用

配置会拒绝两者同时开启。恢复训练仅使用 `--resume`。

### “训练跑起来”等于“复现成功”

最低验收应包括：字段/shape 正确、归一化可逆、loss 有效、checkpoint 可恢复、离线动作分布合理、仿真任务评估可重复。

### 把 Tiny 权重当成官方预训练权重

Tiny 和 Full 只共享架构代码，不共享参数 shape。Tiny checkpoint 只能验证训练系统和部署链路，不能作为 `pi0_base` 的替代品，也不能通过补零/裁剪变成官方 Full 权重。

### 切到 Full 时顺手修改 forward

从本机到服务器只应切换配置和运行资源。如果为了 Full 另外写 attention、loss 或 sampler 分支，Tiny 阶段测试过的就不是最终架构。

## 18. 建议做的源码实验

1. 用很小的布尔数组调用 `make_attn_mask()`，打印矩阵；
2. 固定 observation 和 noise，改变 `num_steps` 比较动作；
3. 固定 observation 与 RNG，验证推理可重复性；
4. 将一批 actions Normalize 再 Unnormalize，检查最大重建误差；
5. 在每层 transform 后断言 key、shape、dtype；
6. 比较训练态图像增强前后，但不要改正式验证输入；
7. 检查缺失 wrist camera 时黑图和 mask 是否成对出现；
8. 加载保存后的 checkpoint，确认相同 noise 下输出一致。

## 19. 端到端复现清单

- [ ] `git submodule update --init --recursive` 成功
- [ ] `uv sync` 与 editable install 成功
- [ ] PyTorch 能看到 NVIDIA GPU
- [ ] Tiny profile 在 4 GB 显存内完成 forward/backward
- [ ] Tiny 与 Full 由同一个 `Pi0Model` 实例化
- [ ] 模型 forward 中不存在基于 profile 名称的算法分支
- [ ] 能解释 π0 prefix/suffix 与 flow matching loss
- [ ] Tiny 保持正式 observation/action 数据契约
- [ ] Tiny 单 batch 与单 episode overfit 成功
- [ ] `compute_norm_stats.py` 成功生成 assets
- [ ] 训练 loss、学习率和吞吐有记录
- [ ] checkpoint 可保存并恢复
- [ ] Tiny policy server、Mock Client 和 action buffer 连通
- [ ] `Tiny_pi0` 的模型 forward 不调用 openpi 官方 forward
- [ ] 自研模型、训练器与部署路径均以 PyTorch 实现
- [ ] 服务器仅切换配置即可实例化 Full，模型源码无改动
- [ ] Full `pi0_base` 权重成功加载
- [ ] 官方参数到自研参数的映射报告无未解释项
- [ ] 固定 batch、`t`、noise 时自研 `v_t` 与官方结果对齐
- [ ] 自研 Euler sampler 的逐步结果与官方结果对齐
- [ ] 自采数据按 episode 划分且 action chunk 不跨 episode
- [ ] Full policy server/client 完成离线回放和仿真验收
- [ ] 能说明为何无法从公开材料重建官方完整基础预训练

## 20. 一条最短官方基线路线

下面命令只负责在服务器建立“官方标准答案”，不应在 4 GB 本机强行执行。你的本机主线是第 5.2、14、15 节的 Tiny profile；服务器阶段再运行以下转换和 Full 基线。

```bash
cd ~/pi0/openpi
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 1. 安装官方 PyTorch 实现需要的 transformers patch
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/

# 2. 验证 PyTorch 训练骨架
uv run scripts/train_pytorch.py debug --exp_name pytorch_debug

# 3. 为真实公开数据计算统计量（该工具可继续复用）
uv run scripts/compute_norm_stats.py --config-name pi0_aloha_sim

# 4. 下载官方 pi0_base
uv run python - <<'PY'
from openpi.shared import download
print(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base"))
PY

# 5. 把官方 Orbax 权重一次性转换为 PyTorch safetensors
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_base \
  --config_name pi0_aloha_sim \
  --output_path ./converted_checkpoints/pi0_base_pytorch \
  --precision float32

# 6. 在自定义 TrainConfig 中设置 pytorch_weight_path 后启动 PyTorch 微调
uv run scripts/train_pytorch.py <你的_pi0配置名> \
  --exp_name pytorch_pi0_finetune
```

其中自定义配置需要包含：

```python
pytorch_weight_path = "./converted_checkpoints/pi0_base_pytorch"
```

完成这条官方 PyTorch 路线后，你获得的是可执行参考；完成第 14 节的独立 PyTorch 实现、参数加载和数值对齐后，才算真正具备端到端实现能力。这套能力也是后续 π0-FAST、π0.5 乃至其他 VLA 会复用的地基：统一 observation/action 契约、机器人数据适配、action chunk、条件生成、预训练权重迁移、归一化资产、checkpoint 和远程 policy serving。
