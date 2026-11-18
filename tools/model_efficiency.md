# 通用双模态模型效率统计与接入说明

入口：`tools/model_efficiency.py`。复用 `models.build.build_model`、原有权重键迁移和实际 `forward`，不单独抽取编码器，不替换注意力，不改写模型来配合 FLOPs 工具。无新增第三方依赖，使用项目现有 PyTorch 2.6 的 eager operator dispatch 接口。

## 运行

从项目根目录执行。默认单卡 `cuda:0`、RGB `[1,3,256,256]`、DSM `[1,256,256]`、FP32、预热 50 次、测量 200 次。

```bash
conda activate testnet
python tools/model_efficiency.py \
  --config configs/cfg_vaihingen_vit-l_3090.jsonc \
  --checkpoint /path/to/scsam/best_miou.pth \
  --model-name ScSAM \
  --output work_dirs/efficiency/scsam.csv
```

同一次运行依次统计多个模型，所有模型共用测量设置。下面以相同 ViT-B 主干为例；权重路径应替换为各自真实文件。最好直接使用每次训练保存的有效配置，确保模型结构、冻结阶段和权重一致。

```bash
conda activate testnet
python tools/model_efficiency.py \
  --config configs/default.jsonc configs/cfg_vaihingen_vit-b.jsonc \
  --model-type mfnet_unetformer testnet_prealign_spmf21 \
  --checkpoint /path/to/mfnet/best_miou.pth /path/to/scsam/best_miou.pth \
  --model-name MFNet ScSAM \
  --device cuda:0 --input-size 256 256 --batch-size 1 \
  --warmup 50 --iterations 200 \
  --output work_dirs/efficiency/comparison.csv
```

`ScSAM` 在当前 `predict.py` 中是输出名称，结构由配置中的 `model.type` 决定。当前主配置使用 `testnet_prealign_spmf21`，MFNet 为 `mfnet_unetformer`。脚本遵循同样规则，`--model-name` 仅标记输出，不能改变结构。各模型可以用不同主干，但必须在报告中明确记录。

参数：

| 参数 | 含义 |
| --- | --- |
| `--config` | 一个或多个现有 JSON/JSONC 训练配置，支持现有 `extends` |
| `--checkpoint` | 与配置逐项对应的完整任务权重；省略时使用构建器初始化及 SAM 预训练权重，结果标为 `initialization` |
| `--model-type` | 可选，逐项覆盖 `model.type`；原配置的训练阶段仍保留，必须与模型兼容 |
| `--model-name` | 可选，逐项指定 CSV 显示名 |
| `--adapter` | 可选，逐项指定 `repository` 或 `module:factory` |
| `--input-size H W` | 原始 patch 尺寸；模型尺寸限制按实际前向自然报错 |
| `--batch-size` | 默认 1；GFLOPs、时延均除以 batch size 得到每 patch 值 |
| `--device` | 默认 `cuda:0`，也支持其他单张 CUDA 卡或 `cpu`，不自动切换设备 |
| `--warmup` / `--iterations` | 预热和正式测量次数，默认 50 / 200 |
| `--train-epoch` | 用于恢复训练冻结策略的 1-based epoch，覆盖权重中的 epoch |
| `--seed` | 默认 0；每个模型重新设置初始化种子并生成相同输入 |
| `--flop-rules` | `module:factory`，返回自定义算子计数规则 |
| `--output` | CSV 路径；覆盖该 CSV，每个模型的详细 JSON 使用唯一文件名 |

默认目标硬件是 RTX 3090。脚本读取并记录实际设备名称；在其他硬件运行时会明确提示，不能将其结果标称为 3090。CPU 结果仅适用于 CPU 比较。

## 参数和冻结策略

- 总参数量按 `model.parameters()` 统计，包含冻结参数、未执行的已注册模块，除以 `10^6` 得到 M。共享的同一个 `Parameter` 只计一次。例如 `sam.image_encoder` 与 `image_encoder` 的别名不会重复计数。参数量不是 `state_dict` 元素数，也不包含 buffers。
- 可训练参数量按训练设置中的 `requires_grad` 统计。不会因为效率测量使用 `inference_mode()` 而将参数全部冻结。
- 先保留构建器的冻结设置，再通过现有 `StageScheduler.resolve_stage` 确定阶段，执行该阶段的 `freeze_modules` 和 `detach_dsm_taps` 设置。模块路径复用项目现有解析规则。
- epoch 优先级：`--train-epoch` → 配置 `efficiency.train_epoch` → checkpoint 的 `epoch`。不会使用 `resume_epoch`，因为这里统计的是保存权重时对应的训练阶段。分阶段配置缺少 epoch 时直接报错；无阶段配置采用构建器冻结设置。
- 权重加载沿用 `predict.py`：默认读取 checkpoint 的 `model` 字段，迁移旧键后 `strict=True`。其他格式可显式设置 `efficiency.checkpoint_key`；例如 `"state_dict"`，或 `null` 表示文件本身是原始 state dict。不会忽略不匹配参数。
- 训练用中间统计记录器关闭，并在详细 JSON 的模型配置中记录；训练阶段的冻结、结构与前向计算保留。

## 完整前向 GFLOPs 和算子审计

在与时延相同的设备、输入、FP32、`eval()`、`inference_mode()` 条件下，单独执行一次完整 eager 前向。审计器观察实际执行的底层算子，不调用反向，不改变计算路径。共享模块每次调用的运算均计入，即使其权重只计一次。

MFNet 和 ScSAM 适配器调用原始 `model(rgb, dsm, mode="Test")`，因此两路模态编码、模型内部 DSM 转换/预对齐、融合/结构分支、特征金字塔、解码器和输出插值均进入统计。模型没有在该推理路径执行的训练辅助头不产生 FLOPs；其注册参数仍计入总参数量。

统一采用 `algorithmic_v1` 算法运算估算口径：**1 MAC = 2 FLOPs**，GFLOPs = FLOPs / `10^9` / batch size。不是直接将某工具的 MACs 改名为 FLOPs，也不是 GPU 实际指令数。

| 运算 | 计数规则 |
| --- | --- |
| 卷积 | `2 × batch × 输出空间元素数 × weight.numel()`，权重形状已体现 groups；有 bias 时加输出元素数。按包含 padding 的稠密卷积口径 |
| 转置卷积 | 上式使用输入空间元素数，并加 bias；不当作普通卷积处理 |
| 矩阵乘、线性层 | `2 × 输出元素数 × 收缩维度`，额外计 bias；`addmm/baddbmm` 计入非默认 alpha/beta 缩放与相加 |
| `einsum` | 由实际分解执行的 `bmm`、逐元素运算等计入，不再重复计高层调用 |
| 标量运算 | 加减乘除、幂、exp、sqrt 等每个输出元素计 1；特殊函数是统一的算法等价估算，不代表硬件代价 |
| 激活 | ReLU 每元素 1，ReLU6/Hardtanh 2，Sigmoid 4，SiLU 5，GELU 默认 erf 近似口径 5、tanh 口径 9；clamp 按实际边界数计比较 |
| 归约 | sum：输入元素数减输出元素数；mean 再加每组一次除法 |
| 稳定 Softmax | `5N − 2G`；LogSoftmax 为 `5N − G`，N 为元素数，G 为归约组数 |
| LayerNorm | `5N + 2G`，可选 affine weight/bias 各加 N；均值、中心化、方差、epsilon、rsqrt 和缩放的算法估算 |
| eval BatchNorm | `2N + 2C`，可选 weight/bias 各加 N；C 为通道数，不执行训练统计更新 |
| 平均/最大池化 | 每个输出分别计 kernel 元素数 / kernel 元素数减 1，包含稠密边界窗口；自适应平均池化按真实分箱覆盖范围计算 |
| 线性/双线性/双三次/三线性插值 | 每输出元素分别计 3 / 7 / 31 / 15，统一采用加权求和估算，忽略坐标与插值系数生成；抗锯齿插值需另补规则 |
| 融合 SDPA | 稠密 QK、AV 加 score 缩放、Softmax 和显式 mask 相加；causal 也按稠密理论口径；仅支持无 dropout 的 BHLD 输入 |
| 内存和索引操作 | reshape、转置、复制、repeat、cat、pad、索引、最近邻插值等显式计 0 浮点运算，仍包含在时延内；索引/条件用比较不计入浮点算术 |

上述口径包含常用非 MAC 运算，可能与只统计卷积/矩阵乘的论文数字不同。激活、归一化、池化、插值等使用明确的理论估算；比较模型应共用同一规则版本。

CSV 中 `flops_complete` 表示该次实际前向没有未分类算子，不能保证其他输入、数据相关分支或第三方自定义内核已被覆盖。每个详细 JSON 包含逐算子的调用次数、FLOPs、排除理由及未覆盖算子的输入形状。

**存在未覆盖算子时**：CSV 的正式 `gflops` 留空，已覆盖部分放入 `counted_gflops`；控制台和 CSV/JSON 均列出未覆盖算子。所有模型结果保存后以非零状态退出。不得将 `counted_gflops` 当作完整模型 GFLOPs 引用。模型执行错误、权重错误和规则错误直接报错。

扩展计数规则示例：

```python
# 项目中的 my_flop_rules.py
def build_rules():
    def custom_gate(arguments, output):
        # 必须根据该算子的实际算法确定公式；此例假定每元素一次乘和一次加。
        return 2 * output.numel()

    return {"my_ops.gate.default": custom_gate}
```

通过 `--flop-rules my_flop_rules:build_rules` 加载。键可用详细报告中的完整 overload 名称（优先匹配），或 packet 名称。函数收到按算子 schema 绑定的参数字典和实际输出，返回该次调用的 FLOPs。规则只读取形状与元数据，不重新执行前向。不支持的高层自定义算子应在这里补充整体规则，不能漏计，也不能与其内部已统计算子重复计数。

## 时延

模型和输入在计时前完成设备放置。输入是固定种子的 CPU 随机 FP32 张量，传输到目标设备后复用；数据集读取、数据集预处理和输入传输不计时。若某模型必须在网络调用前转换通道、缩放或归一化，应放入适配器 `forward`，使它同时进入 FLOPs 和时延。

`eval()` + `torch.inference_mode()`，禁用 autocast 和 TF32；统一关闭 cuDNN benchmark。先预热，每次正式测量前同步选中 GPU，使用 `perf_counter_ns()` 计时，完整适配器前向完成后再次同步。输出张量不搬回 CPU，其释放放在计时外。报告所有正式样本的算术平均值，另保存总体标准差与原始样本。该时延包含 CPU 提交、必需的输入适配与 GPU 执行；批量大于 1 时为批次时延除以 batch size，不代表独立单 patch 的响应时延。

FLOPs 审计在时延测量后单独运行，审计算子拦截开销不计入时延。默认不编译、不做 CUDA Graph 或混合精度优化。公平比较应使用同一设备、软件环境和后台负载条件；CSV 记录实际设备、PyTorch/CUDA/cuDNN 版本、输入尺寸、batch size、精度、预热/测量次数、随机种子及运行选项。

## 接入后续模型

若后续模型已支持 `build_model` 和相同前向接口，无需新增效率统计代码。否则写一个小型适配器，工厂负责构建、严格加载权重和恢复训练冻结策略，`forward` 负责双模态转换并调用实际完整推理入口。

```python
# models/comparison_efficiency.py
from utils.efficiency_adapters import EfficiencyAdapter

class ComparisonAdapter(EfficiencyAdapter):
    def forward(self, rgb, dsm):
        # 根据该模型的真实输入接口适配；转换每次执行，不能预先缓存。
        return self.model(rgb, dsm.unsqueeze(1))

def build_adapter(cfg, checkpoint, train_epoch):
    model = build_comparison_model(cfg["model"])
    load_comparison_weights(model, checkpoint)  # 该模型自己的严格加载入口
    apply_comparison_training_freeze(model, cfg, train_epoch)
    return ComparisonAdapter(model, {
        "model_type": cfg["model"]["type"],
        "task_weights": "checkpoint",
        "train_epoch": train_epoch,
        "freeze_policy": "comparison_training_policy",
    })
```

上例中的三个模型专用函数由比较模型实现。使用 `--adapter models.comparison_efficiency:build_adapter`，或在配置中设置 `"efficiency": {"adapter": "models.comparison_efficiency:build_adapter"}`。适配器的 `.model` 必须持有完整模型及所有可学习的输入适配模块，使参数统计完整；不得只暴露编码器。工厂不放置设备、不包装多卡、不提前计算或缓存输入特征。除接入工厂与前向适配外，参数统计、覆盖审计、计时和 CSV 输出全部共用。
