# 通用双模态模型效率统计与接入说明

统一入口是 `tools/model_efficiency.py`。脚本复用项目的模型构建、权重迁移和完整推理前向。FLOPs 由 `fvcore.nn.FlopCountAnalysis` 统计，时延通过单独的 eager 前向测量。

## 运行

在项目根目录执行。测速环境需要安装 `fvcore`，当前实验使用 `0.1.5.post20221221`。默认单卡 `cuda:0`、RGB `[1,3,256,256]`、DSM `[1,256,256]`、FP32、预热 50 次、测量 200 次。

```bash
conda activate testnet
conda run --no-capture-output -n efficiency01 python tools/model_efficiency.py \
  --config configs/cfg_vaihingen_vit-l_3090.jsonc \
  --checkpoint /path/to/scsam/best_miou.pth \
  --model-name ScSAM \
  --output work_dirs/efficiency/scsam.csv
```

同一次运行依次统计多个模型，所有模型共用测量设置。下面以相同 ViT-B 主干为例；权重路径应替换为各自真实文件。最好直接使用每次训练保存的有效配置，确保模型结构、冻结阶段和权重一致。

```bash
conda activate testnet
conda run --no-capture-output -n efficiency01 python tools/model_efficiency.py \
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
| `--flop-rules` | `module:factory`，返回 fvcore JIT 算子与计数函数的映射 |
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

脚本在与时延相同的设备、输入、FP32、`eval()` 和 `inference_mode()` 条件下，用 fvcore JIT 跟踪一次完整适配器前向。MFNet 和 ScSAM 调用原始 `model(rgb, dsm, mode="Test")`，计数涵盖实际调用的双路编码、融合、解码和输出恢复。

沿用 fvcore 的约定，**一次融合乘加计为 1**。GFLOPs 为总计数除以 `10^9` 和 batch size。卷积、转置卷积、矩阵乘、`einsum` 和常见归一化使用 fvcore 自带规则。逐元素运算、归约和部分插值使用形状估算。整数、比较、张量布局变换和评估模式下的 dropout 不计入浮点运算。报告会记录 fvcore 版本和计数口径。此前采用一次 MAC 计 2 FLOPs 的结果应保留其原口径。 [fvcore 源码](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py)

RS³Mamba 的 `prim::PythonOp.SelectiveScanFn` 使用 VMamba 的前缀扫描估算。审计同时核对 JIT 节点数和实际调用 CUDA selective scan 的次数。节点与原生调用数不一致时，扫描计数标为未覆盖。 [VMamba 计数规则](https://github.com/MzeroMiko/VMamba/blob/main/classification/models/csms6s.py#L932)

`flops_complete` 表示在当前输入和约定下，跟踪图里的算子已被计数或按规则排除。每份 JSON 保留按算子汇总的 FLOPs、未覆盖算子的出现次数和未调用的模块。存在未覆盖算子时，正式 `gflops` 留空，部分计数放在 `counted_gflops`。所有模型结果写出后，脚本以非零状态退出。

通过 `--flop-rules my_flop_rules:build_rules` 加载附加规则。映射键使用 JIT 名称，例如 `aten::my_op` 或 `prim::PythonOp.MyFunction`。计数函数接收 JIT `inputs` 和 `outputs`，并返回数值或按类别统计的 `Counter`。值为 `None` 表示按实验口径有意排除该算子。

```python
from fvcore.nn.jit_handles import elementwise_flop_counter


def build_rules():
    # This example assumes one floating-point operation per output element.
    return {"my_ops::gate": elementwise_flop_counter(0, 1)}
```

## 时延

模型和输入在计时前完成设备放置。输入是固定种子的 CPU 随机 FP32 张量，传输到目标设备后复用；数据集读取、数据集预处理和输入传输不计时。若某模型必须在网络调用前转换通道、缩放或归一化，应放入适配器 `forward`，使它同时进入 FLOPs 和时延。

`eval()` + `torch.inference_mode()`，禁用 autocast 和 TF32；统一关闭 cuDNN benchmark。先预热，每次正式测量前同步选中 GPU，使用 `perf_counter_ns()` 计时，完整适配器前向完成后再次同步。输出张量不搬回 CPU，其释放放在计时外。报告所有正式样本的算术平均值，另保存总体标准差与原始样本。该时延包含 CPU 提交、必需的输入适配与 GPU 执行；批量大于 1 时为批次时延除以 batch size，不代表独立单 patch 的响应时延。

FLOPs 审计在时延测量后单独运行，fvcore 跟踪开销不计入时延。默认不编译、不做 CUDA Graph 或混合精度优化。公平比较应使用同一设备、软件环境和后台负载条件；CSV 记录实际设备、PyTorch/CUDA/cuDNN 版本、输入尺寸、batch size、精度、预热/测量次数、随机种子及运行选项。

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
