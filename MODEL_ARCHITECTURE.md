# 多模态分割项目 Model 模块定义规范

## 模板定位

本文档用于约束“模型结构部分”的实现方式。

目标不是强行统一所有写法，而是给出一套足够稳定、适合长期迭代、又允许接入现成模型代码的模块规范。

本文档默认服务于如下场景：

- 多模态遥感图像分割
- 模型可能接入现成 backbone，例如 SAM、DINOv2、CLIP、MAE、ConvNeXt、ResNet 等
- 项目由个人或小团队维护
- 希望 AI 可以按文档直接完成模型组装与后续重构

本文档不要求严格遵循某个框架风格，但要求职责边界清楚。

## 设计目标

- 模型结构清晰，便于替换 backbone、fusion、decode head
- 兼容外部现成代码，不要求所有模块都重写
- 训练流程与模型结构解耦
- 输入输出格式稳定，方便 Trainer、Evaluator、Loss 对接
- 便于 AI 按统一约定实现新模型

## 总体原则

### 1. 模型模块只负责结构与前向

模型模块负责：

- 定义网络结构
- 装配 backbone、fusion、neck、decode head、aux head
- 执行前向
- 返回训练或推理所需结果

模型模块不负责：

- 日志输出
- checkpoint 保存
- epoch / iter 控制
- dataloader 细节
- 训练主循环

### 2. 优先模块化，不强求过度抽象

允许：

- 一个 `Segmentor` 类中直接组合多个子模块
- 为接入现成模型保留少量适配代码
- 在 builder 中写必要的条件分支

不建议：

- 为了“看起来面向对象”引入过多基类、注册器、工厂层层套娃
- 把简单模型拆得过碎
- 把训练策略硬塞进模型结构类

原则是：抽象服务于替换与复用，不服务于形式感。

### 3. 外部现成模型优先作为组件接入，而不是改造成工程中心

如果使用 SAM、DINOv2 等现成模型：

- 优先将其视作 `backbone` 或 `encoder` 组件
- 在项目侧补一个轻量适配层
- 尽量不要让外部仓库的目录结构主导本项目的模型组织

换句话说：

- 你的项目应该“调用 SAM”
- 而不是“整个项目围着 SAM 原始代码转”

## 推荐目录

推荐目录：

```text
models/
  backbones/
  fusion/
  necks/
  decode_heads/
  losses/
  segmentor.py
  build.py
```

说明：

- `backbones/` 放单模态 backbone、多模态 encoder、外部模型适配器
- `fusion/` 放跨模态交互与融合模块
- `necks/` 放特征变换、通道对齐、多尺度整理模块
- `decode_heads/` 放语义分割头、auxiliary head
- `losses/` 放 loss 定义或组合器
- `segmentor.py` 作为总模型装配入口
- `build.py` 负责按配置构建模型

## 核心模块边界

### `models/segmentor.py`

职责：

- 定义总模型类，例如 `MultiModalSegmentor`
- 组合 backbone、fusion、neck、decode head、aux head
- 接收统一输入格式
- 控制前向时哪些结果需要输出

不负责：

- 解析命令行参数
- 读取数据集文件
- 保存模型
- 打印训练日志

推荐总模型类最少具备：

- `__init__(...)`
- `forward(inputs, targets=None, mode="tensor" | "loss" | "predict")`

也可以采用更直观的分离写法：

- `forward(inputs)`
- `forward_train(inputs, targets)`
- `forward_infer(inputs)`

二选一即可，不要混用得很乱。

### `models/backbones/`

职责：

- 提取单模态或多模态特征
- 输出单尺度或多尺度 feature maps / tokens

允许：

- 封装现成 backbone
- 加少量 adapter、LoRA、prompt、modality projection
- 处理模态输入维度对齐

不建议：

- 在 backbone 内直接做最终分割分类
- 在 backbone 内混入 loss 计算
- 在 backbone 内耦合 Trainer 状态

推荐输出形式之一：

```text
{
  "rgb_feats": [...],
  "aux_feats": [...],
  "fused_feats": [...],
  "tokens": ...,
}
```

如果模型较简单，也可以直接返回：

```text
features = [...]
```

但同一项目内应尽量统一。

### `models/fusion/`

职责：

- 建模模态之间的交互
- 输出融合后的表征，或更新后的双流表征

适合放在这里的内容：

- cross-attention
- gated fusion
- additive fusion
- token interaction
- feature alignment
- boundary-aware fusion

不适合放在这里的内容：

- 最终分类层
- 数据增强逻辑
- 训练阶段调度逻辑

建议 fusion 模块尽量做到：

- 输入输出语义清楚
- 不依赖具体数据集
- 不依赖训练脚本全局变量

### `models/necks/`

职责：

- 统一 backbone 输出尺度
- 调整通道数
- 组织多尺度特征
- 为 decode head 提供标准化输入

典型例子：

- FPN
- feature pyramid adaptation
- token-to-feature projection
- stage feature projection

如果项目较小，`neck` 可以并入 `segmentor` 或 `decode head`，不强制单独拆文件。

### `models/decode_heads/`

职责：

- 将 backbone 或 neck 输出映射为分割 logits
- 支持主头和辅助头

建议：

- 主头只做分割预测
- 辅助头只负责辅助监督，不承担主流程控制
- 尽量保持输入接口稳定

推荐输出：

- 主输出：`seg_logits`
- 辅助输出：`aux_logits` 或字典形式

### `models/losses/`

职责：

- 定义 loss 函数
- 定义多项 loss 的组合方式

建议：

- 单个 loss 写成独立模块
- loss 聚合器单独写
- 如果 loss 很简单，也可以放到 trainer 或 segmentor 外层调用，但不要把复杂 loss 逻辑散落在训练脚本里

## 输入输出规范

### 输入规范

模型统一接收：

```text
inputs = {
  "rgb": ...,
  "dsm": ...,
  "sar": ...,
  ...
}
```

要求：

- `Trainer` 传给模型的是 `inputs` 字典，而不是多个位置参数
- 模型自己读取需要的模态键
- 不要求所有模型支持所有模态，但缺失键要报清楚错误

不推荐：

```text
model(x, y, z, mode="Train")
```

原因：

- 模态顺序难维护
- 后续新增模态时接口容易失控
- 不利于通用 Trainer

### 目标与前向输出规范

训练时建议返回结构化结果，而不是只返回裸 tensor。

推荐：

```text
{
  "seg_logits": ...,
  "aux_logits": {
    "rgb": ...,
    "aux": ...,
  },
  "features": ...,
}
```

如果项目早期想简单一点，也至少建议：

```text
{
  "seg_logits": ...,
}
```

推理时建议返回：

```text
{
  "seg_logits": ...,
  "pred_mask": ...,
}
```

是否在模型内部生成 `pred_mask` 可以灵活处理，但项目内应统一。

## 关于 `mode` 参数

可以使用 `mode`，但不要滥用。

允许：

- `mode="loss"`：返回训练所需结构
- `mode="predict"`：返回推理结果

不建议：

- 在 `mode` 里塞大量训练策略分支
- 用 `mode="Train"` / `mode="Test"` 控制完全不同的模型路径

更推荐的思路是：

- 模型结构前向保持尽量一致
- 训练和推理只在“返回内容”上有差异

## 关于现成模型接入

### 接入原则

接入 SAM、DINOv2 等模型时，优先遵循：

1. 外部代码尽量少改
2. 项目内增加一层适配器
3. 对外暴露统一的 backbone 接口

例如：

- `SAMBackboneAdapter`
- `DinoV2BackboneAdapter`
- `DualModalEncoder`

### 适配器应负责什么

适配器负责：

- 初始化外部模型
- 处理权重加载
- 抽取项目需要的中间层特征
- 将输出整理成项目统一格式

适配器不负责：

- 训练主流程
- checkpoint 策略
- 日志

### 对外部模型代码的修改原则

允许的修改：

- 增加中间层特征导出
- 增加多模态输入投影层
- 增加轻量融合块或 PEFT 模块
- 修正与项目接口不匹配的 forward 输出

不建议的修改：

- 在外部模型源码里掺入项目训练脚本逻辑
- 到处插入数据集特定分支
- 把项目配置读取逻辑写进外部 backbone 类

## 参数初始化与预训练权重加载

建议将以下逻辑放在 builder 或单独初始化函数中：

- 从配置判断使用哪种 backbone
- 加载预训练权重
- 冻结哪些参数
- 打开哪些 adapter / LoRA / prompt 参数训练

不建议直接写在总模型主类里，尤其不建议在 `forward()` 中处理。

原因：

- 初始化逻辑和前向逻辑是两类职责
- 便于复用相同模型结构做不同训练策略

## 配置传递规范

模型类可以接收配置，但建议使用以下方式之一：

- 显式参数
- 配置对象
- 配置字典

不建议：

- 在模型类内部直接调用 `parse_args()`

原因：

- 模型会依赖命令行环境
- 难以测试
- 难以从 notebook、脚本、服务端复用

## AI 实现约束

如果让 AI 编写模型代码，应默认遵循以下约束：

- 不把 Trainer 逻辑写进模型
- 不在模型类内部解析命令行参数
- 不把日志、保存、评估写进 `models/`
- 优先新增适配器，而不是大量改外部 backbone 原始代码
- 模型输入优先使用 `inputs` 字典
- 模型输出优先使用结构化字典
- 主干、融合、解码头、辅助头尽量拆开
- 除非项目已经有统一 registry，否则不要额外造复杂注册系统

## 推荐的最小实现模板

最小建议不是“所有模块都独立文件”，而是至少做到：

```text
models/
  backbones/
    sam_adapter.py
  fusion/
    cross_modal_fusion.py
  decode_heads/
    simple_seg_head.py
  segmentor.py
  build.py
```

其中：

- `sam_adapter.py` 负责接入外部 backbone
- `cross_modal_fusion.py` 负责模态交互
- `simple_seg_head.py` 负责输出分割 logits
- `segmentor.py` 负责组装
- `build.py` 负责根据配置创建模型

如果模型更复杂，再继续细拆。

## 建议的总模型组织方式

推荐总模型遵循下面的装配顺序：

```text
inputs
  -> modality-specific stem / projection
  -> backbone encoder
  -> cross-modal fusion
  -> neck
  -> decode head
  -> seg_logits
```

如果有辅助监督：

```text
intermediate features
  -> aux head
  -> aux_logits
```

## 不必强求的内容

以下内容不必一开始就强行加入：

- 抽象基类层层继承
- 复杂 registry 系统
- hook 机制
- callback 机制
- 插件式模型装配系统
- 过度细分到一个文件只放一个很短的小层

如果将来模型族越来越多，再逐步增强工程抽象。

## 一句话规范

可以灵活接入现成模型，但项目自己的模型边界必须清楚：

- `backbone` 负责提特征
- `fusion` 负责模态交互
- `neck/head` 负责预测
- `segmentor` 负责装配
- `Trainer` 负责训练流程

不要让任何一个类同时承担“结构定义、训练策略、配置入口、日志保存、推理脚本控制”这几类职责。

## 发给 AI 的简版指令

如果后续需要把这份规范直接发给 AI 执行，可附上下面这段：

```text
请按本项目的模型规范实现代码：

1. 模型代码只写在 models/ 下，不要写训练流程、日志、保存逻辑。
2. 总模型以 segmentor.py 为入口，只负责组装 backbone / fusion / neck / decode head。
3. 允许接入现成模型代码，但请优先写 adapter，而不是大改外部源码。
4. 模型输入统一用 inputs 字典，不要设计成多个位置参数。
5. 模型输出优先返回结构化字典，至少包含 seg_logits。
6. 不要在模型类内部 parse_args。
7. 不要引入过重的抽象，保持结构清晰、便于替换模块。
8. 如果是多模态模型，请明确哪些模块负责单模态编码，哪些模块负责跨模态融合。
9. 如果需要辅助监督，请使用 aux head，不要把辅助逻辑散落在 trainer 中。
10. 除非已有现成体系，否则不要额外设计复杂 registry / hook / callback 框架。
```

