# 多模态分割项目 Trainer 模板规范

## 模板定位

本模板固定采用方案 A。

方案 A 定义为：

- `Trainer` 直接持有 `Logger`
- `Trainer` 直接持有 `CheckpointManager`
- `Trainer` 直接调用 `Evaluator`
- 不引入 hook、callback、事件分发或插件式训练框架

本文档不给可选分支，只给一套明确模板。

## 适用范围

本模板面向个人维护的多模态分割项目，适用于如下场景：

- 输入包含两种或多种模态
- 模型结构可能包含独立 backbone、模态交互模块、融合模块、解码头
- 训练阶段需要同时支持按 `iter` 和按 `epoch` 输出
- 项目需要长期迭代，但不希望一开始就做过重的工程抽象

本文档是模板规范，不绑定当前仓库实现，不提供具体代码。

## 设计目标

- 训练主流程清晰
- 模型、数据、训练、日志、保存职责分离
- 支持多模态输入扩展，但不把模态细节写死到 `Trainer`
- 适合后续迁移到新模型或新数据集

## 核心约束

### 1. `Trainer` 只负责流程调度

`Trainer` 负责：

- 控制 epoch / iter 主循环
- 调用 `train_step()`
- 控制日志、验证、保存的触发时机
- 维护训练状态

`Trainer` 不负责：

- 直接实现日志后端
- 直接实现 checkpoint 文件管理策略
- 直接处理具体数据集字段差异
- 直接写模型内部逻辑

### 2. 模型只关心前向与输出

模型模块只负责：

- 接收整理好的多模态输入
- 执行前向
- 返回训练或推理所需输出

模型不应负责：

- 日志打印
- checkpoint 保存
- epoch/iter 控制

### 3. 数据层负责统一 batch 格式

无论使用 RGB、DSM、深度、热红外、SAR 或其他模态，进入 `Trainer` 的 batch 格式都应统一。

推荐统一为字典：

```text
batch = {
  "inputs": {
    "rgb": ...,
    "aux": ...,
  },
  "target": ...,
  "meta": ...
}
```

约束：

- `Trainer` 不写死某个模态名字
- 模型从 `inputs` 中取自己需要的键
- `meta` 只放路径、尺寸、样本标识等辅助信息

### 4. 分离职责，不分离触发点

按 `iter` 和按 `epoch` 的输出都可以保留在训练流程中触发，但实现必须拆开：

- `Trainer` 决定何时输出
- `Logger` 决定如何输出
- `CheckpointManager` 决定如何保存

## 目录模板

```text
project/
  configs/
  datasets/
    transforms/
    multimodal_dataset.py
  models/
    backbones/
    fusion/
    decode_heads/
    losses/
    segmentor.py
  engine/
    trainer.py
    evaluator.py
  utils/
    logger.py
    checkpoint.py
    meter.py
    dist.py
  tools/
    train.py
    test.py
  work_dirs/
```

目录约定：

- `datasets/` 负责多模态样本组织与预处理
- `models/segmentor.py` 作为总模型装配入口
- `engine/trainer.py` 负责训练调度
- `engine/evaluator.py` 负责验证与测试指标汇总
- `utils/` 只放无业务耦合的通用组件

## 模块边界

### `tools/train.py`

职责：

- 读取配置
- 构建 dataloader
- 构建 model / optimizer / scheduler
- 构建 trainer
- 处理 resume 或 finetune 入口
- 调用 `trainer.train()`

约束：

- 不写训练细节
- 不写大段日志逻辑
- 不写具体保存逻辑
- 关键配置字段直接按必填处理，不通过默认值兜底训练行为

### `engine/trainer.py`

职责：

- 维护 `epoch`、`global_step`、最佳指标等状态
- 管理训练主循环
- 触发 iter 日志、epoch 日志、验证、保存
- 调用 AMP、梯度累积、学习率调度等训练策略

固定保留的方法：

- `train()`
- `train_one_epoch()`
- `train_step(batch)`
- `validate()`
- `after_iter(...)`
- `after_epoch(...)`

### `engine/evaluator.py`

职责：

- 聚合验证集或测试集输出
- 计算分割指标
- 返回统一字典结果

统一输出格式：

```text
{
  "loss": ...,
  "miou": ...,
  "f1": ...,
  "oa": ...
}
```

### `utils/logger.py`

职责：

- 控制台日志
- 文本日志
- TensorBoard 或 WandB
- 结构化指标落盘

固定接口：

- 支持 `log_iter()`
- 支持 `log_epoch()`
- 支持 `log_message()`

当前约束：

- `log_iter()` 接收分组后的指标，而不是一个混合 `metrics` 字典
- `log_epoch()` 分别接收 `train_metrics`、`train_state_metrics`、`val_metrics`
- 日志输出按组组织，例如 `running[...]`、`state[...]`、`train[...]`、`val[...]`

### `utils/checkpoint.py`

职责：

- 保存 latest checkpoint
- 保存 best checkpoint
- 按 epoch 保存
- 按 iter 保存
- 恢复训练状态

固定接口：

- `save(...)`
- `save_best(...)`
- `resume(...)`

### `utils/meter.py`

职责：

- 跟踪平均 loss
- 跟踪分项 loss
- 跟踪数据耗时与迭代耗时
- 保存最近一次训练过程指标

约束：

- `RunningMetricTracker` 只负责训练过程中可累积、可平均的指标
- 不把验证指标和训练状态指标混进同一个统计器
- 不要把统计临时变量散落在训练循环中

## Trainer 主流程规范

固定主流程：

```text
before_train
for epoch in max_epochs:
  before_epoch
  train_one_epoch
  if need_validation:
    validate
  after_epoch
after_train
```

其中：

- `train_one_epoch()` 负责 batch 级循环
- `validate()` 负责整轮验证
- `after_epoch()` 统一处理 epoch 末副作用

## iter 级与 epoch 级输出规范

### iter 级输出

iter 级输出放在 `train_one_epoch()` 内触发，常见内容：

- 当前总 loss
- 分项 loss
- 当前 lr
- data time
- iter time
- 显存占用
- 当前 epoch / iter / global_step

固定触发规则：

- 每 `log_interval` 个 iter 调用一次 `logger.log_iter(...)`
- 每 `save_iter_interval` 个 iter 调用一次 `checkpoint.save(...)`

约束：

- 可以在 iter 循环中调用 logger
- 不能在 iter 循环中直接堆大量格式化和存盘细节
- iter 日志应区分运行中指标和训练状态指标
- 例如：`running_metrics={loss, data_time, iter_time}`，`state_metrics={lr}`

### epoch 级输出

epoch 级输出放在 `after_epoch()` 中，常见内容：

- 平均训练 loss
- 验证 loss
- `mIoU`
- `F1`
- `OA`
- 本轮耗时
- 最佳指标是否更新

固定触发规则：

- 每个 epoch 都执行 `logger.log_epoch(...)`
- 每 `val_interval` 个 epoch 验证一次
- 每 `save_epoch_interval` 个 epoch 保存一次
- 若关键指标提升，则额外保存 best model

约束：

- epoch 日志应区分训练平均指标、训练状态指标和验证指标
- 不再通过 `val_` 前缀把验证结果拼进训练指标字典

## 多模态分割项目的特殊约束

### 1. batch 组织不要绑死模态数量

不要在 `Trainer` 里写：

```text
rgb, dsm, label = batch
```

应统一为字典式输入，让模型自行决定读取哪些模态。

### 2. loss 设计按字典返回

多模态分割模型往往不止一个损失项，例如：

- 主分割损失
- 辅助分支损失
- 模态一致性损失
- 融合约束损失

`train_step()` 固定返回：

```text
{
  "loss": ...,
  "loss_seg": ...,
  "loss_aux": ...,
  "loss_fusion": ...
}
```

这样 iter 日志和 epoch 汇总都容易统一。

说明：

- 这些损失项如果发生在训练阶段，进入 `RunningMetricTracker`
- 验证阶段指标由 `Evaluator` 单独返回，不进入训练统计器

### 3. 评估指标统一收口

不要在训练主循环里手写各种分割指标统计。验证阶段的预测收集和指标计算统一放到 `Evaluator` 中。

### 4. 可视化输出单独挂接口

如果后续需要保存可视化结果，例如：

- 输入图像
- 不同模态特征图
- 预测 mask
- 错分区域

如需可视化，单独提供 `Visualizer` 或 `save_visuals()` 接口，不要混入默认训练路径。

## 训练状态规范

`Trainer` 至少维护以下状态：

- `epoch`
- `global_step`
- `max_epochs`
- `best_metric`
- `best_metric_name`
- `is_resumed`

checkpoint 至少保存：

- model state
- optimizer state
- scheduler state
- scaler state
- epoch
- global_step
- best metric

## 配置规范

关键配置字段必须显式提供，缺失时允许进程直接报错，不通过默认值或额外包装校验去修改训练行为。

配置规范只约束原则，不绑定某个仓库当前使用的字段名或层级结构。

建议：

- 模型结构相关参数由模型构建路径显式读取
- 数据相关参数由数据构建路径显式读取
- 优化器与调度器参数由训练入口显式读取
- 训练流程参数由 `Trainer` 显式读取
- 运行时派生出的状态或元信息可以在入口处注入，而不是靠默认值补齐

当前风格约束：

- 关键行为参数使用直接索引，例如 `cfg["train"]`、`train_cfg["max_epochs"]`
- 不使用 `cfg.get(..., default)` 这类会静默改变训练/评估行为的写法
- 不额外包一层 `raise ValueError(...)` 做手写配置兜底，缺失时让底层异常直接暴露

## 推荐命名规范

为保证迁移方便，统一使用以下命名：

- 训练入口：`train()`
- 单轮训练：`train_one_epoch()`
- 单步训练：`train_step()`
- 验证入口：`validate()`
- iter 后处理：`after_iter()`
- epoch 后处理：`after_epoch()`
- 日志模块：`logger`
- 保存模块：`checkpoint_manager`
- 训练过程指标统计：`running_metric_tracker`

## 禁止项

- 不要先上复杂 hook 系统
- 不要先上很多抽象基类
- 不要把所有策略做成注册器
- 不要把 `Trainer` 写成支持所有任务类型的通用框架

这份模板只服务你的多模态分割任务。

## 固定最终形态

固定结构如下：

- 一个 `Trainer`
- 一个 `Evaluator`
- 一个 `Logger`
- 一个 `CheckpointManager`
- 一个 `RunningMetricTracker`

其中：

- iter 日志在 `train_one_epoch()` 中触发
- epoch 汇总在 `after_epoch()` 中触发
- 验证指标由 `Evaluator` 统一返回
- best model 由 `CheckpointManager` 统一管理
- 训练过程指标、训练状态指标、验证指标分组传递，不使用混合 `metrics` 字典

## 模板结论

你的项目模板应保持以下边界：

- 数据层统一多模态 batch
- 模型层只负责前向和输出
- `Trainer` 只负责训练调度
- `Logger` 负责输出
- `CheckpointManager` 负责保存和恢复
- `Evaluator` 负责分割指标汇总

这套结构对多模态分割足够稳定，也足够容易迁移到下一个项目。
