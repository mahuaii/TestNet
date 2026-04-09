from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

import torch

from .evaluator import Evaluator
from .inferencer import Inferencer
from utils.checkpoint import CheckpointManager
from utils.logger import Logger
from utils.stat_tracker import StatTracker
from utils.timer import AnchorTimer


class Trainer(ABC):
    """
    输入：
    - model、optimizer、scheduler
    - train_loader、val_loader
    - logger、checkpoint_manager、evaluator
    - task、inferencer
    - device、cfg

    职责：
    - 统一编排训练、验证、日志和保存流程
    - 串联模型、任务层、推理层和评估层
    - 管理 epoch、step 等训练状态

    输出：
    - 训练过程中的日志与 checkpoint
    - 训练阶段和验证阶段的指标字典
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Any,
        val_loader: Any,
        logger: Logger,
        checkpoint_manager: CheckpointManager,
        evaluator: Evaluator,
        device: torch.device,
        cfg: dict[str, Any],
        inferencer: Inferencer,
        scheduler: Any = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.inferencer = inferencer
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.device = device
        self.cfg = cfg

        self.epoch = 1
        self.global_step = 0
        self.max_epochs = cfg["max_epochs"]
        self.timer = AnchorTimer()

        effective_batch_size = cfg.get("effective_batch_size", self.train_loader.batch_size)
        if effective_batch_size < self.train_loader.batch_size:
            raise ValueError("effective_batch_size must be greater than or equal to batch_size")
        if effective_batch_size % self.train_loader.batch_size != 0:
            raise ValueError("effective_batch_size must be divisible by batch_size")
        self.grad_accum_steps = effective_batch_size // self.train_loader.batch_size
        self.total_steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.grad_accum_steps)
        )

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _load_model_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict):
            if "model" in state_dict:
                model_state = state_dict["model"]
            elif "model_state_dict" in state_dict:
                model_state = state_dict["model_state_dict"]
        self.model.load_state_dict(model_state)

    def _load_training_state(self, path: str) -> None:
        state_dict = self.checkpoint_manager.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler is not None and state_dict["scheduler"] is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        self.epoch = int(state_dict["epoch"])
        self.global_step = int(state_dict["global_step"])

    def train(self) -> None:
        """
        - 执行完整训练流程
        - 处理权重加载、断点恢复、epoch 循环和验证触发
        - 通过 logger 和 checkpoint_manager 产生日志与保存结果
        """
        # 权重加载与断点恢复
        resume_from = self.cfg.get("resume_from")
        load_from = self.cfg.get("load_from")
        if load_from:
            self._load_model_weights(load_from)
        if resume_from:
            self._load_training_state(resume_from)

        # 训练循环
        with self.logger:
            for epoch in range(self.epoch, self.max_epochs + 1):
                self.epoch = epoch
                self.before_epoch()

                self.logger.log_epoch_start(epoch=self.epoch, max_epochs=self.max_epochs)
                self.timer.mark("epoch")

                train_metrics = self.train_one_epoch()  # 训练一个 epoch，得到 epoch 级指标

                train_time_seconds = self.timer.elapsed("epoch")

                self.after_epoch(
                    train_metrics,
                    train_time_seconds=train_time_seconds,
                )

                # 验证（触发条件：配置了 val_epoch_interval 且当前 epoch 满足间隔要求）
                if (
                    self.cfg["val_epoch_interval"] > 0
                    and epoch % self.cfg["val_epoch_interval"] == 0
                ):
                    self.timer.mark("validation")
                    val_metrics = self.validate()
                    validation_time_seconds = self.timer.elapsed("validation")
                    self.after_val(
                        val_metrics,
                        validation_time_seconds=validation_time_seconds,
                    )

    def before_epoch(self) -> None:
        """
        - 预留 epoch 开始前的 hook
        - 由子类决定是否执行 scheduler.step()、计时初始化等 epoch 级行为
        """
        return None

    def train_one_epoch(self) -> dict[str, float]:
        """
        - 执行单个 epoch 的 batch 训练循环
        - 汇总运行指标

        输出：
        - train_metrics：当前 epoch 的平均训练指标dict
        """
        self.model.train()
        self.timer.mark("epoch")
        self.timer.mark("log_interval")

        step = 0
        batch_count_in_step = 0  # 当前 step 已处理的 batch 数
        batch_count_in_epoch = 0  # 当前 epoch 已处理的 batch 数
        epoch_metrics = StatTracker()  # epoch 级统计
        # total_batches 用于识别 epoch 尾部，保证最后一个不完整累积窗口也能正确缩放和提交。
        total_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        step_metrics = StatTracker()  # step 级统计
        # batch循环
        for batch in self.train_loader:
            # 含当前 batch 在内的剩余 batch 数
            remaining_batches = total_batches - batch_count_in_epoch
            # 尾批不足 grad_accum_steps 时，按实际 batch 数缩放 loss，避免最后一次梯度被压小。
            accum_batch_target = min(self.grad_accum_steps, remaining_batches)

            # 前向、反向
            loss, metrics = self.train_forward(batch)
            (loss / accum_batch_target).backward()

            # 当前 step 内指标累积
            step_metrics.update_mean_stats(metrics)
            epoch_metrics.update_mean_stats(metrics)
            batch_count_in_step += 1
            batch_count_in_epoch += 1

            # 满足累积窗口或到达 epoch 末尾时，才提交一次 step。
            if (batch_count_in_step == self.grad_accum_steps) or (
                batch_count_in_epoch == total_batches
            ):
                self._optimize_step()
                step += 1
                self.after_step(
                    step,
                    step_metrics.get_aggregated_stats(),
                    is_last_step_of_epoch=(batch_count_in_epoch == total_batches),
                )

                step_metrics = StatTracker()
                batch_count_in_step = 0

        return epoch_metrics.get_aggregated_stats()

    @abstractmethod
    def train_forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        输入：
        - 单个 batch，推荐格式：
          {
            "inputs": {"rgb": ..., "dsm": ..., ...},
            "target": ...,
            "meta": ...,
          }

        职责：
        - 执行单步训练前向并返回 loss 与指标
        - 保持多模态输入为 `inputs` 字典，不拆为多个位置参数
        - 由具体 Trainer 子类实现项目相关 loss、预测与指标逻辑

        输出：
        - 当前 batch 的 loss tensor 和训练指标字典
        """
        ...

    def after_step(
        self,
        step: int,
        step_stats: dict[str, float],
        is_last_step_of_epoch: bool = False,
    ) -> None:
        """
        - 输出 step 级日志
        - 执行 global step 级 checkpoint 保存
        """
        # 输出 step 级日志（满足间隔要求或 epoch 尾部）
        if step % int(self.cfg["log_step_interval"]) == 0 or is_last_step_of_epoch:
            self.logger.log_train_step(
                epoch=self.epoch,
                max_epochs=self.max_epochs,
                step=step,
                total_steps=self.total_steps_per_epoch,
                global_step=self.global_step,
                step_stats=step_stats,
                interval_time_seconds=self.timer.elapsed("log_interval"),
                epoch_elapsed_seconds=self.timer.elapsed("epoch"),
                lr=self.lr,
            )
            self.timer.mark("log_interval")

        # 保存训练状态（满足 global step 间隔要求）
        save_step_interval = self.cfg["save_step_interval"]
        if save_step_interval > 0 and self.global_step % save_step_interval == 0:
            path = self.checkpoint_manager.save_training_state(
                name=f"global_step_{self.global_step}.pth",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.epoch,
                global_step=self.global_step,
            )
            self.logger.log_checkpoint_saved(path)

    def after_epoch(
        self,
        train_metrics: dict[str, float],
        train_time_seconds: float,
    ) -> None:
        """
        - 输出训练 epoch 级日志
        - 按配置执行 epoch 级 checkpoint 保存
        """
        self.logger.log_epoch_end(
            train_time_seconds=train_time_seconds,
            train_metrics=train_metrics,
            lr=self.lr,
        )

        # 保存训练状态（满足 epoch 间隔要求）
        save_epoch_interval = int(self.cfg["save_epoch_interval"])
        if save_epoch_interval > 0 and self.epoch % save_epoch_interval == 0:
            path = self.checkpoint_manager.save_training_state(
                name=f"epoch_{self.epoch}.pth",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.epoch,
                global_step=self.global_step,
            )
            self.logger.log_checkpoint_saved(path)

    def after_val(
        self,
        val_metrics: dict[str, float],
        validation_time_seconds: float,
    ) -> None:
        """
        - 输出验证阶段日志
        """
        self.logger.log_validation_timing(
            test_time_seconds=validation_time_seconds,
            epoch=self.epoch,
            val_metrics=val_metrics,
        )

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """
        - 执行完整验证流程
        - 调用 inferencer 进行推理
        - 调用 evaluator 聚合验证结果

        输出：
        - 验证指标字典
        """
        outputs = []
        self.model.eval()
        for batch in self.val_loader:
            outputs.append(
                self.inferencer.run_batch_infer(
                    model=self.model,
                    batch=batch,
                    device=self.device,
                )
            )
        val_metrics = self.evaluator.evaluate(
            outputs=outputs,
            model=self.model,
            dataloader=self.val_loader,
            inferencer=self.inferencer,
            trainer=self,
        )
        return val_metrics

    def _optimize_step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1
