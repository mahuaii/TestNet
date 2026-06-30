from __future__ import annotations

from abc import ABC, abstractmethod
import re
from pathlib import Path
from typing import Any

import torch

from tools.update_experiments_tsv import (
    update_experiments_tsv_best_metrics,
    update_experiments_tsv_status,
)

from .evaluator import Evaluator
from utils import (
    OPTIMIZER_GROUP_METADATA_KEYS,
    normalize_legacy_optimizer_state_dict,
    normalize_legacy_state_dict_keys,
    restore_optimizer_group_metadata,
)
from utils.checkpoint_manager import CheckpointManager
from utils.logger import Logger
from utils.stat_tracker import StatTracker
from utils.timer import AnchorTimer


class Trainer(ABC):
    """
    输入：
    - model、optimizer、scheduler
    - train_loader、val_loader
    - logger、evaluator
    - inferencer
    - device、cfg

    职责：
    - 统一编排训练、验证、日志和保存流程
    - 串联模型、推理层和评估层
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
        evaluator: Evaluator,
        device: torch.device,
        cfg: dict[str, Any],
        inferencer: Any,
        scheduler: Any = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.inferencer = inferencer
        self.logger = logger
        self.device = device
        self.cfg = cfg

        self.epoch = 1
        self.global_step = 0
        self.best_miou = 0.0
        self.pending_validation_epoch: int | None = None
        self.max_epochs = cfg["max_epochs"]
        self.timer = AnchorTimer()
        self.total_steps_per_epoch = max(1, len(self.train_loader))

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _load_model_weights(self, path: str):
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict):
            if "model" in state_dict:
                model_state = state_dict["model"]
            elif "model_state_dict" in state_dict:
                model_state = state_dict["model_state_dict"]
        self.model.load_state_dict(normalize_legacy_state_dict_keys(model_state))

    def _load_training_state(self, path: str):
        state_dict = CheckpointManager.load(path)
        self.model.load_state_dict(normalize_legacy_state_dict_keys(state_dict["model"]))
        optimizer_group_metadata = [
            {
                key: group[key]
                for key in OPTIMIZER_GROUP_METADATA_KEYS
                if key in group
            }
            for group in self.optimizer.param_groups
        ]
        self.optimizer.load_state_dict(normalize_legacy_optimizer_state_dict(state_dict["optimizer"]))
        restore_optimizer_group_metadata(self.optimizer, optimizer_group_metadata)
        if self.scheduler is not None and state_dict["scheduler"] is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        checkpoint_epoch = int(state_dict["epoch"])
        resume_epoch = self._resolve_resume_epoch(path=path, state_dict=state_dict)
        checkpoint_global_step = int(state_dict["global_step"])
        completed_epoch = checkpoint_epoch if resume_epoch > checkpoint_epoch else checkpoint_epoch - 1
        self.logger.truncate_after_completed_epoch(max(0, completed_epoch))
        self.logger.purge_tensorboard_after_global_step(checkpoint_global_step)
        self.epoch = resume_epoch
        self.global_step = checkpoint_global_step
        self.best_miou = float(state_dict.get("best_miou", 0.0))
        self.pending_validation_epoch = (
            checkpoint_epoch
            if bool(state_dict.get("validation_pending", False))
            else None
        )

    @staticmethod
    def _resolve_resume_epoch(path: str, state_dict: dict[str, Any]) -> int:
        if "resume_epoch" in state_dict:
            return int(state_dict["resume_epoch"])

        checkpoint_epoch = int(state_dict["epoch"])
        checkpoint_name = Path(path).name
        if (
            checkpoint_name == "latest.pth"
            or checkpoint_name == "best_miou.pth"
            or re.fullmatch(r"epoch_\d+\.pth", checkpoint_name) is not None
        ):
            return checkpoint_epoch + 1
        return checkpoint_epoch

    def train(self):
        """
        - 执行完整训练流程
        - 处理权重加载、断点恢复、epoch 循环和验证触发
        - 通过 logger 和 CheckpointManager 产生日志与保存结果
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
            if self.pending_validation_epoch is not None:
                self._validate_and_finalize_epoch(
                    validation_epoch=self.pending_validation_epoch,
                    resume_epoch=self.epoch,
                )
                self.pending_validation_epoch = None

            for epoch in range(self.epoch, self.max_epochs + 1):
                self.epoch = epoch
                self.before_epoch()

                self.logger.log_epoch_start(epoch=self.epoch, max_epochs=self.max_epochs)
                self.after_epoch_start_logged()
                self.timer.mark("epoch")

                train_metrics = self.train_one_epoch()  # 训练一个 epoch，得到 epoch 级指标

                train_time_seconds = self.timer.elapsed("epoch")

                val_epoch_interval = int(self.cfg.get("val_epoch_interval", 0))
                validation_pending = (
                    val_epoch_interval > 0 and epoch % val_epoch_interval == 0
                )
                self.after_epoch(
                    train_metrics,
                    train_time_seconds=train_time_seconds,
                    validation_pending=validation_pending,
                )

                if validation_pending:
                    self._validate_and_finalize_epoch(
                        validation_epoch=epoch,
                        resume_epoch=epoch + 1,
                    )

    def before_epoch(self):
        """
        - 预留 epoch 开始前的 hook
        - 由子类决定是否执行 scheduler.step()、计时初始化等 epoch 级行为
        """
        update_experiments_tsv_status(
            work_dir=self.cfg["work_dir"],
            epoch=self.epoch,
            max_epochs=self.max_epochs,
        )

    def after_epoch_start_logged(self) -> None:
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
        epoch_metrics = StatTracker()  # epoch 级统计
        log_window_metrics = StatTracker()  # 日志窗口级统计
        self.optimizer.zero_grad()

        # batch循环
        for batch in self.train_loader:
            # 前向、反向
            loss, metrics = self._run_train_forward(batch)
            loss.backward()

            log_window_metrics.update_mean_stats(metrics)
            epoch_metrics.update_mean_stats(metrics)
            self.optimize_step()
            step += 1
            self.after_step(
                step,
                log_window_metrics,
                is_last_step_of_epoch=(step == len(self.train_loader)),
            )
            if step % int(self.cfg["log_step_interval"]) == 0 or step == len(self.train_loader):
                log_window_metrics = StatTracker()

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

    def before_train_forward(self) -> None:
        intermediate_stats = getattr(self.model, "intermediate_stats", None)
        if intermediate_stats is not None:
            intermediate_stats.clear()

    def after_train_forward(self, metrics: dict[str, float]) -> dict[str, float]:
        intermediate_stats = getattr(self.model, "intermediate_stats", None)
        if intermediate_stats is not None:
            metrics.update(intermediate_stats.snapshot(reset=True))
        return metrics

    def _run_train_forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        self.before_train_forward()
        loss, metrics = self.train_forward(batch)
        return loss, self.after_train_forward(metrics)

    def after_step(
        self,
        step: int,
        step_stats_tracker: StatTracker,
        is_last_step_of_epoch: bool = False,
    ):
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
                step_stats=step_stats_tracker.get_aggregated_stats(),
                interval_time_seconds=self.timer.elapsed("log_interval"),
                epoch_elapsed_seconds=self.timer.elapsed("epoch"),
                lr=self.lr,
            )
            self.timer.mark("log_interval")

        # 保存训练状态
        save_step_interval = self.cfg["save_step_interval"]
        if save_step_interval > 0 and self.global_step % save_step_interval == 0:
            self._save_training_state(
                name=f"global_step_{self.global_step}.pth",
                resume_epoch=self.epoch,
            )

    def after_epoch(
        self,
        train_metrics: dict[str, float],
        train_time_seconds: float,
        validation_pending: bool = False,
    ):
        """
        - 输出训练 epoch 级日志
        - 按配置执行 epoch 级 checkpoint 保存
        """
        self.logger.log_epoch_end(
            train_time_seconds=train_time_seconds,
            train_metrics=train_metrics,
            lr=self.lr,
        )

        self._save_epoch_checkpoints(
            resume_epoch=self.epoch + 1,
            validation_pending=validation_pending,
        )

    def _save_epoch_checkpoints(
        self,
        *,
        resume_epoch: int,
        validation_pending: bool,
    ) -> None:
        self._save_training_state(
            name="latest.pth",
            resume_epoch=resume_epoch,
            validation_pending=validation_pending,
        )
        save_epoch_interval = int(self.cfg["save_epoch_interval"])
        if save_epoch_interval > 0 and self.epoch % save_epoch_interval == 0:
            self._save_training_state(
                name=f"epoch_{self.epoch}.pth",
                resume_epoch=resume_epoch,
                validation_pending=validation_pending,
            )

    def _validate_and_finalize_epoch(
        self,
        *,
        validation_epoch: int,
        resume_epoch: int,
    ) -> None:
        current_epoch = self.epoch
        self.epoch = validation_epoch
        try:
            update_experiments_tsv_status(
                work_dir=self.cfg["work_dir"],
                epoch=self.epoch,
                max_epochs=self.max_epochs,
                phase="val",
            )
            self.validate()
            update_experiments_tsv_status(
                work_dir=self.cfg["work_dir"],
                epoch=self.epoch,
                max_epochs=self.max_epochs,
            )
            self._save_epoch_checkpoints(
                resume_epoch=resume_epoch,
                validation_pending=False,
            )
        finally:
            self.epoch = current_epoch

    def after_val(
        self,
        val_metrics: dict[str, float],
        validation_time_seconds: float,
    ):
        """
        - 输出验证阶段日志
        """
        self.logger.log_validation_timing(
            test_time_seconds=validation_time_seconds,
            epoch=self.epoch,
            val_metrics=val_metrics,
        )

        if "MIoU" in val_metrics:
            current_miou = float(val_metrics["MIoU"])
            if (
                current_miou > self.best_miou
                and "accuracy" in val_metrics
                and "F1Score" in val_metrics
            ):
                update_experiments_tsv_best_metrics(
                    work_dir=self.cfg["work_dir"],
                    epoch=self.epoch,
                    miou=current_miou,
                    oa=float(val_metrics["accuracy"]),
                    f1=float(val_metrics["F1Score"]),
                )
            self._update_save_best_miou(current_miou)
        self.logger.log_val_best_metric("MIoU_best", self.best_miou)

    @torch.no_grad()
    def validate(self):
        """
        - 执行完整验证流程
        - 调用 inferencer 进行推理
        - 调用 evaluator 聚合验证结果
        - 输出验证阶段日志与计时
        """
        self.timer.mark("validation")
        self.model.eval()

        outputs = [
            self.inferencer.run_batch_infer(
                model=self.model,
                batch=batch,
                device=self.device,
            )
            for batch in self.val_loader
        ]
        val_metrics = self.evaluator.evaluate(outputs=outputs, trainer=self)
        validation_time_seconds = self.timer.elapsed("validation")
        self.after_val(
            val_metrics,
            validation_time_seconds=validation_time_seconds,
        )
        self.model.train()

    def optimize_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    def _save_training_state(
        self,
        name: str,
        resume_epoch: int | None = None,
        validation_pending: bool = False,
    ) -> None:
        path = CheckpointManager.save_training_state(
            path=Path(self.cfg["work_dir"]) / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            global_step=self.global_step,
            best_miou=self.best_miou,
            resume_epoch=resume_epoch,
            validation_pending=validation_pending,
        )
        self.logger.log_checkpoint_saved(path)

    def _update_save_best_miou(self, miou: float) -> None:
        if miou > self.best_miou:
            self.best_miou = miou

            self._save_training_state(
                name="best_miou.pth",
                resume_epoch=self.epoch + 1,
            )
