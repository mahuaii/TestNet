from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix
import torch


class Evaluator:
    """
    输入：
    - 验证阶段收集到的 outputs
    - 其他可选上下文参数

    职责：
    - 聚合验证结果
    - 计算并返回指标

    输出：
    - 指标字典
    """

    @staticmethod
    def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        correct = int(torch.count_nonzero(pred == target).item())
        total = int(target.numel())
        return 100.0 * float(correct) / float(total)

    def evaluate(
        self,
        outputs: list[Any],
        num_classes: int | None = None,
        metric_classes: int = 5,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        输入：
        - outputs
        - kwargs

        职责：
        - 预留验证结果聚合与指标计算入口

        输出：
        - 验证指标字典
        """
        if num_classes is None:
            num_classes = int(kwargs["trainer"].cfg["num_classes"])
        predictions = np.concatenate([self._to_numpy(output["pred"]).ravel() for output in outputs])
        gts = np.concatenate([self._to_numpy(output["target"]).ravel() for output in outputs])

        cm = confusion_matrix(gts, predictions, labels=range(num_classes))

        total = np.sum(cm)
        accuracy = np.trace(cm)
        accuracy *= 100 / float(total)

        f1_score = np.zeros(num_classes)
        for class_index in range(num_classes):
            f1_score[class_index] = (
                2.0
                * cm[class_index, class_index]
                / (np.sum(cm[class_index, :]) + np.sum(cm[:, class_index]))
            )

        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        kappa = (pa - pe) / (1 - pe)

        miou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))

        return {
            "MIoU": float(np.nanmean(miou[:metric_classes])),
            "accuracy": float(accuracy),
            "F1Score": float(np.nanmean(f1_score[:metric_classes])),
            "kappa": float(kappa),
        }

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)
