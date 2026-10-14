from __future__ import annotations

from typing import Any

import numpy as np
import torch

ISPRS_LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]


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
        outputs,
        num_classes: int | None = None,
        metric_classes: int = 5,
        label_values: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
        label_values = tuple(label_values or ISPRS_LABELS[:num_classes])

        # Streaming confusion matrix: accumulate per-batch to avoid
        # holding all predictions/gts in RAM at once.
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        total = 0
        for output in outputs:
            pred = self._to_numpy(output["pred"]).ravel()
            gt = self._to_numpy(output["target"]).ravel()
            batch_cm = self._confusion_matrix(
                targets=gt,
                predictions=pred,
                num_classes=num_classes,
            )
            cm += batch_cm
            total += int(np.sum(batch_cm))
            # Free batch arrays immediately
            del pred, gt, batch_cm
        accuracy = np.trace(cm)
        accuracy *= 100 / float(total)

        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_accuracy = np.diag(cm) / cm.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            f1_score = 2.0 * np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0))

        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        kappa = (pa - pe) / (1 - pe)

        with np.errstate(divide="ignore", invalid="ignore"):
            miou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))

        return {
            "MIoU": float(np.nanmean(miou[:metric_classes])),
            "accuracy": float(accuracy),
            "F1Score": float(np.nanmean(f1_score[:metric_classes])),
            "kappa": float(kappa),
            "confusion_matrix": cm,
            "pixels_processed": int(total),
            "class_names": list(label_values),
            "per_class_accuracy": per_class_accuracy,
            "per_class_recall": per_class_accuracy,
            "per_class_f1": f1_score,
            "per_class_iou": miou,
        }

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _confusion_matrix(
        targets: np.ndarray,
        predictions: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        valid_mask = (targets >= 0) & (targets < num_classes) & (predictions >= 0) & (predictions < num_classes)
        target_labels = targets[valid_mask].astype(np.int64, copy=False)
        prediction_labels = predictions[valid_mask].astype(np.int64, copy=False)
        encoded = num_classes * target_labels + prediction_labels
        return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
