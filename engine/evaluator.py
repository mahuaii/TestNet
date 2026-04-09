from __future__ import annotations

from typing import Any

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
        accuracy = float((pred == target).float().mean().item() * 100.0)
        return accuracy

    def evaluate(self, outputs: list[Any], **kwargs: Any) -> dict[str, float]:
        """
        输入：
        - outputs
        - kwargs

        职责：
        - 预留验证结果聚合与指标计算入口

        输出：
        - 验证指标字典
        """
        del kwargs
        metrics = {"num_outputs": float(len(outputs))}
        return metrics
