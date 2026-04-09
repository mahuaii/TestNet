from __future__ import annotations

from typing import Any


class Inferencer:
    """
    输入：
    - model
    - batch 或 dataloader
    - device

    职责：
    - 承载验证/测试阶段的推理流程
    - 为后续普通推理、滑窗推理、多尺度推理预留位置

    输出：
    - 单 batch 或整套数据的推理结果
    """

    def run_batch_infer(self, model: Any, batch: Any, device: Any) -> Any:
        """
        输入：
        - model、batch、device

        职责：
        - 预留单个 batch 的推理入口

        输出：
        - 单个 batch 的推理结果
        """
        del model, device
        return {"batch": batch}

    def run_dataset(self, model: Any, dataloader: Any, device: Any) -> Any:
        """
        输入：
        - model、dataloader、device

        职责：
        - 预留整套数据集级别的推理入口

        输出：
        - 数据集级别的推理结果
        """
        del model, device
        return [self.run_batch_infer(model=None, batch=batch, device=None) for batch in dataloader]
