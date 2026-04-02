from __future__ import annotations

import torch


class Evaluator:
    def evaluate(self, outputs: list[dict[str, torch.Tensor | float]], num_classes: int) -> dict[str, float]:
        total_loss = 0.0
        total_pixels = 0
        total_correct = 0
        total_intersection = 0.0
        total_union = 0.0
        total_pred_fg = 0.0
        total_target_fg = 0.0

        for output in outputs:
            logits = output["seg_logits"]
            target = output["target"]
            total_loss += float(output["loss"])

            if num_classes == 1:
                pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
                target_int = target.long()
            else:
                pred = torch.argmax(logits, dim=1)
                target_int = target.long()

            total_correct += int((pred == target_int).sum().item())
            total_pixels += int(target_int.numel())

            pred_fg = pred > 0
            target_fg = target_int > 0
            intersection = (pred_fg & target_fg).sum().item()
            union = (pred_fg | target_fg).sum().item()
            total_intersection += intersection
            total_union += union
            total_pred_fg += pred_fg.sum().item()
            total_target_fg += target_fg.sum().item()

        mean_loss = total_loss / max(len(outputs), 1)
        oa = total_correct / max(total_pixels, 1)
        miou = total_intersection / max(total_union, 1.0)
        f1 = 0.0 if total_intersection == 0 else 2 * total_intersection / max(
            total_pred_fg + total_target_fg,
            1.0,
        )
        return {"loss": mean_loss, "miou": miou, "f1": f1, "oa": oa}
