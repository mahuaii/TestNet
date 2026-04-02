from __future__ import annotations


class RunningMetricTracker:
    def __init__(self) -> None:
        self.metric_sums: dict[str, float] = {}
        self.metric_counts: dict[str, int] = {}
        self.latest_metrics: dict[str, float] = {}

    def update(self, **metrics: float) -> None:
        for key, value in metrics.items():
            scalar = float(value)
            if key not in self.metric_sums:
                self.metric_sums[key] = 0.0
            self.metric_sums[key] += scalar

            if key not in self.metric_counts:
                self.metric_counts[key] = 0
            self.metric_counts[key] += 1

            self.latest_metrics[key] = scalar

    def get_latest_metrics(self) -> dict[str, float]:
        return dict(self.latest_metrics)

    def get_average_metrics(self) -> dict[str, float]:
        return {
            key: self.metric_sums[key] / self.metric_counts[key]
            for key in self.metric_sums
        }
