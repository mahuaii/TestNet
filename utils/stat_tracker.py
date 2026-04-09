from __future__ import annotations


class StatTracker:
    def __init__(self) -> None:
        self.mean_stat_sums: dict[str, float] = {}
        self.mean_stat_counts: dict[str, int] = {}
        self.point_stats: dict[str, float] = {}

    def update_mean_stats(self, stats: dict[str, float]) -> None:
        for key, value in stats.items():
            scalar = float(value)
            self.point_stats[key] = scalar
            self.mean_stat_sums[key] = self.mean_stat_sums.get(key, 0.0) + scalar
            self.mean_stat_counts[key] = self.mean_stat_counts.get(key, 0) + 1

    def update_point_stats(self, stats: dict[str, float]) -> None:
        for key, value in stats.items():
            scalar = float(value)
            self.point_stats[key] = scalar

    def get_point_stats(self) -> dict[str, float]:
        return dict(self.point_stats)

    def get_mean_stats(self) -> dict[str, float]:
        return {
            key: self.mean_stat_sums[key] / self.mean_stat_counts[key]
            for key in self.mean_stat_sums
        }

    def get_aggregated_stats(self) -> dict[str, float]:
        aggregated_stats = self.get_point_stats()
        for key, value in self.get_mean_stats().items():
            aggregated_stats[key] = value
        return aggregated_stats
