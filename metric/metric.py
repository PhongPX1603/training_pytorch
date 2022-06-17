import torch
from torch import nn
from collections import defaultdict
from typing import Callable, Dict, Any, Tuple


class MetricBase(nn.Module):
    def __init__(self, metric_fn: nn.Module, output_transform: Callable = lambda x: x):
        super(MetricBase, self).__init__()
        self.output_transform = output_transform
        self.metric_fn = metric_fn

    def forward(self, *output):
        output = self.output_transform(output)
        return self.metric_fn(*output)


class Metric:
    def __init__(
        self,
        metric: Dict[str, Callable],
        output_transform: Callable = lambda x: x,
    ):
        super(Metric, self).__init__()
        self.metric = metric
        self.output_transform = output_transform
        self.metric_tracker = defaultdict(list)

    def reset(self):
        self.metric_tracker = defaultdict(list)

    def update(self, metric: Dict[str, float]) -> None:
        for metric_name, metric_value in metric.items():
            self.metric_tracker[metric_name].append(metric_value)

    def iteration_compute(self, evaluator_name: str, output: tuple) -> Dict[str, float]:
        iteration_metric = dict()
        output = self.output_transform(output)
        for metric_name, metric_fn in self.metric.items():
            value = metric_fn(*output)
            if isinstance(value, torch.Tensor):
                value = value.item()

            iteration_metric[f'{evaluator_name}_{metric_name}'] = value

        return iteration_metric

    def epoch_compute(self) -> Dict[str, float]:
        epoch_metric = dict()
        for metric_name, metric_values in self.metric_tracker.items():
            epoch_metric[metric_name] = sum(metric_values) / len(metric_values)

        return epoch_metric
