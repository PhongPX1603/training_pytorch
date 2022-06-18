from ast import Call
import torch
import torch.nn as nn
from typing import Callable, Any

from .metric import MetricBase


class Metric(MetricBase):
    def __init__(self, metric_fn: Any, output_transform: Callable = lambda x: x):
        super(Metric, self).__init__(output_transform)
        self.metric_fn = metric_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output: Any) -> float:
        value = self.metric_fn(*output)

        if len(value.shape) != 0:
            raise ValueError('metric_fn did not return the average accuracy.')

        N = output[0].shape[0]
        self._sum += value.item() * N
        self._num_examples += N

        return value.item()

    def compute(self):
        if self._num_examples == 0:
            raise ValueError('Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(preds, dim=1)
        correct = (preds == targets).sum()
        return torch.true_divide(correct, targets.shape[0])


class ConfusionMatrix(MetricBase):
    def __init__(self, num_classes: int, output_transform: Callable = lambda x: x):
        super(ConfusionMatrix, self).__init__(output_transform)
        self.num_classes = num_classes

    def reset(self):
        self.confusion_matrix = torch.zeros(size=(self.num_classes, self.num_classes), dtype=torch.int16)

    def update(self, output: Any) -> None:
        preds, targets = output
        preds = torch.argmax(preds, dim=1)
        for target, pred in zip(targets.view(-1), preds.view(-1)):
            self.confusion_matrix[target.long(), pred.long()] += 1

    def compute(self):
        return self.confusion_matrix
