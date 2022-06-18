from ast import Call
import torch
import torch.nn as nn
from typing import Callable, Any

from .metric_base import MetricBase


class Accuracy(MetricBase):
    def __init__(self, output_transform: Callable = lambda x: x):
        super(Accuracy, self).__init__(output_transform)

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def accuracy_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(preds, dim=1)
        assert preds.shape[0] == len(targets)
        correct = 0
        correct += torch.sum(preds == targets)
        return correct / len(targets)

    def update(self, output: Any) -> None:
        average_accuracy = self.accuracy_fn(*output)

        if len(average_accuracy.shape) != 0:
            raise ValueError('accuracy_fn did not return the average accuracy.')

        N = output[0].shape[0]
        self._sum += average_accuracy.item() * N
        self._num_examples += N

        return average_accuracy.item()

    def compute(self):
        if self._num_examples == 0:
            raise ValueError('Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples


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
