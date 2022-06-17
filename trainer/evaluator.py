import os
import time
import torch
import numpy as np
import torch.nn as nn

from pathlib import Path
from collections import defaultdict
from typing import Optional, Callable, Dict, Tuple, List

from .utils import prepare_device


class Evaluator(nn.Module):
    def __init__(
        self,
        data: nn.Module = None,
        model: nn.Module = None,
        metric: Callable = None,
    ):
        super(Evaluator, self).__init__()
        self.data = data
        self.model = model
        self.metric = metric

    def eval_epoch(self, evaluator_name: str, dataloader: nn.Module = None) -> Dict[str, float]:
        self.model.eval()
        self.metric.reset()
        with torch.no_grad():
            for batch in dataloader:
                params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
                params[0] = self.model(params[0])

                iteration_metric = self.metric.iteration_compute(evaluator_name=evaluator_name, output=params)
                self.metric.update(metric=iteration_metric)    

        return self.metric.epoch_compute()

    def eval(self, checkpoint_path: Optional[str] = None, num_gpus: int = 0):
        # Load weight
        if checkpoint_path is None:
            raise ValueError('No checkpoint to load.')

        # Set Device for Model: prepare for (multi-device) GPU training
        self.device, gpu_indices = prepare_device(num_gpus)
        self.model = self.model.to(self.device)
        if len(gpu_indices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_indices).module

        # Load weight
        state_dict = torch.load(f=checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict=state_dict)

        # Start to evaluate
        print(f'{time.asctime()} - STARTED')
        metrics = self.eval_epoch(evaluator_name='test', dataloader=self.data)
        messages = [f'{metric_name}: {metric_value:.5f}' for metric_name, metric_value in metrics.items()]
        print(f"\t[Info] {' - '.join(messages)}")
        print(f'{time.asctime()} - COMPLETED')
