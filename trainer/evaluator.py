import os
import time
import torch
import numpy as np
import torch.nn as nn

from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional, Callable, Dict, Tuple, List

from .utils import prepare_device


class Evaluator(nn.Module):
    def __init__(
        self,
        model: nn.Module = None,
        metric: Callable = None,
        test_loader: nn.Module = None,
        logger: Callable = None,
        writer: Callable = None,
    ):
        super(Evaluator, self).__init__()
        self.model = model
        self.metric = metric
        self.test_loader = test_loader
        self.iteration_counters = defaultdict(int)
        self.writer = writer
        self.logger = logger.get_logger(log_name='evaluation')

    def eval_epoch(self, evaluator_name: str, dataloader: nn.Module = None) -> Dict[str, float]:
        self.model.eval()
        self.metric.reset()
        with torch.no_grad():
            for batch in dataloader:
                params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
                params[0] = self.model(params[0])

                iteration_metric = self.metric.iteration_compute(evaluator_name=evaluator_name, output=params)
                self.metric.update(metric=iteration_metric)    

                for metric_name, metric_value in iteration_metric.items():
                    self.writer.write(
                        name=metric_name, value=metric_value, step=self.iteration_counters[evaluator_name]
                    )

                self.iteration_counters[evaluator_name] += 1

        return self.metric.epoch_compute()

    def verbose(self, message: str) -> None:
        self.logger.info(message)
        print(message)

    def forward(self, save_dir: str, checkpoint_path: Optional[str] = None, num_gpus: int = 0):
        # Set Device for Model: prepare for (multi-device) GPU training
        self.device, gpu_indices = prepare_device(num_gpus)
        self.model = self.model.to(self.device)
        if len(gpu_indices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_indices).module

        # Load weight
        if checkpoint_path is not None:
            state_dict = torch.load(f=checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict=state_dict)

        # Save Directory for Checkpoint and Backup
        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # Start to evaluate
        self.verbose(message=f'{time.asctime()} - STARTED')
        metrics = self.eval_epoch(evaluator_name='valid', dataloader=self.test_loader)
        messages = [f'{metric_name}: {metric_value:.5f}' for metric_name, metric_value in metrics.items()]
        self.verbose(message=f"\t{' - '.join(messages)}")
        self.verbose(message=f'{time.asctime()} - COMPLETED')
