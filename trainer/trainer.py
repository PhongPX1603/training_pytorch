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


class Trainer:
    def __init__(
        self,
        model: nn.Module = None,
        data: Dict[str, Callable] = None,
        loss: nn.Module = None,
        optim: nn.Module = None,
        metric: Callable = None,
        early_stopping: nn.Module = None,
        lr_scheduler: nn.Module = None,
        logger: Callable = None,
        writer: Callable = None,
        save_dir: str = None
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.data = data
        self.loss = loss
        self.optim = optim
        self.metric = metric
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.iteration_counters = defaultdict(int)

        # Logger and Tensorboard
        self.writer = writer
        self.logger = logger.get_logger(log_name='training')

        # Save Directory for Checkpoint and Backup
        self.save_dir = Path(save_dir) / datetime.now().strftime(r'%y%m%d%H%M')
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

    def train_epoch(self, evaluator_name: str = 'train', dataloader: nn.Module = None) -> Dict[str, float]:
        self.model.train()
        self.metric.reset()
        for batch in dataloader:
            self.optim.zero_grad()
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            params[0] = self.model(params[0])
            loss = self.loss(*params)
            loss.backward()
            self.optim.step()

            iteration_metric = self.metric.iteration_compute(evaluator_name=evaluator_name, output=params)
            self.metric.update(metric=iteration_metric)    

            for metric_name, metric_value in iteration_metric.items():
                self.writer.write(
                    name=metric_name, value=metric_value, step=self.iteration_counters[evaluator_name]
                )

            self.iteration_counters[evaluator_name] += 1

        return self.metric.epoch_compute()

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

    def verbose(self, message: str, _print: bool = True) -> None:
        self.logger.info(message)
        if _print:
            print(message)

    def train(
        self,
        num_epochs: int,
        resume_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        num_gpus: int = 0,
    ) -> None:
        # Set Device for Model: prepare for (multi-device) GPU training
        self.device, gpu_indices = prepare_device(num_gpus)
        self.model = self.model.to(self.device)
        if len(gpu_indices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_indices).module

        # Load pretrained weight
        if checkpoint_path is not None:
            state_dict = torch.load(f=checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict=state_dict)

        # Resume Mode
        if resume_path is not None:
            self.verbose(message=f'{time.asctime()} - RESUME')
            checkpoint = torch.load(f=resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optim.load_state_dict(checkpoint['optim'])
            start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            score_name = checkpoint['score_name']
            mode = checkpoint['mode']
        else:
            start_epoch = 0
            mode = self.early_stopping.mode
            score_name = self.early_stopping.score_name
            best_score = -np.Inf if mode == 'min' else 0

        # Start to train
        self.verbose(message=f'{time.asctime()} - STARTED')
        for epoch in range(start_epoch, num_epochs):
            self.verbose(message=f'Epoch #{epoch} - {time.asctime()}')
            train_metrics = self.train_epoch(evaluator_name='train', dataloader=self.data['train'])
            train_eval_metrics = self.eval_epoch(evaluator_name='train_eval', dataloader=self.data['train_eval'])
            valid_metrics = self.eval_epoch(evaluator_name='valid', dataloader=self.data['valid'])

            messages = [f'{metric_name}: {metric_value:.5f}' for metric_name, metric_value in train_metrics.items()]
            self.verbose(message=f"\t[Info] {' - '.join(messages)}")

            messages = [f'{metric_name}: {metric_value:.5f}' for metric_name, metric_value in train_eval_metrics.items()]
            self.verbose(message=f"\t[Info] {' - '.join(messages)}")

            messages = [f'{metric_name}: {metric_value:.5f}' for metric_name, metric_value in valid_metrics.items()]
            self.verbose(message=f"\t[Info] {' - '.join(messages)}")

            # update learning scheduler
            self.lr_scheduler.step(valid_metrics[f'valid_loss'])

            # update early stopping
            self.early_stopping(valid_metrics)
            if self.early_stopping.early_stop:
                self.logger.info('__Stop Training__ Model can not improve.')
                break

            # save backup checkpoint
            if self.save_dir.joinpath(f'backup_epoch_{epoch - 1}.pth').exists():
                os.remove(str(self.save_dir.joinpath(f'backup_epoch_{epoch - 1}.pth')))

            backup_checkpoint = {
                'epoch': epoch,
                'best_score': best_score,
                'score_name': self.early_stopping.score_name,
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
            }

            save_backup_path = self.save_dir.joinpath(f'backup_epoch_{epoch}.pth')
            torch.save(obj=backup_checkpoint, f=str(save_backup_path))
            self.verbose(message=f'\t[__Saving Backup Checkpoint__] {str(save_backup_path)}', _print=False)

            score = -valid_metrics[f'valid_{score_name}'] if mode == 'min' else valid_metrics[f'valid_{score_name}']
            if score > best_score:                
                if self.save_dir.joinpath(f'best_valid_{score_name}_{best_score}.pth').exists():
                    os.remove(str(self.save_dir.joinpath(f'best_valid_{score_name}_{best_score}.pth')))

                best_score = score
                save_path = self.save_dir.joinpath(f'best_model_{epoch}_{score_name}_{best_score}.pth')
                self.verbose(message=f'\t[__Saving Checkpoint__] {str(save_path)}', _print=False)
                torch.save(obj=self.model.state_dict(), f=str(save_path))

        self.verbose(message=f'{time.asctime()} - COMPLETED')
