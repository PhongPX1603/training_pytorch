from distutils.command.config import config
import os
import time
import torch
import yaml
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional, Callable, Dict, Any

from .utils import prepare_device


class Trainer:
    def __init__(
        self,
        project_name: str = None,
        model: nn.Module = None,
        data: Dict[str, Callable] = None,
        loss: nn.Module = None,
        optim: nn.Module = None,
        metric: Callable = None,
        lr_scheduler: nn.Module = None,
        early_stopping: Callable = None,
        model_inspection: Callable = None,
        logger: Callable = None,
        writer: Callable = None,
        plotter: Callable = None,
        save_dir: str = None
    ):
        super(Trainer, self).__init__()
        self.logger = logger.get_logger(log_name=project_name)
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
        self.plotter = plotter

        # get model info
        if model_inspection is not None:
            model_inspection(self.model, self.logger)

        # Save Directory for Checkpoint and Backup
        self.save_dir = Path(save_dir) / datetime.now().strftime(r'%y%m%d%H%M')
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        # training information
        self.training_info = dict()
        
    def train_epoch(self, evaluator_name: str = 'train', dataloader: nn.Module = None) -> Dict[str, float]:
        self.model.train()
        self.metric.started(evaluator_name)
        for batch in tqdm(dataloader, total=len(dataloader)):
            self.optim.zero_grad()
            params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
            params[0] = self.model(params[0])
            loss = self.loss(*params)
            loss.backward()
            self.optim.step()

            # log learning_rate
            self.writer.add_scalar(
                name='learning_rate', value=self.optim.param_groups[0]['lr'], step=self.iteration_counters[evaluator_name]
            )
 
            iteration_metric = self.metric.iteration_completed(output=params)  

            for metric_name, metric_value in iteration_metric.items():
                self.writer.add_scalar(
                    name=metric_name, value=metric_value, step=self.iteration_counters[evaluator_name]
                )

            self.iteration_counters[evaluator_name] += 1

        return self.metric.epoch_completed()

    def eval_epoch(self, evaluator_name: str, dataloader: nn.Module = None) -> Dict[str, float]:
        self.model.eval()
        self.metric.started(evaluator_name)
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                params = [param.to(self.device) if torch.is_tensor(param) else param for param in batch]
                params[0] = self.model(params[0])

                iteration_metric = self.metric.iteration_completed(output=params)

                for metric_name, metric_value in iteration_metric.items():
                    self.writer.add_scalar(
                        name=metric_name, value=metric_value, step=self.iteration_counters[evaluator_name]
                    )

                self.iteration_counters[evaluator_name] += 1

        return self.metric.epoch_completed()

    def verbose(self, message: str, _print: bool = True) -> None:
        self.logger.info(message)
        if _print:
            print(message)

    def verbose_metric(self, metric: Dict[str, Any], _print: bool = True) -> None:
        messages = []
        for metric_name, metric_value in metric.items():
            if isinstance(metric_value, float):
                messages.append(f'{metric_name}: {metric_value:.5f}')
                # save metric value to plotter
                self.plotter.add_scalar(metric_name, metric_value)

        message = ' - '.join(messages)
        self.verbose(message=f'{message}', _print=_print)

    def test(self) -> None:
        self.verbose(message='\n____________________________________TESTING____________________________________\n')
        # Load weight
        checkpoint_path = self.training_info['checkpoint_path']
        state_dict = torch.load(f=checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict=state_dict)

        # Start to evaluate
        print(f'{time.asctime()} - STARTED')
        metrics = self.eval_epoch(evaluator_name='test', dataloader=self.data['test'])
        messages = [f'{metric_name}: {metric_value:.5f}' for metric_name, metric_value in metrics.items()]
        print(f"[Info] {' - '.join(messages)}")
        print(f'{time.asctime()} - COMPLETED')

    def train(
        self,
        num_epochs: int,
        resume_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        num_gpus: int = 0,
    ) -> None:
        self.verbose(message='\n____________________________________TRAINING___________________________________\n')
        # Set Device for Model: prepare for (multi-device) GPU training
        self.device, gpu_indices = prepare_device(num_gpus)
        self.model = self.model.to(self.device)
        if len(gpu_indices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_indices)

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
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']
            score_name = checkpoint['score_name']
            mode = self.early_stopping.mode
        else:
            start_epoch = 0
            mode = self.early_stopping.mode
            score_name = self.early_stopping.score_name
            best_score = -np.Inf if mode == 'min' else 0

        # Initialize checkpoint path for saving checkpoint
        _checkpoint_path = self.save_dir / f'best_model_{start_epoch}_{score_name}_{best_score}.pth'

        # Start to train
        self.verbose(message=f'{time.asctime()} - STARTED')
        for epoch in range(start_epoch, num_epochs + start_epoch):
            self.verbose(message=f'Epoch #{epoch} - {time.asctime()}')
            train_metrics = self.train_epoch(evaluator_name='train', dataloader=self.data['train'])
            train_eval_metrics = self.eval_epoch(evaluator_name='train_eval', dataloader=self.data['train_eval'])
            valid_metrics = self.eval_epoch(evaluator_name='valid', dataloader=self.data['valid'])

            # update learning scheduler
            self.lr_scheduler._step(valid_metrics)

            # update early stopping
            self.early_stopping(valid_metrics)
            # check early stopping flag
            if self.early_stopping.early_stop:
                self.verbose(message=f'{time.asctime()} - EARLY STOPPING.')
                break

            # export training information
            self.verbose_metric(train_metrics)
            self.verbose_metric(train_eval_metrics)
            self.verbose_metric(valid_metrics)

            # save backup checkpoint
            if self.save_dir.joinpath(f'backup_epoch_{epoch - 1}.pth').exists():
                os.remove(str(self.save_dir.joinpath(f'backup_epoch_{epoch - 1}.pth')))

            backup_checkpoint = {
                'epoch': epoch,
                'best_score': best_score,
                'score_name': score_name,
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
            }

            backup_checkpoint_path = self.save_dir / f'backup_epoch_{epoch}.pth'
            torch.save(obj=backup_checkpoint, f=str(backup_checkpoint_path))
            self.verbose(message=f'__Saving Backup Checkpoint__ {str(backup_checkpoint_path)}', _print=False)

            score = -valid_metrics[f'valid_{score_name}'] if mode == 'min' else valid_metrics[f'valid_{score_name}']
            if score > best_score:
                best_score = score                
                if _checkpoint_path.exists():
                    os.remove(str(_checkpoint_path))
                _checkpoint_path = self.save_dir / f'best_model_{epoch}_{score_name}_{best_score}.pth'
                torch.save(obj=self.model.state_dict(), f=str(_checkpoint_path))
                self.verbose(message=f'__Saving Checkpoint__ {str(_checkpoint_path)}', _print=False)

                self.training_info['checkpoint_path'] = str(_checkpoint_path)
                self.training_info['best_score'] = best_score
                self.training_info['epoch'] = epoch

        self.training_info['score_name'] = score_name
        self.training_info['backup_checkpoint_path'] = str(backup_checkpoint_path)

        self.plotter.draw()
        self.verbose(message=f'{time.asctime()} - COMPLETED')

    def __call__(self,
        num_epochs: int,
        resume_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        num_gpus: int = 0,
    ):
        self.train(num_epochs, resume_path, checkpoint_path, num_gpus)
        if self.data.get('test', None) is not None:
            self.test()