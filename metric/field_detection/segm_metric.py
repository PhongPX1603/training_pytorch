import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from typing import Any, List, Callable

from ..metric_base import MetricBase


class SegmMetric(MetricBase):
    def __init__(self, metric_name: str = None, output_transform=lambda x: x):
        super(SegmMetric, self).__init__(output_transform)
        self.metric_name = metric_name

    def reset(self):
        self._sum = 0
        self._num_samples = 0

    def update(self, output):
        '''
        Args:
            preds, targets, image_infos = output
            preds: torch.Tensor [B, num_classes, H, W]
            targets: torch.Tensor [B, H, W]
            image_infos: List[Tuple(image_path, (w, h))]
        Outputs:
            ...
        '''
        assert self.metric_name in ['pixel_accuracy', 'mean_pixel_accuracy',
                                    'mean_iou', 'frequence_weighted_IU'], f'metric: {self.metric_name} not supported'

        preds, targets, image_infos = output
        _, image_sizes = image_infos
        image_sizes = [(w.item(), h.item()) for w, h in zip(*image_sizes)]

        targets = targets.to(preds.dtype).unsqueeze(dim=1)  # B, 1, H, W
        preds = torch.argmax(preds, dim=1, keepdims=True).to(targets.dtype)  # B, 1, H, W

        value = 0
        for i in range(len(image_sizes)):
            pred, target, image_size = preds[i:i + 1], targets[i:i + 1], image_sizes[i]
            pred = torch.nn.functional.interpolate(pred, size=image_size[::-1], mode='nearest')  # 1, 1, H, W
            target = torch.nn.functional.interpolate(target, size=image_size[::-1], mode='nearest')  # 1, 1, H, W
            pred, target = pred.squeeze(dim=0).squeeze(dim=0), target.squeeze(dim=0).squeeze(dim=0)  # pred, target: H, w
            if self.metric_name == 'pixel_accuracy':
                metric = self._pixel_accuracy(pred, target)
            elif self.metric_name == 'mean_pixel_accuracy':
                metric = self._mean_pixel_accuracy(pred, target)
            elif self.metric_name == 'mean_iou':
                metric = self._mean_IU(pred, target)
            elif self.metric_name == 'frequence_weighted_IU':
                metric = self._frequency_weighted_IU(pred, target)

            value += metric
            self._sum += metric
            self._num_samples += 1
        
        return value

    def compute(self):
        return self._sum / self._num_samples

    def _pixel_accuracy(self, pred, target):
        '''pixel_accuracy = sum_i(n_ii) / sum_i(t_i)
        Args:
            pred: torch.Tensor [H, W]
            target: torch.Tensor [H, W]
        Outputs:
            pixel_accuracy: float
        '''
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))
        
        sum_n_ii, sum_t_i = 0, 0
        for category in categories:
            sum_n_ii += ((target == category) & (pred == category)).sum().item()
            sum_t_i += (target == category).sum().item()

        pixel_accuracy = sum_n_ii / sum_t_i if sum_t_i != 0 else 0.

        return pixel_accuracy

    def _mean_pixel_accuracy(self, pred, target):
        '''mean_pixel_accuracy = (1/n_cl) * sum_i(n_ii / t_i)
        Args:
            pred: torch.Tensor [H, W]
            target: torch.Tensor [H, W]
        Outputs:
            mean_pixel_accuracy: float
        '''
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))

        pixel_accs = []
        for category in categories:
            n_ii = ((target == category) & (pred == category)).sum().item()
            t_i = (target == category).sum().item()
            pixel_acc = n_ii / t_i if t_i != 0 else 0.
            pixel_accs.append(pixel_acc)

        mean_pixel_accuracy = sum(pixel_accs) / len(pixel_accs) if len(pixel_accs) != 0 else 0.

        return mean_pixel_accuracy

    def _mean_IU(self, pred, target):
        '''mean_iou = (1 / n_cl) * sum_i(n_ii / (t_i + sum_j(n_ij) - n_ii)
        Args:
            pred: torch.Tensor [H, W]
            target: torch.Tensor [H, W]
        Outputs:
            mean_iou: float
        '''
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))

        ious = []
        for category in categories:
            n_ii = ((target == category) & (pred == category)).sum().item()
            t_i = (target == category).sum().item()
            n_ij = (pred == category).sum().item()
            iou = n_ii / (t_i + n_ij - n_ii)
            ious.append(iou)

        mean_iou = sum(ious) / len(ious) if len(ious) != 0 else 0.

        return mean_iou

    def _frequency_weighted_IU(self, pred, target):
        '''frequence_weighted_IU = (1 / sum_k(t_k)) * sum_i(t_i * n_ii / (t_i + sum_j(n_ij) - n_ii)
        Args:
            pred: torch.Tensor [H, W]
            target: torch.Tensor [H, W]
        Outputs:
            frequence_weighted_IU: float
        '''
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))

        freq_ious = []
        for category in categories:
            n_ii = ((target == category) & (pred == category)).sum().item()
            t_i = (target == category).sum().item()
            n_ij = (pred == category).sum().item()
            freq_iou = (t_i * n_ii) / (t_i + n_ij - n_ii)
            freq_ious.append(freq_iou)

        sum_k_t_k = target.shape[0] * target.shape[1]

        fw_iou = sum(freq_ious) / sum_k_t_k

        return fw_iou
    
class ConfusionMatrix(MetricBase):
    def __init__(self, save_dir: str, classes: List[str], output_transform: Callable = lambda x: x):
        super(ConfusionMatrix, self).__init__(output_transform)
        self.save_dir = Path(save_dir) / datetime.now().strftime(r'%y%m%d%H%M') / 'plot'
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.num_classes = len(classes)
        self.classes = classes

    def reset(self):
        self.confusion_matrix = torch.zeros(size=(self.num_classes, self.num_classes), dtype=torch.int16)

    def update(self, output: Any) -> None:
        preds, targets = output
        targets = targets.to(preds.dtype).unsqueeze(dim=1)
        preds = torch.argmax(preds, dim=1, keepdims=True).to(targets.dtype)
        for target, pred in zip(targets, preds):
            target = target.squeeze(0).squeeze(0)
            pred = pred.squeeze(0).squeeze(0)
            print(target.shape, pred.shape)
            for tar, pre in zip(target.view(-1), pred.view(-1)):
                self.confusion_matrix[tar.long(), pre.long()] += 1
        
    def compute(self):
        # plt.figure(figsize=(15,10))

        # class_names = self.classes
        # df_cm = pd.DataFrame(self.confusion_matrix, index=class_names, columns=class_names).astype(int)
        # heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        # heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        # heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.savefig(str(self.save_dir.joinpath(f'confusion_matrix.png')))
        
        return self.confusion_matrix
