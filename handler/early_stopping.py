from torch import nn
from typing import Dict


class EarlyStopping(nn.Module):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(
        self,
        evaluator_name: str = 'valid',
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        mode: str = 'min',
        score_name: str = 'loss'
    ) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): 'min' - use with valid loss and 'max' use for valid acc
            monitor (str): 'valid_loss' or 'valid_accuracy' to take value in metrics
        """
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.evaluator_name = evaluator_name
        self.score_name = score_name

    def forward(self, metrics: Dict[str, dict]) -> None:
        score = metrics[f'{self.evaluator_name}_{self.score_name}']
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early Stopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Training Stop !!!')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
