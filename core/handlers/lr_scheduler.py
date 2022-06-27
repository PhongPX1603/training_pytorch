from torch.optim import lr_scheduler
from typing  import Dict


class ReduceLROnPlateau:
    def __init__(self, score_name, evaluator_name, optim, **kwargs):
        super(ReduceLROnPlateau, self).__init__()
        self.score_name = score_name
        self.evaluator_name = evaluator_name
        self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optim, **kwargs)

    def _step(self, metrics: Dict[str, float]):
        self.scheduler.step(metrics[f'{self.evaluator_name}_{self.score_name}'])