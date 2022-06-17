from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, save_dir: str):
        tb_dir = Path(save_dir) / datetime.now().strftime(r'%y%m%d%H%M') / 'tensorboard'
        if not tb_dir.exists():
            tb_dir.mkdir(parents=True)

        self.writer = SummaryWriter(str(tb_dir))

    def write(self, name: str, value: float, step: int = 0) -> None:
        self.writer.add_scalar(name, value, global_step=step)
