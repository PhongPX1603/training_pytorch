from collections import defaultdict
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from collections import defaultdict


class Plot:
    def __init__(self, save_dir: str):
        self.plot_dir = Path(save_dir) / datetime.now().strftime(r'%y%m%d%H%M') / 'plot'
        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True)
        self.plot_metric = defaultdict(list)
    
    def update(self, monitor: str, value):
        self.plot_metric[monitor].append(value)
        
    def finish(self):
        for plot_name, plot_value in self.plot_metric.items():
            plt.plot(plot_value)
            plt.title(plot_name)
            plt.xlabel('Epoch')
            plt.ylabel('Loss' if 'loss' in plot_name else "Acc")
            plt.savefig(str(self.plot_dir.joinpath(f'{plot_name}.png')))
            plt.close()