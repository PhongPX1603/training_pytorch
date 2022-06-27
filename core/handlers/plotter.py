from collections import defaultdict
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from collections import defaultdict


class Plotter:
    def __init__(self, save_dir: str):
        self.plot_dir = Path(save_dir) / datetime.now().strftime(r'%y%m%d%H%M') / 'plot'
        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True)

        self.data_plot = defaultdict(list)
    
    def add_scalar(self, monitor: str, value: float) -> None:
        self.data_plot[monitor].append(value)
        
    def draw(self):
        for plot_name, plot_value in self.data_plot.items():
            plt.plot(plot_value)
            plt.title(plot_name)
            plt.xlabel('Epoch')
            plt.ylabel('Loss' if 'loss' in plot_name else "Acc")
            plt.savefig(str(self.plot_dir.joinpath(f'{plot_name}.png')))
            plt.close()