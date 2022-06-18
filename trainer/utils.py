import copy
import time
import torch
from torch import nn
from thop.profile import profile  # using to get params and flops
from typing import Callable, Tuple, List


def prepare_device(n_gpu_use: int = 0) -> Tuple[str, List[int]]:
    n_gpu = torch.cuda.device_count()  # get all GPU indices of system.

    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine.")
        n_gpu_use = n_gpu

    device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
    gpu_indices = list(range(n_gpu_use))

    return device, gpu_indices


class ModelInfo:
    def __init__(self, verbose: bool = False, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.verbose = verbose
        self.input_shape = input_shape

    def __call__(self, model: nn.Module, logger: Callable) -> None:
        # the number of parameters
        n_params = sum(param.numel() for param in model.parameters())
        # the number of gradients
        n_grads = sum(param.numel() for param in model.parameters() if param.requires_grad)

        if self.verbose:
            message = '___MODEL INFOMATION___\n'
            message += '\tModel Detail:\n'
            for i, (name, params) in enumerate(model.named_parameters()):
                name = name.replace('module_list.', '')
                message += f"\t  [...] layer: {i}, name: {name}, gradient: {params.requires_grad}, params: {params.numel()}, "
                message += f"shape: {list(params.shape)}, mu: {params.mean().item()}, sigma: {params.std().item()}\n"
        # get FLOPs
        try:
            device = next(model.parameters()).device
            dummy_image = torch.zeros(size=(1, self.input_shape[2], self.input_shape[0], self.input_shape[1]), device=device)
            total_ops, total_params = profile(copy.deepcopy(model), inputs=(dummy_image,), verbose=False)  # MACs, params
            total_ops, total_params = round(total_ops / 1e9, 2), round(total_params / 1e6, 2)  # GMACs, Mparams
        except (ImportError, Exception):
            total_ops, total_params = '-', '-'

        message += f"\tModel Summary:\n"
        message += f"\t  [...] Layers: {len(list(model.modules()))}, Parameters: {n_params}, Gradients: {n_grads}\n"
        message += f"\t  [...] Params (M): {total_params}, MACs (G): {total_ops}\n"

        logger.info(message)
        print(message)


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class CustomDataParallel(nn.DataParallel):
    """force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """
    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = [f'cuda:{x}' for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return (
            [
                (
                    inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                    inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True)
                )
                for device_idx in range(len(devices))
            ],
            [kwargs] * len(devices)
        )
