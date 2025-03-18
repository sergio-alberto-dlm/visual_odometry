import torch
import torch.multiprocessing as mp


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        