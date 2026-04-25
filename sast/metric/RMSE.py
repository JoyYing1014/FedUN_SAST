
import sast as fs
import torch


class RMSE(fs.Metric):
    def __init__(self):
        super().__init__(name='RMSE')

    @staticmethod
    def calc(network_output, target):
        rmse = torch.sqrt(torch.sum(torch.abs((network_output - target)**2)) / len(target))  
        return rmse.item()
