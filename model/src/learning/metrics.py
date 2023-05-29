import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from util import pytorch_ssim

class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self): pass
    def add(self):   pass
    def value(self): pass


def img_metrics(target, pred, var=None, pixelwise=True):
    rmse = torch.sqrt(torch.mean(torch.square(target - pred)))
    psnr = 20 * torch.log10(1 / rmse)
    mae = torch.mean(torch.abs(target - pred))
    
    # spectral angle mapper
    mat = target * pred
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(target * target, 1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(pred * pred, 1)))
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1))*torch.tensor(180)/torch.pi)

    ssim = pytorch_ssim.ssim(target, pred)

    metric_dict = {'RMSE': rmse.cpu().numpy().item(),
                   'MAE': mae.cpu().numpy().item(),
                   'PSNR': psnr.cpu().numpy().item(),
                   'SAM': sam.cpu().numpy().item(),
                   'SSIM': ssim.cpu().numpy().item()}

    # evaluate the (optional) variance maps
    if var is not None:
        error = target - pred
        # average across the spectral dimensions
        se = torch.square(error)
        ae = torch.abs(error)

        # collect sample-wise error, AE, SE and uncertainties
 
        # define a sample as 1 image and provide image-wise statistics
        errvar_samplewise = {'error': error.nanmean().cpu().numpy().item(),
                            'mean ae': ae.nanmean().cpu().numpy().item(),
                            'mean se': se.nanmean().cpu().numpy().item(),
                            'mean var': var.nanmean().cpu().numpy().item()}
        if pixelwise:
            # define a sample as 1 multivariate pixel and provide image-wise statistics
            errvar_samplewise = {**errvar_samplewise, **{'pixelwise error': error.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise ae': ae.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise se': se.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise var': var.nanmean(0).nanmean(0).flatten().cpu().numpy()}}

        metric_dict     = {**metric_dict, **errvar_samplewise}

    return metric_dict

class avg_img_metrics(Metric):
    def __init__(self):
        super().__init__()
        self.n_samples = 0
        self.metrics   = ['RMSE', 'MAE', 'PSNR','SAM','SSIM']
        self.metrics  += ['error', 'mean se', 'mean ae', 'mean var']

        self.running_img_metrics = {}
        self.running_nonan_count = {}
        self.reset()

    def reset(self):
        for metric in self.metrics: 
            self.running_nonan_count[metric] = 0
            self.running_img_metrics[metric] = np.nan

    def add(self, metrics_dict):
        for key, val in metrics_dict.items():
            # skip variables not registered
            if key not in self.metrics: continue
            # filter variables not translated to numpy yet
            if torch.is_tensor(val): continue
            if isinstance(val, tuple): val=val[0]

            # only keep a running mean of non-nan values
            if np.isnan(val): continue

            if not self.running_nonan_count[key]: 
                self.running_nonan_count[key] = 1
                self.running_img_metrics[key] = val
            else: 
                self.running_nonan_count[key]+= 1
                self.running_img_metrics[key] = (self.running_nonan_count[key]-1)/self.running_nonan_count[key] * self.running_img_metrics[key] \
                                                + 1/self.running_nonan_count[key] * val

    def value(self):
        return self.running_img_metrics