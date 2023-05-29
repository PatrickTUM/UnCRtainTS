"""
 Python script to obtain Deep Ensemble predictions by collecting each instance's pre-computed predictions.
    Each member's predictions are first meant to be pre-computed via test_reconstruct.py, with the outputs exported,
    and read again in this script. Online ensembling is currently not implemented as this may exceed hardware constraints.
    For every ensemble member, the path to its output directory has to be specified in the list 'ensemble_paths'.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from natsort import natsorted

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from data.dataLoader import SEN12MSCR, SEN12MSCRTS
from src.learning.metrics import img_metrics, avg_img_metrics
from train_reconstruct import recursive_todevice, compute_uce_auce, export, plot_img, save_results

epoch       = 1
root		= '/home/data/' # path to directory containing dataset
mode        = 'test'        # split to evaluate on
in_time     = 3             # length of input time series
region      = 'all'         # region of areas of interest
max_samples = 1e9           # maximum count of samples to consider
uncertainty = 'both'        # e.g. 'aleatoric', 'epistemic', 'both' --- only matters if ensemble==True
ensemble    = True          # whether to compute ensemble mean and var or not
pixelwise   = True          # whether to summarize errors and variances for image-based AUCE and UCE or keep pixel-based statistics
export_path = None          # where to export ensemble statistics, set to None if no writing to files is desired

# define path to find the individual ensembe member's predictions in
ensemble_paths = [os.path.join(dirname, 'inference', f'diagonal_1/export/epoch_{epoch}/{mode}'),
                  os.path.join(dirname, 'inference', f'diagonal_2/export/epoch_{epoch}/{mode}'),
                  os.path.join(dirname, 'inference', f'diagonal_3/export/epoch_{epoch}/{mode}'),
                  os.path.join(dirname, 'inference', f'diagonal_4/export/epoch_{epoch}/{mode}'), 
                  os.path.join(dirname, 'inference', f'diagonal_5/export/epoch_{epoch}/{mode}'),
                  ]

n_ensemble = len(ensemble_paths)
print('Ensembling over model predictions:')
for instance in ensemble_paths: print(instance)

if export_path:
    plot_dir    = os.path.join(export_path, 'plots', f'epoch_{epoch}', f'{mode}')
    export_dir  = os.path.join(export_path, 'export', f'epoch_{epoch}', f'{mode}')


def prepare_data_multi(batch, device, batch_size=1, use_sar=True):
    in_S2     = recursive_todevice(torch.tensor(batch['input']['S2']), device)
    in_S2_td  = recursive_todevice(torch.tensor(batch['input']['S2 TD']), device)
    if batch_size>1: in_S2_td = torch.stack((in_S2_td)).T
    in_m      = recursive_todevice(torch.tensor(batch['input']['masks']), device) 
    target_S2 = recursive_todevice(torch.tensor(batch['target']['S2']), device)
    y         = target_S2 

    if use_sar: 
        in_S1 = recursive_todevice(torch.tensor(batch['input']['S1']), device)
        in_S1_td = recursive_todevice(torch.tensor(batch['input']['S1 TD']), device)
        if batch_size>1: in_S1_td = torch.stack((in_S1_td)).T
        x     = torch.cat((torch.stack(in_S1,dim=1), torch.stack(in_S2,dim=1)),dim=2)
        dates = torch.stack((torch.tensor(in_S1_td),torch.tensor(in_S2_td))).float().mean(dim=0).to(device)
    else:
        x     = in_S2 # torch.stack(in_S2,dim=1)
        dates = torch.tensor(in_S2_td).float().to(device)
    
    return x.unsqueeze(dim=0), y.unsqueeze(dim=0), in_m.unsqueeze(dim=0), dates


def main():

    # list all predictions of the first ensemble member
    dataPath = ensemble_paths[0]
    samples  = natsorted([os.path.join(dataPath, f) for f in os.listdir(dataPath) if (os.path.isfile(os.path.join(dataPath, f)) and "_pred.npy" in f)])

    # collect sample-averaged uncertainties and errors
    img_meter  = avg_img_metrics()
    vars_aleatoric = []
    errs, errs_se, errs_ae = [], [], []

    import_data_path = os.path.join(os.getcwd(), 'util', 'precomputed', f'generic_{in_time}_{mode}_{region}_s2cloudless_mask.npy')
    import_data_path = import_data_path if os.path.isfile(import_data_path) else None
    dt_test          = SEN12MSCRTS(os.path.join(root, 'SEN12MSCRTS'), split=mode, region=region, sample_type="cloudy_cloudfree" , n_input_samples=in_time, import_data_path=import_data_path)
    if len(dt_test.paths) != len(samples): raise AssertionError

    # iterate over the ensemble member's mean predictions
    for idx, sample_mean in enumerate(tqdm(samples)):
        if idx >= max_samples: break # exceeded desired sample count

        # fetch target data and cloud masks of idx-th sample from
        batch = dt_test.getsample(idx) # ... in order to compute metrics
        x, y, in_m, _ = prepare_data_multi(batch, 'cuda', batch_size=1, use_sar=False)

        try:
            mean, var = [], []
            for path in ensemble_paths: # for each ensemble member ...
                # ... load the member's mean predictions and ...
                mean.append(np.load(os.path.join(path, os.path.basename(sample_mean))))
                # ... load the member's covariance or var predictions
                sample_var  = sample_mean.replace('_pred', '_covar') 
                if not os.path.isfile(os.path.join(path, os.path.basename(sample_var))):
                    sample_var  = sample_mean.replace('_pred', '_var') 
                var.append(np.load(os.path.join(path, os.path.basename(sample_var))))
        except:
            # skip any sample for which not all members provide predictions
            # (note: we also next'ed the dataloader's sample already)
            print(f'Skipped sample {idx}, missing data.')
            continue
        mean, var = np.array(mean), np.array(var)

        # get the variances from the covariance matrix
        if len(var.shape) > 4: # loaded covariance matrix
            var = np.moveaxis(np.diagonal(var, axis1=1, axis2=2), -1, 1)

        # combine predictions

        if ensemble:
            # get ensemble estimate and epistemic uncertainty,
            # approximate 1 Gaussian by mixture parameter ensembling
            mean_ensemble = 1/n_ensemble * np.sum(mean, axis=0)
            
            if uncertainty == 'aleatoric':
                # average the members' aleatoric uncertainties
                var_ensemble  = 1/n_ensemble * np.sum(var, axis=0)
            elif uncertainty == 'epistemic':     
                # compute average variance of ensemble predictions      
                var_ensemble  = 1/n_ensemble * np.sum(mean**2, axis=0) - mean_ensemble**2
            elif uncertainty == 'both':
                # combine both
                var_ensemble  = 1/n_ensemble * np.sum(var + mean**2, axis=0) - mean_ensemble**2
            else: raise NotImplementedError
        else: mean_ensemble, var_ensemble = mean[0], var[0]

        mean_ensemble = torch.tensor(mean_ensemble).cuda()
        var_ensemble  = torch.tensor(var_ensemble).cuda()

        # compute test metrics on ensemble prediction
        extended_metrics = img_metrics(y[0], mean_ensemble.unsqueeze(dim=0),
                                       var=var_ensemble.unsqueeze(dim=0), 
                                       pixelwise=pixelwise)
        img_meter.add(extended_metrics) # accumulate performances over the entire split

        if pixelwise: # collect variances and errors
            vars_aleatoric.extend(extended_metrics['pixelwise var']) 
            errs.extend(extended_metrics['pixelwise error']) 
            errs_se.extend(extended_metrics['pixelwise se']) 
            errs_ae.extend(extended_metrics['pixelwise ae']) 
        else:
            vars_aleatoric.append(extended_metrics['mean var']) 
            errs.append(extended_metrics['error'])
            errs_se.append(extended_metrics['mean se'])
            errs_ae.append(extended_metrics['mean ae'])

        if export_path: # plot and export ensemble predictions
            plot_img(mean_ensemble.unsqueeze(dim=0), 'pred', plot_dir, file_id=idx)
            plot_img(x[0], 'in', plot_dir, file_id=idx)
            plot_img(var_ensemble.mean(dim=0, keepdims=True).expand(3, *var_ensemble.shape[1:]).unsqueeze(dim=0), 'var', plot_dir, file_id=idx)
            export(mean_ensemble[None], 'pred', export_dir, file_id=idx)
            export(var_ensemble[None], 'var', export_dir, file_id=idx)


    # compute UCE and AUCE
    uce_l2, auce_l2 = compute_uce_auce(vars_aleatoric, errs, len(vars_aleatoric), percent=5, l2=True, mode=mode, step=0)

    # no need for a running mean here
    img_meter.value()['UCE SE']  = uce_l2.cpu().numpy().item()
    img_meter.value()['AUCE SE'] = auce_l2.cpu().numpy().item()

    print(f'{mode} split image metrics: {img_meter.value()}')
    if export_path:
        np.save(os.path.join(export_path, f'pred_var_{uncertainty}.npy'), vars_aleatoric)
        np.save(os.path.join(export_path, 'errors.npy'), errs)
        save_results(img_meter.value(), export_path, split=mode)
        print(f'Exported predictions to path {export_path}')


if __name__ == "__main__":
    main()
    exit()