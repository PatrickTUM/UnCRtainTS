"""
 Python script to pre-compute cloud coverage statistics on the data of SEN12MS-CR-TS.
    The data loader performs online sampling of input and target patches depending on its flags
    (e.g.: split, region, n_input_samples, min_cov, max_cov, ) and the patches' calculated cloud coverage.
    If using sampler='random', patches can also vary across epochs to act as data augmentation mechanism.

    However, online computing of cloud masks can slow down data loading. A solution is to pre-compute
    cloud coverage an relief the dataloader from re-computing each sample, which is what this script offers. 
    Currently, pre-calculated statistics are exported in an *.npy file, a collection of which is readily
    available for download via https://syncandshare.lrz.de/getlink/fiHhwCqr7ch3X39XoGYaUGM8/splits
    
    Pre-computed statistics can be imported via the dataloader's "import_data_path" argument.
"""

import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# see: https://docs.python.org/3/library/resource.html#resource.RLIM_INFINITY
resource.setrlimit(resource.RLIMIT_NOFILE, (int(1024*1e3), rlimit[1]))

import torch
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))
from data.dataLoader import SEN12MSCRTS

# fix all RNG seeds
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)


pathify = lambda path_list: [os.path.join(*path[0].split('/')[-6:]) for path in path_list]

if __name__ == '__main__':
    # main parameters for instantiating SEN12MS-CR-TS
    root                = '/home/data/SEN12MSCRTS'                              # path to your copy of SEN12MS-CR-TS
    split               = 'test'                                                # ROI to sample from, belonging to splits [all | train | val | test]
    input_t             = 3                                                     # number of input time points to sample (irrelevant if choosing sample_type='generic')
    region              = 'all'                                                 # choose the region of data input. [all | africa | america | asiaEast | asiaWest | europa]
    sample_type         = 'generic'                                             # type of samples returned [cloudy_cloudfree | generic]
    import_data_path    = None                                                  # path to importing the suppl. file specifying what time points to load for input and output, e.g. os.path.join(os.getcwd(), 'util', '3_test_s2cloudless_mask.npy')
    export_data_path    = os.path.join(dirname, 'precomputed')                  # e.g. ...'/3_all_train_vary_s2cloudless_mask.npy'
    vary                = 'random' if split!='test' else 'fixed'                # whether to vary samples across epoch or not
    n_epochs            = 1 if vary=='fixed' or sample_type=='generic' else 30  # if not varying dates across epochs, then a single epoch is sufficient
    max_samples         = int(1e9)

    shuffle             = False
    if export_data_path is not None:                # if exporting data indices to file then need to disable DataLoader shuffling, else pdx are not sorted (they may still be shuffled when importing)
        shuffle = False                             # ---for importing, shuffling may change the order from that of the exported file (which may or may not be desired)

    sen12mscrts         = SEN12MSCRTS(root, split=split, sample_type=sample_type, n_input_samples=input_t, region=region, sampler=vary, import_data_path=import_data_path)
    # instantiate dataloader, note: worker_init_fn is needed to get reproducible random samples across runs if vary_samples=True
    # note: if using 'export_data_path' then keep batch_size at 1 (unless moving data writing out of dataloader)
    #                                   and shuffle=False (processes patches in order, but later imports can still shuffle this)
    dataloader          = torch.utils.data.DataLoader(sen12mscrts, batch_size=1, shuffle=shuffle, worker_init_fn=seed_worker, generator=g, num_workers=0)
    
    if export_data_path is not None: 
        data_pairs  = {}  # collect pre-computed dates in a dict to be exported
        epoch_count = 0   # count, for loading time points that vary across epochs
    collect_var = []      # collect variance across S2 intensities

    # iterate over data to pre-compute indices for e.g. training or testing
    start_timer = time.time()
    for epoch in range(1, n_epochs + 1):
        print(f'\nCurating indices for {epoch}. epoch.')
        for pdx, patch in enumerate(tqdm(dataloader)):
            # stop sampling when sample count is exceeded
            if pdx>=max_samples: break
            
            if sample_type == 'generic':
                # collect variances in all samples' S2 intensities, finally compute grand average variance
                collect_var.append(torch.stack(patch['S2']).var())

                if export_data_path is not None:
                    if sample_type == 'cloudy_cloudfree':
                        # compute epoch-sensitive index, such that exported dates can differ across epochs 
                        adj_pdx = epoch_count*dataloader.dataset.__len__() + pdx
                        # performs repeated writing to file, only use this for processes dedicated for exporting
                        # and if so, only use a single thread of workers (--num_threads 1), this ain't thread-safe
                        data_pairs[adj_pdx] = {'input':     patch['input']['idx'], 'target': patch['target']['idx'],
                                               'coverage':  {'input': patch['input']['coverage'],
                                                             'output': patch['output']['coverage']},
                                               'paths':     {'input':  {'S1': pathify(patch['input']['S1 path']),
                                                                        'S2': pathify(patch['input']['S2 path'])},
                                                             'output': {'S1': pathify(patch['target']['S1 path']),
                                                                        'S2': pathify(patch['target']['S2 path'])}}}
                    elif sample_type == 'generic':
                        # performs repeated writing to file, only use this for processes dedicated for exporting
                        # and if so, only use a single thread of workers (--num_threads 1), this ain't thread-safe
                        data_pairs[pdx] = {'coverage':  patch['coverage'],
                                           'paths':     {'S1': pathify(patch['S1 path']),
                                                         'S2': pathify(patch['S2 path'])}}
        if sample_type == 'generic':    
            # export collected dates
            # eiter do this here after each epoch or after all epochs
            if export_data_path is not None:
                ds = dataloader.dataset
                if os.path.isdir(export_data_path):
                    export_here = os.path.join(export_data_path, f'{sample_type}_{input_t}_{split}_{region}_{ds.cloud_masks}.npy')
                else:
                    export_here = export_data_path
                np.save(export_here, data_pairs)
                print(f'\nEpoch {epoch_count+1}/{n_epochs}: Exported pre-computed dates to {export_here}')

                # bookkeeping at the end of epoch
                epoch_count += 1

    print(f'The grand average variance of S2 samples in the {split} split is: {torch.mean(torch.tensor(collect_var))}')

    if export_data_path is not None: print('Completed exporting data.')

    # benchmark speed of dataloader when (not) using 'import_data_path' flag
    elapsed = time.time() - start_timer
    print(f'Elapsed time is {elapsed}')